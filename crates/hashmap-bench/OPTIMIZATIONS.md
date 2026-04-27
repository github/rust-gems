# Missing Optimizations in PrefixHashMap vs. Rust Swiss Table (hashbrown)

## Executive Summary

The `PrefixHashMap` in this repository is a minimal, insertion-only hash map specialized for pre-hashed `u32` keys. While it borrows the core Swiss table concept of control-byte-based group scanning, it omits a large number of optimizations present in the production [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) Swiss table implementation. The most impactful missing optimizations are: **SIMD-accelerated group scanning** (SSE2/NEON), **open-addressing with triangular probing** (instead of overflow chaining), **SoA memory layout** separating control bytes from data for cache efficiency, **in-place rehashing** to reclaim tombstones, **DELETED tombstone support** for element removal, and **over-allocation utilization**. This report catalogs every significant optimization gap across architecture, probing, memory layout, SIMD, resize strategy, and API completeness.

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────────┐
│                    hashbrown Swiss Table                          │
│                                                                   │
│   Single contiguous allocation:                                   │
│   [Padding] [T_n ... T_1  T_0] [CT_0 CT_1 ... CT_n] [CT_extra]  │
│                 data (SoA)          control bytes     (mirrored)  │
│                                                                   │
│   • Open addressing, triangular probing                           │
│   • 16-byte groups (SSE2) or 8-byte groups (NEON/generic)         │
│   • SIMD parallel group scan                                      │
│   • EMPTY / DELETED / FULL tag states                             │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│                     PrefixHashMap                                 │
│                                                                   │
│   Vec<Group> where each Group:                                    │
│   { ctrl: [u8; 8], keys: [u32; 8], values: [MaybeUninit<V>; 8],  │
│     overflow: u32 }                                               │
│                                                                   │
│   • Overflow chaining (linked Group structs)                      │
│   • Fixed 8-byte groups, scalar bit-manipulation                  │
│   • EMPTY / FULL tag states only (no DELETED)                     │
└───────────────────────────────────────────────────────────────────┘
```

---

## 1. SIMD-Accelerated Group Scanning

**Status: Missing from PrefixHashMap**

This is arguably the most impactful optimization gap.

### hashbrown

hashbrown provides three SIMD backends selected at compile time[^1]:

| Platform | Backend | Group Width | Instructions Used |
|----------|---------|-------------|-------------------|
| x86/x86_64 with SSE2 | `sse2.rs` | 16 bytes | `_mm_cmpeq_epi8`, `_mm_movemask_epi8` |
| AArch64 with NEON | `neon.rs` | 8 bytes | `vceq_u8`, `vcltz_s8`, `vreinterpret_u64_u8` |
| Fallback | `generic.rs` | 8 bytes (u64) | Scalar bit tricks |

On x86_64, the SSE2 `match_tag` compiles to just 2 instructions: a `pcmpeqb` and a `pmovmskb`, producing a 16-bit mask where each bit directly indicates a matching slot[^2]. This means **16 slots are scanned in a single operation**.

### PrefixHashMap

PrefixHashMap uses only the scalar approach, operating on 8 control bytes packed into a `u64`[^3]:

```rust
fn match_byte(ctrl: &[u8; GROUP_SIZE], byte: u8) -> u64 {
    let word = u64::from_ne_bytes(*ctrl);
    let broadcast = 0x0101010101010101u64 * (byte as u64);
    let xor = word ^ broadcast;
    (xor.wrapping_sub(0x0101010101010101)) & !xor & 0x8080808080808080
}
```

This is essentially the same algorithm as hashbrown's `generic.rs` fallback[^4], but:

- **Fixed at 8-byte groups** — never benefits from the SSE2 16-byte group scan available on most modern x86 machines.
- **No platform-specific fast paths** — no NEON, no SSE2, no LoongArch LSX.

**Impact**: On x86_64, hashbrown scans 2× more slots per group operation using native SIMD instructions that are lower latency than the scalar bit-manipulation chain.

---

## 2. Probing Strategy: Triangular Probing vs. Overflow Chaining

**Status: Missing from PrefixHashMap**

### hashbrown

hashbrown uses **triangular probing**, a variant of open addressing where each successive probe jumps by one more group width[^5]:

```rust
struct ProbeSeq { pos: usize, stride: usize }
impl ProbeSeq {
    fn move_next(&mut self, bucket_mask: usize) {
        self.stride += Group::WIDTH;
        self.pos += self.stride;
        self.pos &= bucket_mask;
    }
}
```

This is mathematically guaranteed to visit every group exactly once in a power-of-two-sized table[^6]. All probing occurs within a single contiguous allocation, enabling excellent spatial locality.

### PrefixHashMap

PrefixHashMap uses **overflow chaining**: when a primary group is full, an overflow group is allocated at the end of the `Vec<Group>` and linked via an index pointer[^7]:

```rust
overflow: u32, // index into groups vec, or NO_OVERFLOW
```

**Missing benefits of triangular probing**:

- **Spatial locality**: Triangular probing accesses nearby memory regions (the next group is typically in the same or adjacent cache line). Overflow groups are appended at the end of the vector, potentially far from the primary group.
- **No pointer chasing**: Triangular probing computes the next position arithmetically; overflow chaining follows an indirection.
- **Probe termination guarantee**: Triangular probing terminates when it encounters an EMPTY slot. Overflow chaining must check the `overflow` field and follow links.

---

## 3. Memory Layout: SoA vs. AoS

**Status: Missing from PrefixHashMap**

### hashbrown

hashbrown uses a **Structure-of-Arrays (SoA)** layout within a single allocation[^8]:

```
[Padding] [T_n, ..., T_1, T_0] [CT_0, CT_1, ..., CT_n, CT_extra...]
           ^^^ data part ^^^     ^^^ control bytes (contiguous) ^^^
```

All control bytes are stored contiguously at the end of the allocation. When probing, the initial scan only touches control bytes — the data is only accessed after a tag match. This means:

- **Control byte scans stay in L1 cache**: For a table with 1024 entries, all 1024 control bytes fit in ~1KB, likely fitting entirely in L1 cache.
- **Data is only accessed on hits**: Cache pollution from data access is minimized.

### PrefixHashMap

PrefixHashMap uses an **Array-of-Structures (AoS)** layout[^9]:

```rust
struct Group<V> {
    ctrl: [u8; 8],      // 8 bytes
    keys: [u32; 8],     // 32 bytes
    values: [MaybeUninit<V>; 8], // 8 * size_of::<V>() bytes
    overflow: u32,       // 4 bytes
}
```

For a `V` of 8 bytes (e.g., `usize`), each Group is 8 + 32 + 64 + 4 = 108 bytes (plus alignment padding). Scanning the control bytes of sequential groups requires jumping over all the key/value data, degrading cache utilization when doing multi-group probing.

---

## 4. Control Byte Mirroring for Wrap-Around

**Status: Missing from PrefixHashMap (but less needed due to overflow chaining)**

### hashbrown

hashbrown allocates `num_buckets + Group::WIDTH` control bytes. The first `Group::WIDTH` control bytes are replicated at the end[^10]:

```rust
fn set_ctrl(&mut self, index: usize, ctrl: Tag) {
    let index2 = ((index.wrapping_sub(Group::WIDTH)) & self.bucket_mask) + Group::WIDTH;
    *self.ctrl(index) = ctrl;
    *self.ctrl(index2) = ctrl;  // mirror
}
```

This ensures that a group load starting near the end of the table can safely wrap around without a branch or special case.

### PrefixHashMap

Not implemented — not needed because PrefixHashMap doesn't use open addressing. Each group is self-contained with its own control byte array.

---

## 5. Tombstone / DELETED Support and In-Place Rehashing

**Status: Missing from PrefixHashMap**

### hashbrown

hashbrown has three control byte states[^11]:

| State | Encoding | Meaning |
|-------|----------|---------|
| `EMPTY` | `0xFF` (1111_1111) | Slot never occupied or fully reclaimed |
| `DELETED` | `0x80` (1000_0000) | Tombstone — element removed, probing must continue past |
| `FULL` | `0x00..0x7F` | Occupied — top 7 bits of hash |

When elements are removed, the control byte is set to `DELETED` rather than `EMPTY`. This preserves the probe chain for other elements. When the ratio of deleted entries gets too high, hashbrown performs an **in-place rehash**[^12]:

1. Convert all FULL → DELETED, DELETED → EMPTY via `convert_special_to_empty_and_full_to_deleted()`
2. Walk through each DELETED (originally FULL) entry and swap it into its ideal position
3. If both old and new positions are in the same probe group, just update the control byte in place

This avoids a full reallocation when many deletes have fragmented the table.

### PrefixHashMap

PrefixHashMap only has two states[^13]:

| State | Encoding |
|-------|----------|
| `EMPTY` | `0x00` |
| `FULL` | `key_byte \| 0x80` |

There is **no deletion support at all** — the map is described as "insertion-only"[^14]. This means:
- No `remove()` method
- No tombstones
- No in-place rehash optimization
- If an entry needs to be removed, the entire map must be rebuilt

---

## 6. Tag / Hash Encoding

**Status: Different approach in PrefixHashMap (not necessarily worse, but different trade-offs)**

### hashbrown

Uses the **top 7 bits** of the 64-bit hash as the tag, stored with the high bit clear (range `0x00..0x7F`)[^15]:

```rust
pub(crate) const fn full(hash: u64) -> Tag {
    let top7 = hash >> (MIN_HASH_LEN * 8 - 7);
    Tag((top7 & 0x7f) as u8)
}
```

The high bit is reserved for EMPTY/DELETED sentinel detection. This means `EMPTY`/`DELETED` can be distinguished from `FULL` with a single bit test.

### PrefixHashMap

Forces bit 7 high and uses the low 7 bits of the key[^16]:

```rust
fn tag(key: u32) -> u8 {
    (key as u8) | 0x80
}
```

EMPTY is `0x00`. This inverts the hashbrown convention — FULL entries have bit 7 set, EMPTY has bit 7 clear. The `match_empty` function checks for zero bytes[^17]:

```rust
fn match_empty(ctrl: &[u8; GROUP_SIZE]) -> u64 {
    let word = u64::from_ne_bytes(*ctrl);
    !word & 0x8080808080808080
}
```

**Key difference**: PrefixHashMap cannot distinguish DELETED from FULL because all non-zero control bytes have bit 7 set. This is a deliberate simplification for the insertion-only use case.

---

## 7. Load Factor and Growth Strategy

**Status: Different and less sophisticated in PrefixHashMap**

### hashbrown

Uses an **87.5% maximum load factor** (7/8) with a `growth_left` counter[^18]:

```rust
fn bucket_mask_to_capacity(bucket_mask: usize) -> usize {
    if bucket_mask < 8 { bucket_mask }
    else { ((bucket_mask + 1) / 8) * 7 }
}
```

Growth is triggered when `growth_left` reaches 0, which tracks insertions minus the capacity. The `growth_left` field is decremented only when inserting into an EMPTY slot (not a DELETED one)[^19].

### PrefixHashMap

Uses overflow group exhaustion as the growth trigger[^20]:

```rust
let max_overflow = self.num_primary / 8 + 1;
let num_overflow = self.groups.len() as u32 - self.num_primary;
if num_overflow >= max_overflow {
    self.grow();
    return self.insert(key, value);
}
```

This reserves 12.5% extra groups for overflow. Growth happens when the overflow area is full. This is a coarser signal than hashbrown's per-slot tracking and can lead to:
- **Premature growth** if unlucky hash distribution fills overflow disproportionately
- **Delayed growth** if hash distribution is uniform (overflow area may never fill even at high load)

---

## 8. Resize Strategy

**Status: Significantly less optimized in PrefixHashMap**

### hashbrown

hashbrown's resize has multiple optimizations:

1. **Over-allocation utilization**[^21]: When the allocator returns more memory than requested, hashbrown uses the extra space for additional buckets:
   ```rust
   if block.len() != layout.size() {
       let x = maximum_buckets_in(block.len(), table_layout, Group::WIDTH);
       // Use larger capacity...
   }
   ```

2. **In-place rehashing** when fragmentation from deletions is high (described in §5).

3. **Efficient element copying** using `ptr::copy_nonoverlapping` with layout-aware size calculations[^22].

4. **Panic-safe resize** using `ScopeGuard` to ensure the old table is freed even if the hasher panics[^23].

### PrefixHashMap

PrefixHashMap's grow is simpler and less efficient[^24]:

```rust
fn grow(&mut self) {
    let old_groups = std::mem::take(&mut self.groups);
    self.n_bits += 1;
    // ... allocate new groups ...
    for group in old_groups {
        for i in 0..GROUP_SIZE {
            if group.ctrl[i] != CTRL_EMPTY {
                self.insert(key, value);  // full re-insertion
            }
        }
        std::mem::forget(group);
    }
}
```

Missing optimizations:
- **Always doubles** — no option for in-place rehash
- **Re-inserts via the public API** — each element goes through the full insert path including overflow chain traversal, whereas hashbrown uses a fast `prepare_insert_index` that skips duplicate checking
- **No over-allocation utilization**
- **Limited panic safety** — uses `mem::forget` on old groups but doesn't guard against panics during re-insertion

---

## 9. Branch Prediction Hints

**Status: Missing from PrefixHashMap**

### hashbrown

hashbrown extensively uses `likely()` and `unlikely()` hints to guide the CPU's branch predictor[^25]:

```rust
if unlikely(self.table.growth_left == 0 && old_ctrl.special_is_empty()) {
    self.reserve(1, hasher);
}
```

```rust
if likely(eq(index)) {
    return Some(index);
}
```

### PrefixHashMap

No branch hints are used anywhere in the implementation. On modern CPUs, this can affect branch prediction accuracy for cold paths like growth and overflow traversal.

---

## 10. Slot Hint / Preferred Slot

**Status: Present in PrefixHashMap but NOT in hashbrown (PrefixHashMap advantage)**

PrefixHashMap has a unique optimization not present in hashbrown: a **preferred slot hint** derived from additional hash bits[^26]:

```rust
fn slot_hint(key: u32) -> usize {
    ((key >> 7) & 0x7) as usize
}
```

Before scanning the group, PrefixHashMap first checks the preferred slot directly[^27]:

```rust
let c = group.ctrl[hint];
if c == CTRL_EMPTY {
    // Direct insert without scanning
}
if c == tag && group.keys[hint] == key {
    // Direct hit without scanning
}
```

This is a fast path that avoids the scalar group scan entirely when the preferred slot is available. hashbrown does not have this optimization — it always does a full group scan via SIMD/scalar.

---

## 11. Additional Missing Features and Optimizations

| Feature | hashbrown | PrefixHashMap |
|---------|-----------|---------------|
| Custom allocator support | Yes (`Allocator` trait)[^28] | No (uses `Vec` with global allocator) |
| ZST (Zero-Sized Type) handling | Optimized special case[^29] | Not supported |
| `#[cold]` / `#[inline(never)]` on slow paths | Yes (e.g., `reserve_rehash`)[^30] | Not used |
| `Entry` API | Full entry API | Not provided |
| Iterator support | `RawIter`, `RawDrain`, `RawIntoIter` | Not provided |
| `shrink_to` / `shrink_to_fit` | Yes | Not provided |
| Generic over key type | Yes (any `K: Hash + Eq`) | Fixed `u32` keys only |
| `remove` / `erase` | Yes, with tombstones | Not supported |
| Monomorphization reduction | Uses `dyn Fn` for inner functions[^31] | Not applicable (simpler API) |
| Small table optimization | Min capacity thresholds based on layout/group width[^32] | Minimum 2 primary groups |

---

## 12. Summary of Impact

The missing optimizations can be categorized by their likely performance impact:

### High Impact
1. **SIMD group scanning** — 2× more slots per scan on SSE2; lower-latency instructions
2. **SoA memory layout** — dramatically better cache behavior for control byte scanning
3. **Open addressing with triangular probing** — eliminates pointer chasing in overflow chains
4. **Resize without re-insertion** — hashbrown copies elements directly without re-probing

### Medium Impact
5. **In-place rehashing** — avoids allocation when table is fragmented by deletions (N/A for insert-only)
6. **Over-allocation utilization** — free extra capacity from allocator rounding
7. **Branch hints** — guides CPU branch predictor for common vs. rare paths
8. **Load factor tracking** — precise growth triggering vs. overflow-area exhaustion

### Lower Impact (or N/A for the use case)
9. **Control byte mirroring** — needed for open addressing wrap-around (not needed with chaining)
10. **Tombstone/DELETED support** — only matters if deletion is needed
11. **Custom allocators** — not needed for most use cases
12. **ZST handling** — irrelevant for `u32` keys

### PrefixHashMap Advantages (Not in hashbrown)
- **Slot hint fast path** — direct preferred-slot check before group scan
- **No hashing overhead** — keys are pre-hashed `u32` values
- **Simpler implementation** — ~250 lines vs. ~5000+ lines, easier to reason about

---

## Confidence Assessment

- **High confidence**: All claims about both implementations are verified directly from source code. The hashbrown analysis is based on the current `main` branch (commit `420e83ba`), and the PrefixHashMap analysis is from the local `crates/hashmap-bench/prefix_map.rs`.
- **Moderate confidence**: Performance impact assessments are based on algorithmic analysis and known CPU architecture properties (cache line sizes, SIMD throughput) rather than measured benchmarks. Actual impact depends on workload, key distribution, and hardware.
- **Assumption**: The PrefixHashMap is intentionally minimal — many "missing" features are deliberate design choices for simplicity in a benchmarking context, not oversights.

---

## Footnotes

[^1]: `src/control/group/mod.rs` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — compile-time cfg selection of SSE2, NEON, LSX, or generic backend
[^2]: `src/control/group/sse2.rs:80-93` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — `match_tag` using `_mm_cmpeq_epi8` + `_mm_movemask_epi8`
[^3]: `crates/hashmap-bench/prefix_map.rs:50-56` — scalar `match_byte` function
[^4]: `src/control/group/generic.rs:96-104` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — generic `match_tag` using same bit-trick
[^5]: `src/raw.rs:80-97` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — `ProbeSeq` struct and `move_next`
[^6]: Blog post cited in hashbrown source: https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n/
[^7]: `crates/hashmap-bench/prefix_map.rs:12` — `overflow: u32` field in Group struct
[^8]: `src/raw.rs` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — `TableLayout::calculate_layout_for` computes `ctrl_offset = size * buckets` (data then control bytes)
[^9]: `crates/hashmap-bench/prefix_map.rs:8-13` — Group struct with interleaved ctrl/keys/values
[^10]: `src/raw.rs` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — `set_ctrl` method mirrors control bytes
[^11]: `src/control/tag.rs:5-9` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — Tag EMPTY=0xFF, DELETED=0x80
[^12]: `src/raw.rs` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — `rehash_in_place` method
[^13]: `crates/hashmap-bench/prefix_map.rs:4` — `CTRL_EMPTY: u8 = 0x00`
[^14]: `crates/hashmap-bench/prefix_map.rs:26` — doc comment: "Insertion-only hash map"
[^15]: `src/control/tag.rs:36-47` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — `Tag::full` method
[^16]: `crates/hashmap-bench/prefix_map.rs:40-42` — `tag` function
[^17]: `crates/hashmap-bench/prefix_map.rs:59-62` — `match_empty` function
[^18]: `src/raw.rs` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — `bucket_mask_to_capacity` function
[^19]: `src/raw.rs` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — `record_item_insert_at` decrements `growth_left` only for EMPTY
[^20]: `crates/hashmap-bench/prefix_map.rs:148-154` — overflow exhaustion check triggering `grow()`
[^21]: `src/raw.rs` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — `new_uninitialized` over-allocation handling
[^22]: `src/raw.rs` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — `resize_inner` uses `ptr::copy_nonoverlapping`
[^23]: `src/raw.rs` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — `prepare_resize` returns `ScopeGuard`
[^24]: `crates/hashmap-bench/prefix_map.rs:216-241` — `grow` method
[^25]: `src/raw.rs` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — uses `likely()` / `unlikely()` from `crate::util`
[^26]: `crates/hashmap-bench/prefix_map.rs:45-47` — `slot_hint` function
[^27]: `crates/hashmap-bench/prefix_map.rs:98-114` — fast path check in `insert`
[^28]: `src/raw.rs` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — `RawTable<T, A: Allocator>`
[^29]: `src/raw.rs` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — `IS_ZERO_SIZED` special cases throughout
[^30]: `src/raw.rs` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — `#[cold] #[inline(never)]` on `reserve_rehash`
[^31]: `src/raw.rs` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — `find_inner` uses `&mut dyn FnMut(usize) -> bool`
[^32]: `src/raw.rs` in [rust-lang/hashbrown](https://github.com/rust-lang/hashbrown) — `capacity_to_buckets` with `min_cap` thresholds
