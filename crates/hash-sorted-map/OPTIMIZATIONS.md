# HashSortedMap vs. Rust Swiss Table (hashbrown): Optimization Analysis

## Executive Summary

`HashSortedMap` is a Swiss-table-inspired hash map that uses **overflow
chaining** (instead of open addressing), **SIMD group scanning** (NEON/SSE2),
and an **optimized growth strategy**. It is generic over key type, value type,
and hash builder.

This document analyzes the design trade-offs versus
[hashbrown](https://github.com/rust-lang/hashbrown) — the Swiss-table
implementation that backs `std::collections::HashMap` — and records the
experimental results that guided the current design. The benchmark suite
drives `std::HashMap` directly with various hashers.

---

## Architecture Comparison

```
┌──────────────────────────────────────────────────────────────────┐
│                   hashbrown Swiss Table                          │
│                                                                  │
│  Single contiguous allocation (SoA):                             │
│  [Padding] [T_n ... T_1  T_0] [CT_0 CT_1 ... CT_n] [CT_extra]    │
│                data               control bytes    (mirrored)    │
│                                                                  │
│  • Open addressing, triangular probing                           │
│  • 16-byte groups (SSE2) or 8-byte groups (NEON/generic)         │
│  • EMPTY / DELETED / FULL tag states                             │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                      HashSortedMap                               │
│                                                                  │
│  Vec<Group<K,V>> where each Group (AoS):                         │
│  { ctrl: [u8; 8], keys: [MaybeUninit<K>; 8],                     │
│    values: [MaybeUninit<V>; 8], overflow: u32 }                  │
│                                                                  │
│  • Overflow chaining (linked groups)                             │
│  • 8-byte groups with NEON/SSE2/scalar SIMD scan                 │
│  • EMPTY / FULL tag states only (insertion-only, no deletion)    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Optimizations Investigated

### 1. SIMD Group Scanning ✅ Implemented

Platform-specific SIMD for control byte matching:
- **aarch64**: NEON `vceq_u8` + `vreinterpret_u64_u8` (8-byte groups)
- **x86_64**: SSE2 `_mm_cmpeq_epi8` + `_mm_movemask_epi8` (16-byte groups)
- **Fallback**: Scalar u64 zero-byte detection trick

**Benchmark result**: ~5% faster than scalar on Apple M-series. The gain is
modest because the slot-hint fast path often skips the group scan entirely.

### 2. Open Addressing with Triangular Probing ❌ Rejected

This is not really an option for this hash map, since it would prevent efficient sorting.
Additionally, we didn't observe any performance improvement in comparison to the linked overflow buffer approach.
The biggest benefit of triangular probing is that it allows a much higher load factor, i.e. reduces memory consumption which isn't our main concern though.

**Benchmark result**: **40% slower** than overflow chaining. With the AoS
layout, each group is ~112 bytes, so probing to the next group jumps over
large memory regions. Overflow chaining with the slot-hint fast path is
faster because most inserts land in the first group.

### 3. SoA Memory Layout ❌ Rejected

Tested a SoA variant (`SoaHashSortedMap`) with separate control byte and
key/value arrays, combined with triangular probing.

**Benchmark result**: **Slowest variant** — even slower than AoS open
addressing. The two-Vec SoA layout doubles TLB/cache pressure versus
hashbrown's single-allocation layout. Without the single-allocation trick,
SoA is worse than AoS for this use case.

### 4. Capacity Sizing ✅ Implemented

Without the correct sizing, there was always the penality of a grow operation.

**Fix**: Changed to ~70% max load factor. This was the **single biggest improvement** — HashSortedMap went from 2× slower to matching hashbrown.

### 5. Optimized Growth ✅ Implemented

The original `grow()` called the full `insert()` for each element (including
duplicate checking and overflow traversal). hashbrown uses:
- `find_insert_index` (skip duplicate check)
- `ptr::copy_nonoverlapping` (raw memory copy)
- Bulk counter updates

**Fix**: Added `insert_for_grow()` that skips duplicate checking, uses raw
pointer copies, and iterates occupied slots via bitmask.

**Benchmark result**: Growth is now **2× faster** than hashbrown (4.8 µs vs
9.8 µs for 3 resize rounds).

### 6. Branch Prediction Hints ⚠️ Mixed Results

Added `likely()`/`unlikely()` annotations and `#[cold] #[inline(never)]` on
the overflow path.

**Benchmark result**: Helped the scalar version (~2–6% faster) but **hurt the
SIMD version** by pessimizing NEON code generation. Removed from the SIMD
implementation, kept in the scalar version.

### 7. Slot Hint Fast Path ❌ Removed

Originally, HashSortedMap checked a preferred slot before scanning the group:
```rust
let hint = slot_hint(hash);  // 3 bits from hash → slot index
if ctrl[hint] == EMPTY { /* direct insert */ }
if ctrl[hint] == tag && keys[hint] == key { /* direct hit */ }
```

**Experimental finding**: This scalar check **hurts performance** on random
workloads. The branch predictor cannot help because random keys map to random
slots, making the hint check a 50/50 branch that pollutes the branch
predictor. SIMD-only scanning (match_tag + match_empty) is uniformly fast
regardless of key distribution.

**Structural benefit of removal**: Without the slot hint, inserts always
append to the first empty slot. This guarantees that occupied slots are
**packed contiguously from the beginning** of each group (no gaps). This
invariant enables:
- `count_occupied()`: a single `leading_zeros()` on the ctrl word replaces
  bitmask scanning to find the next free slot or count entries
- Simpler `insert_for_grow()`: just write at position `count_occupied()`
- Simpler iteration: occupied slots are always `0..count_occupied()`
- Simpler `sort_by_hash()`: no need to compact gaps before sorting

**Current state**: Slot hint is fully removed. All paths use SIMD group
scanning for lookups and `count_occupied()` for finding the insertion point.

### 8. Overflow Reserve Sizing ✅ Validated

Tested overflow reserves from 0% to 100% of primary groups:

| Reserve | Growth scenario (µs) |
|---------|----------------------|
| m/8 (12.5%, default) |  8.04   |
| m/4 (25%)            |  8.33   |
| m/2 (50%)            |  8.93   |
| m/1 (100%)           | 10.31   |
| 0 (grow immediately) |  6.96   |

**Conclusion**: Smaller reserves are faster — growing early is cheaper than
traversing overflow chains.

### 9. IdentityHasher Fix ✅ Implemented

The original `IdentityHasher` zero-extended u32 to u64, putting zeros in the
top 32 bits. Since hashbrown derives the 7-bit tag from `hash >> 57`, every
entry got the same tag — completely defeating control byte filtering.

**Fix**: Use `folded_multiply` to expand u32 keys to u64 with independent
entropy in both halves. Also changed trigram generation to use
`folded_multiply` instead of murmur3.

---

## Optimizations Not Implemented (and Why)

| Optimization                    | Reason                                   |
|---------------------------------|------------------------------------------|
| **Tombstone / DELETED support** | Insertion-only map — no deletions needed |
| **In-place rehashing**          | No tombstones to reclaim                 |
| **Control byte mirroring**      | Not needed with overflow chaining (no wrap-around) |
| **Custom allocator support**    | Out of scope for benchmarking            |
| **Over-allocation utilization** | Uses `Vec` (no raw allocator control)    |

---

## Summary of Impact

| Change                          | Effect                              |
|---------------------------------|-------------------------------------|
| Capacity sizing fix             | **−50%** insert time (biggest win)  |
| Optimized growth path           | **2× faster** growth than hashbrown |
| SIMD group scanning             | **−5%** insert time                 |
| Slot hint removal               | **−25%** merge latency, contiguous packing |
| Branch hints (scalar only)      | **−2–6%**                           |
| IdentityHasher fix              | Enabled fair comparison             |

---

## Benchmark Results (local x86_64 snapshot)

Hardware used for the current local snapshot:

- CPU: Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
- Architecture: x86_64
- Topology: 1 socket, 1 core, 2 threads
- CPU frequency range: 800 MHz to 2800 MHz
- Memory: 7.8 GiB RAM

### Insert (1000 trigrams, pre-sized)

| Implementation              | Time (µs) | vs `std::HashMap+Identity` |
|-----------------------------|-----------|----------------------------|
| `std::HashMap+FoldHash`     | 13.88     | −4%                        |
| `FxHashMap`                 | 14.60     | +1%                        |
| `std::HashMap+Identity`     | 14.44     | baseline                   |
| `std::HashMap+FNV`          | 15.55     | +8%                        |
| `std::HashMap+AHash`        | 15.59     | +8%                        |
| **`HashSortedMap`**         | **9.40**  | **−35%**                   |
| `std::HashMap` (RandomState)| 25.26     | +75%                       |

### Reinsert (1000 trigrams, all keys exist)

| Implementation          | Time (µs) |
|-------------------------|-----------|
| **`HashSortedMap`**     | **6.59**  |
| `std::HashMap+Identity` | 6.95      |

### Growth (128 → 1000 trigrams, 3 resize rounds)

| Implementation          | Time (µs) |
|-------------------------|-----------|
| `std::HashMap+Identity` | 26.66     |
| **`HashSortedMap`**     | **27.50** |

### Count (4000 trigrams, mixed insert/update)

| Implementation                          | Time (µs) |
|-----------------------------------------|-----------|
| `std::HashMap+Identity` `entry()`       | 15.49     |
| **`HashSortedMap get_or_default`**      | **15.88** |
| **`HashSortedMap entry().or_default()`**| **16.15** |

### Iteration (1000 trigrams)

| Implementation                       | Time (µs) |
|--------------------------------------|-----------|
| **`HashSortedMap iter()`**           | **3.02**  |
| `std::HashMap+Identity` `iter()`     | 3.04      |
| **`HashSortedMap into_iter()`**      | **3.03**  |
| `std::HashMap+Identity` `into_iter()`| 3.56      |

### Sort (100K trigrams)

| Implementation                 | Time (ms) |
|--------------------------------|-----------|
| **`HashSortedMap sort_by_hash`** | **1.66** |
| `Vec::sort_unstable`           | 2.20      |

### Merge (100 maps × 100K keys each → sorted output)

| Implementation                                | Time (ms) | vs HSM merge+sort |
|-----------------------------------------------|-----------|--------------------|
| `std::HashMap+Identity` merge presized        | 160.79    | +6%                |
| **`HashSortedMap` merge presized**            | **117.01**| **−23%**           |
| **`HashSortedMap` merge (no sort)**           | **141.57**| **−7%**            |
| `std::HashMap+Identity` merge                 | 163.59    | +7%                |
| **`HashSortedMap` merge + sort**              | **152.34**| **baseline**       |
| `std::HashMap+Identity` merge + Vec sort      | 193.37    | +27%               |
| k-way merge sorted vecs                       | 445       | +192%              |

**Key takeaways:**
- Pre-sized insert is **~35% faster** than `std::HashMap+Identity`
- Reinsert and iter paths are now close to parity with `std::HashMap+Identity`
- Growth path is currently **~3% slower** than `std::HashMap+Identity`
- sort_by_hash is **~24% faster** than `Vec::sort_unstable`
- merge + sort is **~21% faster** than `std::HashMap+Identity` merge + Vec sort
