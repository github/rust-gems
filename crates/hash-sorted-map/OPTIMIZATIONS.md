# HashSortedMap vs. Rust Swiss Table (hashbrown): Optimization Analysis

## Executive Summary

`HashSortedMap` is a Swiss-table-inspired hash map that uses **overflow
chaining** (instead of open addressing), **SIMD group scanning** (NEON/SSE2),
and an **optimized growth strategy**. It is generic over key type, value type,
and hash builder.

This document analyzes the design trade-offs versus
[hashbrown](https://github.com/rust-lang/hashbrown) and records the
experimental results that guided the current design.

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

### 7. Slot Hint Fast Path ⚠️ Removed from Lookup Paths

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

**Results of removing slot_hint from different paths:**
- `find_or_insertion_slot` (entry API): **−25% latency** on merge benchmark
- `get_hashed`: **−4.4%** improvement (SIMD scan is faster than branch+scalar)
- `insert_hashed`: **+7%** regression on presized insert (the hint genuinely
  helps when inserting into a mostly-empty group), but accepted for code
  simplicity since the merge workload matters more

**Current state**: slot_hint is **only** used in `insert_for_grow()`, where
the map is guaranteed sparse after a resize (groups are mostly empty, so the
hint slot is very likely free). For all other paths, SIMD-only scanning is
used.

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
| Slot hint removal (entry/get)   | **−25%** merge latency              |
| Branch hints (scalar only)      | **−2–6%**                           |
| IdentityHasher fix              | Enabled fair comparison             |

---

## Benchmark Results (Apple M-series, aarch64 NEON)

### Insert (1000 trigrams, pre-sized)

| Implementation       | Time (µs) | vs hashbrown |
|----------------------|-----------|--------------|
| FoldHashMap          | 2.44      | −11%         |
| FxHashMap            | 2.61      | −5%          |
| hashbrown+Identity   | 2.63      | baseline     |
| hashbrown::HashMap   | 2.74      | +4%          |
| std::HashMap+FNV     | 3.18      | +21%         |
| AHashMap             | 3.38      | +29%         |
| **HashSortedMap**    | **3.46**  | **+32%**     |
| std::HashMap         | 8.65      | +229%        |

### Reinsert (1000 trigrams, all keys exist)

| Implementation       | Time (µs) |
|----------------------|-----------|
| hashbrown+Identity   | 2.50      |
| **HashSortedMap**    | **2.70**  |

### Growth (128 → 1000 trigrams, 3 resize rounds)

| Implementation       | Time (µs) |
|----------------------|-----------|
| **HashSortedMap**    | **5.35**  |
| hashbrown+Identity   | 10.12     |

### Count (4000 trigrams, mixed insert/update)

| Implementation                   | Time (µs) |
|----------------------------------|-----------|
| hashbrown+Identity entry()       | 4.89      |
| **HashSortedMap entry().or_default()** | **5.44** |
| **HashSortedMap get_or_default** | **5.48**  |

### Iteration (1000 trigrams)

| Implementation                | Time (ns) |
|-------------------------------|-----------|
| **HashSortedMap iter()**      | **794**   |
| **HashSortedMap into_iter()** | **998**   |
| hashbrown+Identity iter()     | 1,067     |
| hashbrown+Identity into_iter()| 1,060     |

### Sort (100K trigrams)

| Implementation              | Time (µs) |
|-----------------------------|-----------|
| **HashSortedMap sort_by_hash** | **706** |
| Vec::sort_unstable          | 984       |

### Merge (100 maps × 100K keys each → sorted output)

| Implementation                    | Time (ms) | vs HSM merge+sort |
|-----------------------------------|-----------|--------------------|
| hashbrown merge presized          | 30.4      | −46%               |
| **HashSortedMap merge presized**  | **37.3**  | **−33%**           |
| **HashSortedMap merge (no sort)** | **44.0**  | **−21%**           |
| hashbrown merge                   | 45.4      | −19%               |
| **HashSortedMap merge + sort**    | **55.9**  | **baseline**       |
| hashbrown merge + Vec sort        | 58.7      | +5%                |
| k-way merge sorted vecs           | 445       | +696%              |

**Key takeaways:**
- HashSortedMap has **2× faster growth** than hashbrown
- **25% faster iteration** than hashbrown (dense group layout)
- **sort_by_hash is 28% faster** than Vec::sort_unstable (data is partially sorted by group)
- **merge + sort is 5% faster** than hashbrown merge + Vec sort (the primary use case)
- Pre-sized insert is 32% slower than hashbrown (trade-off for sort/merge efficiency)
