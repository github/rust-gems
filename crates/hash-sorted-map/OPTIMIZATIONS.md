# HashSortedMap vs. Rust Swiss Table (hashbrown): Optimization Analysis

## Executive Summary

`HashSortedMap` is a Swiss-table-inspired hash map that uses **overflow
chaining** (instead of open addressing), **SIMD group scanning** (NEON/SSE2),
a **slot-hint fast path**, and an **optimized growth strategy**. It is generic
over key type, value type, and hash builder.

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
│  [Padding] [T_n ... T_1  T_0] [CT_0 CT_1 ... CT_n] [CT_extra]  │
│                data               control bytes      (mirrored) │
│                                                                  │
│  • Open addressing, triangular probing                           │
│  • 16-byte groups (SSE2) or 8-byte groups (NEON/generic)         │
│  • EMPTY / DELETED / FULL tag states                             │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                      HashSortedMap                               │
│                                                                  │
│  Vec<Group<K,V>> where each Group (AoS):                         │
│  { ctrl: [u8; 8], keys: [MaybeUninit<K>; 8],                    │
│    values: [MaybeUninit<V>; 8], overflow: u32 }                  │
│                                                                  │
│  • Overflow chaining (linked groups)                             │
│  • 8-byte groups with NEON/SSE2/scalar SIMD scan                 │
│  • EMPTY / FULL tag states only (insertion-only, no deletion)    │
│  • Slot-hint fast path                                           │
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

Tested an open-addressing variant (`OpenHashSortedMap`) with triangular
probing over AoS groups.

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

The original `with_capacity` allocated `capacity / 8` groups, giving ~100%
slot utilization. hashbrown uses `capacity * 8 / 7`, giving ~50% load.

**Fix**: Changed to `capacity * 8 / 7` (87.5% max load factor), matching
hashbrown. This was the **single biggest improvement** — HashSortedMap went
from 2× slower to matching hashbrown.

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

### 7. Slot Hint Fast Path (Unique to HashSortedMap)

HashSortedMap checks a preferred slot before scanning the group:
```rust
let hint = slot_hint(hash);  // 3 bits from hash → slot index
if ctrl[hint] == EMPTY { /* direct insert */ }
if ctrl[hint] == tag && keys[hint] == key { /* direct hit */ }
```

hashbrown does **not** have this optimization — it always does a full SIMD
group scan. At ~50% load, the hint hits ~58% of the time, avoiding the scan
entirely.

### 8. Overflow Reserve Sizing ✅ Validated

Tested overflow reserves from 0% to 100% of primary groups:

| Reserve | Growth scenario (µs) |
|---------|---------------------|
| m/8 (12.5%, default) | 8.04 |
| m/4 (25%) | 8.33 |
| m/2 (50%) | 8.93 |
| m/1 (100%) | 10.31 |
| 0 (grow immediately) | 6.96 |

**Conclusion**: Smaller reserves are faster — growing early is cheaper than
traversing overflow chains. The `m/8` default implicitly enforces ~62.5% max
load, which aligns with the mathematical analysis (Poisson model, 3σ
confidence).

### 9. IdentityHasher Fix ✅ Implemented

The original `IdentityHasher` zero-extended u32 to u64, putting zeros in the
top 32 bits. Since hashbrown derives the 7-bit tag from `hash >> 57`, every
entry got the same tag — completely defeating control byte filtering.

**Fix**: Use `folded_multiply` to expand u32 keys to u64 with independent
entropy in both halves. Also changed trigram generation to use
`folded_multiply` instead of murmur3.

---

## Optimizations Not Implemented (and Why)

| Optimization | Reason |
|---|---|
| **Tombstone / DELETED support** | Insertion-only map — no deletions needed |
| **In-place rehashing** | No tombstones to reclaim |
| **Control byte mirroring** | Not needed with overflow chaining (no wrap-around) |
| **Custom allocator support** | Out of scope for benchmarking |
| **Over-allocation utilization** | Uses `Vec` (no raw allocator control) |

---

## Summary of Impact

| Change | Effect on insert time |
|---|---|
| Capacity sizing fix (`*8/7`) | **−50%** (biggest win) |
| Optimized growth path | **−10%** on growth scenarios |
| SIMD group scanning | **−5%** |
| Branch hints (scalar only) | **−2–6%** |
| IdentityHasher fix | Enabled fair comparison |

The current HashSortedMap **matches hashbrown+FxHash** on pre-sized inserts,
**beats all hashbrown variants** on overwrites, and has **2× faster growth**.
