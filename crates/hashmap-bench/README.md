# hashmap-bench

Benchmarks comparing the custom `PrefixHashMap` (an insertion-only hash map for
pre-hashed `u32` keys) against Rust's standard library and several third-party
hash map implementations.

## Design

`PrefixHashMap` is a Swiss-table-inspired hash map optimized for the case where
keys are already well-distributed `u32` hashes (e.g. trigram fingerprints). It
skips the hash function entirely and uses the key bits directly for bucket
selection and tag matching.

Key design choices:

- **Overflow chaining** instead of open addressing — groups that fill up link
  to overflow groups rather than probing into neighbours.
- **Slot hint** — a preferred slot index derived from the key, checked before
  scanning the group. Gives a direct hit on most inserts at low load.
- **AoS group layout** — each group stores its control bytes, keys, and values
  together, keeping a single insert's data within 1–2 cache lines.
- **Optimized growth** — during resize, elements are re-inserted without
  duplicate checking and copied via raw pointers.

`SimdPrefixHashMap` adds platform-specific SIMD for the control byte scan
(NEON on aarch64, SSE2 on x86\_64, scalar fallback elsewhere).

## Benchmark results

All benchmarks insert 1000 random trigram hashes (scrambled with
`folded_multiply`) into maps with various configurations. Measured on Apple
M-series (aarch64).

### Insert 1000 trigrams — pre-sized, no growth

| Rank | Map | Time (µs) | vs best |
|------|-----|-----------|---------|
| 🥇 | FoldHashMap | 2.31 | — |
| 🥈 | **SimdPrefixHashMap** | **2.51** | +9% |
| 🥉 | FxHashMap | 2.65 | +15% |
| 4 | hashbrown::HashMap | 2.67 | +16% |
| 5 | hashbrown+Identity | 2.72 | +18% |
| 6 | NoHintSimd | 2.76 | +19% |
| 7 | **PrefixHashMap** | **3.00** | +30% |
| 8 | std::HashMap+FNV | 3.10 | +34% |
| 9 | AHashMap | 3.33 | +44% |
| 10 | GxHashMap | 3.74 | +62% |
| 11 | std::HashMap | 8.52 | +269% |

### Re-insert same keys (all overwrites)

| Map | Time (µs) |
|-----|-----------|
| **SimdPrefixHashMap** | **2.15** ✅ |
| hashbrown+Identity | 2.33 |
| PrefixHashMap | 3.24 |

### Growth from small (`with_capacity(128)`, 3 resize rounds)

| Map | Time (µs) | Growth cost |
|-----|-----------|-------------|
| **SimdPrefixHashMap** | **7.21** | +4.70 |
| **PrefixHashMap** | **7.68** | +4.68 |
| hashbrown+Identity | 10.05 | +7.33 |

### Overflow reserve sizing (from small, 3 resize rounds)

| Reserve | Time (µs) |
|---------|-----------|
| 0 (grow immediately) | 6.96 |
| m/8 (12.5%, default) | 8.04 |
| m/4 (25%) | 8.33 |
| m/2 (50%) | 8.93 |
| m/1 (100%) | 10.31 |
| hashbrown+Identity | 9.86 |

### Key takeaways

- **SimdPrefixHashMap beats every hashbrown variant** except FoldHashMap on
  first-time inserts, and is **the fastest** for overwrites.
- **Growth is ~40% cheaper** than hashbrown thanks to the optimized
  `insert_for_grow` path that skips duplicate checking and uses raw copies.
- **Smaller overflow reserves are faster** — growing early is cheaper than
  traversing overflow chains.
- The remaining ~9% gap to FoldHashMap comes from hashbrown's highly optimized
  code generation (branch hints, `#[cold]` paths, monomorphization reduction)
  and its SoA memory layout advantage for SIMD group scans.

## Running

```sh
cargo bench --bench hashmap_insert
```
