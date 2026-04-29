# hashmap-bench

Benchmarks comparing `PrefixHashMap` — an insertion-only hash map with
overflow chaining and SIMD group scanning — against Rust's standard library
and several third-party hash map implementations.

## Design

`PrefixHashMap<K, V, S>` is a Swiss-table-inspired hash map that uses:

- **Overflow chaining** instead of open addressing — groups that fill up link
  to overflow groups rather than probing into neighbours.
- **Slot hint** — a preferred slot index derived from the hash, checked before
  scanning the group. Gives a direct hit on most inserts at low load.
- **SIMD group scanning** — uses NEON on aarch64, SSE2 on x86\_64, and a
  scalar fallback elsewhere to scan 8–16 control bytes in parallel.
- **AoS group layout** — each group stores its control bytes, keys, and values
  together, keeping a single insert's data within 1–2 cache lines.
- **Optimized growth** — during resize, elements are re-inserted without
  duplicate checking and copied via raw pointers.
- **Generic key/value/hasher** — supports any `K: Hash + Eq`, any
  `S: BuildHasher`, and `Borrow<Q>`-based lookups.

## Benchmark results

All benchmarks insert 1000 random trigram hashes (scrambled with
`folded_multiply`) into maps with various configurations. Measured on Apple
M-series (aarch64).

### Insert 1000 trigrams — pre-sized, no growth

| Rank | Map | Time (µs) | vs best |
|------|-----|-----------|---------|
| 🥇 | FoldHashMap | 2.44 | — |
| 🥈 | FxHashMap | 2.61 | +7% |
| 🥉 | hashbrown::HashMap | 2.67 | +9% |
| 4 | **PrefixHashMap** | **2.71** | +11% |
| 5 | hashbrown+Identity | 2.74 | +12% |
| 6 | std::HashMap+FNV | 3.27 | +34% |
| 7 | AHashMap | 3.22 | +32% |
| 8 | GxHashMap | 3.69 | +51% |
| 9 | std::HashMap | 8.49 | +248% |

### Re-insert same keys (all overwrites)

| Map | Time (µs) |
|-----|-----------|
| **PrefixHashMap** | **2.36** ✅ |
| hashbrown+Identity | 2.58 |

### Growth from small (`with_capacity(128)`, 3 resize rounds)

| Map | Time (µs) | Growth penalty |
|-----|-----------|----------------|
| **PrefixHashMap** | **4.85** | +2.14 |
| hashbrown+Identity | 9.77 | +7.03 |

### Key takeaways

- **PrefixHashMap matches the fastest hashbrown configurations** on pre-sized
  first-time inserts and is **the fastest for overwrites**.
- **Growth is ~2× faster** than hashbrown thanks to the optimized
  `insert_for_grow` path that skips duplicate checking and uses raw copies.
- The remaining gap to FoldHashMap (~11%) comes from foldhash's extremely
  efficient hash function that pipelines well with hashbrown's SIMD scan.

## Running

```sh
cargo bench --bench hashmap_insert
```
