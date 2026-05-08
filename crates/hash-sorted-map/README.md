# hash-sorted-map

A hash map whose groups are ordered by hash prefix, enabling efficient
sorted-order iteration and linear-time merging of two maps.

## Motivation

In a search index, each document produces a **term map** (term → frequency).
At index time, term maps from many documents must be **merged** into a single
posting list, and the result is **serialized in hash-key order** so that
lookups can use a skip-list approach, leveraging the hash ordering to
efficiently jump to the right region of the serialized data.

A conventional hash map stores entries in arbitrary order, so merging two maps
requires collecting, sorting, and reshuffling all entries — an expensive step
that dominates indexing time for large term maps typical of code search, where
documents contain massive numbers of tokens.

`HashSortedMap` avoids this by organizing its groups by hash prefix.
Iterating through the groups in order yields entries sorted by their hashed
keys, which means:

- **Merging** two maps is a single linear scan (like merge-sort's merge step).
- **Serialization** in hash-key order requires no extra sorting or copying.

## Design

`HashSortedMap<K, V, S>` is a Swiss-table-inspired hash map that uses:

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

Latest local Criterion snapshot from this repository's
`target/criterion` outputs (lower is better):

| Scenario                                     | HashSortedMap | Comparison                             | Result      |
| :------------------------------------------- | ------------: | :------------------------------------- | :---------- |
| Insert 1000 trigrams (pre-sized)             |       7.34 µs | hashbrown::HashMap: 12.88 µs           | ~43% faster |
| Grow from capacity 128                       |      20.54 µs | hashbrown+Identity: 23.17 µs           | ~11% faster |
| Count 4000 trigrams (`entry().or_default()`) |      12.70 µs | hashbrown+Identity `entry()`: 13.53 µs | ~6% faster  |
| Iterate 1000 trigrams (`iter()`)             |       3.93 µs | hashbrown+Identity `iter()`: 2.87 µs   | ~37% slower |
| Sort 100000 trigrams by hash                 |       1.83 ms | `Vec::sort_unstable`: 2.09 ms          | ~12% faster |
| Merge 100 sorted maps + final sort           |     161.93 ms | hashbrown merge + vec sort: 234.70 ms  | ~31% faster |

Key takeaways:

- `HashSortedMap` is strongest on insert-heavy and merge/sort-heavy paths.
- Iteration throughput is currently behind `hashbrown+Identity`.
- In workloads that need deterministic hash-order serialization, the merge and
  sort advantages can outweigh the iteration gap.

## Running

```sh
cargo bench --bench hashmap_insert
```
