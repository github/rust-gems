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
- **Contiguous packing** — occupied slots are always packed from position 0
  with no gaps, enabling a single `leading_zeros()` to find the next free slot.
- **SIMD group scanning** — uses NEON on aarch64, SSE2 on x86\_64, and a
  scalar fallback elsewhere to scan 8–16 control bytes in parallel.
- **AoS group layout** — each group stores its control bytes, keys, and values
  together, keeping a single insert's data within 1–2 cache lines.
- **Optimized growth** — during resize, elements are re-inserted without
  duplicate checking and copied via raw pointers.
- **Generic key/value/hasher** — keys need only `Eq` (`Ord` to sort).
  Customise hashing with the single-method [`SortingHash`] trait; any
  standard `S: BuildHasher` works out of the box via a blanket impl, and
  `Borrow<Q>`-based lookups are supported.

## Benchmark results

Latest local Criterion snapshot from this repository's
`target/criterion` outputs (lower is better):

Hardware used for this snapshot:

- CPU: Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
- Architecture: x86_64
- Topology: 1 socket, 1 core, 2 threads
- CPU frequency range: 800 MHz to 2800 MHz
- Memory: 7.8 GiB RAM

| Scenario                                     | HashSortedMap | Comparison                                | Result      |
| :------------------------------------------- | ------------: | :---------------------------------------- | :---------- |
| Insert 1000 trigrams (pre-sized)             |       9.40 µs | `std::HashMap+FoldHash`: 14.55 µs         | ~35% faster |
| Grow from capacity 128                       |      27.50 µs | `std::HashMap+Identity`: 26.66 µs         | ~3% slower  |
| Count 4000 trigrams (`entry().or_default()`) |      16.15 µs | `std::HashMap+Identity` `entry()`: 15.49 µs | ~4% slower |
| Iterate 1000 trigrams (`iter()`)             |       3.02 µs | `std::HashMap+Identity` `iter()`: 3.04 µs | ~1% faster  |
| Sort 100000 trigrams by hash                 |       1.66 ms | `Vec::sort_unstable`: 2.20 ms             | ~24% faster |
| Merge 100 sorted maps + final sort           |     152.34 ms | `std::HashMap+Identity` merge + vec sort: 193.37 ms | ~21% faster |

> Note: `std::collections::HashMap` is `hashbrown` under the hood, so the benchmark drives `std::collections::HashMap` directly with the same custom `BuildHasher`s that were previously passed to `hashbrown`.

Key takeaways:

- Pre-sized inserts, sorting, and merge+sort remain the strongest paths.
- Iteration is now roughly on par with `std::HashMap+Identity`.
- Growth and count/update workloads are currently slightly slower than
  `std::HashMap+Identity` in this run.

## Running

```sh
cargo bench --bench hashmap_insert
```
