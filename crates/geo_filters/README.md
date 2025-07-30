# Geometric Counting Filters

This crate implements probabilistic data structures that solve the [Distinct Count Problem](https://en.wikipedia.org/wiki/Count-distinct_problem) using geometric filters.
Two variants are implemented, which differ in the way new elements are added to the filter:

- `GeoDiffCount` adds elements through symmetric difference. Elements can be added and later removed.
  Supports estimating the size of the symmetric difference of two sets with a precision related to the estimated size and not relative to the union of the original sets.
- `GeoDistinctCount` adds elements through union. Elements can be added, duplicates are ignored. The union of two sets can be estimated with precision.
  Supports estimating the size of the union of two sets with a precision related to the estimated size.
  It has some similar properties as related filters like HyperLogLog, MinHash, etc, but uses less space.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
geo_filters = "0.1"
```

A simple example using a default configuration for distinct count:

```rust
use geo_filters::distinct_count::GeoDistinctCount13;
use geo_filters::Count;

let mut c1 = GeoDistinctCount13::default();
c1.push(1);
c1.push(2);

let mut c2 = GeoDistinctCount13::default();
c2.push(2);
c2.push(3);

let estimated_size = c1.size_with_sketch_real(&c2);
assert!(estimated_size >= 3.0_f32 * 0.9 &&
        estimated_size <= 3.0_f32 * 1.1);
```

## Background

The idea of using geometric filters to estimate set size was born in the project to develop GitHub's new code search [^1].
We were looking for an efficient way to approximate the similarity between the sets of files in different repositories.
Alexander Neubeck, with careful reviewing by Pavel Avgustinov, developed the math for what became the geometric diff count filter.
The filter allowed us to compute approximate minimum spanning trees based on repository similarity, which gave us repository processing schedules that reduced duplicate work and increased data sharing.
This was an important part of the new code search's indexing pipeline in production [^2].

For this open source library, Alexander Neubeck simplified the math compared to our original implementation, Hendrik van Antwerpen streamlined the code and evaluation experiments.

[^1]: <https://github.blog/2021-12-08-improving-github-code-search/>
[^2]: <https://github.blog/2023-02-06-the-technology-behind-githubs-new-code-search/>

## Comparison to HLL++

Most projects rely on [`HLL++`](https://en.wikipedia.org/wiki/HyperLogLog), a successor of the `HyperLogLog` algorithm, which is very efficient w.r.t. memory and CPU usage.
The `GeoDistinctCount` solution proposed in this crate reduces the memory consumption by another 25% for large sets and even more for smaller sets while keeping updates fast.
In contrast to `HLL++`, the proposed algorithm uses the same representation for all set sizes and is thus also conceptually simpler.
When the most significant bucket ids are encoded with variable delta encoding, then only 50% of HLL++ memory is required to achieve the same accuracy.

For simplicity, we benchmarked our `GeoDistinctCount` implementation against an existing HLL++ port to Rust which wasn't optimised for speed.
Our implementation is at least as fast as the HLL++ port for all sizes.
The `GeoDistinctCount` solution only needs to count the number of occupied buckets and the offset of the first bucket.
Both numbers are tracked during incremental updates, such that the distinct count can be computed in constant time.

## Evaluation

Accuracy and performance evaluations for the predefined filter configurations are included in the repository:

- [accuracy results](evaluation/accuracy.md)
- [performance results](evaluation/performance.md)

The predefined configurations should be sufficient for most users.
If you want to evaluate a custom configuration, or reproduce these results, see the [instructions](evaluation/README.md).
