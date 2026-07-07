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

let estimated_size = c1.size_with_sketch(&c2);
assert_eq!(estimated_size, 3);
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

## Sorting filters by masked similarity

`GeoDiffCount::cmp_masked` compares two filters after applying a repeated bit mask, a locality-sensitive projection that induces a total order in which similar filters end up close together.
Sorting or bucketing a large collection this way is dominated by the cost of the pairwise `cmp_masked` calls.

`GeoDiffCount::masked_sort_key` builds a compact `u64` sort key from the most significant masked bits of a filter.
Comparing two keys numerically reproduces the `cmp_masked` order whenever the keys differ, so a sort only falls back to the (much slower) `cmp_masked` on the rare key ties:

```rust
use geo_filters::diff_count::GeoDiffCount7;
use geo_filters::Count;

// A repeated 20-bit mask with 3 bits set acts as a locality-sensitive projection.
const MASK: u64 = 0b0000_0100_0000_1000_0001;
const MASK_SIZE: usize = 20;

let filters: Vec<_> = (0..5u64)
    .map(|s| {
        let mut f = GeoDiffCount7::default();
        (0..1000).for_each(|i| f.push(s * 1000 + i));
        f
    })
    .collect();

// Precompute one key per filter, then sort by comparing keys, only using the exact
// `cmp_masked` to break ties.
let keys: Vec<u64> = filters.iter().map(|f| f.masked_sort_key(MASK, MASK_SIZE)).collect();
let mut order: Vec<usize> = (0..filters.len()).collect();
order.sort_by(|&i, &j| {
    keys[i]
        .cmp(&keys[j])
        .then_with(|| filters[i].cmp_masked(&filters[j], MASK, MASK_SIZE))
});
assert_eq!(order.len(), filters.len());
```

Sorting 1,000 filters of 1,000 items each with the 20-bit mask above, measured with `cargo bench --bench masked_sort`:

| operation | `GeoDiffConfig7` | `GeoDiffConfig13` |
| --- | --- | --- |
| sort via `cmp_masked` | 675 µs | 862 µs |
| sort via `masked_sort_key` (incl. key construction) | 25 µs | 18 µs |
| speed-up | ~27× | ~47× |
| key construction only | 13 µs | 7 µs |

## Nearest-neighbor similarity metric

`GeoDiffCount` also doubles as a compact similarity sketch. The number of differing one-bits between
two filters — the Hamming distance of their bit representations — is a simple, uncalibrated distance
that grows with the true symmetric-difference size. This is useful for nearest-neighbor search, e.g.
to find the most similar repository or document.

[`GeoDiffMetric`](https://docs.rs/geo_filters/latest/geo_filters/diff_count/struct.GeoDiffMetric.html)
wraps a `GeoDiffCount`, caches its one-bit count, and implements the `MetricSpace` trait:

- `size()` returns the filter's one-bit count as a `Metric` value;
- `symmetric_diff_size(other, bound)` returns the exact one-bit distance to another filter, but
  abandons the computation once it reaches `bound` (returning an "infinite" value), so far-away
  candidates are rejected cheaply while scanning a candidate list;
- `size().abs_diff(&other.size())` is an `O(1)` reverse-triangle lower bound on that distance.

```rust
use geo_filters::diff_count::{GeoDiffCount7, GeoDiffMetric, OnesMetric};
use geo_filters::{Count, Metric, MetricSpace};

let mut a = GeoDiffCount7::default();
(0..1000u64).for_each(|i| a.push(i));
let mut b = GeoDiffCount7::default();
(500..1500u64).for_each(|i| b.push(i));

let a = GeoDiffMetric::new(a);
let b = GeoDiffMetric::new(b);

// O(1) lower bound from the cached sizes, then the exact distance (unbounded).
let lower_bound = a.size().abs_diff(&b.size());
let distance = a.symmetric_diff_size(&b, OnesMetric::infinite());
assert!(lower_bound <= distance);
```

Time to compare a query filter (1M items) with a candidate, given the distance to a known near
neighbor as the pruning bound. A `far` candidate is disjoint (farther than the bound, so it is
rejected), while a `near` candidate is a closer near-duplicate (below the bound, so it becomes the
new best and is scanned in full). Release build; absolute numbers are hardware-dependent:

| configuration     | candidate | `estimate` | `symmetric_diff_size` | `symmetric_diff_size` (capped) |
| ----------------- | --------- | ---------- | --------------------- | ------------------------------ |
| `GeoDiffConfig7`  | far       | 115 ns     | 60 ns                 | 12 ns                          |
| `GeoDiffConfig7`  | near      | 149 ns     | 50 ns                 | 51 ns                          |
| `GeoDiffConfig10` | far       | 633 ns     | 374 ns                | 74 ns                          |
| `GeoDiffConfig10` | near      | 918 ns     | 298 ns                | 300 ns                         |
| `GeoDiffConfig13` | far       | 4.69 µs    | 2.44 µs               | 337 ns                         |
| `GeoDiffConfig13` | near      | 6.92 µs    | 1.90 µs               | 1.92 µs                        |

`estimate` is the calibrated `size_with_sketch` estimate and `symmetric_diff_size` the exact bit
distance; the capped variant abandons once the running distance reaches the bound. For a `far`
candidate it abandons almost immediately (~7× faster than the exact distance, ~14× faster than the
estimate), whereas a `near` candidate is below the bound and therefore scanned in full — the capped
and exact costs match. Since a nearest-neighbor scan rejects far more candidates than it keeps, the
early abandon dominates the search cost. Reproduce with `cargo bench --bench nearest_neighbor`.

### Precision and configuration choice

Ranking candidates by their one-bit distance is a *noisy* estimate of the true symmetric-difference
size (calibrating it to an item count with `Metric::to_f32` is order-preserving, so it does not
change the ranking). The **relative error** below is the standard deviation of `estimate / true − 1`
— i.e. relative to the true number of differing items. It is smallest for small differences and grows
**slowly, like √(ln n)**, as the difference size `n` increases: the dense low bits saturate to a
random ½ (encoding only the *parity* of many items, so they add variance without carrying size
information). Modelling the buckets as independent Bernoulli variables, the count's variance is
exactly `½·expected_diff_buckets(2n)`, which is logarithmic in `n`; the resulting relative error is
about `√((γ + ln(4·(1−φ)·n)) / (2·S))` with `S = 2^b / (2·ln 2)` and `γ` the Euler–Mascheroni
constant. For `GeoDiffConfig7` that formula gives ~8.6% at `n = 100`, ~18% at `10⁴`, ~24% at `10⁶`,
close to the measured ~8% / ~18% / ~27% (real-filter bucket correlations push it a little higher at
the top end).

| configuration     | relative error (1σ), small … 1M-item difference | may reorder candidates within |
| ----------------- | ----------------------------------------------- | ----------------------------- |
| `GeoDiffConfig7`  | ~8% … ~27%                                      | ~2.4×                         |
| `GeoDiffConfig10` | ~2.5% … ~6%                                     | ~1.3×                         |
| `GeoDiffConfig13` | ~0.8% … ~2%                                     | ~1.1×                         |

The one-bit count is *not* exact even for a few items — two items can hash to the same bucket — but
such collisions are rare while the difference is small, so it is a low-variance estimate there.
Compared to the standard windowed estimator `GeoDiffCount::size()` (and `size_with_sketch`), the
one-bit count matches it for small differences but is noisier for large ones: for `GeoDiffConfig7` at
a one-million-item difference it is ~27% here versus ~14% for `size()`, which reads only a fixed
window near the fringe and ignores the saturated bits. Use `size_with_sketch` when you need an
accurate absolute difference size; the one-bit metric exists for fast *ranking* of near neighbors,
where its early-abandon speed matters more than the last few percent of precision.

Because of the coarse resolution, **the metric is best used with `GeoDiffConfig10` or
`GeoDiffConfig13` (`b ≥ 10`)**. `GeoDiffConfig7` reorders candidates whose true distances lie within
roughly a factor of two of each other, so over a large candidate set it returns an *approximate*
nearest neighbor rather than the exact one — though it never promotes a genuinely far candidate over
a close one, since the noise cannot bridge more than the factor in the last column, even across
hundreds of millions of candidates.

For an exact result, use the metric as a cheap first pass to shortlist the top *k* candidates, then
re-rank the shortlist with the more robust `Count::size_with_sketch`, which does not accumulate this
variance.

## Evaluation

Accuracy and performance evaluations for the predefined filter configurations are included in the repository:

- [accuracy results](evaluation/accuracy.md)
- [performance results](evaluation/performance.md)

The predefined configurations should be sufficient for most users.
If you want to evaluate a custom configuration, or reproduce these results, see the [instructions](evaluation/README.md).
