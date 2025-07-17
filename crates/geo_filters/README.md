# Geometric Counting Filters

This crate implements probabilistic data structures that solve the [Distinct Count Problem](https://en.wikipedia.org/wiki/Count-distinct_problem) using geometric filters.
Two variants are implemented, which differ in the way new elements are added to the filter:

- `GeoDiffCount` adds elements through symmetric difference. Elements can be added and later removed.
  Supports estimating the size of the symmetric difference of two sets with a precision related to the estimated size and not relative to the union of the original sets.
- `GeoDistinctCount` adds elements through union. Elements can be added, duplicates are ignored. The union of two sets can be estimated with precision.
  Supports estimating the size of the union of two sets with a precision related to the estimated size.
  It has some similar properties as related filters like HyperLogLog, MinHash, etc, but uses less space.

<details>
<summary>Data Structure Analogy</summary>
If you're not familiar with probabilistic data structures then getting an intuition for how this data structure works
using a real world example might be helpful. 

Imagine you wanted to count how many rain drops fall from the sky and landing in certain area. A single rain drop will
fall randomly hitting the ground in the area you wish to analyze. Trying to count every single drop that hits the 
concrete would be very difficult, there are simply far too many.

Perhaps you can estimate the number of rain drops?

Instead of counting each and every drop of rain you could instead lay out a grid of buckets and then count how many buckets
have *any* rain drops in them at all. For this thought experiment we're not considering how _much_ water is in them, only if
there is a non-zero amount of water. Uniformly sized buckets might work ok for a small shower, but you'd quickly run
into an issue where most of your buckets have some amount rain in them. Because of this, you would not be able to differentiate between a gentle shower and a downpour; either way most of the buckets have _some_ water in them.

By varying the size of the buckets you reduce the probability that a rain drop will land in the smaller ones. You can
then estimate the number of droplets by adding up the probabilities that a given bucket has a rain drop in it. Smaller
ones are much less likely to have a droplet in so if you've got a lot of smaller buckets with drop lets in, that would imply
that there was a lot of rain. If those buckets are mostly dry, then it would imply that there was only a small amount
of drizzle. You still need a wide range of bucket sizes to be able to tell the difference between having no rain and a small
amount of rain.

You can estimate the difference in the amount of rain fall on two areas by counting the number of buckets where the matching
bucket size has rain in it in one area but not the other.

This data structure works in a similar way. Items are hashed to produce a "random" number which we assign to a bucket. The
bucket "sizes" are arranged to follow a geometric distribution to allow us to calculate an estimate of the number of items
using well known formulas.
</details>

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
