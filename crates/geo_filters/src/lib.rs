//! This crate implements probabilistic data structures that solve the [Distinct Count Problem](https://en.wikipedia.org/wiki/Count-distinct_problem) using geometric filters.
//! Two variants are implemented, which differ in the way new elements are added to the filter:
//!
//! - [`GeoDiffCount`](diff_count::GeoDiffCount) adds elements through symmetric difference. Elements can be added and later removed.
//!   Supports estimating the size of the symmetric difference of two sets with a precision related to the estimated size and not relative to the union of the original sets.
//! - [`GeoDistinctCount`](distinct_count::GeoDistinctCount) adds elements through union. Elements can be added, duplicates are ignored. The union of two sets can be estimated with precision.
//!   Supports estimating the size of the union of two sets with a precision related to the estimated size.
//!   It has some similar properties as related filters like HyperLogLog, MinHash, etc, but uses less space.

pub mod build_hasher;
pub mod config;
pub mod diff_count;
pub mod distinct_count;
#[cfg(feature = "evaluation")]
pub mod evaluation;
#[cfg(test)]
mod test_rng;

use std::hash::Hash;
use std::ops::Add;

/// Marker trait to indicate the variant implemented by a [`Count`] instance.
pub trait Method: Clone + Eq + PartialEq + Send + Sync {}

/// Indicates a diff count estimation, which allows addition and removal of items, and combines values using symmetric difference.
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Diff {}
impl Method for Diff {}

/// Indicates a distinct count estimation, which allows addition of items, and combines values using union.
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Distinct {}
impl Method for Distinct {}

/// A comparable distance value with the few operations needed by nearest-neighbor search: addition,
/// the zero distance, an "unreachable"/infinite value, the absolute difference of two values, and
/// conversion to and from a floating-point value (the calibrated size the metric represents).
pub trait Metric: Ord + Add<Output = Self> + Sized {
    /// The zero distance.
    fn zero() -> Self;
    /// An unreachable distance, larger than any real one.
    fn infinite() -> Self;
    /// The absolute difference between two distances.
    fn abs_diff(&self, other: &Self) -> Self;
    /// The metric as a floating-point value.
    fn as_f32(&self) -> f32;
    /// The metric closest to the given floating-point value.
    fn from_f32(value: f32) -> Self;
}

/// A type that can be measured (its [`size`](MetricSpace::size)) and compared to another value of
/// the same type (their [`symmetric_diff_size`](MetricSpace::symmetric_diff_size)), both yielding a value of
/// the associated [`Metric`] type. This abstracts the filter-based similarity metrics from the
/// nearest-neighbor search that consumes them.
pub trait MetricSpace {
    /// The metric type produced by [`Self::size`] and [`Self::symmetric_diff_size`].
    type Metric: Metric;

    /// The size of this value.
    fn size(&self) -> Self::Metric;

    /// The symmetric difference between this value and `other`, abandoned once it reaches `bound`
    /// (in which case a value `>= bound`, e.g. [`Metric::infinite`], is returned). Pass
    /// [`Metric::infinite`] to always compute the exact value.
    fn symmetric_diff_size(&self, other: &Self, bound: Self::Metric) -> Self::Metric;
}

/// Trait for types solving the set cardinality estimation problem.
pub trait Count<M: Method> {
    /// Add the given hash to the set.
    fn push_hash(&mut self, hash: u64);

    /// Add the hash of the given item, computed with the configured hasher, to the set.
    fn push<I: Hash>(&mut self, item: I);

    /// Add the given sketch to this one.
    /// If only the size of the combined set is needed, [`Self::size_with_sketch`] is more efficient and should be used.
    fn push_sketch(&mut self, other: &Self);

    /// Return the estimated set size rounded to the nearest unsigned integer.
    fn size(&self) -> usize {
        let size = self.size_f32().round();
        debug_assert_f32s_in_range(size);
        size as usize
    }

    /// Return the estimated set size as a real number.
    fn size_f32(&self) -> f32;

    /// Return the estimated set size when combined with the given sketch rounded to the nearest unsigned integer.
    /// If the combined set itself is not going to be used, this method is more efficient than using [`Self::push_sketch`] and [`Self::size`].
    fn size_with_sketch(&self, other: &Self) -> usize {
        let size = self.size_with_sketch_f32(other).round();
        debug_assert_f32s_in_range(size);
        size as usize
    }

    /// Return the estimated set size when combined with the given sketch as a real number.
    /// If the combined set itself is not going to be used, this method is more efficient than using [`Self::push_sketch`] and [`Self::size`].
    fn size_with_sketch_f32(&self, other: &Self) -> f32;

    /// Returns the number of bytes in memory used to represent this filter.
    fn bytes_in_memory(&self) -> usize;
}

#[inline]
fn debug_assert_f32s_in_range(v: f32) {
    // The geometric filter should never produce these values.
    // These assertions failing indicates that there is a bug.
    debug_assert!(v.is_finite(), "Estimated size must be finite, got {v}");
    debug_assert!(v >= 0.0, "Estimated size must be non-negative, got {v}");
    debug_assert!(
        v <= usize::MAX as f32,
        "Estimated size {v} exceeds usize::MAX",
    );
}

#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDocTests;
