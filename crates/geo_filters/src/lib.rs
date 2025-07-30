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
        let size = self.size_real().round();
        debug_assert_f32s_in_range(size);
        size as usize
    }

    /// Return the estimated set size as a real number.
    fn size_real(&self) -> f32;

    /// Return the estimated set size when combined with the given sketch rounded to the nearest unsigned integer.
    /// If the combined set itself is not going to be used, this method is more efficient than using [`Self::push_sketch`] and [`Self::size`].
    fn size_with_sketch(&self, other: &Self) -> usize {
        let size = self.size_with_sketch_real(other).round();
        debug_assert_f32s_in_range(size);
        size as usize
    }

    /// Return the estimated set size when combined with the given sketch as a real number.
    /// If the combined set itself is not going to be used, this method is more efficient than using [`Self::push_sketch`] and [`Self::size`].
    fn size_with_sketch_real(&self, other: &Self) -> f32;

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
