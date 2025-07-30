//! Geometric filter configuration types.

use std::{marker::PhantomData, sync::Arc};

use crate::{build_hasher::ReproducibleBuildHasher, Method};

mod bitchunks;
mod buckets;
mod estimation;
mod lookup;

pub(crate) use bitchunks::*;
pub(crate) use buckets::*;
pub(crate) use estimation::*;
pub(crate) use lookup::*;
use once_cell::sync::Lazy;

/// Trait that configures geo filters and determines counting precision.
///
/// In order to achieve a desired counting precision, one has to choose a couple of parameters:
///   * granularity (defined by phi)
///   * heap size
///
/// Those parameters define a set of mathematical operations that are needed by any actual
/// implementation, like:
///   * converting an item into a bucket id.
///   * estimating the number items for an observed number of filled buckets.
///   * ...
///
/// Those conversions can be shared across multiple geo filter instances. This way, the
/// conversions can also be optimized via e.g. lookup tables without paying the cost with every
/// new geo filter instance again and again.
pub trait GeoConfig<M: Method>: Clone + Eq + Sized {
    type BucketType: IsBucketType + 'static;
    type BuildHasher: ReproducibleBuildHasher;

    /// The number of most-significant bits that are stored sparsely as positions.
    fn max_msb_len(&self) -> usize;

    /// The maximum number of bytes used to estimate the size.
    fn max_bytes(&self) -> usize;

    fn bits_per_level(&self) -> usize;

    /// The granularity of the geometric buckets.
    /// The size of the i-th bucket is determined by the formula:
    ///    (1 - phi) * phi^i
    fn phi(&self) -> f32;

    /// The granularity of the geometric buckets.
    /// See [`Self::phi`].
    fn phi_f64(&self) -> f64;

    /// A (deterministic) mapping from a hash value to a bucket id.
    /// The default mapping uses the formula:
    ///     floor(log(hash / u64::MAX) / log(phi))
    /// Actual implementations may use a lookup table instead to speed up the computation.
    fn hash_to_bucket(&self, hash: u64) -> Self::BucketType;

    /// Estimates the number of items given the count of filled buckets.
    fn expected_items(&self, buckets: usize) -> f32;

    /// If the k most significant buckets were dropped, i.e. k is the lowest included bucket id,
    /// then the estimation can simply be upscaled with the following function.
    #[inline]
    fn upscale_estimate(&self, k: usize, estimate: f32) -> f32 {
        let q = self.phi().powf(k as f32);
        (estimate + 1.0) / q - 1.0
    }
}

/// Geometric filter configuration using static lookup tables.
///
/// Configuration is determined through the generic parameters:
/// - `T`: The type used for storing bucket indices.
/// - `B`: (1 << B) many buckets will be used to cover 50% of the hash space.
/// - `BYTES`: BYTES many bytes will be used to estimate size.
/// - `MSB`: MSB many of the most significant filled buckets will be encoded sparsely.
///
/// Instantiating this type may panic if `T` is too small to hold the maximum possible
/// bucket id determined by `B`, or `B` is larger than the largest statically defined
/// lookup table.
#[derive(Clone)]
pub struct FixedConfig<
    M: Method,
    T,
    const B: usize,
    const BYTES: usize,
    const MSB: usize,
    H: ReproducibleBuildHasher,
> {
    _phantom: PhantomData<(M, T, H)>,
}

impl<
        M: Method + Lookups,
        T: IsBucketType + 'static,
        const B: usize,
        const BYTES: usize,
        const MSB: usize,
        H: ReproducibleBuildHasher,
    > GeoConfig<M> for FixedConfig<M, T, B, BYTES, MSB, H>
{
    type BucketType = T;
    type BuildHasher = H;

    #[inline]
    fn max_msb_len(&self) -> usize {
        MSB
    }

    #[inline]
    fn max_bytes(&self) -> usize {
        BYTES
    }

    #[inline]
    fn bits_per_level(&self) -> usize {
        1 << B
    }

    #[inline]
    fn phi(&self) -> f32 {
        phi(B)
    }

    #[inline]
    fn phi_f64(&self) -> f64 {
        phi_f64(B)
    }

    /// Maps a 64-bit hash into a bit index where bit buckets follow a geometric distribution.
    /// The mapping is monotonically decreasing, i.e. u64::MAX gets mapped to bit 0.
    #[inline]
    fn hash_to_bucket(&self, hash: u64) -> Self::BucketType {
        Self::BucketType::from_usize(M::get_lookups()[B].hash_to_bucket.lookup(hash))
    }

    #[inline]
    fn expected_items(&self, buckets: usize) -> f32 {
        M::get_lookups()[B].estimation.interpolate(buckets)
    }
}

pub trait Lookups {
    fn get_lookups() -> &'static [Lazy<Lookup>];
    fn new_lookup(b: usize) -> Lookup;
}

pub struct Lookup {
    pub(crate) hash_to_bucket: HashToBucketLookup,
    pub(crate) estimation: EstimationLookup,
}

impl<
        M: Method + Lookups,
        T: IsBucketType,
        const B: usize,
        const BYTES: usize,
        const MSB: usize,
        H: ReproducibleBuildHasher,
    > Default for FixedConfig<M, T, B, BYTES, MSB, H>
{
    fn default() -> Self {
        assert_bucket_type_large_enough::<T>(B);
        assert_buckets_within_estimation_bound(B, BYTES * BITS_PER_BYTE);

        assert!(
            B < M::get_lookups().len(),
            "B = {} is not available for fixed config, requires B < {}",
            B,
            M::get_lookups().len()
        );

        Self {
            _phantom: PhantomData,
        }
    }
}

impl<
        M: Method + Lookups,
        T: IsBucketType,
        const B: usize,
        const BYTES: usize,
        const MSB: usize,
        H: ReproducibleBuildHasher,
    > PartialEq for FixedConfig<M, T, B, BYTES, MSB, H>
{
    fn eq(&self, _other: &Self) -> bool {
        H::debug_assert_hashers_eq();

        // The values of the fixed config are provided at compile time
        // so no runtime computation is required
        true
    }
}

impl<
        M: Method + Lookups,
        T: IsBucketType,
        const B: usize,
        const BYTES: usize,
        const MSB: usize,
        H: ReproducibleBuildHasher,
    > Eq for FixedConfig<M, T, B, BYTES, MSB, H>
{
}

/// Geometric filter configuration using dynamic lookup tables.
#[derive(Clone)]
pub struct VariableConfig<M: Method, T, H: ReproducibleBuildHasher> {
    b: usize,
    bytes: usize,
    msb: usize,
    _phantom: PhantomData<(M, T, H)>,
    lookup: Arc<Lookup>,
}

impl<M: Method, T, H: ReproducibleBuildHasher> Eq for VariableConfig<M, T, H> {}

impl<M: Method, T, H: ReproducibleBuildHasher> PartialEq for VariableConfig<M, T, H> {
    fn eq(&self, other: &Self) -> bool {
        H::debug_assert_hashers_eq();

        self.b == other.b && self.bytes == other.bytes && self.msb == other.msb
    }
}

impl<M: Method + Lookups, T: IsBucketType, H: ReproducibleBuildHasher> VariableConfig<M, T, H> {
    /// Returns a new configuration value. See [`FixedConfig`] for the meaning
    /// of the parameters. This functions computes a new lookup table every time
    /// it is invoked, so make sure to share the resulting value as much as possible.
    pub fn new(b: usize, bytes: usize, msb: usize) -> Self {
        assert_bucket_type_large_enough::<T>(b);
        assert_buckets_within_estimation_bound(b, bytes * BITS_PER_BYTE);
        Self {
            b,
            bytes,
            msb,
            _phantom: PhantomData,
            lookup: Arc::new(M::new_lookup(b)),
        }
    }

    #[inline]
    pub fn b(&self) -> usize {
        self.b
    }
}

impl<M: Method, T: IsBucketType + 'static, H: ReproducibleBuildHasher> GeoConfig<M>
    for VariableConfig<M, T, H>
{
    type BucketType = T;
    type BuildHasher = H;

    #[inline]
    fn max_msb_len(&self) -> usize {
        self.msb
    }

    #[inline]
    fn max_bytes(&self) -> usize {
        self.bytes
    }

    #[inline]
    fn bits_per_level(&self) -> usize {
        1 << self.b
    }

    #[inline]
    fn phi(&self) -> f32 {
        phi(self.b)
    }

    #[inline]
    fn phi_f64(&self) -> f64 {
        phi_f64(self.b)
    }

    /// Maps a 64-bit hash into a bit index where bit buckets follow a geometric distribution.
    /// The mapping is monotonically decreasing, i.e. u64::MAX gets mapped to bit 0.
    #[inline]
    fn hash_to_bucket(&self, hash: u64) -> Self::BucketType {
        Self::BucketType::from_usize(self.lookup.hash_to_bucket.lookup(hash))
    }

    #[inline]
    fn expected_items(&self, buckets: usize) -> f32 {
        self.lookup.estimation.interpolate(buckets)
    }
}

#[inline]
/// Returns phi for the given bit count `b`.
pub(crate) fn phi(b: usize) -> f32 {
    0.5f32.powf(1.0 / (1 << b) as f32)
}

#[inline]
/// Returns phi for the given bit count `b`.
pub(crate) fn phi_f64(b: usize) -> f64 {
    0.5f64.powf(1.0 / (1 << b) as f64)
}

#[allow(dead_code)]
#[inline]
/// Returns the bucket for a given hash code.
pub(crate) fn hash_to_bucket(phi: f64, hash: u64) -> usize {
    (hash as f64 / u64::MAX as f64).log(phi).floor() as usize
}

/// n specifies the zero-based index of the desired one bit where ones are enumerated from least significant
/// to most significant bits.
/// The function returns the zero-based bit index of that one bit.
/// If no such one bit exist, the function will return 64.
#[cfg(target_arch = "x86_64")]
pub(crate) fn nth_one(value: u64, n: u32) -> u32 {
    unsafe { std::arch::x86_64::_pdep_u64(1 << n, value).trailing_zeros() }
}

/// n specifies the zero-based index of the desired one bit where ones are enumerated from least significant
/// to most significant bits.
/// The function returns the zero-based bit index of that one bit.
/// If no such one bit exist, the function will return 64.
#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn nth_one(mut value: u64, mut n: u32) -> u32 {
    while n > 0 && value != 0 {
        value ^= 1 << value.trailing_zeros();
        n -= 1;
    }
    value.trailing_zeros()
}

/// Take a number of elements from an iterator without consuming it.
pub(crate) fn take_ref<I: Iterator>(iter: &mut I, n: usize) -> impl Iterator<Item = I::Item> + '_ {
    struct TakeRef<'a, I: Iterator>(usize, &'a mut I);
    impl<I: Iterator> Iterator for TakeRef<'_, I> {
        type Item = I::Item;
        fn next(&mut self) -> Option<Self::Item> {
            if self.0 > 0 {
                self.0 -= 1;
                self.1.next()
            } else {
                None
            }
        }
    }
    TakeRef(n, iter)
}

#[cfg(test)]
pub(crate) mod tests {
    use rand::RngCore;
    use rand_chacha::ChaCha12Rng;

    use crate::{Count, Method};

    /// Runs estimation trials and returns the average precision and variance.
    pub(crate) fn test_estimate<M: Method, C: Count<M>>(
        rnd: &mut ChaCha12Rng,
        f: impl Fn() -> C,
    ) -> (f32, f32) {
        let cnt = 10000usize;
        let mut avg_precision = 0.0;
        let mut avg_var = 0.0;
        let trials = 500;
        for _ in 0..trials {
            let mut m = f();
            // Insert cnt many random items.
            for _ in 0..cnt {
                m.push_hash(rnd.next_u64());
            }
            // Compute the relative error between estimate and actually inserted items.
            let high_precision = m.size() / cnt as f32 - 1.0;
            // Take the average over trials many attempts.
            avg_precision += high_precision / trials as f32;
            avg_var += high_precision.powf(2.0) / trials as f32;
        }
        (avg_precision, avg_var)
    }
}
