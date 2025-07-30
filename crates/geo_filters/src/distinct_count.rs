//! Geometric filter implementation for distinct count.

use std::collections::VecDeque;
use std::hash::BuildHasher as _;
use std::mem::{size_of, size_of_val};

use crate::config::{
    count_ones_from_bitchunks, iter_bit_chunks, iter_ones, max_lsb_bytes, or_bit_chunks, take_ref,
    BitChunk, GeoConfig, IsBucketType,
};
use crate::distinct_count::bitdeque::BitDeque;
use crate::{Count, Distinct};

mod bitdeque;
mod config;

pub use config::{GeoDistinctConfig13, GeoDistinctConfig7};

/// Distinct count filter with a relative error standard deviation of ~0.065.
/// Uses at most 168 bytes of memory.
pub type GeoDistinctCount7<'a> = GeoDistinctCount<'a, GeoDistinctConfig7>;

/// Distinct count filter with a relative error standard deviation of ~0.0075.
/// Uses at most 9248 bytes of memory.
pub type GeoDistinctCount13<'a> = GeoDistinctCount<'a, GeoDistinctConfig13>;

/// Probabilistic distinct count data structure based on geometric filters.
///
/// The [`GeoDistinctCount`] falls into the category of probabilistic set size estimators.
/// It has some similar properties as related filters like HyperLogLog, MinHash, etc, but uses less space.
#[derive(Eq, PartialEq)]
pub struct GeoDistinctCount<'a, C: GeoConfig<Distinct>> {
    config: C,
    msb: VecDeque<C::BucketType>,
    lsb: BitDeque<'a>,
}

impl<C: GeoConfig<Distinct> + Default> Default for GeoDistinctCount<'_, C> {
    fn default() -> Self {
        Self::new(C::default())
    }
}

impl<C: GeoConfig<Distinct>> std::fmt::Debug for GeoDistinctCount<'_, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "~{} (msb: {:?}, |lsb|: {:?})",
            self.size(),
            self.msb,
            self.lsb.bit_range(),
        )
    }
}

impl<C: GeoConfig<Distinct>> GeoDistinctCount<'_, C> {
    pub fn new(config: C) -> Self {
        let msb = Default::default();
        let lsb = BitDeque::new(max_lsb_bytes::<C::BucketType>(
            config.max_bytes(),
            config.max_msb_len(),
        ));
        Self { config, msb, lsb }
    }

    fn from_bit_chunks<I: Iterator<Item = BitChunk>>(config: C, chunks: I) -> Self {
        let mut ones = iter_ones::<C::BucketType, _>(chunks.peekable());

        let mut msb = VecDeque::default();
        take_ref(&mut ones, config.max_msb_len() - 1).for_each(|bucket| {
            msb.push_back(bucket);
        });
        let smallest_msb = ones
            .next()
            .map(|bucket| {
                msb.push_back(bucket);
                bucket
            })
            .unwrap_or_default();

        let lsb = BitDeque::from_bit_chunks(
            ones.into_bitchunks(),
            smallest_msb.into_usize(),
            max_lsb_bytes::<C::BucketType>(config.max_bytes(), config.max_msb_len()),
        );

        let result = Self { config, msb, lsb };
        result.debug_assert_invariants();
        result
    }

    fn bit_chunks(&self) -> impl Iterator<Item = BitChunk> + '_ {
        iter_bit_chunks(
            self.msb.iter().map(|b| b.into_usize()),
            self.lsb.bit_chunks(),
        )
    }

    pub fn union(a: &Self, b: &Self) -> Self {
        or(a, b)
    }

    fn set_bit(&mut self, bucket: C::BucketType) {
        if bucket.into_usize() < self.lsb.bit_range().start {
            // below LSB range, ignore
        } else if bucket.into_usize() < self.lsb.bit_range().end {
            // within LSB range, insert
            self.insert_into_lsb(bucket.into_usize());
        } else if let Err(idx) = self.msb.binary_search_by(|k| bucket.cmp(k)) {
            // above LSB range, insert as MSB
            self.msb.insert(idx, bucket);
            if self.msb.len() > self.config.max_msb_len() {
                let smallest = self
                    .msb
                    .pop_back()
                    .expect("there must be one!")
                    .into_usize();
                // pad vector to cover the smallest MSB
                self.lsb.resize(smallest + 1);
                self.insert_into_lsb(smallest);
                // pad vector up to the new smallest MSB
                let new_smallest = self.msb.back().expect("there must be one!").into_usize();
                if self.lsb.bit_range().end < new_smallest {
                    self.lsb.resize(new_smallest);
                }
            }
            self.debug_assert_invariants();
        }
    }
}

impl<C: GeoConfig<Distinct>> Count<Distinct> for GeoDistinctCount<'_, C> {
    fn push_hash(&mut self, hash: u64) {
        self.set_bit(self.config.hash_to_bucket(hash));
    }

    fn push<I: std::hash::Hash>(&mut self, item: I) {
        let build_hasher = C::BuildHasher::default();
        self.push_hash(build_hasher.hash_one(item));
    }

    fn push_sketch(&mut self, other: &Self) {
        *self = or(self, other)
    }

    fn size_f32(&self) -> f32 {
        let lowest_bucket = self.lsb.bit_range().start;
        let total = self.msb.len()
            + self
                .lsb
                .bit_chunks()
                .map(|c| c.block.count_ones() as usize)
                .sum::<usize>();
        if lowest_bucket > 0 {
            self.config
                .upscale_estimate(lowest_bucket, self.config.expected_items(total))
        } else {
            self.config.expected_items(total)
        }
    }

    fn size_with_sketch_f32(&self, other: &Self) -> f32 {
        assert!(
            self.config == other.config,
            "combined filters must have the same configuration"
        );
        let (total, lowest_bucket) = count_ones_from_bitchunks::<C::BucketType>(
            or_bit_chunks(self.bit_chunks(), other.bit_chunks()).peekable(),
            self.config.max_bytes(),
            self.config.max_msb_len(),
        );
        if lowest_bucket.into_usize() > 0 {
            self.config.upscale_estimate(
                lowest_bucket.into_usize(),
                self.config.expected_items(total),
            )
        } else {
            self.config.expected_items(total)
        }
    }

    fn bytes_in_memory(&self) -> usize {
        let Self { config, msb, lsb } = self;
        size_of_val(config) + msb.len() * size_of::<C::BucketType>() + lsb.bytes_in_memory()
    }
}

impl<C: GeoConfig<Distinct>> GeoDistinctCount<'_, C> {
    fn insert_into_lsb(&mut self, bucket: usize) {
        if !self.lsb.test_bit(bucket) {
            self.lsb.set_bit(bucket);
        }
    }

    #[inline]
    fn debug_assert_invariants(&self) {
        debug_assert!(
            self.msb.len() <= self.config.max_msb_len(),
            "msb exceeds maximum"
        );
        debug_assert!(
            self.lsb.bit_range().is_empty() || self.msb.len() == self.config.max_msb_len(),
            "lsb non-empty wile msb is not fully used: msb = {:?} |lsb| = {:?}",
            self.msb,
            self.lsb.bit_range(),
        );
        debug_assert!(
            self.lsb.bit_range().is_empty()
                || self.msb.back().map_or(0, |b| b.into_usize()) == self.lsb.bit_range().end,
            "msb and lsb non-contiguous: msb = {:?} |lsb| = {:?}",
            self.msb,
            self.lsb.bit_range(),
        );
    }
}

fn or<C: GeoConfig<Distinct>>(
    a: &GeoDistinctCount<'_, C>,
    b: &GeoDistinctCount<'_, C>,
) -> GeoDistinctCount<'static, C> {
    assert!(
        a.config == b.config,
        "combined filters must have the same configuration"
    );

    GeoDistinctCount::<'static, C>::from_bit_chunks(
        a.config.clone(),
        or_bit_chunks(a.bit_chunks(), b.bit_chunks()),
    )
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::RngCore;

    use crate::build_hasher::UnstableDefaultBuildHasher;
    use crate::config::{iter_ones, tests::test_estimate, FixedConfig, VariableConfig};
    use crate::evaluation::simulation::simulate;
    use crate::test_rng::prng_test_harness;

    use super::*;

    #[test]
    fn test_lookup_table() {
        let c =
            FixedConfig::<Distinct, u32, 13, 10000, 1000, UnstableDefaultBuildHasher>::default();
        for i in 0..c.max_bytes() * 4 {
            let hash = (c.phi_f64().powf(i as f64 + 0.5) * u64::MAX as f64).round() as u64;
            let a = c.hash_to_bucket(hash);
            let b = (hash as f64 / u64::MAX as f64).log(c.phi_f64()).floor() as u32;
            assert_eq!(a, b, "{i} {hash}");
        }
    }

    #[test]
    fn test_geo_count() {
        // Pairs of (n, expected) where n is the number of inserted items
        // and expected is the expected size of the GeoDistinctCount.
        // The output matching the expected values is dependent on the configuration
        // and hashing function. Changes to these will lead to different results and the
        // test will need to be updated.
        for (n, result) in [
            (10, 10.0021105),
            (100, 100.21153),
            (1000, 1001.81635),
            (10000, 9951.017),
            (30000, 29927.705),
            (100000, 99553.24),
            (1000000, 1003824.1),
            (10000000, 10071972.0),
        ] {
            let mut geo_count = GeoDistinctCount13::default();
            (0..n).for_each(|i| geo_count.push(i));
            assert_eq!(result, geo_count.size_f32());
        }
    }

    #[test]
    fn test_or() {
        let a = GeoDistinctCount7::from_ones(Default::default(), 0..300);
        let b = GeoDistinctCount7::from_ones(Default::default(), 200..500);
        let c = or(&a, &b);
        let d = or(&a, &b);
        assert_eq!(a.iter_ones().count(), 300);
        assert_eq!(b.iter_ones().count(), 300);
        assert_eq!((0..500).rev().collect_vec(), c.iter_ones().collect_vec());
        assert_eq!(c.iter_ones().collect_vec(), d.iter_ones().collect_vec());
    }

    #[test]
    fn test_set_bit() {
        let mut m = GeoDistinctCount7::default();
        m.set_bit(10);
        assert_eq!(m.iter_ones().collect_vec(), vec![10]);
        m.set_bit(10);
        assert_eq!(m.iter_ones().collect_vec(), vec![10]);

        let mut m = GeoDistinctCount7::from_ones(Default::default(), vec![0, 2, 4, 8, 16, 32]);
        assert_eq!(m.iter_ones().count(), 6);
        m.set_bit(1);
        assert_eq!(m.iter_ones().count(), 7);
        m.set_bit(3);
        assert_eq!(m.iter_ones().count(), 8);
        m.set_bit(4);
        assert_eq!(m.iter_ones().count(), 8);
    }

    #[test]
    fn test_estimate_fast() {
        prng_test_harness(1, |rnd| {
            let (avg_precision, avg_var) = test_estimate(rnd, GeoDistinctCount7::default);
            println!(
                "avg precision: {} with standard deviation: {}",
                avg_precision,
                avg_var.sqrt(),
            );
            // Make sure that the estimate converges to the correct value.
            assert!(avg_precision.abs() < 0.04);
            // We should theoretically achieve a standard deviation of about 0.065
            assert!(avg_var.sqrt() < 0.08);
        })
    }

    #[test]
    fn test_estimate_union_size_fast() {
        prng_test_harness(1, |rnd| {
            let mut a = GeoDistinctCount7::default();
            let mut b = GeoDistinctCount7::default();
            for _ in 0..10000 {
                a.push_hash(rnd.next_u64());
            }
            for _ in 0..1000 {
                b.push_hash(rnd.next_u64());
            }
            let c = or(&a, &b);

            assert_eq!(c.size(), a.size_with_sketch(&b));
            assert_eq!(c.size(), b.size_with_sketch(&a));
        })
    }

    fn golden_section_min<F: Fn(f32) -> f32>(min: f32, max: f32, f: F) -> f32 {
        let phi = (1.0 + 5.0f32.sqrt()) / 2.0;
        let mut a = min;
        let mut b = max;
        let mut c = b - (b - a) / phi;
        let mut d = a + (b - a) / phi;
        let mut fc = f(c);
        let mut fd = f(d);
        while a + 1.0 < b {
            println!("f({c}) = {fc} .. f({d}) = {fd}");
            if fc < fd {
                b = d;
                d = c;
                fd = fc;
                c = b - (b - a) / phi;
                fc = f(c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + (b - a) / phi;
                fd = f(d);
            }
        }
        c
    }

    #[ignore]
    #[test]
    fn test_optimal_msb() {
        // Run this test in order to search for an optimal msb parameter for a predefined amount of memory.
        // Note: the simulation has some variation and thus the function is not strictly decreasing.
        // Therefore, the search may get stuck in a local minimum. So, treat results with some salt of grain.
        let msb = golden_section_min(1.0, 1000.0, |msb| {
            simulate(
                || {
                    Box::new(GeoDistinctCount::new(VariableConfig::<
                        _,
                        u32,
                        UnstableDefaultBuildHasher,
                    >::new(
                        13,
                        7800,
                        (7800 - (msb.round() as usize) * 8) / 3,
                    )))
                },
                3000,
                &[100000],
            )[0]
            .upscaled_relative_stddev() as f32
        });
        println!("optimal msb: {msb}");
    }

    #[test]
    fn test_bit_chunks() {
        prng_test_harness(100, |rnd| {
            let mut expected = GeoDistinctCount7::default();
            for _ in 0..1000 {
                expected.push_hash(rnd.next_u64());
            }
            let actual =
                GeoDistinctCount::from_bit_chunks(expected.config.clone(), expected.bit_chunks());
            assert_eq!(expected, actual);
        })
    }

    #[test]
    fn test_msb_order() {
        let mut a = GeoDistinctCount7::default();
        a.set_bit(7);
        a.set_bit(17);
        a.set_bit(11);
        assert_eq!(vec![17, 11, 7], a.msb.iter().copied().collect_vec());
    }

    impl<C: GeoConfig<Distinct>> GeoDistinctCount<'_, C> {
        fn from_ones(config: C, ones: impl IntoIterator<Item = C::BucketType>) -> Self {
            let mut result = Self::new(config);
            for one in ones {
                result.set_bit(one);
            }
            result
        }

        fn iter_ones(&self) -> impl Iterator<Item = C::BucketType> + '_ {
            iter_ones(self.bit_chunks().peekable()).map(C::BucketType::from_usize)
        }
    }
}
