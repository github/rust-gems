//! Geometric filter implementation for diff count.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::hash::BuildHasher as _;
use std::mem::{size_of, size_of_val};

use crate::config::{
    count_ones_from_bitchunks, count_ones_from_msb_and_lsb, iter_bit_chunks, iter_ones,
    mask_bit_chunks, take_ref, xor_bit_chunks, BitChunk, GeoConfig, IsBucketType,
};
use crate::{Count, Diff};

mod bitvec;
mod config;
mod sim_hash;

use bitvec::*;
pub use config::{GeoDiffConfig13, GeoDiffConfig7};

/// Diff count filter with a relative error standard deviation of ~0.125.
pub type GeoDiffCount7<'a> = GeoDiffCount<'a, GeoDiffConfig7>;

/// Diff count filter with a relative error standard deviation of ~0.015.
pub type GeoDiffCount13<'a> = GeoDiffCount<'a, GeoDiffConfig13>;

/// Probabilistic diff count data structure based on geometric filters.
///
/// The [`GeoDiffCount`] falls into the category of probabilistic set size estimators.
/// It has some special properties that makes it distinct from related filters like
/// HyperLogLog, MinHash, etc.
///
/// 1.  It supports insertion and deletion which are both expressed by simply toggling
///     the bit associated with the hash of the original item.
///
/// 2.  Because of that property, the symmetric difference between two sets
///     is also expressed by a simple xor operation of the two associated GeoDiffCounts.
///     But, it doesn't support the union operation of two sets, since it would treat items
///     that occur in both sets as deletions!
///
/// 3.  In contrast to other set estimators, the precision of the estimated size of the
///     symmetric difference between two sets is relative to the estimate size and not
///     relative to the union of the original two sets!
///
/// 4.  This is possible, since the GeoDiffCount keeps for large sets all the bits necessary
///     to find potentially single item differences with another large set. This increases
///     the size of the GeoDiffCount, but it will do so only for large sets. I.e. in contrast
///     to most other estimators, the size of the GeoDiffCount grows logarithmically with the
///     size of the represented set. This means that the GeoDiffCount will always be much
///     smaller than the original set and the size advantage improves the larger the set becomes.
///
/// 5.  With the choice of 128 bits per level, the standard deviation of the estimated relative
///     error is ~0.12 in the worst case. It achieves this precision with less than 300 bytes
///     for a set containing 1 million items.
///
///     Comparing two sets with 1 million items which differ in only 10k items would be estimated
///     with an error of +/-1200 items using the GeoDiffCount. A HyperLogLog data structure with
///     1500 bytes achieves a precision of ~0.022 and could estimate the 10k items with an error
///     of +/-22k items in the best case which is 20 times worse despite using 5 times more space!
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct GeoDiffCount<'a, C: GeoConfig<Diff>> {
    config: C,
    /// The bit positions are stored from largest to smallest.
    msb: Cow<'a, [C::BucketType]>,
    lsb: BitVec<'a>,
}

impl<C: GeoConfig<Diff>> std::fmt::Debug for GeoDiffCount<'_, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "~{} (msb: {:?}, |lsb|: {:?})",
            self.estimate_size(),
            self.msb,
            self.lsb.num_bits(),
        )
    }
}

impl<C: GeoConfig<Diff>> GeoDiffCount<'_, C> {
    pub fn new(config: C) -> Self {
        Self {
            config,
            msb: Default::default(),
            lsb: Default::default(),
        }
    }

    /// `BitChunk`s can be processed much more efficiently than individual one bits!
    /// This function makes it possible to construct a GeoDiffCount instance directly from
    /// `BitChunk`s. It will extract the most significant bits first and then put the remainder
    /// into a `BitVec` representation.
    ///
    /// Note: we need a peekable iterator, such that we can extract the most significant bits without
    /// having to construct another iterator with the remaining `BitChunk`s.
    fn from_bit_chunks<I: Iterator<Item = BitChunk>>(config: C, chunks: I) -> Self {
        let mut ones = iter_ones::<C::BucketType, _>(chunks.peekable());
        let mut msb = Vec::default();
        take_ref(&mut ones, config.max_msb_len() - 1).for_each(|bucket| {
            msb.push(bucket);
        });
        let smallest_msb = ones
            .next()
            .inspect(|bucket| {
                msb.push(*bucket);
            })
            .unwrap_or_default();
        let lsb = BitVec::from_bit_chunks(ones.into_bitchunks(), smallest_msb.into_usize());
        let result = Self {
            config,
            msb: Cow::from(msb),
            lsb,
        };
        result.debug_assert_invariants();
        result
    }

    /// Compare two geometric filters after applying the specified mask.
    ///
    /// To reduce the number of operations, the implementation first xors the bit chunks together,
    /// then applies the mask (without compressing the mask), and finally looks for the first
    /// modified bit. This bit must be set in the larger geometric filter.
    pub fn cmp_masked(&self, other: &Self, mask: u64, mask_size: usize) -> Ordering {
        let mut differing_bit = mask_bit_chunks(
            xor_bit_chunks(self.bit_chunks(), other.bit_chunks()),
            mask,
            mask_size,
        );
        match differing_bit.next() {
            None => Ordering::Equal,
            Some(bit_chunk) => {
                let msb =
                    C::BucketType::from_block_msb(bit_chunk.block).into_bucket(bit_chunk.index);
                if self.test_bit(msb) {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
        }
    }

    fn test_bit(&self, index: C::BucketType) -> bool {
        if index.into_usize() < self.lsb.num_bits() {
            self.lsb.test_bit(index.into_usize())
        } else {
            self.msb.binary_search_by(|k| index.cmp(k)).is_ok()
        }
    }

    /// Instead of computing the full xor, we can take a short-cut when estimating the size by
    /// counting the ones in the equivalent of up to max_bytes of data from the xor iterator.
    fn estimate_diff_size(&self, other: &Self) -> f32 {
        assert!(
            self.config == other.config,
            "combined filters must have the same configuration"
        );
        let (total, lowest_bucket) = count_ones_from_bitchunks::<C::BucketType>(
            xor_bit_chunks(self.bit_chunks(), other.bit_chunks()).peekable(),
            self.config.max_bytes(),
            self.config.max_msb_len(),
        );
        if lowest_bucket > 0 {
            self.config
                .upscale_estimate(lowest_bucket, self.config.expected_items(total))
        } else {
            self.config.expected_items(total)
        }
    }

    /// Converts the GeoDiffCount into `BitChunk`s. Thereby, `BitChunk`s are returned from
    /// most to least significant, overlapping `BitChunk`s got merged, and empty ones filtered.
    fn bit_chunks(&self) -> impl Iterator<Item = BitChunk> + '_ {
        iter_bit_chunks(
            self.msb.iter().map(|b| b.into_usize()),
            self.lsb.bit_chunks(),
        )
    }

    /// Estimates the size of the represented set using the specified number of one bits.
    /// E.g. with ones = 20, a relative precision of ~0.22 is achieved.
    /// With ones = 128, a relative precision of ~0.12 is achieved.
    /// Using numbers larger than 128 won't increase the precision!
    fn estimate_size(&self) -> f32 {
        let (total, lowest_bucket) = count_ones_from_msb_and_lsb::<C::BucketType>(
            self.msb.len(),
            self.msb.last().copied().unwrap_or_default(),
            self.lsb.bit_chunks().peekable(),
            self.config.max_bytes(),
            self.config.max_msb_len(),
        );
        if lowest_bucket > 0 {
            self.config
                .upscale_estimate(lowest_bucket, self.config.expected_items(total))
        } else {
            self.config.expected_items(total)
        }
    }

    /// xor-ing random bits has amortized complexity O(1).
    /// The reason is that the probability to execute the expensive else branch happens with a geometric
    /// distribution. I.e. it will be hit O(log(n)) number of times out of n calls. And each of these
    /// executions has a constant complexity, but one that is rather high compared to the fast path. In total
    /// that makes the cost of the else case negligible.
    fn xor_bit(&mut self, bucket: C::BucketType) {
        if bucket.into_usize() < self.lsb.num_bits() {
            self.lsb.toggle(bucket.into_usize());
        } else {
            let msb = self.msb.to_mut();
            match msb.binary_search_by(|k| bucket.cmp(k)) {
                Ok(idx) => {
                    msb.remove(idx);
                    let first = {
                        let mut lsb = iter_ones(self.lsb.bit_chunks().peekable());
                        lsb.next()
                    };
                    if let Some(smallest) = first {
                        msb.push(C::BucketType::from_usize(smallest));
                        self.lsb.resize(smallest);
                    } else {
                        self.lsb.resize(0);
                    };
                }
                Err(idx) => {
                    msb.insert(idx, bucket);
                    if msb.len() > self.config.max_msb_len() {
                        let smallest = msb
                            .pop()
                            .expect("we should have at least one element!")
                            .into_usize();
                        // ensure vector covers smallest
                        let new_smallest = msb
                            .last()
                            .expect("should have at least one element")
                            .into_usize();
                        self.lsb.resize(new_smallest);
                        self.lsb.toggle(smallest);
                    } else if msb.len() == self.config.max_msb_len() {
                        let smallest = msb
                            .last()
                            .expect("should have at least one element")
                            .into_usize();
                        self.lsb.resize(smallest);
                    }
                }
            }
            self.debug_assert_invariants();
        }
    }

    // Consumers the current GeoDiffCount and produces an owned GeoDiffCount that
    // can live arbitrarily.
    pub fn into_owned(self) -> GeoDiffCount<'static, C> {
        GeoDiffCount {
            config: self.config,
            lsb: self.lsb.into_owned(),
            msb: Cow::Owned(self.msb.into_owned()),
        }
    }

    pub fn symmetric_difference(a: &Self, b: &Self) -> Self {
        xor(a, b)
    }

    #[inline]
    fn debug_assert_invariants(&self) {
        debug_assert!(
            self.msb.len() <= self.config.max_msb_len(),
            "msb exceeds maximum"
        );
        debug_assert!(
            self.lsb.num_bits() == 0 || self.msb.len() == self.config.max_msb_len(),
            "lsb non-empty wile msb is not fully used: msb = {:?} |lsb| = {:?}",
            self.msb,
            self.lsb.num_bits(),
        );
        debug_assert!(
            self.lsb.num_bits() == 0
                || self.msb.last().map_or(0, |b| b.into_usize()) == self.lsb.num_bits(),
            "msb and lsb non-contiguous: msb = {:?} |lsb| = {:?}",
            self.msb,
            self.lsb.num_bits(),
        );
    }
}

/// Applies a repeated bit mask to the underlying filter.
/// E.g. given the bit mask `0b110100` with modulus 6, we filter the bitset of the geometric filter as follows:
/// bitset of the geometric filter: 011010 101101 001010
/// repeated bit mask             : 110100 110100 110100
/// masked bitset                 : 010000 100100 000000
/// after compression             : 01 0   10 1   00 0
/// bitset of the returned filter :           010 101000
#[cfg(test)]
pub(crate) fn masked<C: GeoConfig<Diff>>(
    diff_count: &GeoDiffCount<'_, C>,
    mask: usize,
    modulus: usize,
) -> GeoDiffCount<'static, C> {
    GeoDiffCount::from_bit_chunks(
        diff_count.config.clone(),
        mask_bit_chunks(diff_count.bit_chunks(), mask as u64, modulus).peekable(),
    )
}

/// Computes an xor of the two underlying bitsets.
/// This operation corresponds to computing the symmetric difference of the two
/// sets represented by the GeoDiffCounts.
///
/// # Panics
///
/// Panics if the configuration of the geofilters is not identical.
pub(crate) fn xor<C: GeoConfig<Diff>>(
    diff_count: &GeoDiffCount<'_, C>,
    other: &GeoDiffCount<'_, C>,
) -> GeoDiffCount<'static, C> {
    assert!(
        diff_count.config == other.config,
        "combined filters must have the same configuration"
    );

    GeoDiffCount::from_bit_chunks(
        diff_count.config.clone(),
        xor_bit_chunks(diff_count.bit_chunks(), other.bit_chunks()).peekable(),
    )
}

impl<C: GeoConfig<Diff>> Count<Diff> for GeoDiffCount<'_, C> {
    fn push_hash(&mut self, hash: u64) {
        self.xor_bit(self.config.hash_to_bucket(hash));
    }

    fn push<I: std::hash::Hash>(&mut self, item: I) {
        let build_hasher = C::BuildHasher::default();
        self.push_hash(build_hasher.hash_one(item));
    }

    fn push_sketch(&mut self, other: &Self) {
        *self = xor(self, other);
    }

    fn size(&self) -> f32 {
        self.estimate_size()
    }

    fn size_with_sketch(&self, other: &Self) -> f32 {
        assert!(
            self.config == other.config,
            "combined filters must have the same configuration"
        );
        self.estimate_diff_size(other)
    }

    fn bytes_in_memory(&self) -> usize {
        let Self { config, msb, lsb } = self;

        size_of_val(config) + msb.len() * size_of::<C::BucketType>() + lsb.bytes_in_memory()
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::RngCore;

    use crate::{
        build_hasher::UnstableDefaultBuildHasher,
        config::{iter_ones, tests::test_estimate, FixedConfig},
        test_rng::prng_test_harness,
    };

    use super::*;

    /// Count filter with a relative error standard deviation of ~0.25.
    // Precision evaluation:
    //
    //     scripts/accuracy -n 10000 geo_diff/u16/b=7/bytes=50/msb=10
    //
    type GeoDiffCount7_50<'a> =
        GeoDiffCount<'a, FixedConfig<Diff, u16, 7, 50, 10, UnstableDefaultBuildHasher>>;

    #[test]
    fn test_geo_count() {
        for (n, result) in [
            (10, 10.0042095),
            (100, 100.422966),
            (1000, 1002.95654),
            (10000, 9876.323),
            (30000, 29577.656),
            (100000, 97392.945),
            (1000000, 1021517.56),
            (10000000, 10194611.0),
        ] {
            let mut geo_count = GeoDiffCount13::default();

            (0..n).for_each(|i| geo_count.push(i));
            assert_eq!(result, geo_count.size());
        }
    }

    #[test]
    fn test_xor() {
        let a = GeoDiffCount7::from_ones(Default::default(), 0..1000);
        let b = GeoDiffCount7::from_ones(Default::default(), 10..1010);
        let c = xor(&a, &b);
        let d = xor(&a, &b);
        assert_eq!(a.iter_ones().count(), 1000);
        assert_eq!(b.iter_ones().count(), 1000);
        assert_eq!(
            (0..10).chain(1000..1010).rev().collect_vec(),
            c.iter_ones().collect_vec(),
        );
        assert_eq!(c.iter_ones().collect_vec(), d.iter_ones().collect_vec());
    }

    #[test]
    fn test_xor_bit() {
        let mut m = GeoDiffCount7::default();
        m.xor_bit(10);
        assert_eq!(m.iter_ones().collect_vec(), vec![10]);
        m.xor_bit(10);
        assert!(m.iter_ones().collect_vec().is_empty());

        let mut m = GeoDiffCount7::from_ones(Default::default(), 0..100);
        assert_eq!(m.iter_ones().count(), 100);
        m.xor_bit(10);
        assert_eq!(m.iter_ones().count(), 99);
        m.xor_bit(99);
        assert_eq!(m.iter_ones().count(), 98);
        m.xor_bit(99);
        assert_eq!(m.iter_ones().count(), 99);
        m.xor_bit(10);
        assert_eq!(m.iter_ones().count(), 100);
        m.xor_bit(100);
        assert_eq!(m.iter_ones().count(), 101);
    }

    #[test]
    fn test_estimate_fast() {
        prng_test_harness(1, |rnd| {
            let (avg_precision, avg_var) = test_estimate(rnd, GeoDiffCount7::default);
            println!(
                "avg precision: {} with standard deviation: {}",
                avg_precision,
                avg_var.sqrt(),
            );
            // Make sure that the estimate converges to the correct value.
            assert!(avg_precision.abs() < 0.04);
            // We should theoretically achieve a standard deviation of about 0.12
            assert!(avg_var.sqrt() < 0.14);
        })
    }

    #[test]
    fn test_estimate_fast_low_precision() {
        prng_test_harness(1, |rnd| {
            let (avg_precision, avg_var) = test_estimate(rnd, GeoDiffCount7_50::default);
            println!(
                "avg precision: {} with standard deviation: {}",
                avg_precision,
                avg_var.sqrt(),
            );
            // Make sure that the estimate converges to the correct value.
            assert!(avg_precision.abs() < 0.15);
            // We should theoretically achieve a standard deviation of about 0.25
            assert!(avg_var.sqrt() < 0.4);
        });
    }

    #[test]
    fn test_estimate_diff_size_fast() {
        prng_test_harness(1, |rnd| {
            let mut a_p = GeoDiffCount7_50::default();
            let mut a_hp = GeoDiffCount7::default();
            let mut b_p = GeoDiffCount7_50::default();
            let mut b_hp = GeoDiffCount7::default();
            for _ in 0..10000 {
                let hash = rnd.next_u64();
                a_p.push_hash(hash);
                a_hp.push_hash(hash);
            }
            for _ in 0..1000 {
                let hash = rnd.next_u64();
                b_p.push_hash(hash);
                b_hp.push_hash(hash);
            }
            let c_p = xor(&a_p, &b_p);
            let c_hp = xor(&a_hp, &b_hp);

            assert_eq!(c_p.size(), a_p.size_with_sketch(&b_p));
            assert_eq!(c_p.size(), b_p.size_with_sketch(&a_p));

            assert_eq!(c_hp.size(), a_hp.size_with_sketch(&b_hp));
            assert_eq!(c_hp.size(), b_hp.size_with_sketch(&a_hp));
        });
    }

    #[test]
    fn test_masking() {
        // bit index                     :     12      6      0
        // bitset of the geometric filter: 011010 101101 001010
        // repeated bit mask             : 110100 110100 110100
        // masked bitset                 : 010000 100100 000000
        // after compression             : 01 0   10 1   00 0
        // bitset of the returned filter :           010 101000
        let m = GeoDiffCount7::from_ones(Default::default(), [16, 15, 13, 11, 9, 8, 6, 3, 1]);
        let n = masked(&m, 0b110100, 6);
        assert_eq!(n.iter_ones().collect_vec(), vec![16, 11, 8]);

        for i in 0..100 {
            let m = GeoDiffCount7::from_ones(Default::default(), (0..i).collect_vec());
            let n = masked(&m, 0b111, 3);
            assert_eq!(m, n);
        }

        for i in 0..300 {
            let m = GeoDiffCount7::from_ones(Default::default(), (0..i).collect_vec());
            let slow =
                GeoDiffCount::from_ones(Default::default(), masked(&m, 0b110, 3).iter_ones());
            let n = masked(&m, 0b110, 3);
            assert_eq!(slow, n, "in iteration: {i}");
        }
    }

    #[test]
    fn test_xor_plus_mask() {
        prng_test_harness(1000, |rnd| {
            let mask_size = 12;
            let mask = 0b100001100000;
            let mut a = GeoDiffCount7::default();
            for _ in 0..10000 {
                a.xor_bit(a.config.hash_to_bucket(rnd.next_u64()));
            }
            let mut expected = GeoDiffCount7::default();
            let mut b = a.clone();
            for _ in 0..1000 {
                let hash = rnd.next_u64();
                b.xor_bit(b.config.hash_to_bucket(hash));
                expected.xor_bit(expected.config.hash_to_bucket(hash));
                assert_eq!(expected, xor(&a, &b));
                let masked_a = masked(&a, mask, mask_size);
                let masked_b = masked(&b, mask, mask_size);
                let masked_expected = masked(&expected, mask, mask_size);
                assert_eq!(masked_expected, xor(&masked_a, &masked_b));
            }
        });
    }

    #[test]
    fn test_bit_chunks() {
        prng_test_harness(100, |rnd| {
            let mut expected = GeoDiffCount7::default();
            for _ in 0..1000 {
                expected.push_hash(rnd.next_u64());
            }
            let actual = GeoDiffCount::from_bit_chunks(
                expected.config.clone(),
                expected.bit_chunks().peekable(),
            );
            assert_eq!(expected, actual);
        });
    }

    #[test]
    fn test_msb_order() {
        let mut a = GeoDiffCount7::default();
        a.xor_bit(7);
        a.xor_bit(17);
        a.xor_bit(11);
        assert_eq!(vec![17, 11, 7], a.msb.iter().copied().collect_vec());
    }

    impl<C: GeoConfig<Diff>> GeoDiffCount<'_, C> {
        fn from_ones(config: C, ones: impl IntoIterator<Item = C::BucketType>) -> Self {
            let mut result = Self::new(config);
            for one in ones {
                result.xor_bit(one);
            }
            result
        }

        fn iter_ones(&self) -> impl Iterator<Item = C::BucketType> + '_ {
            iter_ones(self.bit_chunks().peekable()).map(C::BucketType::from_usize)
        }
    }
}
