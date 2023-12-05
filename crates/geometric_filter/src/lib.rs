//! A data structure, like HyperLogLog, that lossily represents a collection of hashable
//! values. Useful for:
//! * estimating the number of distinct elements in a collection;
//! * comparing collections, estimating how many elements they share.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::Peekable;
use std::ops::{Deref, Range};

use dataview::PodMethods;
use fnv::FnvHasher;
use github_lock_free::SelfContained;
use github_pspack::Serializable;
use github_stable_hash::StableHash;
use itertools::Itertools;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

use crate::bitvec::nth_one;
use crate::bitvec::BitVec;

mod bitvec;

pub const HIGH_PRECISION_ONES: usize = 128;
const ONES_U16_ENCODED: usize = 10;
const BITS_PER_LEVEL: usize = 128;
const BIT_MAPPING_BUCKETS_BITS: usize = 8;
const BIT_MAPPING_BUCKETS: usize = 1 << BIT_MAPPING_BUCKETS_BITS;
static PHI_F64: Lazy<f64> = Lazy::new(|| 0.5f64.powf(1.0 / BITS_PER_LEVEL as f64));
static PHI_F32: Lazy<f32> = Lazy::new(|| *PHI_F64 as f32);

static BIT_MAPPING: Lazy<[(usize, usize); BIT_MAPPING_BUCKETS]> = Lazy::new(build_bit_mapping);

/// Number of bits covered by each SimHash bucket.
pub const SIM_BUCKET_SIZE: usize = 6;
/// Number of consecutive SimHash buckets used for searching.
pub const SIM_BUCKETS: usize = 20;

type BucketId = usize;

/// The i-th bit covers the interval (phi^(i+1); phi^i] in the (0; 1] hash space where i >= 0. phi
/// is chosen such that phi^BITS_PER_LEVEL == 0.5.
///
/// In order to translate a random number r in the range (0; 1] into its corresponding bit position
/// i, we first determine its level l where each level covers the interval (0.5^(l+1); 0.5^l].
///
/// Given the binary representation 0.xxxxxxxxx of r, the level can e.g. be determined by looking
/// at the number of leading zeros after the dot (which can be derived from the exponent in the
/// float representation). Within one level, we compute a lookup table consisting of equally sized
/// buckets. Each bucket stores the bit to which it corresponds. The bucket size is chosen such
/// that at most two bit intervals fall into each. In such a case, we store the lower interval
/// bound of the smaller bit id as part of the bucket. If the random number falls now below that
/// lower bound, the bit id associated with the bucket has to be incremented by one.
///
/// Bit Layout:
///
/// ```text
///            0                          0.5                            1
///   bit ids: |                           |n-1|     ...    |  1  |  0   |
///                                    phi^n            phi^2         phi^0
///                                         phi^(n-1)           phi^1
/// ```
fn build_bit_mapping() -> [(usize, usize); BIT_MAPPING_BUCKETS] {
    let mut buckets = [(0, 0); BIT_MAPPING_BUCKETS];
    let mut last_filled_bucket = BIT_MAPPING_BUCKETS;
    for bucket in 0..BITS_PER_LEVEL {
        let lower_bucket_limit = PHI_F64.powf((bucket + 1) as f64);
        let lower_hash_limit = ((lower_bucket_limit - 0.5) * 2.0f64.powf(33.0)) as usize;
        let lower_hash_bucket = lower_hash_limit >> (32 - BIT_MAPPING_BUCKETS_BITS);
        assert!(lower_hash_bucket < last_filled_bucket);
        while last_filled_bucket > lower_hash_bucket {
            last_filled_bucket -= 1;
            buckets[last_filled_bucket] = (bucket, lower_hash_limit);
        }
    }
    assert_eq!(last_filled_bucket, 0);
    buckets
}

/// The GeometricFilter falls into the category of probabilistic set size estimators.
/// It has some special properties that makes it distinct from related filters like
/// HyperLogLog, MinHash, etc.
///
/// 1.  It supports insertion and deletion which are both expressed by simply toggling
///     the bit associated with the hash of the original item.
///
/// 2.  Because of that property, the symmetric difference between two sets
///     is also expressed by a simple xor operation of the two associated GeometricFilters.
///     But, it doesn't support the union operation of two sets, since it would treat items
///     that occur in both sets as deletions!
///
/// 3.  In contrast to other set estimators, the precision of the estimated size of the
///     symmetric difference between two sets is relative to the estimate size and not
///     relative to the union of the original two sets!
///
/// 4.  This is possible, since the GeometricFilter keeps for large sets all the bits necessary
///     to find potentially single item differences with another large set. This increases
///     the size of the GeometricFilter, but it will do so only for large sets. I.e. in contrast
///     to most other estimators, the size of the GeometricFilter grows logarithmically with the
///     size of the represented set. This means that the GeometricFilter will always be much
///     smaller than the original set and the size advantage improves the larger the set becomes.
///
/// 5.  With the choice of 128 bits per level, the standard deviation of the estimated relative
///     error is about 0.12 in the worst case. It achieves this precision with less than 300 bytes
///     for a set containing 1 million items.
///
///     Comparing two sets with 1 million items which differ in only 10k items would be estimated
///     with an error of +/-1200 items using the GeometricFilter. A HyperLogLog data structure with
///     1500 bytes achieves a precision of ~0.022 and could estimate the 10k items with an error
///     of +/-22k items in the best case which is 20 times worse despite using 5 times more space!
///
/// Note: There is a related GeometricOrFilter which supports the same features as HyperLogLog
/// using less space!
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct GeometricFilter<'a> {
    /// The bit positions are stored from largest to smallest.
    /// The largest bit position that we can encounter is 32*128 = 4096 which easily fits into u16.
    most_significant_bits: Cow<'a, [u16]>,
    least_significant_bits: BitVec<'a>,
}

/// SimHash is a hash computed over a continuous range of bits from a GeometricFilter.
/// It is used to quickly find similar sets with a reverse index.
#[derive(
    Copy, Clone, Default, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct SimHash(pub u64);

impl SelfContained for SimHash {}

impl StableHash for SimHash {
    fn stable_hash(&self) -> u64 {
        self.0
    }
}

impl SimHash {
    /// Constructs a `SimHash` from the bit set extracted from the `GeometricFilter`.
    /// Since the same bitset can occur for differently shifted buckets, we also
    /// take the bucket_id into account when computing the `SimHash`.
    /// Note: We can shrink the hash space to e.g. 32-bits if the `SimHash` index
    /// becomes too large. Collisions will slightly increase the chance of false
    /// positives.
    pub fn new(bucket_id: BucketId, bits: u64) -> Self {
        let mut hasher = FnvHasher::default();
        bucket_id.hash(&mut hasher);
        bits.hash(&mut hasher);
        let hash = hasher.finish();

        // Don't use SimHash(0) since it is used to mark empty hash entries. See `<u64 as
        // AHashKey>::empty()`, `impl KeyMap for SimHash`.
        Self(hash.max(1))
    }

    /// Constructs a SimHash from its hash code, not hashing actual data. This is public for
    /// debugging -- it's way too low level to be useful normally.
    pub fn from_bits(bits: u64) -> Self {
        SimHash(bits)
    }
}

impl<'a> Debug for GeometricFilter<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let mut bytes = Vec::<u8>::new();
        self.write(&mut bytes)
            .expect("serialization should only fail on I/O errors");
        write!(
            f,
            "{} (size: ~{}, msb: {:?})",
            hex::encode(bytes),
            self.estimate_size(20).round() as usize,
            self.most_significant_bits
        )
    }
}

impl<'a> GeometricFilter<'a> {
    /// This function interprets the bit sequence of the geometric filters as a gray code.
    /// And compares the rank of two such gray codes.
    /// This can be done much more efficiently than actually determining the rank of a gray code,
    /// since we just need to find the first differing bit and whether the common prefix has
    /// an odd or even number of one bits. When there are odd many preceding one bits, then the
    /// standard binary ordering is simply reversed.
    pub fn cmp_gray_rank(&self, other: &Self) -> Ordering {
        let msb_len = self
            .most_significant_bits
            .len()
            .min(other.most_significant_bits.len());
        for i in 0..msb_len {
            if self.most_significant_bits[i] != other.most_significant_bits[i] {
                let result = self.most_significant_bits[i].cmp(&other.most_significant_bits[i]);
                return if i & 1 != 0 { result.reverse() } else { result };
            }
        }
        let result = if self.most_significant_bits.len() != other.most_significant_bits.len() {
            self.most_significant_bits
                .len()
                .cmp(&other.most_significant_bits.len())
        } else {
            self.least_significant_bits
                .cmp_gray_rank(&other.least_significant_bits)
        };
        if msb_len & 1 != 0 {
            result.reverse()
        } else {
            result
        }
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_ones(mut ones: Vec<usize>) -> Self {
        ones.sort_by(|a, b| b.cmp(a));
        Self::from_descending_iter(ones.into_iter())
    }

    /// `BitChunk`s can be processed much more efficiently than individual one bits!
    /// This function makes it possible to construct a GeometricFilter instance directly from
    /// `BitChunk`s. It will extract the most significant bits first and then put the remainder
    /// into a `BitVec` representation.
    ///
    /// Note: we need a peekable iterator, such that we can extract the most significant bits without
    /// having to construct another iterator with the remaining `BitChunk`s.
    pub fn from_bit_chunks<I: Iterator<Item = BitChunk>>(
        mut iter: Peekable<I>,
    ) -> GeometricFilter<'static> {
        let mut most_significant_bits = vec![];
        let mut num_bits = 0;
        // Note: we wrap the nested loops into a lambda function so that we can exit the inner
        // most loop directly.
        (|| {
            while let Some(BitChunk(block_idx, block)) = iter.peek_mut() {
                while *block != 0 {
                    // Determine the next most significant bit and clear it in the current `BitChunk`.
                    let mut bit_idx = 63 - block.leading_zeros() as u16;
                    *block ^= 1 << bit_idx;
                    bit_idx += *block_idx * 64;
                    most_significant_bits.push(bit_idx);
                    if most_significant_bits.len() == ONES_U16_ENCODED {
                        // Once, the desired number of most significant bits has been collected.
                        // we can stop iterating. We just need to ensure that we drop the last block
                        // if it became empty.
                        if *block == 0 {
                            iter.next();
                        }
                        num_bits = bit_idx as usize;
                        return;
                    }
                }
                iter.next();
            }
        })();
        GeometricFilter {
            most_significant_bits: Cow::from(most_significant_bits),
            least_significant_bits: BitVec::from_bit_chunks(num_bits, iter),
        }
    }

    /// Applies a repeated bit mask to the underlying filter.
    /// E.g. given the bit mask `0b110100` with modulus 6, we filter the bitset of the geometric filter as follows:
    /// bitset of the geometric filter: 011010 101101 001010
    /// repeated bit mask             : 110100 110100 110100
    /// masked bitset                 : 010000 100100 000000
    /// after compression             : 01 0   10 1   00 0
    /// bitset of the returned filter :           010 101000
    pub fn masked(&self, mask: usize, modulus: usize) -> GeometricFilter<'static> {
        GeometricFilter::from_bit_chunks(
            mask_bit_chunks(self.bit_chunks(), mask as u64, modulus as u16).peekable(),
        )
    }

    /// The provided iterator must list the one bit positions in descending order.
    fn from_descending_iter<I: Iterator<Item = usize>>(mut iter: I) -> GeometricFilter<'static> {
        let mut most_significant_bits = vec![];
        let mut least_significant_bits = BitVec::new();
        for bit in iter.by_ref() {
            most_significant_bits.push(bit as u16);
            if most_significant_bits.len() >= ONES_U16_ENCODED {
                least_significant_bits.resize(bit);
                break;
            }
        }
        for bit in iter {
            least_significant_bits.toggle(bit);
        }
        GeometricFilter {
            most_significant_bits: Cow::from(most_significant_bits),
            least_significant_bits,
        }
    }

    /// Returns indices of set bits from most significant to least significant.
    fn iter_ones(&'_ self) -> impl Iterator<Item = usize> + '_ {
        self.most_significant_bits
            .deref()
            .iter()
            .map(|bit| *bit as usize)
            .chain(self.least_significant_bits.iter_ones())
    }

    /// Returns masked indices of set bits from most significant to least significant.
    /// While the implementation is pretty straight-forward, it is pretty slow and in production code
    /// one should rather use the `masked_bit_chunks` version which processes 64-bits in one go.
    ///
    /// Nevertheless, this simple implementation is very handy for verifying the correctness of the
    /// more complex chunked implementation.
    #[cfg(test)]
    fn iter_ones_masked(&'_ self, mask: u64, modulus: u16) -> impl Iterator<Item = usize> + '_ {
        self.most_significant_bits
            .deref()
            .iter()
            .cloned()
            .chain(self.least_significant_bits.iter_ones().map(|a| a as u16))
            .filter_map(move |bit| {
                if ((1 << (bit % modulus)) & mask) != 0 {
                    Some(bit as usize)
                } else {
                    None
                }
            })
    }

    /// Returns an iterator which outputs the set bits from most significant to least significant
    /// when applying the xor operation on the two bitsets.
    pub fn iter_xor<'b>(&'b self, other: &'b GeometricFilter) -> impl Iterator<Item = usize> + 'b {
        XorIterator {
            iter: self
                .iter_ones()
                .merge_by(other.iter_ones(), |x, y| x > y)
                .peekable(),
        }
    }

    /// n specifies the desired zero-based index of the most significant one.
    /// The zero-based index of the desired one bit is returned.
    fn nth_most_significant_one(&self, mut n: usize) -> Option<usize> {
        if n < self.most_significant_bits.len() {
            Some(self.most_significant_bits[n] as usize)
        } else {
            n -= self.most_significant_bits.len();
            self.least_significant_bits.nth_most_significant_one(n)
        }
    }

    /// Returns the total number of one bits.
    fn count_ones(&self) -> usize {
        self.most_significant_bits.len() + self.least_significant_bits.count_ones()
    }

    /// Compare two geometric filters after applying the specified mask.
    ///
    /// To reduce the number of operations, the implementation first xors the bit chunks together,
    /// then applies the mask (without compressing the mask), and finally looks for the first
    /// modified bit. This bit must be set in the larger geometric filter.
    pub fn cmp_masked(&self, other: &Self, mask: u64, mask_size: u16) -> Ordering {
        let mut differing_bit = mask_bit_chunks(
            xor_bit_chunks(self.bit_chunks(), other.bit_chunks()),
            mask,
            mask_size,
        );
        match differing_bit.next() {
            None => Ordering::Equal,
            Some(bit_chunk) => {
                let msb = 63 - bit_chunk.1.leading_zeros() as usize + bit_chunk.0 as usize * 64;
                if self.test_bit(msb) {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
        }
    }

    fn test_bit(&self, index: usize) -> bool {
        if index < self.least_significant_bits.len() {
            self.least_significant_bits.test_bit(index)
        } else {
            self.most_significant_bits
                .binary_search_by(|k| index.cmp(&(*k as usize)))
                .is_ok()
        }
    }

    /// This computes the mean estimate for up to 256 one bits.
    /// The polynomial was derived by fitting a polynomial through the first 256 estimated means
    /// as described in the book chapter about Geometric Filters.
    /// For fitting a polynomial Google Sheets trendline feature was used.
    ///
    /// TODO: change return type from f32 to usize
    fn mean(ones: f32) -> f32 {
        (0.18 + (0.87 + (0.00642 + (-0.0000268 + 0.000000154 * ones) * ones) * ones) * ones).round()
    }

    #[allow(dead_code)]
    fn std(ones: f32) -> f32 {
        0.273 + (0.0556 + (0.00086 + (-0.0000042 + 0.0000000269 * ones) * ones) * ones) * ones
    }

    /// For more large number of one bits, we upscale the mean and standard deviation from
    /// the most significant one bits. q specifies the fraction of the hash space for which
    /// the mean is known.
    fn upscale_mean(q: f32, mean: f32) -> f32 {
        (mean + 1.0) / q - 1.0
    }

    #[allow(dead_code)]
    fn upscale_std(q: f32, mean: f32, std: f32) -> f32 {
        (std.powf(2.0) + (1.0 - q) * (mean + 1.0)).sqrt() / q
    }

    /// Instead of computing the full xor, we can take a short-cut when estimating the size from
    /// just the a small number of one bits. For this, we simply find the position of the n-th one bit
    /// via the xor iterator.
    pub fn estimate_diff_size(&self, other: &Self, ones: usize) -> f32 {
        let (cnt, bit) = self.nth_diff(other, ones + 1);
        if cnt == ones + 1 {
            let q = PHI_F32.powf(bit as f32 + 1.0);
            Self::upscale_mean(q, Self::mean(ones as f32))
        } else {
            Self::mean(cnt as f32)
        }
    }

    /// Converts the GeometricFilter into `BitChunk`s. Thereby, `BitChunk`s are returned from
    /// most to least significant, overlapping `BitChunk`s got merged, and empty ones filtered.
    pub fn bit_chunks(&self) -> impl Iterator<Item = BitChunk> + '_ {
        BitChunkIterator {
            trailing: self.least_significant_bits.bit_chunks().peekable(),
            leading: self.most_significant_bits.iter().copied().peekable(),
        }
    }

    /// Fast way to determine the nth most significant bit that differs between the two GeometricFilters.
    fn nth_diff(&self, other: &Self, ones: usize) -> (usize, usize) {
        // Note: we don't treat the most significant bits differently here, since that would result
        // in quite a lot of cases. Instead we convert both GeometricFilters into `BitChunk`s for which
        // the xor operation is trivial.
        let merged = XorBitChunksIterator {
            a: self.bit_chunks().peekable(),
            b: other.bit_chunks().peekable(),
        };

        let mut remaining = ones;
        for BitChunk(idx, block) in merged {
            // With sorted xor `BitChunk`s we can simply count the number of ones in each block
            // and determine the nth differing bit, once we reached the correct block.
            let block_ones = block.count_ones() as usize;
            if block_ones < remaining {
                remaining -= block_ones;
            } else {
                return (
                    ones,
                    idx as usize * 64 + nth_one(block, (block_ones - remaining) as u32) as usize,
                );
            }
        }
        (ones - remaining, 0)
    }

    /// Estimates the size of the represented set using the specified number of one bits.
    /// E.g. with ones = 20, a relative precision of ~0.22 is achieved.
    /// With ones = 128, a relative precision of ~0.12 is achieved.
    /// Using numbers larger than 128 won't increase the precision!
    pub fn estimate_size(&self, ones: usize) -> f32 {
        if let Some(bit) = self.nth_most_significant_one(ones) {
            let q = PHI_F32.powf(bit as f32 + 1.0);
            Self::upscale_mean(q, Self::mean(ones as f32))
        } else {
            Self::mean(self.count_ones() as f32)
        }
    }

    /// Computes an xor of the two underlying bitsets.
    /// This operation corresponds to computing the symmetric difference of the two
    /// sets represented by the GeometricFilters.
    pub fn xor(&self, other: &Self) -> GeometricFilter<'static> {
        let mut most_significant_bits = vec![];
        let mut least_significant_bits = BitVec::new();
        let mut largest_lsb = 0;
        for bit in self.iter_xor(other) {
            most_significant_bits.push(bit as u16);
            if most_significant_bits.len() >= ONES_U16_ENCODED {
                largest_lsb = bit;
                break;
            }
        }
        least_significant_bits.resize(largest_lsb);

        // Flip bits
        for filter in [self, other] {
            for bit in filter
                .most_significant_bits
                .iter()
                .rev()
                .map(|bit| *bit as usize)
            {
                if bit >= largest_lsb {
                    break;
                }
                least_significant_bits.toggle(bit);
            }
        }

        least_significant_bits.xor(&self.least_significant_bits);
        least_significant_bits.xor(&other.least_significant_bits);
        GeometricFilter {
            most_significant_bits: Cow::from(most_significant_bits),
            least_significant_bits,
        }
    }

    /// xor-ing random bits has amortized complexity O(1).
    /// The reason is that the probability to execute the expensive else branch happens with a geometric
    /// distribution. I.e. it will be hit O(log(n)) number of times out of n calls. And each of these
    /// executions has a constant complexity, but one that is rather high compared to the fast path. In total
    /// that makes the cost of the else case negligible.
    pub fn xor_bit(&mut self, bit_idx: usize) {
        if bit_idx < self.least_significant_bits.len() {
            self.least_significant_bits.toggle(bit_idx);
        } else {
            let msb = self.most_significant_bits.to_mut();
            match msb.binary_search_by(|k| bit_idx.cmp(&(*k as usize))) {
                Ok(idx) => {
                    msb.remove(idx);
                    let lsb_bit = self.least_significant_bits.iter_ones().next();
                    // Push the next most significant bit into the vector, such that the invariance
                    // of having the first ONES_DELTA_ENCODED one bits in the vector is satisfied.
                    if let Some(bit) = lsb_bit {
                        msb.push(bit as u16);
                        // If we get an lsb_bit, then it must be smaller than lsb.len()!
                        // As a result the subsequent shrinking of the lsb vector will implicitly clear this bit.
                    }
                    if msb.len() == ONES_U16_ENCODED {
                        self.least_significant_bits
                            .resize(msb[ONES_U16_ENCODED - 1] as usize);
                    } else {
                        self.least_significant_bits.resize(0);
                    }
                }
                Err(idx) => {
                    msb.insert(idx, bit_idx as u16);
                    match msb.len().cmp(&ONES_U16_ENCODED) {
                        Ordering::Greater => {
                            let msb_bit = msb
                                .pop()
                                .expect("we should have at least ONES_DELTA_ENCODED many entries!")
                                as usize;
                            let num_bits = msb[ONES_U16_ENCODED - 1] as usize;
                            self.least_significant_bits.resize(num_bits);
                            self.least_significant_bits.toggle(msb_bit);
                            assert_eq!(msb.len(), ONES_U16_ENCODED);
                        }
                        Ordering::Equal => {
                            let num_bits = msb[ONES_U16_ENCODED - 1] as usize;
                            self.least_significant_bits.resize(num_bits);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Consumers the current GeometricFilter and produces an owned GeometricFilter that
    // can live arbitrarily.
    pub fn into_owned(self) -> GeometricFilter<'static> {
        GeometricFilter {
            least_significant_bits: self.least_significant_bits.into_owned(),
            most_significant_bits: Cow::Owned(self.most_significant_bits.into_owned()),
        }
    }

    /// Collects the bits for the specified range into a u64.
    /// Non-existant bits will be filled in with zeros.
    /// The range must not be larger than 64.
    pub fn bit_range(&self, range: &Range<usize>) -> u64 {
        let mut result = self.least_significant_bits.bit_range(range);
        if self.least_significant_bits.len() < range.end {
            for bit in self.most_significant_bits.iter() {
                let bit = *bit as usize;
                if range.contains(&bit) {
                    result ^= 1 << (bit - range.start);
                }
            }
        }
        result
    }

    /// Given the expected size of a diff, this function returns the range of bucket ids which should
    /// be searched for in order to find geometric filters of the desired similarity. If at least half
    /// of the buckets in the range match, one found a match that has the expected diff size (or better).
    pub fn sim_hash_range(expected_diff_size: usize) -> Range<BucketId> {
        // Compute i, such that:
        // Self::upscale_mean(PHI_F32.powf(i), Self::mean(SIM_BUCKETS / 2.0)) == expected_diff_size
        let i = ((expected_diff_size as f32 + 1.0).log2()
            - (Self::mean(SIM_BUCKETS as f32 / 2.0) + 1.0).log2())
            * BITS_PER_LEVEL as f32;
        let start = i / SIM_BUCKET_SIZE as f32;
        let start = start.floor() as usize;
        start..(start + SIM_BUCKETS)
    }

    /// Given a geometric filter, we want to index roughly O(log(size)) many tokens while being able to
    /// find the document with geometric filters that differ by size. This function returns the range
    /// which satisfies those two properties.
    pub fn sim_hash_indexing_range(&self) -> Range<BucketId> {
        let msb = self.nth_most_significant_one(SIM_BUCKETS / 2 - 1);
        let last_bucket_id = msb.map(|b| b / SIM_BUCKET_SIZE).unwrap_or(0) + SIM_BUCKETS;
        0..last_bucket_id
    }

    /// Returns an iterator which produces all `SimHashes` of the `GeometricFilter`.
    /// The first argument in the tuple is the bucket id of the `SimHash` which can be used
    /// to select a certain subset of `SimHashes`. SimHashes are returned in decreasing order
    /// of bucket ids, since that's their natural construction order.
    pub fn sim_hashes(&self) -> impl Iterator<Item = (BucketId, SimHash)> + '_ {
        SimHashIterator::new(self)
    }

    pub fn sim_hashes_indexing(&self) -> impl Iterator<Item = SimHash> + '_ {
        let range = self.sim_hash_indexing_range();
        self.sim_hashes()
            .skip_while(move |(bucket_id, _)| *bucket_id >= range.end)
            .take_while(move |(bucket_id, _)| *bucket_id >= range.start)
            .map(|(_, sim_hash)| sim_hash)
    }

    pub fn sim_hashes_search(
        &self,
        expected_diff_size: usize,
    ) -> impl Iterator<Item = SimHash> + '_ {
        let range = Self::sim_hash_range(expected_diff_size);
        self.sim_hashes()
            .skip_while(move |(bucket_id, _)| *bucket_id >= range.end)
            .take_while(move |(bucket_id, _)| *bucket_id >= range.start)
            .map(|(_, sim_hash)| sim_hash)
    }

    pub fn toggle_git_file(&mut self, file: &GitFile) {
        self.xor_bit(file.bit_index())
    }
}

/// Maps a 64-bit hash into a bit index where bit buckets follow a geometric distribution.
/// The mapping is monotonically decreasing, i.e. u64::MAX gets mapped to bit 0.
pub fn bit_from_hash(hash: u64) -> usize {
    let levels = hash.leading_zeros() as usize;
    // Take the most significant non-zero 32 bits from the hash (and drop the first leading 1).
    let hash = (if levels > 31 {
        // Note: in this case we don't have 32 significant bits. So, we take the bits
        // that we actually have. This case is extremely unlikely to be hit and therefore the
        // resulting inaccuracies are not really relevant for us.
        hash << (levels - 31)
    } else {
        hash >> (31 - levels)
    } & 0xFFFFFFFF) as usize;
    // From those, the first BIT_MAPPING_BUCKETS_BITS bits determine the bucket index in our lookup table.
    let idx = hash >> (32 - BIT_MAPPING_BUCKETS_BITS);
    let offset = (hash < BIT_MAPPING[idx].1) as usize;
    offset + BIT_MAPPING[idx].0 + BITS_PER_LEVEL * levels
}

#[cfg(any(test, feature = "test-support"))]
pub fn random_filter(items: usize) -> GeometricFilter<'static> {
    use rand::{RngCore, SeedableRng};

    let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(items as u64);
    let mut filter = GeometricFilter::new();
    for _ in 0..items {
        filter.xor_bit(bit_from_hash(rng.next_u64()))
    }
    filter
}

pub struct GitFile<'a> {
    pub path: &'a str,
    pub blob_sha: &'a [u8],
}

impl<'a> GitFile<'a> {
    pub fn bit_index(&self) -> usize {
        bit_from_hash(self.stable_hash())
    }
}

impl<'a> StableHash for GitFile<'a> {
    fn stable_hash(&self) -> u64 {
        let mut hasher = FnvHasher::default();
        self.path.hash(&mut hasher);
        self.blob_sha.hash(&mut hasher);
        hasher.finish()
    }
}

/// Helper iterator which drops two consecutive items if they are equal.
/// Note: this is different from the unique operation! E.g. the sequence [1, 1, 1, 2, 2]
/// is converted by the XorIterator into the sequence [1], whereas the unique operation
/// would output [1, 2].
struct XorIterator<I: Iterator<Item = usize>> {
    iter: Peekable<I>,
}

impl<I: Iterator<Item = usize>> Iterator for XorIterator<I> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.iter.next() {
                Some(bit) if Some(&bit) == self.iter.peek() => {
                    self.iter.next();
                }
                x => return x,
            }
        }
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct BitChunk(pub u16, pub u64);

struct BitChunkIterator<I: Iterator<Item = BitChunk>, J: Iterator<Item = u16>> {
    trailing: Peekable<I>,
    leading: Peekable<J>,
}

impl<I: Iterator<Item = BitChunk>, J: Iterator<Item = u16>> Iterator for BitChunkIterator<I, J> {
    type Item = BitChunk;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(bit) = self.leading.next() {
            // Test whether it can be merged with trailing and more leading bits.
            let block_idx = bit / 64;
            let mut block = 1u64 << (bit % 64);
            loop {
                match self.leading.peek() {
                    Some(bit) if *bit / 64 == block_idx => {
                        block ^= 1u64 << (*bit % 64);
                        self.leading.next();
                    }
                    _ => break,
                }
            }
            match self.trailing.peek() {
                Some(BitChunk(other_idx, other)) if *other_idx == block_idx => {
                    block ^= *other;
                    self.trailing.next();
                }
                _ => {}
            }
            Some(BitChunk(block_idx, block))
        } else {
            self.trailing.next()
        }
    }
}

fn xor_bit_chunks(
    a: impl Iterator<Item = BitChunk>,
    b: impl Iterator<Item = BitChunk>,
) -> impl Iterator<Item = BitChunk> {
    XorBitChunksIterator {
        a: a.peekable(),
        b: b.peekable(),
    }
}

/// Xor-s two `BitChunk` iterators. Both iterators have to output `BitChunk`s from most
/// to least significant and must not report duplicates!
struct XorBitChunksIterator<I: Iterator<Item = BitChunk>, J: Iterator<Item = BitChunk>> {
    a: Peekable<I>,
    b: Peekable<J>,
}

impl<I: Iterator<Item = BitChunk>, J: Iterator<Item = BitChunk>> Iterator
    for XorBitChunksIterator<I, J>
{
    type Item = BitChunk;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match (self.a.peek(), self.b.peek()) {
                (None, _) => return self.b.next(),
                (_, None) => return self.a.next(),
                (Some(x), Some(y)) if x == y => {
                    self.a.next();
                    self.b.next();
                }
                (Some(BitChunk(x, ma)), Some(BitChunk(y, mb))) => {
                    return match x.cmp(y) {
                        Ordering::Equal => {
                            let res = Some(BitChunk(*x, *ma ^ *mb));
                            self.a.next();
                            self.b.next();
                            res
                        }
                        Ordering::Less => self.b.next(),
                        Ordering::Greater => self.a.next(),
                    }
                }
            }
        }
    }
}

/// ANDs the bit chunks with the repetitive mask without compressing the result to the one bits.
/// The result will only contain non-zero BitChunks!
fn mask_bit_chunks(
    iter: impl Iterator<Item = BitChunk>,
    mask: u64,
    modulus: u16,
) -> impl Iterator<Item = BitChunk> {
    MaskedBitChunksIterator::new(iter, mask, modulus)
}

struct MaskedBitChunksIterator<I: Iterator<Item = BitChunk>> {
    iter: I,
    block_mask: u64, // The mask repeated to fill the full block.
    modulus: u16,    // Number of bits the mask consists of.
}

impl<I: Iterator<Item = BitChunk>> MaskedBitChunksIterator<I> {
    fn new(iter: I, mask: u64, modulus: u16) -> Self {
        let mut block_mask = mask;
        let mut i = modulus;
        while i < 64 {
            block_mask |= block_mask << i;
            i *= 2;
        }
        Self {
            iter,
            block_mask,
            modulus,
        }
    }
}

impl<I: Iterator<Item = BitChunk>> Iterator for MaskedBitChunksIterator<I> {
    type Item = BitChunk;

    fn next(&mut self) -> Option<Self::Item> {
        for mut bit_chunk in &mut self.iter {
            let index = bit_chunk.0 * 64;
            let shift = index % self.modulus;
            let curr_mask =
                (self.block_mask >> shift) | (self.block_mask << (self.modulus - shift));
            bit_chunk.1 &= curr_mask;
            if bit_chunk.1 != 0 {
                return Some(bit_chunk);
            }
        }
        None
    }
}

impl<'a> Serializable<'a> for GeometricFilter<'a> {
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        if self.most_significant_bits.is_empty() {
            return Ok(0);
        }
        let mut encoded_bytes = self.most_significant_bits.as_bytes().write(writer)?;
        // Assert on invariances which are required by the deserialization!
        assert!(self.most_significant_bits.len() <= ONES_U16_ENCODED);
        assert!(
            self.most_significant_bits.len() == ONES_U16_ENCODED
                || self.least_significant_bits.is_empty()
        );
        encoded_bytes += self.least_significant_bits.write(writer)?;
        Ok(encoded_bytes)
    }

    fn from_bytes(buf: &'a [u8]) -> Self {
        if buf.is_empty() {
            return Self::default();
        }
        let msb_len = (buf.len() / 2).min(ONES_U16_ENCODED);
        let msb = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u16, msb_len) };
        Self {
            most_significant_bits: Cow::from(msb),
            least_significant_bits: BitVec::from_bytes(&buf[msb_len * 2..]),
        }
    }
}

impl<'a> From<Vec<u8>> for GeometricFilter<'a> {
    fn from(bytes: Vec<u8>) -> Self {
        GeometricFilter::from_bytes(&bytes).into_owned()
    }
}

impl<'a> From<GeometricFilter<'a>> for Vec<u8> {
    fn from(geo_filter: GeometricFilter<'a>) -> Self {
        let mut buffer = Vec::new();
        geo_filter
            .write(&mut buffer)
            .expect("writing to a vec never returns an error");
        buffer
    }
}

/// Given a bucket id i, all bit positions which satisfy the following inequality
/// for some non-negative k fall into that bucket:
/// (i + k * SIM_BUCKETS) * SIM_BUCKET_SIZE <= i < (i + k * SIM_BUCKETS + 1) * SIM_BUCKET_SIZE
///
/// Note:
///  1. Every bit falls into "infinitely" many buckets.
///  2. The bucket i covers all the bits of bucket (i + SIM_BUCKETS), plus SIM_BUCKET_SIZE additional ones.
///  3. The buckets i, ..., i + SIM_BUCKETS - 1 cover all bits starting at position i * SIM_BUCKET_SIZE
///     exactly once.
///
/// Property 2 is used to construct all sim hashes efficiently.
/// Property 3 is used to search for similar GeometricFilters.
struct SimHashIterator<'a> {
    filter: &'a GeometricFilter<'a>,
    prev_bucket_id: BucketId,
    sim_hash: [u64; SIM_BUCKETS],
}

impl<'a> SimHashIterator<'a> {
    pub fn new(filter: &'a GeometricFilter<'a>) -> Self {
        let msb = filter.nth_most_significant_one(0);
        let prev_bucket_id = msb.map(|b| b / SIM_BUCKET_SIZE).unwrap_or(0) + SIM_BUCKETS;
        Self {
            filter,
            prev_bucket_id,
            sim_hash: [0; SIM_BUCKETS],
        }
    }
}

impl<'a> Iterator for SimHashIterator<'a> {
    type Item = (BucketId, SimHash);

    fn next(&mut self) -> Option<Self::Item> {
        if self.prev_bucket_id == 0 {
            return None;
        }
        self.prev_bucket_id -= 1;
        let bit_start = self.prev_bucket_id * SIM_BUCKET_SIZE;
        let bit_range = self
            .filter
            .bit_range(&(bit_start..(bit_start + SIM_BUCKET_SIZE)));
        let bucket = self.prev_bucket_id % SIM_BUCKETS;
        self.sim_hash[bucket] =
            bit_range ^ self.sim_hash[bucket].rotate_left(SIM_BUCKET_SIZE as u32);
        Some((
            self.prev_bucket_id,
            SimHash::new(self.prev_bucket_id, self.sim_hash[bucket]),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::str::FromStr;

    use super::*;
    use github_sha::Sha;

    #[test]
    fn test_xor() {
        let a = GeometricFilter::from_ones((0..1000).collect_vec());
        let b = GeometricFilter::from_ones((10..1010).collect_vec());
        let c = a.xor(&b);
        let d = b.xor(&a);
        assert_eq!(a.iter_ones().count(), 1000);
        assert_eq!(b.iter_ones().count(), 1000);
        assert_eq!(
            c.iter_ones().collect_vec(),
            (0..10).chain(1000..1010).rev().collect_vec()
        );
        assert_eq!(c.iter_ones().collect_vec(), d.iter_ones().collect_vec());
    }

    #[test]
    fn test_bit_from_hash() {
        assert_eq!(bit_from_hash(u64::MAX), 0);
        assert_eq!(bit_from_hash(0), BITS_PER_LEVEL * 65 - 1);
        // Note: The test fails when going down to more than 40 leading zeros, since don't
        // have the required 32 significant bits. As a result the rounding fails.
        // Also, this is the only range that is practically relevant. All smaller hash
        // values are only relevant for bit sets with more than trillions of entries!
        for bit in 0..(BITS_PER_LEVEL * 40) {
            let lower_bound = 2f64.powf(64f64) * PHI_F64.powf((bit + 1) as f64);
            // Note: due to rounding issues we are testing hash values slight above or below the
            // computed limit.
            assert_eq!(bit_from_hash((lower_bound * 1.0000001) as u64), bit);
            assert_eq!(bit_from_hash((lower_bound * 0.9999999) as u64), bit + 1);
        }
    }

    #[test]
    fn test_xor_bit() {
        let mut m = GeometricFilter::new();
        m.xor_bit(10);
        assert_eq!(m.iter_ones().collect_vec(), vec![10]);
        m.xor_bit(10);
        assert!(m.iter_ones().collect_vec().is_empty());

        let mut m = GeometricFilter::from_ones((0..100).collect_vec());
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
        let cnt = 10000usize;
        let mut avg_precision = 0.0;
        let mut avg_high_precision = 0.0;
        let mut avg_var = 0.0;
        let mut avg_high_var = 0.0;
        let trials = 50;
        for _ in 0..trials {
            let mut m = GeometricFilter::new();
            // Insert cnt many random items.
            for i in 0..cnt {
                let mut hasher = DefaultHasher::new();
                i.hash(&mut hasher);
                let bit = bit_from_hash(hasher.finish());
                m.xor_bit(bit);
            }
            // Compute the relative error between estimate and actually inserted items.
            let precision = m.estimate_size(20) / cnt as f32 - 1.0;
            let high_precision = m.estimate_size(HIGH_PRECISION_ONES) / cnt as f32 - 1.0;
            // Take the average over trials many attempts.
            avg_precision += precision / trials as f32;
            avg_high_precision += high_precision / trials as f32;
            avg_var += precision.powf(2.0) / trials as f32;
            avg_high_var += high_precision.powf(2.0) / trials as f32;
        }
        println!(
            "avg precision: {} avg standard deviation: {} avg high precision: {} std: {}",
            avg_precision,
            avg_var.sqrt(),
            avg_high_precision,
            avg_high_var.sqrt(),
        );
        // Make sure that the estimate converges to the correct value.
        assert!(avg_precision.abs() < 0.05);
        // We should theoretically achieve a standard deviation of about 0.3
        assert!(avg_var.sqrt() < 0.4);
    }

    #[test]
    fn test_estimate_diff_size_fast() {
        let mut a = GeometricFilter::new();
        let mut b = GeometricFilter::new();
        for _ in 0..10000 {
            a.xor_bit(bit_from_hash(rand::random::<u64>()));
        }
        for _ in 0..1000 {
            b.xor_bit(bit_from_hash(rand::random::<u64>()));
        }
        let c = a.xor(&b);
        assert_eq!(c.estimate_size(20), a.estimate_diff_size(&b, 20));
        assert_eq!(c.estimate_size(20), b.estimate_diff_size(&a, 20));

        assert_eq!(c.estimate_size(128), a.estimate_diff_size(&b, 128));
        assert_eq!(c.estimate_size(128), b.estimate_diff_size(&a, 128));
    }

    #[test]
    fn test_serialization() {
        let mut m = GeometricFilter::new();
        for i in 0..1000 {
            m.xor_bit(i * 3);
        }

        let mut buffer = Vec::new();
        m.write(&mut buffer)
            .expect("writing into a buffer should never fail!");
        let d = GeometricFilter::from_bytes(buffer.as_slice());
        let x = m.xor(&d);
        assert_eq!(x.count_ones(), 0);
        assert_eq!(buffer.len(), 397);
    }

    #[test]
    fn test_toggle_git_file() {
        let mut m = GeometricFilter::new();

        let path = "foo.txt";
        let sha = Sha::from_str("2e2892743c977370ed75316f6038d8fd563bf9a6").unwrap();
        m.toggle_git_file(&GitFile {
            blob_sha: sha.as_bytes(),
            path,
        });

        assert_eq!(m, GeometricFilter::from_bytes(&[158, 0]));
    }

    #[test]
    fn test_clearing_of_bitvec() {
        // This triggers a deserialization bug.
        let mut m = GeometricFilter::new();
        for i in 100..=(100 + ONES_U16_ENCODED) {
            m.xor_bit(i);
        }
        // At this point we have ONES_U16_ENCODED + 1 many bits set.
        // Only one of them exists in the bitvec, the others are stored explicitly.
        m.xor_bit(100);
        // At this point, exactly ONES_U16_ENCODED bits are set.
        // The bitvec has no bits set, but isn't empty.
        m.xor_bit(101);
        // At this point, only ONES_U16_ENCODED - 1 bits are set.
        // Now the bitvec must be empty or serialization will fail!
        let mut buffer = Vec::new();
        m.write(&mut buffer).unwrap();
        let filter = GeometricFilter::from_bytes(&buffer);
        assert_eq!(m, filter);
        assert_eq!(m.count_ones(), filter.count_ones());
    }

    #[test]
    fn test_empty_bitvec_with_xor() {
        // This test verifies that when the xor has at most ONES_U16_ENCODED many bits
        // set, that the bitvec is empty. Otherwise, deserialization would fail.
        let mut m1 = GeometricFilter::new();
        for i in 100..=(100 + ONES_U16_ENCODED) {
            m1.xor_bit(i);
        }
        let mut m2 = GeometricFilter::new();
        m2.xor_bit(100);
        m2.xor_bit(101);
        let m3 = m1.xor(&m2);
        let mut buffer = Vec::new();
        m3.write(&mut buffer).unwrap();
        let filter = GeometricFilter::from_bytes(&buffer);
        assert_eq!(m3, filter);
        assert_eq!(m3.count_ones(), filter.count_ones());
    }

    #[test]
    fn test_random_bit_flips() {
        let mut m = GeometricFilter::new();
        for _ in 0..1000000 {
            let i = rand::random::<usize>() % 200;
            m.xor_bit(i);
            let mut buffer = Vec::new();
            m.write(&mut buffer).unwrap();
            let filter = GeometricFilter::from_bytes(&buffer);
            assert_eq!(m, filter);
            assert_eq!(m.count_ones(), filter.count_ones());
        }
    }

    #[test]
    fn test_masking() {
        // bit index                     :     12      6      0
        // bitset of the geometric filter: 011010 101101 001010
        // repeated bit mask             : 110100 110100 110100
        // masked bitset                 : 010000 100100 000000
        // after compression             : 01 0   10 1   00 0
        // bitset of the returned filter :           010 101000
        let m = GeometricFilter::from_ones(vec![16, 15, 13, 11, 9, 8, 6, 3, 1]);
        let masked = m.masked(0b110100, 6);
        assert_eq!(masked.iter_ones().collect_vec(), vec![16, 11, 8]);

        for i in 0..100 {
            let m = GeometricFilter::from_ones((0..i).collect_vec());
            let masked = m.masked(0b111, 3);
            assert_eq!(m, masked);
        }

        for i in 0..300 {
            let m = GeometricFilter::from_ones((0..i).collect_vec());
            let slow = GeometricFilter::from_descending_iter(m.iter_ones_masked(0b110, 3));
            let masked = m.masked(0b110, 3);
            assert_eq!(slow, masked, "in iteration: {i}");
        }
    }

    #[test]
    fn test_gray_code() {
        fn cmp(a: &[usize], b: &[usize]) -> Ordering {
            GeometricFilter::from_ones(a.to_vec())
                .cmp_gray_rank(&GeometricFilter::from_ones(b.to_vec()))
        }
        assert_eq!(cmp(&[], &[]), Ordering::Equal);
        assert_eq!(cmp(&[10], &[]), Ordering::Greater);
        assert_eq!(cmp(&[10], &[9]), Ordering::Greater);
        assert_eq!(cmp(&[10], &[11]), Ordering::Less);
        assert_eq!(cmp(&[10], &[10]), Ordering::Equal);
        assert_eq!(cmp(&[10, 8], &[10]), Ordering::Less);
        assert_eq!(cmp(&[10, 8], &[10, 7]), Ordering::Less);
        assert_eq!(cmp(&[10, 8], &[10, 9]), Ordering::Greater);
        assert_eq!(cmp(&[10, 8], &[10, 8]), Ordering::Equal);
        assert_eq!(cmp(&[10, 8, 6], &[10, 8]), Ordering::Greater);
        assert_eq!(cmp(&[10, 8, 6], &[10, 8, 5]), Ordering::Greater);
        assert_eq!(cmp(&[10, 8, 6], &[10, 8, 7]), Ordering::Less);
        assert_eq!(cmp(&[10, 8, 6], &[10, 8, 6]), Ordering::Equal);

        fn cmp2(a: &[usize], b: &[usize]) -> Ordering {
            let a = (1000usize..1020usize)
                .chain(a.iter().cloned())
                .collect_vec();
            let b = (1000usize..1020usize)
                .chain(b.iter().cloned())
                .collect_vec();
            GeometricFilter::from_ones(a).cmp_gray_rank(&GeometricFilter::from_ones(b))
        }
        assert_eq!(cmp2(&[], &[]), Ordering::Equal);
        assert_eq!(cmp2(&[10], &[]), Ordering::Greater);
        assert_eq!(cmp2(&[10], &[9]), Ordering::Greater);
        assert_eq!(cmp2(&[10], &[11]), Ordering::Less);
        assert_eq!(cmp2(&[10], &[10]), Ordering::Equal);
        assert_eq!(cmp2(&[10, 8], &[10]), Ordering::Less);
        assert_eq!(cmp2(&[10, 8], &[10, 7]), Ordering::Less);
        assert_eq!(cmp2(&[10, 8], &[10, 9]), Ordering::Greater);
        assert_eq!(cmp2(&[10, 8], &[10, 8]), Ordering::Equal);
        assert_eq!(cmp2(&[10, 8, 6], &[10, 8]), Ordering::Greater);
        assert_eq!(cmp2(&[10, 8, 6], &[10, 8, 5]), Ordering::Greater);
        assert_eq!(cmp2(&[10, 8, 6], &[10, 8, 7]), Ordering::Less);
        assert_eq!(cmp2(&[10, 8, 6], &[10, 8, 6]), Ordering::Equal);

        assert_eq!(cmp2(&[], &[]), Ordering::Equal);
        assert_eq!(cmp2(&[100], &[]), Ordering::Greater);
        assert_eq!(cmp2(&[100], &[1]), Ordering::Greater);
        assert_eq!(cmp2(&[100], &[200]), Ordering::Less);
        assert_eq!(cmp2(&[100], &[100]), Ordering::Equal);
        assert_eq!(cmp2(&[100, 20], &[100]), Ordering::Less);
        assert_eq!(cmp2(&[100, 20], &[100, 10]), Ordering::Less);
        assert_eq!(cmp2(&[100, 20], &[100, 30]), Ordering::Greater);
        assert_eq!(cmp2(&[100, 20], &[100, 20]), Ordering::Equal);
    }

    #[test]
    fn test_xor_plus_mask() {
        let mask_size = 12;
        let mask = 0b100001100000;
        let mut a = GeometricFilter::new();
        for _ in 0..10000 {
            a.xor_bit(bit_from_hash(rand::random::<u64>()));
        }
        let mut expected = GeometricFilter::new();
        let mut b = a.clone();
        for _ in 0..1000 {
            let bit = bit_from_hash(rand::random::<u64>());
            b.xor_bit(bit);
            expected.xor_bit(bit);
            assert_eq!(expected, a.xor(&b));

            let masked_a = a.masked(mask, mask_size);
            let masked_b = b.masked(mask, mask_size);
            let masked_expected = expected.masked(mask, mask_size);
            assert_eq!(masked_expected, masked_a.xor(&masked_b));
        }
    }
}
