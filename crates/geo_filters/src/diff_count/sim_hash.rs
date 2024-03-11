//! Similarity hashing allows quickly finding similar sets with a reverse index.

use std::hash::{Hash, Hasher};
use std::ops::Range;

use fnv::FnvHasher;

use crate::config::{BitChunk, GeoConfig, IsBucketType};
use crate::diff_count::GeoDiffCount;
use crate::Diff;

use super::BitVec;

/// Number of bits covered by each SimHash bucket.
pub(crate) const SIM_BUCKET_SIZE: usize = 6;
/// Number of consecutive SimHash buckets used for searching.
pub(crate) const SIM_BUCKETS: usize = 20;

pub type BucketId = usize;

/// SimHash is a hash computed over a continuous range of bits from a GeoDiffCount.
/// It is used to quickly find similar sets with a reverse index.
#[derive(Copy, Clone, Default, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct SimHash(pub u64);

impl SimHash {
    /// Constructs a `SimHash` from the bit set extracted from the `GeoDiffCount`.
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
}

impl<C: GeoConfig<Diff>> GeoDiffCount<'_, C> {
    /// Given the expected size of a diff, this function returns the range of bucket ids which should
    /// be searched for in order to find geometric filters of the desired similarity. If at least half
    /// of the buckets in the range match, one found a match that has the expected diff size (or better).
    pub fn sim_hash_range(&self, expected_diff_size: usize) -> Range<BucketId> {
        // Compute i, such that:
        // Self::upscale_mean(PHI_F32.powf(i), Self::mean(SIM_BUCKETS / 2.0)) == expected_diff_size
        let i = ((expected_diff_size as f32 + 1.0).log2()
            - (self.config.expected_items(SIM_BUCKETS / 2) + 1.0).log2())
            * self.config.bits_per_level() as f32;
        let start = i / SIM_BUCKET_SIZE as f32;
        let start = start.floor() as usize;
        start..(start + SIM_BUCKETS)
    }

    /// Given a GeoDiffCount, we want to index roughly O(log(size)) many tokens while being able to
    /// find the document with geometric filters that differ by size. This function returns the range
    /// which satisfies those two properties.
    pub fn sim_hash_indexing_range(&self) -> Range<BucketId> {
        let msb = self.nth_most_significant_one(SIM_BUCKETS / 2 - 1);
        let last_bucket_id =
            msb.map(|b| b.into_usize() / SIM_BUCKET_SIZE).unwrap_or(0) + SIM_BUCKETS;
        0..last_bucket_id
    }

    /// Returns an iterator which produces all `SimHashes` of the `GeoDiffCount`.
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
        let range = self.sim_hash_range(expected_diff_size);
        self.sim_hashes()
            .skip_while(move |(bucket_id, _)| *bucket_id >= range.end)
            .take_while(move |(bucket_id, _)| *bucket_id >= range.start)
            .map(|(_, sim_hash)| sim_hash)
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
/// Property 3 is used to search for similar GeoDiffCounts.
struct SimHashIterator<'a, C: GeoConfig<Diff>> {
    filter: &'a GeoDiffCount<'a, C>,
    prev_bucket_id: BucketId,
    sim_hash: [u64; SIM_BUCKETS],
}

impl<'a, C: GeoConfig<Diff>> SimHashIterator<'a, C> {
    pub fn new(filter: &'a GeoDiffCount<'a, C>) -> Self {
        let msb = filter.nth_most_significant_one(0);
        let prev_bucket_id =
            msb.map(|b| b.into_usize() / SIM_BUCKET_SIZE).unwrap_or(0) + SIM_BUCKETS;
        Self {
            filter,
            prev_bucket_id,
            sim_hash: [0; SIM_BUCKETS],
        }
    }
}

impl<'a, C: GeoConfig<Diff>> Iterator for SimHashIterator<'a, C> {
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

impl<'a, C: GeoConfig<Diff>> GeoDiffCount<'a, C> {
    /// n specifies the desired zero-based index of the most significant one.
    /// The zero-based index of the desired one bit is returned.
    fn nth_most_significant_one(&self, mut n: usize) -> Option<C::BucketType> {
        if n < self.msb.len() {
            Some(self.msb[n])
        } else {
            n -= self.msb.len();
            self.lsb
                .nth_most_significant_one(n)
                .map(C::BucketType::from_usize)
        }
    }

    /// Collects the bits for the specified range into a u64.
    /// Non-existant bits will be filled in with zeros.
    /// The range must not be larger than 64.
    fn bit_range(&self, range: &Range<usize>) -> u64 {
        let mut result = self.lsb.bit_range(range);
        if self.lsb.num_bits() < range.end {
            for bit in self.msb.iter() {
                let bit = bit.into_usize();
                if range.contains(&bit) {
                    result ^= (bit - range.start).into_block();
                }
            }
        }
        result
    }
}

impl<'a> BitVec<'a> {
    /// n specifies the desired zero-based index of the most significant one.
    /// The zero-based index of the desired one bit is returned.
    pub fn nth_most_significant_one(&self, mut n: usize) -> Option<usize> {
        for BitChunk { index, block } in self.bit_chunks() {
            let ones = block.count_ones() as usize;
            if n < ones {
                return Some(usize::from_block_nth_msb(block, n).into_bucket(index));
            } else {
                n -= ones;
            }
        }
        None
    }
}
