//! Geometric filter implementation for diff count.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::hash::BuildHasher as _;
use std::mem::{size_of, size_of_val};
use std::ops::Deref as _;

use crate::config::{
    count_ones_from_bitchunks, count_ones_from_msb_and_lsb, iter_bit_chunks, iter_ones,
    mask_bit_chunks, take_ref, xor_bit_chunks, BitChunk, GeoConfig, IsBucketType, BITS_PER_BLOCK,
};
use crate::{Count, Diff};

mod bitvec;
mod config;
mod metric;
mod sim_hash;

use bitvec::*;
pub use config::{GeoDiffConfig10, GeoDiffConfig13, GeoDiffConfig7};
pub use metric::{GeoDiffMetric, OnesMetric};
pub use sim_hash::SimHash;

/// Diff count filter with a relative error standard deviation of ~0.125.
pub type GeoDiffCount7<'a> = GeoDiffCount<'a, GeoDiffConfig7>;

/// Diff count filter with a relative error standard deviation of ~0.04.
pub type GeoDiffCount10<'a> = GeoDiffCount<'a, GeoDiffConfig10>;

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

impl<'a, C: GeoConfig<Diff>> GeoDiffCount<'a, C> {
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

    /// Builds a sort key from the most significant bits of the masked filter.
    ///
    /// The key packs the largest bucket positions of the masked filter into a single `u64`,
    /// most significant position first. Because masking distributes over the xor used by
    /// [`Self::cmp_masked`], comparing two keys numerically yields the same ordering as
    /// [`Self::cmp_masked`] whenever the keys differ. When two keys are equal the ordering is
    /// undetermined and the caller must fall back to [`Self::cmp_masked`], e.g.
    /// `a_key.cmp(&b_key).then_with(|| a.cmp_masked(b, mask, mask_size))`.
    ///
    /// Each position occupies `C::BucketType::BITS` bits, so the key holds
    /// `64 / C::BucketType::BITS` positions (4 for `u16`, 2 for `u32`).
    pub fn masked_sort_key(&self, mask: u64, mask_size: usize) -> u64 {
        assert!(
            (1..u64::BITS as usize).contains(&mask_size),
            "mask_size must be in 1..=63 (got {mask_size})"
        );
        let bits = C::BucketType::BITS;
        debug_assert!(
            (1..=32).contains(&bits) && u64::BITS % bits == 0,
            "sort key packing requires a bucket type of at most 32 bits"
        );
        let per_word = (u64::BITS / bits) as usize;

        // The most significant bits are stored sparsely and sorted from largest to smallest, so we
        // can test each of them against the periodic mask directly, avoiding the more expensive
        // merge performed by `bit_chunks`. The least significant bits are dense and always below
        // every most significant bit, so we mask whole 64-bit blocks at once. Chaining the two
        // therefore yields the masked one-bit positions from largest to smallest.
        let msb = self
            .msb
            .iter()
            .map(|&p| p.into_usize())
            .filter(move |&pos| (mask >> (pos % mask_size)) & 1 != 0)
            .map(|pos| pos as u64);
        let lsb = iter_ones::<C::BucketType, _>(
            mask_bit_chunks(self.lsb.bit_chunks(), mask, mask_size).peekable(),
        )
        .map(|b| b.into_usize() as u64);
        let mut positions = msb.chain(lsb);

        let mut key = 0u64;
        for _ in 0..per_word {
            key = (key << bits) | positions.next().unwrap_or(0);
        }
        key
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
            // The bit being toggled is within our LSB bit vector
            // so toggle it directly.
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
                        // We have too many values in the MSB sparse index vector,
                        // let's move the smalles MSB value into the LSB bit vector
                        let smallest = msb
                            .pop()
                            .expect("we should have at least one element!")
                            .into_usize();
                        let new_smallest = msb
                            .last()
                            .expect("should have at least one element")
                            .into_usize();
                        // ensure LSB bit vector has the space for `smallest`
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

    // Serialization:
    //
    // Since most of our target platforms are little endian there are more optimised approaches
    // for little endian platforms, just splatting the bytes into the writer. This is contrary
    // to the usual "network endian" approach where big endian is the default, but most of our
    // consumers are little endian so it makes sense for this to be the optimal approach.
    //
    // For now we do not support big endian platforms. In the future we might add a big endian
    // platform specific implementation which is able to read the little endian serialized
    // representation. For now, if you attempt to serialize a filter on a big endian platform
    // you get a panic.

    /// Create a new [`GeoDiffCount`] from a slice of bytes
    #[cfg(target_endian = "little")]
    pub fn from_bytes_with_config(c: C, buf: &'a [u8]) -> Self {
        if buf.is_empty() {
            return Self::new(c);
        }
        // The number of most significant bits stores in the MSB sparse repr
        let msb_len = (buf.len() / size_of::<C::BucketType>()).min(c.max_msb_len());
        let msb = unsafe {
            std::mem::transmute::<&[u8], &[C::BucketType]>(std::slice::from_raw_parts(
                buf.as_ptr(),
                msb_len,
            ))
        };
        // The number of bytes representing the MSB - this is how many bytes we need to
        // skip over to reach the LSB
        let msb_bytes_len = msb_len * size_of::<C::BucketType>();
        Self {
            config: c,
            msb: Cow::Borrowed(msb),
            lsb: BitVec::from_bytes(&buf[msb_bytes_len..]),
        }
    }

    #[cfg(target_endian = "little")]
    pub fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        if self.msb.is_empty() {
            return Ok(0);
        }
        let msb_buckets = self.msb.deref();
        let msb_bytes = unsafe {
            std::slice::from_raw_parts(msb_buckets.as_ptr() as *const u8, size_of_val(msb_buckets))
        };
        writer.write_all(msb_bytes)?;
        let mut bytes_written = msb_bytes.len();
        bytes_written += self.lsb.write(writer)?;
        Ok(bytes_written)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn from_ones_with_config(config: C, ones: impl IntoIterator<Item = C::BucketType>) -> Self {
        let mut result = Self::new(config);
        for one in ones {
            result.xor_bit(one);
        }
        result
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn iter_ones(&self) -> impl Iterator<Item = C::BucketType> + '_ {
        iter_ones(self.bit_chunks().peekable()).map(C::BucketType::from_usize)
    }

    /// Generate a pseudo-random filter. The RNG used to build the filter
    /// is seeded using the number of items so for a given number of items
    /// the resulting geofilter should always be the same.
    #[cfg(any(test, feature = "test-support"))]
    pub fn pseudorandom_filter_with_config(config: C, items: usize) -> Self {
        use rand_chacha::rand_core::{Rng, SeedableRng};

        let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(items as u64);
        let mut filter = Self::new(config);
        for _ in 0..items {
            filter.push_hash(rng.next_u64());
        }
        filter
    }
}

impl<'a, C: GeoConfig<Diff> + Default> GeoDiffCount<'a, C> {
    #[cfg(target_endian = "little")]
    pub fn from_bytes(buf: &'a [u8]) -> Self {
        Self::from_bytes_with_config(C::default(), buf)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn from_ones(ones: impl IntoIterator<Item = C::BucketType>) -> Self {
        Self::from_ones_with_config(C::default(), ones)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn pseudorandom_filter(items: usize) -> Self {
        Self::pseudorandom_filter_with_config(C::default(), items)
    }
}

/// Applies a repeated bit mask to the underlying filter.
/// E.g. given the bit mask `0b110100` with modulus 6, we filter the bitset of the geometric filter as follows:
/// bitset of the geometric filter: 011010 101101 001010
/// repeated bit mask             : 110100 110100 110100
/// masked bitset                 : 010000 100100 000000
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

/// Estimates the split bucket separating the sparse most-significant buckets ("numbers") from
/// the dense least-significant buckets ("bits") for a filter built from `n` hashes.
///
/// The expected number of hashes falling into buckets `>= s` is `n * phi^s`. We target about
/// `max_msb_len / 2` such hashes: the most-significant buckets do *not* need to be fully supplied
/// by the collected numbers, since [`GeoDiffCount::from_bit_chunks`] re-splits the combined stream
/// and pulls the remainder from the dense bits. Because the buckets are geometric, raising the
/// split by one `bits_per_level` roughly halves the collected set, so a small target keeps the
/// sort cheap while only marginally enlarging the bit vector. Correctness does not depend on the
/// estimate.
fn estimate_split_bucket<C: GeoConfig<Diff>>(config: &C, n: usize) -> usize {
    let target = config.max_msb_len() / 2;
    if n <= target {
        // Every hash ends up in `numbers` (split == 0).
        return 0;
    }
    let ratio = target as f64 / n as f64;
    ((ratio.ln() / config.phi_f64().ln()).floor() as usize)
        // No bucket can ever exceed this bound, so never allocate a larger bit vector.
        .min(64 * config.bits_per_level())
}

/// Splits a descending stream of set buckets into the new msb (the top `max_msb_len`) and folds
/// the remaining buckets into `lsb`, resizing it to the new boundary. If the stream is too short
/// to fill the msb, the highest bits of `lsb` are pulled back out to refill it (and `lsb` is
/// truncated accordingly, or emptied if it could not be refilled). Returns the new msb.
fn split_into_msb<T: IsBucketType>(
    mut buckets: impl Iterator<Item = T>,
    lsb: &mut BitVec<'_>,
    max_msb_len: usize,
) -> Vec<T> {
    let mut msb: Vec<T> = Vec::with_capacity(max_msb_len);
    msb.extend(buckets.by_ref().take(max_msb_len));
    if msb.len() == max_msb_len {
        // The msb is full: its smallest entry is the new boundary, the rest folds into the bits.
        let smallest = msb[max_msb_len - 1].into_usize();
        lsb.resize(smallest);
        let mut toggler = lsb.toggler();
        for bucket in buckets {
            toggler.toggle(bucket.into_usize());
        }
    } else {
        // Refill the msb from the highest bits, then truncate the bits to the new boundary.
        let need = max_msb_len - msb.len();
        let pulled: Vec<T> = iter_ones::<T, _>(lsb.bit_chunks().peekable())
            .take(need)
            .collect();
        let smallest = if pulled.len() == need {
            pulled[need - 1].into_usize()
        } else {
            0
        };
        msb.extend(pulled);
        lsb.resize(smallest);
    }
    msb
}

/// Incrementally builds a [`GeoDiffCount`] from a known number of pushes.
///
/// Hashes are added one at a time via [`Self::push_hash`] / [`Self::push`], or in bulk via
/// [`Self::extend_by_hashes`]. Reserve the expected number of pushes with [`Self::with_capacity`]
/// so the dense/sparse split can be estimated and the buffers presized. The most-significant
/// buckets accumulate in a plain vector without enforcing the `max_msb_len` limit; that limit, and
/// the filter invariants, are applied only once when [`Self::build`] turns the builder into a
/// [`GeoDiffCount`]. Pushing more (or fewer) hashes than reserved stays correct — only the presizing
/// is then less accurate. If the final count is not known up front, call [`Self::reserve`] as it
/// grows.
pub struct GeoDiffCountBuilder<C: GeoConfig<Diff>> {
    config: C,
    /// Running total of pushes reserved for; drives the split estimate.
    expected: usize,
    /// Buckets at or above `split` accumulate in `numbers` (with duplicates, and transiently some
    /// below `split` after a [`GeoDiffCountBuilder::reserve`]); buckets below `split` are folded
    /// (xor) into `blocks`. [`GeoDiffCountBuilder::cleanup`] reconciles the two.
    split: usize,
    numbers: Vec<usize>,
    blocks: Vec<u64>,
}

impl<C: GeoConfig<Diff>> GeoDiffCountBuilder<C> {
    /// Creates a builder reserving space for roughly `expected` pushes.
    ///
    /// `expected` only positions the dense/sparse split; the `numbers` buffer is a fixed
    /// `2 * max_msb_len` working set that is compacted in place once full (see [`Self::push_hash`]),
    /// so it never needs to be sized to the number of pushes.
    pub fn with_capacity(config: C, expected: usize) -> Self {
        let split = estimate_split_bucket(&config, expected);
        let capacity = 2 * config.max_msb_len();
        Self {
            config,
            expected,
            split,
            numbers: Vec::with_capacity(capacity),
            blocks: vec![0; split.div_ceil(BITS_PER_BLOCK)],
        }
    }

    /// Reserves space for `additional` further pushes.
    ///
    /// This only advances the estimated split (growing the bit space to match) so that subsequent
    /// pushes of low buckets fold straight into the bits. The numbers already collected below the
    /// new split are *not* migrated here — they are folded in lazily the next time the buffer is
    /// compacted or built (see [`Self::cleanup`]). The resulting filter is unaffected.
    pub fn reserve(&mut self, additional: usize) {
        self.expected = self.expected.saturating_add(additional);
        let new_split = estimate_split_bucket(&self.config, self.expected);
        if new_split > self.split {
            self.split = new_split;
            self.blocks.resize(new_split.div_ceil(BITS_PER_BLOCK), 0);
        }
    }

    /// Sorts `numbers` and reduces it to the distinct buckets that still belong above the split:
    /// even occurrences cancel (xor), and any bucket below the current split is folded into the bit
    /// space. Afterwards `numbers` is sorted in descending order with no duplicates and no entries
    /// below `split`. Shared by [`Self::compact`] and [`Self::build`].
    fn cleanup(&mut self) {
        self.numbers.sort_unstable_by(|a, b| b.cmp(a));
        let split = self.split;
        let blocks = &mut self.blocks;
        let numbers = &mut self.numbers;
        let mut write = 0;
        let mut read = 0;
        while read < numbers.len() {
            let bucket = numbers[read];
            let mut next = read + 1;
            while next < numbers.len() && numbers[next] == bucket {
                next += 1;
            }
            // An odd number of occurrences leaves the bucket set; an even number cancels.
            if (next - read) % 2 == 1 {
                if bucket < split {
                    let (index, bit) = bucket.into_index_and_bit();
                    blocks[index] ^= bit.into_block();
                } else {
                    numbers[write] = bucket;
                    write += 1;
                }
            }
            read = next;
        }
        numbers.truncate(write);
    }

    /// Processes a full `numbers` buffer in place rather than letting it grow. [`Self::cleanup`]
    /// first collapses duplicates and any sub-split entries; if that already frees half the buffer
    /// the split stays put. Otherwise the split is advanced in whole levels — each level halves the
    /// expected number of buckets at or above it — until at most half the buffer remains, folding
    /// the now-sub-split buckets into the bit space. The buffer is therefore never reallocated.
    fn compact(&mut self) {
        let target = self.numbers.capacity() / 2;
        self.cleanup();
        if self.numbers.len() <= target {
            return;
        }
        // `numbers` is sorted descending, so the count at or above a split is a prefix length.
        let bits_per_level = self.config.bits_per_level();
        let mut new_split = self.split;
        let mut keep = self.numbers.len();
        while keep > target {
            new_split += bits_per_level;
            keep = self.numbers.partition_point(|&b| b >= new_split);
        }
        self.blocks.resize(new_split.div_ceil(BITS_PER_BLOCK), 0);
        let blocks = &mut self.blocks;
        for &bucket in &self.numbers[keep..] {
            let (index, bit) = bucket.into_index_and_bit();
            blocks[index] ^= bit.into_block();
        }
        self.numbers.truncate(keep);
        self.split = new_split;
    }

    /// Adds the given hash to the filter being built.
    #[inline]
    pub fn push_hash(&mut self, hash: u64) {
        let bucket = self.config.hash_to_bucket(hash).into_usize();
        if bucket >= self.split {
            // Compact the buffer in place once it is full rather than reallocating it. Compacting
            // may advance the split past this bucket, in which case it lands in `numbers` below the
            // split; the next `cleanup` simply folds it into the bits, so this stays correct.
            if self.numbers.len() == self.numbers.capacity() {
                self.compact();
            }
            self.numbers.push(bucket);
        } else {
            // `bucket < split`, so the block index is always in range; toggling cancels repeats.
            let (index, bit) = bucket.into_index_and_bit();
            self.blocks[index] ^= bit.into_block();
        }
    }

    /// Adds the hash of the given item, computed with the configured hasher, to the filter.
    pub fn push<I: std::hash::Hash>(&mut self, item: I) {
        let build_hasher = C::BuildHasher::default();
        self.push_hash(build_hasher.hash_one(item));
    }

    /// Inserts a batch of hashes, reserving room for them up front via the size estimator.
    ///
    /// Unlike a loop of [`Self::push_hash`] calls — which must re-resolve `self` on every call —
    /// this folds the dense low buckets into the bit space in a tight loop that hoists the bit
    /// storage out of the per-hash work, only re-acquiring it after the rare in-place compaction.
    /// It can be mixed freely with [`Self::push_hash`], and further pushes remain possible after.
    pub fn extend_by_hashes(&mut self, mut hashes: impl ExactSizeIterator<Item = u64>) {
        self.reserve(hashes.len());
        loop {
            let split = self.split;
            let filled = {
                let config = &self.config;
                let blocks = &mut self.blocks;
                let numbers = &mut self.numbers;
                let mut filled = false;
                for hash in hashes.by_ref() {
                    let bucket = config.hash_to_bucket(hash).into_usize();
                    if bucket >= split {
                        numbers.push(bucket);
                        // Stop exactly at capacity so the buffer is never reallocated.
                        if numbers.len() == numbers.capacity() {
                            filled = true;
                            break;
                        }
                    } else {
                        let (index, bit) = bucket.into_index_and_bit();
                        blocks[index] ^= bit.into_block();
                    }
                }
                filled
            };
            // The iterator is either exhausted or the buffer filled; compact and continue if full.
            if !filled {
                break;
            }
            self.compact();
        }
    }

    /// Finalizes the builder into a [`GeoDiffCount`], applying the `max_msb_len` constraint and
    /// re-establishing the filter invariants.
    pub fn build(mut self) -> GeoDiffCount<'static, C> {
        let max_msb_len = self.config.max_msb_len();
        // `cleanup` leaves `numbers` sorted descending, deduplicated, and free of sub-split entries.
        self.cleanup();
        let mut lsb = BitVec::from_blocks(self.blocks, self.split);
        let msb = split_into_msb(
            self.numbers.iter().map(|&b| C::BucketType::from_usize(b)),
            &mut lsb,
            max_msb_len,
        );
        let result = GeoDiffCount {
            config: self.config,
            msb: Cow::from(msb),
            lsb,
        };
        result.debug_assert_invariants();
        result
    }
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

    fn size_f32(&self) -> f32 {
        self.estimate_size()
    }

    fn size_with_sketch_f32(&self, other: &Self) -> f32 {
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
    use std::io::Write;

    use itertools::Itertools;
    use rand::{seq::IteratorRandom, Rng as RngCore};
    use rand_chacha::ChaCha12Rng;

    use crate::{
        build_hasher::UnstableDefaultBuildHasher,
        config::{tests::test_estimate, FixedConfig},
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
            assert_eq!(result, geo_count.size_f32());
        }
    }

    #[test]
    fn test_xor() {
        let a = GeoDiffCount7::from_ones(0..1000);
        let b = GeoDiffCount7::from_ones(10..1010);
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

        let mut m = GeoDiffCount7::from_ones(0..100);
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

    /// Building a filter via `GeoDiffCountBuilder` must produce exactly the same filter as pushing
    /// the hashes one by one, regardless of how accurately the capacity was reserved.
    #[test]
    fn test_builder() {
        fn assert_builder_matches<C: GeoConfig<Diff> + Default>(hashes: &[u64], reserve: usize) {
            let mut expected: GeoDiffCount<'static, C> = GeoDiffCount::new(C::default());
            for &hash in hashes {
                expected.push_hash(hash);
            }
            let mut builder = GeoDiffCountBuilder::with_capacity(C::default(), reserve);
            for &hash in hashes {
                builder.push_hash(hash);
            }
            let actual = builder.build();
            let label = (hashes.len(), reserve);
            assert_eq!(expected, actual, "filter mismatch for {label:?}");
            assert_eq!(
                expected.iter_ones().collect_vec(),
                actual.iter_ones().collect_vec(),
                "ones mismatch for {label:?}",
            );
        }

        // Starts with a tiny reservation and grows it while pushing, which moves the split forward
        // and exercises the number-migration path in `reserve`.
        fn assert_grown_builder_matches<C: GeoConfig<Diff> + Default>(hashes: &[u64]) {
            let mut expected: GeoDiffCount<'static, C> = GeoDiffCount::new(C::default());
            for &hash in hashes {
                expected.push_hash(hash);
            }
            let mut builder = GeoDiffCountBuilder::with_capacity(C::default(), 1);
            for (i, &hash) in hashes.iter().enumerate() {
                if i % 64 == 0 {
                    builder.reserve(64);
                }
                builder.push_hash(hash);
            }
            assert_eq!(expected, builder.build(), "grown builder mismatch");
        }

        prng_test_harness(4, |rnd| {
            for n in [0usize, 1, 5, 50, 500, 5000, 50000] {
                let pool: Vec<u64> = (0..n.div_ceil(2).max(1)).map(|_| rnd.next_u64()).collect();
                let hashes: Vec<u64> = (0..n)
                    .map(|_| *pool.iter().choose(rnd).expect("pool is non-empty"))
                    .collect();
                // Reserve exactly, far too little (split too low), and far too much (split too high).
                assert_builder_matches::<GeoDiffConfig7>(&hashes, n);
                assert_builder_matches::<GeoDiffConfig13>(&hashes, n);
                assert_builder_matches::<GeoDiffConfig13>(&hashes, n / 4);
                assert_builder_matches::<GeoDiffConfig13>(&hashes, n * 4);
                // Reserve nothing so the split starts at 0 and every bucket initially lands in
                // `numbers`, forcing repeated compaction once the fixed-size buffer fills. This
                // hammers the lazy-flush path, including buckets that land below the split a
                // compaction just advanced past.
                assert_builder_matches::<GeoDiffConfig7>(&hashes, 0);
                assert_builder_matches::<GeoDiffConfig13>(&hashes, 0);
                assert_grown_builder_matches::<GeoDiffConfig7>(&hashes);
                assert_grown_builder_matches::<GeoDiffConfig13>(&hashes);
            }
        });
    }

    /// `GeoDiffCountBuilder::extend_by_hashes` (alone, or mixed with `push_hash`) must produce
    /// exactly the same filter as pushing every hash one by one.
    #[test]
    fn test_builder_extend() {
        fn assert_extend_matches<C: GeoConfig<Diff> + Default>(hashes: &[u64]) {
            let mut expected: GeoDiffCount<'static, C> = GeoDiffCount::new(C::default());
            for &hash in hashes {
                expected.push_hash(hash);
            }

            // Extend a fresh builder in one batch (auto-reserves for the batch size).
            let mut batched = GeoDiffCountBuilder::with_capacity(C::default(), 0);
            batched.extend_by_hashes(hashes.iter().copied());
            assert_eq!(expected, batched.build(), "extend-from-empty mismatch");

            // Push a prefix one by one, then extend with the remainder.
            let mid = hashes.len() / 2;
            let mut mixed = GeoDiffCountBuilder::with_capacity(C::default(), 0);
            for &hash in &hashes[..mid] {
                mixed.push_hash(hash);
            }
            mixed.extend_by_hashes(hashes[mid..].iter().copied());
            assert_eq!(expected, mixed.build(), "push+extend mismatch");
        }

        prng_test_harness(4, |rnd| {
            for n in [0usize, 1, 5, 50, 500, 5000, 50000] {
                // Draw from a smaller pool so buckets repeat, exercising xor cancellation.
                let pool: Vec<u64> = (0..n.div_ceil(2).max(1)).map(|_| rnd.next_u64()).collect();
                let hashes: Vec<u64> = (0..n)
                    .map(|_| *pool.iter().choose(rnd).expect("pool is non-empty"))
                    .collect();
                assert_extend_matches::<GeoDiffConfig7>(&hashes);
                assert_extend_matches::<GeoDiffConfig13>(&hashes);
            }
        });
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
        let m = GeoDiffCount7::from_ones([16, 15, 13, 11, 9, 8, 6, 3, 1]);
        let n = masked(&m, 0b110100, 6);
        assert_eq!(n.iter_ones().collect_vec(), vec![16, 11, 8]);

        for i in 0..100 {
            let m = GeoDiffCount7::from_ones((0..i).collect_vec());
            let n = masked(&m, 0b111, 3);
            assert_eq!(m, n);
        }

        for i in 0..300 {
            let m = GeoDiffCount7::from_ones((0..i).collect_vec());
            let slow = GeoDiffCount::from_ones(masked(&m, 0b110, 3).iter_ones());
            let n = masked(&m, 0b110, 3);
            assert_eq!(slow, n, "in iteration: {i}");
        }
    }

    #[test]
    fn test_xor_plus_mask() {
        prng_test_harness(10, |rnd| {
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
    fn test_masked_sort_key() {
        let masks: &[(u64, usize)] = &[
            (0b1, 1),   // keeps every bit, i.e. a full comparison
            (0b10, 2),  // keeps every other bit
            (0b110, 3), // keeps two out of every three bits
            (0b110100, 6),
            (0b100001100000, 12),
        ];

        fn check<C: GeoConfig<Diff> + Default>(rnd: &mut ChaCha12Rng, masks: &[(u64, usize)]) {
            let mut build = || {
                let mut f = GeoDiffCount::<C>::new(C::default());
                for _ in 0..1000 {
                    f.push_hash(rnd.next_u64());
                }
                f
            };
            let a = build();
            let b = build();
            for &(mask, mask_size) in masks {
                let ka = a.masked_sort_key(mask, mask_size);
                let kb = b.masked_sort_key(mask, mask_size);
                let expected = a.cmp_masked(&b, mask, mask_size);
                // The key comparison plus fall back must always agree with the exact comparison.
                assert_eq!(
                    ka.cmp(&kb).then_with(|| a.cmp_masked(&b, mask, mask_size)),
                    expected,
                    "keyed comparison mismatch for mask {mask:b}/{mask_size}",
                );
                // Whenever the keys already differ, they alone must yield the exact order.
                if ka != kb {
                    assert_eq!(
                        ka.cmp(&kb),
                        expected,
                        "key ordering mismatch for mask {mask:b}/{mask_size}",
                    );
                }
            }
        }

        prng_test_harness(20, |rnd| {
            check::<GeoDiffConfig7>(rnd, masks);
            check::<GeoDiffConfig13>(rnd, masks);
        });
    }

    #[test]
    fn test_bit_chunks() {
        prng_test_harness(100, |rnd| {
            let mut expected = GeoDiffCount7::default();
            for _ in 0..1000 {
                expected.push_hash(rnd.next_u64());
            }
            let actual =
                GeoDiffCount::from_bit_chunks(expected.config.clone(), expected.bit_chunks());
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

    #[test]
    fn test_serialization_empty() {
        let before = GeoDiffCount7::default();

        let mut writer = vec![];
        before.write(&mut writer).unwrap();

        assert_eq!(writer.len(), 0);

        let after = GeoDiffCount7::from_bytes_with_config(before.config.clone(), &writer);

        assert_eq!(before, after);
    }

    // This helper exists in order to easily test serializing types with different
    // bucket types in the MSB sparse bit field representation. See tests below.
    #[cfg(target_endian = "little")]
    fn serialization_round_trip<C: GeoConfig<Diff> + Default>(rnd: &mut ChaCha12Rng) {
        // Run 100 simulations of random values being put into
        // a diff counter. "Serializing" to a vector to emulate
        // writing to a disk, and then deserializing and asserting
        // the filters are equal.
        let mut before = GeoDiffCount::<'_, C>::default();
        // Select a random number of items to insert.
        let items = (1..1000).choose(rnd).unwrap();
        for _ in 0..items {
            before.push_hash(rnd.next_u64());
        }
        let mut writer = vec![];
        // Insert some padding to emulate alignment issues with the slices.
        // A previous version of this test never panicked even though we were
        // violating the alignment preconditions for the `from_raw_parts` function.
        let padding = [0_u8; 8];
        let pad_amount = (0..8).choose(rnd).unwrap();
        writer.write_all(&padding[..pad_amount]).unwrap();
        before.write(&mut writer).unwrap();
        let after = GeoDiffCount::<'_, C>::from_bytes_with_config(
            before.config.clone(),
            &writer[pad_amount..],
        );
        assert_eq!(before, after);
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn test_serialization_round_trip_7() {
        prng_test_harness(100, |rnd| {
            // Uses a u16 for MSB buckets.
            serialization_round_trip::<GeoDiffConfig7>(rnd);
        });
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn test_serialization_round_trip_13() {
        prng_test_harness(100, |rnd| {
            // Uses a u32 for MSB buckets.
            serialization_round_trip::<GeoDiffConfig13>(rnd);
        });
    }
}
