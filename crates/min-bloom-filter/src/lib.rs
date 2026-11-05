//! A split-block Bloom filter with four-bit values.
//!
//! Every block consists of eight [`u32`] words. Each word contains eight
//! nibbles, and a hash selects one nibble in every word. Inserting a value
//! raises each selected nibble to the maximum of its current and inserted
//! values. Retrieving a hash returns the minimum of its eight selected
//! nibbles.

const WORDS_PER_BLOCK: usize = 8;
const MAX_VALUE: u8 = 0x0f;

// The salts are the constants from the Parquet split-block Bloom filter.
const SALT: [u32; WORDS_PER_BLOCK] = [
    0x47b6_137b,
    0x4497_4d91,
    0x8824_ad5b,
    0xa2b7_289d,
    0x7054_95c7,
    0x2df1_424b,
    0x9efc_4947,
    0x5c6b_fb31,
];

/// A split-block min Bloom filter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MinBloomFilter {
    blocks: Box<[[u32; WORDS_PER_BLOCK]]>,
}

impl MinBloomFilter {
    /// Creates a filter sized for `expected_entries` and `false_positive_rate`.
    ///
    /// The rate must be finite and strictly between zero and one. The filter
    /// uses the standard Bloom-filter sizing equation with eight probes, then
    /// rounds up to a whole 32-byte split block.
    #[must_use]
    pub fn new(expected_entries: usize, false_positive_rate: f64) -> Self {
        assert!(
            expected_entries > 0,
            "expected_entries must be greater than zero"
        );
        assert!(
            false_positive_rate.is_finite()
                && false_positive_rate > 0.0
                && false_positive_rate < 1.0,
            "false_positive_rate must be finite and between zero and one"
        );

        let block_count =
            (expected_entries as f64 / mean_block_occupancy(false_positive_rate)).ceil() as usize;

        Self {
            blocks: vec![[0; WORDS_PER_BLOCK]; block_count].into_boxed_slice(),
        }
    }

    /// Creates an empty filter with exactly `block_count` split blocks.
    #[must_use]
    pub fn with_block_count(block_count: usize) -> Self {
        assert!(block_count > 0, "block_count must be greater than zero");
        Self {
            blocks: vec![[0; WORDS_PER_BLOCK]; block_count].into_boxed_slice(),
        }
    }

    /// Raises the eight nibbles selected by `hash` to at least `value`.
    ///
    /// # Panics
    ///
    /// Panics when `value` is greater than 15.
    #[inline]
    pub fn insert(&mut self, hash: u64, value: u8) {
        assert!(value <= MAX_VALUE, "value must fit in a nibble");

        let block_index = block_index(self.blocks.len(), hash);
        let block = &mut self.blocks[block_index];
        insert_block(block, hash as u32, value);
    }

    /// Returns the minimum of the eight nibbles selected by `hash`.
    #[inline]
    #[must_use]
    pub fn get(&self, hash: u64) -> u8 {
        let block = &self.blocks[block_index(self.blocks.len(), hash)];
        get_block(block, hash as u32)
    }

    /// Returns the number of 32-byte split blocks in this filter.
    #[inline]
    #[must_use]
    pub const fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the filter's allocated data size in bytes.
    #[inline]
    #[must_use]
    pub const fn len_bytes(&self) -> usize {
        self.blocks.len() * size_of::<[u32; WORDS_PER_BLOCK]>()
    }

    /// Returns `true` when all nibbles are zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.iter().flatten().all(|word| *word == 0)
    }

    /// Resets every nibble to zero.
    pub fn clear(&mut self) {
        self.blocks.fill([0; WORDS_PER_BLOCK]);
    }
}

fn mean_block_occupancy(false_positive_rate: f64) -> f64 {
    let mut low = 0.0;
    let mut high = 64.0;
    while split_block_false_positive_probability(high) < false_positive_rate {
        high *= 2.0;
    }
    for _ in 0..64 {
        let middle = (low + high) / 2.0;
        if split_block_false_positive_probability(middle) < false_positive_rate {
            low = middle;
        } else {
            high = middle;
        }
    }
    low
}

fn split_block_false_positive_probability(mean_occupancy: f64) -> f64 {
    let mut probability = (-mean_occupancy).exp();
    let mut total = 0.0;
    let mut occupancy = 0_u32;
    loop {
        let selected = 1.0 - (7.0_f64 / 8.0).powf(f64::from(occupancy));
        total += probability * selected.powi(WORDS_PER_BLOCK as i32);

        occupancy += 1;
        probability *= mean_occupancy / f64::from(occupancy);
        if probability < f64::EPSILON && occupancy as f64 > mean_occupancy {
            return total;
        }
    }
}

#[inline]
fn block_index(block_count: usize, hash: u64) -> usize {
    (((hash >> 32) * block_count as u64) >> 32) as usize
}

#[cfg(any(not(target_arch = "aarch64"), test))]
#[inline]
fn nibble_shift(hash: u32, salt: u32) -> u32 {
    (hash.wrapping_mul(salt) >> 29) * 4
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn insert_block(block: &mut [u32; WORDS_PER_BLOCK], hash: u32, value: u8) {
    // SAFETY: NEON is mandatory on AArch64, and `block` points to eight valid u32 values.
    unsafe { neon::insert(block.as_mut_ptr(), hash, value) }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn insert_block(block: &mut [u32; WORDS_PER_BLOCK], hash: u32, value: u8) {
    for (word, salt) in block.iter_mut().zip(SALT) {
        let shift = nibble_shift(hash, salt);
        let current = ((*word >> shift) & u32::from(MAX_VALUE)) as u8;
        if current < value {
            *word = (*word & !(u32::from(MAX_VALUE) << shift)) | (u32::from(value) << shift);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn get_block(block: &[u32; WORDS_PER_BLOCK], hash: u32) -> u8 {
    // SAFETY: NEON is mandatory on AArch64, and `block` points to eight valid u32 values.
    unsafe { neon::get(block.as_ptr(), hash) }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn get_block(block: &[u32; WORDS_PER_BLOCK], hash: u32) -> u8 {
    block
        .iter()
        .zip(SALT)
        .map(|(word, salt)| {
            let shift = nibble_shift(hash, salt);
            ((word >> shift) & u32::from(MAX_VALUE)) as u8
        })
        .min()
        .unwrap_or_default()
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::{
        int32x4_t, uint32x4_t, vandq_u32, vbicq_u32, vdupq_n_u32, vld1q_u32, vmaxq_u32, vminq_u32,
        vminvq_u32, vmulq_u32, vnegq_s32, vorrq_u32, vreinterpretq_s32_u32, vshlq_u32, vshrq_n_u32,
        vst1q_u32,
    };

    use super::{MAX_VALUE, SALT};

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn shifts(hash: u32) -> (uint32x4_t, uint32x4_t) {
        unsafe {
            let hash = vdupq_n_u32(hash);
            let low = vmulq_u32(vld1q_u32(SALT.as_ptr()), hash);
            let high = vmulq_u32(vld1q_u32(SALT.as_ptr().add(4)), hash);
            let four = vdupq_n_u32(4);
            (
                vmulq_u32(vshrq_n_u32(low, 29), four),
                vmulq_u32(vshrq_n_u32(high, 29), four),
            )
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn selected_values(words: uint32x4_t, shifts: int32x4_t) -> uint32x4_t {
        vandq_u32(
            vshlq_u32(words, vnegq_s32(shifts)),
            vdupq_n_u32(u32::from(MAX_VALUE)),
        )
    }

    #[target_feature(enable = "neon")]
    #[inline]
    pub unsafe fn insert(block: *mut u32, hash: u32, value: u8) {
        unsafe {
            let shifts = shifts(hash);
            let low_shifts = vreinterpretq_s32_u32(shifts.0);
            let high_shifts = vreinterpretq_s32_u32(shifts.1);
            let low = vld1q_u32(block);
            let high = vld1q_u32(block.add(4));
            let nibble_mask = vdupq_n_u32(u32::from(MAX_VALUE));
            let value = vdupq_n_u32(u32::from(value));

            let low_values = vmaxq_u32(selected_values(low, low_shifts), value);
            let high_values = vmaxq_u32(selected_values(high, high_shifts), value);
            let low_mask = vshlq_u32(nibble_mask, low_shifts);
            let high_mask = vshlq_u32(nibble_mask, high_shifts);
            let low_values = vshlq_u32(low_values, low_shifts);
            let high_values = vshlq_u32(high_values, high_shifts);

            vst1q_u32(block, vorrq_u32(vbicq_u32(low, low_mask), low_values));
            vst1q_u32(
                block.add(4),
                vorrq_u32(vbicq_u32(high, high_mask), high_values),
            );
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    pub unsafe fn get(block: *const u32, hash: u32) -> u8 {
        unsafe {
            let shifts = shifts(hash);
            let low = selected_values(vld1q_u32(block), vreinterpretq_s32_u32(shifts.0));
            let high = selected_values(vld1q_u32(block.add(4)), vreinterpretq_s32_u32(shifts.1));
            vminvq_u32(vminq_u32(low, high)) as u8
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_filter_returns_zero() {
        let filter = MinBloomFilter::with_block_count(1);

        assert_eq!(filter.get(42), 0);
        assert!(filter.is_empty());
    }

    #[test]
    fn insert_and_retrieve_value() {
        let mut filter = MinBloomFilter::with_block_count(16);

        filter.insert(42, 7);

        assert_eq!(filter.get(42), 7);
        assert!(!filter.is_empty());
    }

    #[test]
    fn insert_only_raises_values() {
        let mut filter = MinBloomFilter::with_block_count(1);

        filter.insert(42, 11);
        filter.insert(42, 3);
        assert_eq!(filter.get(42), 11);

        filter.insert(42, 15);
        assert_eq!(filter.get(42), 15);
    }

    #[test]
    fn retrieval_uses_minimum_selected_nibble() {
        let mut filter = MinBloomFilter::with_block_count(1);
        let hash = 42;
        let shifts = SALT.map(|salt| nibble_shift(hash as u32, salt));

        for (index, (word, shift)) in filter.blocks[0].iter_mut().zip(shifts).enumerate() {
            *word = ((index as u32 + 3) & u32::from(MAX_VALUE)) << shift;
        }

        assert_eq!(filter.get(hash), 3);
    }

    #[test]
    fn clear_resets_filter() {
        let mut filter = MinBloomFilter::with_block_count(2);
        filter.insert(42, 9);

        filter.clear();

        assert_eq!(filter.get(42), 0);
        assert!(filter.is_empty());
    }

    #[test]
    fn sizing_meets_requested_rate() {
        let entries = 3_000_000;
        let one_percent = MinBloomFilter::new(entries, 0.01);
        let tenth_percent = MinBloomFilter::new(entries, 0.001);

        assert_eq!(one_percent.len_bytes(), 20_149_216);
        assert_eq!(tenth_percent.len_bytes(), 38_009_632);
        assert!(tenth_percent.len_bytes() > one_percent.len_bytes());
        assert_eq!(one_percent.len_bytes() % 32, 0);
        assert_eq!(tenth_percent.len_bytes() % 32, 0);
    }

    #[test]
    #[should_panic(expected = "value must fit in a nibble")]
    fn rejects_values_larger_than_a_nibble() {
        MinBloomFilter::with_block_count(1).insert(42, 16);
    }
}
