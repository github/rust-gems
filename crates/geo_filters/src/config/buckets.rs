use std::fmt::Debug;
use std::ops::{Add, Rem};

use crate::config::nth_one;

pub(crate) const BITS_PER_BLOCK: usize = 64;
pub(crate) const BITS_PER_BYTE: usize = 8;
pub(crate) const BYTES_PER_BLOCK: usize = 8;

pub trait IsBucketType: Copy + Debug + Default + Ord + Send + Sync
where
    Self: Add<Output = Self> + Rem<Output = Self>,
{
    const BITS: u32;

    fn from_usize(v: usize) -> Self;
    fn into_usize(self) -> usize;

    /// Returns a block with the bit in this bit position set to one.
    #[inline]
    fn into_block(self) -> u64 {
        assert!(
            self.into_usize() < BITS_PER_BLOCK,
            "position in block must be less then 64, got {self:?}",
        );
        1u64 << self.into_usize()
    }

    /// Returns the bit position for the most-significant bit in the block.
    /// The block should be non-zero!
    #[inline]
    fn from_block_msb(block: u64) -> Self {
        assert_ne!(block, 0);
        Self::from_usize(BITS_PER_BLOCK - 1 - block.leading_zeros() as usize)
    }

    /// Returns the bit position for the nth most-significant bit in the block.
    /// The block should be non-zero!
    fn from_block_nth_msb(block: u64, n: usize) -> Self {
        assert_ne!(block, 0);
        let ones = block.count_ones() as usize;
        assert!(n < ones);
        Self::from_usize(nth_one(block, (ones - n - 1) as u32) as usize)
    }

    /// Returns the bucket position given a block index and a block position.
    #[inline]
    fn into_bucket(self, block_idx: usize) -> Self {
        assert!(self.into_usize() < BITS_PER_BLOCK);
        Self::from_usize(block_idx * BITS_PER_BLOCK) + self
    }

    /// Returns the block index and bit position for this bucket position.
    #[inline]
    fn into_index_and_bit(self) -> (usize, Self) {
        (self.into_index(), self.into_bit())
    }

    /// Returns the block index for this bucket position.
    #[inline]
    fn into_index(self) -> usize {
        self.into_usize() / BITS_PER_BLOCK
    }

    /// Returns the block position for this bucket position.
    #[inline]
    fn into_bit(self) -> Self {
        self % Self::from_usize(BITS_PER_BLOCK)
    }

    /// Returns a mask matching the bits below this block position.
    #[inline]
    fn into_mask(self) -> u64 {
        self.into_block() - 1
    }
}

impl IsBucketType for u8 {
    const BITS: u32 = Self::BITS;

    #[inline]
    fn from_usize(v: usize) -> Self {
        v as Self
    }

    #[inline]
    fn into_usize(self) -> usize {
        self as usize
    }
}

impl IsBucketType for u16 {
    const BITS: u32 = Self::BITS;

    #[inline]
    fn from_usize(v: usize) -> Self {
        v as Self
    }

    #[inline]
    fn into_usize(self) -> usize {
        self as usize
    }
}

impl IsBucketType for u32 {
    const BITS: u32 = Self::BITS;

    #[inline]
    fn from_usize(v: usize) -> Self {
        v as Self
    }

    #[inline]
    fn into_usize(self) -> usize {
        self as usize
    }
}

impl IsBucketType for u64 {
    const BITS: u32 = Self::BITS;

    #[inline]
    fn from_usize(v: usize) -> Self {
        v as Self
    }

    #[inline]
    fn into_usize(self) -> usize {
        self as usize
    }
}

impl IsBucketType for usize {
    const BITS: u32 = Self::BITS;

    #[inline]
    fn from_usize(v: usize) -> Self {
        v
    }

    #[inline]
    fn into_usize(self) -> usize {
        self
    }
}

/// Computes the largest bucket index for 64-bit hashes given that (1 << B) bits cover half
/// the hash space.
///
/// (1 << B) buckets cover half the hash space, i.e., buckets [k * (1<<B), (k+1) * (1<<B) cover
/// the hashes with k leading zeros. For a 64-bit hash, this gives us 64 * (1<<B) buckets.
#[inline]
pub(crate) fn largest_bucket(b: usize) -> usize {
    64 * (1 << b)
}

#[inline]
pub(crate) fn assert_bucket_type_large_enough<T: IsBucketType>(b: usize) {
    assert!(
        largest_bucket(b).ilog2() < T::BITS,
        "bucket type has {} bits, which is too small for B = {}, requires bits > {}",
        T::BITS,
        b,
        largest_bucket(b).ilog2()
    );
}
