use std::borrow::Cow;
use std::cmp::Ordering;
use std::iter::Peekable;
use std::mem::{size_of, size_of_val};
use std::ops::{Deref as _, Index, Range};

use crate::config::IsBucketType;
use crate::config::BITS_PER_BLOCK;
use crate::config::{BitChunk, BYTES_PER_BLOCK};

/// A bit vector where every bit occupies exactly one bit (in contrast to `Vec<bool>` where each
/// bit consumes 1 byte). It only implements the minimum number of operations that we need for our
/// GeoDiffCount implementation. In particular it supports xor-ing of two bit vectors and
/// iterating through one bits.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub(crate) struct BitVec<'a> {
    num_bits: usize,
    blocks: Cow<'a, [u64]>,
}

impl Ord for BitVec<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.num_bits.cmp(&other.num_bits) {
            Ordering::Equal => self.blocks.iter().rev().cmp(other.blocks.iter().rev()),
            ord => ord,
        }
    }
}

impl PartialOrd for BitVec<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl BitVec<'_> {
    /// Takes an iterator of `BitChunk` items as input and returns the corresponding `BitVec`.
    /// The order of `BitChunk`s doesn't matter for this function and `BitChunk` may be hitting
    /// the same block. In this case, the function will simply xor them together.
    ///
    /// NOTE: If the bitchunks iterator is empty, the result is NOT sized to `num_bits` but will
    ///       be EMPTY instead.
    pub fn from_bit_chunks<I: Iterator<Item = BitChunk>>(
        mut chunks: Peekable<I>,
        num_bits: usize,
    ) -> Self {
        // if there are no chunks, we keep the size zero
        let num_bits = chunks.peek().map(|_| num_bits).unwrap_or_default();
        let mut result = Self::default();
        result.resize(num_bits);
        let blocks = result.blocks.to_mut();
        for BitChunk { index, block } in chunks {
            blocks[index] ^= block;
        }
        result.clear_superfluous_bits();
        result
    }

    /// Resize the vector such that the top block contains the given bucket.
    pub fn resize(&mut self, num_bits: usize) {
        let num_blocks = num_bits.div_ceil(BITS_PER_BLOCK);
        if num_blocks != self.blocks.len() {
            self.blocks.to_mut().resize(num_blocks, 0);
        }
        self.num_bits = num_bits;
        self.clear_superfluous_bits();
    }

    fn clear_superfluous_bits(&mut self) {
        let bit = self.num_bits.into_bit();
        if bit > 0 {
            self.blocks
                .to_mut()
                .last_mut()
                .into_iter()
                .for_each(|block| *block &= bit.into_mask());
        }
    }

    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    pub fn is_empty(&self) -> bool {
        self.num_bits() == 0
    }

    /// Tests the bit specified by the provided zero-based bit position.
    pub fn test_bit(&self, index: usize) -> bool {
        assert!(index < self.num_bits);
        let (block_idx, bit_idx) = index.into_index_and_bit();
        (self.blocks[block_idx] & bit_idx.into_block()) != 0
    }

    /// Toggles the bit specified by the provided zero-based bit position.
    pub fn toggle(&mut self, index: usize) {
        assert!(index < self.num_bits);
        let (block_idx, bit_idx) = index.into_index_and_bit();
        self.blocks.to_mut()[block_idx] ^= bit_idx.into_block();
    }

    /// Returns an iterator over all blocks in reverse order.
    /// The blocks are represented as `BitChunk`s.
    pub fn bit_chunks(&self) -> impl Iterator<Item = BitChunk> + '_ {
        self.blocks
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, block)| **block != 0)
            .map(|(index, block)| BitChunk::new(index, *block))
    }

    // Returns an owned BitVec with static lifetime that can be used for cases when we want
    // BitVec instance to live arbitrarily long.
    pub fn into_owned(self) -> BitVec<'static> {
        BitVec {
            blocks: Cow::Owned(self.blocks.into_owned()),
            num_bits: self.num_bits,
        }
    }

    /// Collects the bits for the specified range into a u64.
    /// Non-existant bits will be filled in with zeros.
    /// The range must not be larger than 64.
    pub fn bit_range(&self, range: &Range<usize>) -> u64 {
        if range.is_empty() {
            return 0;
        }
        assert!(
            range.len() <= 64,
            "Only ranges up to length 64 are supported, got: {range:?}"
        );
        let (block_start, start_idx) = range.start.into_index_and_bit();
        if block_start >= self.blocks.len() {
            return 0;
        }
        let mut result = self.blocks[block_start] >> start_idx;
        let block_end = (range.end - 1) / BITS_PER_BLOCK;
        if block_end < self.blocks.len() && block_end != block_start {
            result |= self.blocks[block_end] << (BITS_PER_BLOCK - start_idx);
        }
        result & (u64::MAX >> (BITS_PER_BLOCK - range.len()))
    }

    pub fn bytes_in_memory(&self) -> usize {
        let Self { num_bits, blocks } = self;
        size_of_val(num_bits) + blocks.len() * size_of::<u64>()
    }

    pub fn from_bytes(mut buf: &[u8]) -> Self {
        if buf.is_empty() {
            return Self::default();
        }

        // The first byte of the serialized BitVec is used to indicate how many
        // of the bits in the left-most byte are *unoccupied*.
        // See [`BitVec::write`] implementation for how this is done.
        assert!(
            buf[0] < 64,
            "Number of unoccupied bits should be <64, got {}",
            buf[0]
        );

        let num_bits = (buf.len() - 1) * 8 - buf[0] as usize;
        buf = &buf[1..];

        assert_eq!(
            buf.len() % BYTES_PER_BLOCK,
            0,
            "buffer should be a multiple of 8 bytes, got {}",
            buf.len()
        );

        let blocks = unsafe {
            std::mem::transmute(std::slice::from_raw_parts(
                buf.as_ptr(),
                buf.len() / BYTES_PER_BLOCK,
            ))
        };
        let blocks = Cow::Borrowed(blocks);

        Self { num_bits, blocks }
    }

    pub fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        if self.is_empty() {
            return Ok(0);
        }

        // First serialize the number of unoccupied bits in the last block as one byte.
        let unoccupied_bits = 63 - ((self.num_bits - 1) % 64) as u8;
        writer.write_all(&[unoccupied_bits])?;

        let blocks = self.blocks.deref();
        let block_bytes = unsafe {
            std::slice::from_raw_parts(blocks.as_ptr() as *const u8, blocks.len() * BYTES_PER_BLOCK)
        };

        writer.write_all(block_bytes)?;

        Ok(block_bytes.len() + 1)
    }
}

impl Index<usize> for BitVec<'_> {
    type Output = bool;

    /// Returns the value of the bit corresponding to the provided zero-based bit position.
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.num_bits);
        let (block_idx, bit_idx) = index.into_index_and_bit();
        match (self.blocks[block_idx] & bit_idx.into_block()) != 0 {
            true => &true,
            false => &false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constructs a BitVec instance with num_bits and all bits initialized to the provided default value.
    fn bitvec_from_element<'a>(num_bits: usize, value: bool) -> BitVec<'a> {
        let num_blocks = num_bits.div_ceil(BITS_PER_BLOCK) + 1;
        let blocks = vec![if value { u64::MAX } else { u64::MIN }; num_blocks];
        BitVec {
            blocks: Cow::from(blocks),
            num_bits,
        }
    }

    #[test]
    fn test_nth_most_significant_bit() {
        let mut bitvec = bitvec_from_element(100, false);
        (0..10).for_each(|bit| bitvec.toggle(bit * 10));
        for i in 0..10 {
            assert_eq!(bitvec.nth_most_significant_one(i), Some(90 - i * 10));
        }
    }

    #[test]
    fn test_bitvec_index() {
        let mut bitvec = bitvec_from_element(100, true);
        for i in 0..100 {
            assert!(bitvec[i]);
        }
        bitvec.toggle(50);
        for i in 0..100 {
            assert_eq!(bitvec[i], i != 50);
        }
    }

    #[test]
    fn test_bitvec_resize() {
        let mut bitvec = BitVec::default();
        bitvec.resize(128);
        bitvec.toggle(14);
        bitvec.toggle(48);
        bitvec.toggle(72);
        bitvec.toggle(120);

        bitvec.resize(110);
        {
            assert!(bitvec.test_bit(14));
            assert!(bitvec.test_bit(48));
            assert!(bitvec.test_bit(72));
            let last = bitvec.blocks.last().expect("there must be a block");
            assert!(last & 120usize.into_bit().into_block() == 0);
        }

        bitvec.resize(72);
        {
            assert!(bitvec.test_bit(14));
            assert!(bitvec.test_bit(48));
            let last = bitvec.blocks.last().expect("there must be a block");
            assert!(last & 72usize.into_bit().into_block() == 0);
        }

        bitvec.resize(64);
        {
            assert!(bitvec.test_bit(14));
            assert!(bitvec.test_bit(48));
        }
    }

    #[test]
    #[should_panic]
    fn test_bitvec_index_out_of_bounds() {
        let bitvec = bitvec_from_element(100, true);
        assert!(!bitvec[100]);
    }

    #[test]
    fn test_bitvec_range() {
        let mut bitvec = bitvec_from_element(120, false);
        bitvec.toggle(7);
        bitvec.toggle(66);
        assert_eq!(4, bitvec.bit_range(&(64..128)));
        assert_eq!(1, bitvec.bit_range(&(66..128)));
        assert_eq!(0, bitvec.bit_range(&(67..128)));
        assert_eq!(1, bitvec.bit_range(&(66..130)));
    }

    #[test]
    fn test_bitvec_range_one_bit() {
        for i in 0..64 {
            let mut bitvec = bitvec_from_element(128, false);
            // Set a single bit and move all possible 64 sized ranges over that bit and
            // test that only the expected bit is set.
            bitvec.toggle(63 + i);
            for j in 0..64 {
                assert_eq!(1 << j, bitvec.bit_range(&((63 + i - j)..(63 + i - j + 64))));
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_bitvec_range_too_large() {
        let bitvec = bitvec_from_element(128, false);
        bitvec.bit_range(&(0..65));
    }
}
