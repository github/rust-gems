use std::borrow::Cow;
use std::cmp::Ordering;
use std::ops::{Deref, Index, Range};

use dataview::PodMethods;
use github_pspack::Serializable;

use crate::BitChunk;

/// A bit vector where every bit occupies exactly one bit (in contrast to `Vec<bool>` where each
/// bit consumes 1 byte). It only implements the minimum number of operations that we need for our
/// GeometricFilter implementation. In particular it supports xor-ing of two bit vectors and
/// iterating through one bits.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct BitVec<'a> {
    num_bits: usize,
    blocks: Cow<'a, [u64]>,
}

const BITS_PER_BLOCK: usize = 64;

impl<'a> Ord for BitVec<'a> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.num_bits.cmp(&other.num_bits) {
            Ordering::Equal => self.blocks.iter().rev().cmp(other.blocks.iter().rev()),
            ord => ord,
        }
    }
}

impl<'a> PartialOrd for BitVec<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> BitVec<'a> {
    pub fn cmp_gray_rank(&self, other: &Self) -> Ordering {
        match self.num_bits.cmp(&other.num_bits) {
            Ordering::Equal => {
                let mut ones = 0;
                for i in (0..self.blocks.len()).rev() {
                    match self.blocks[i].cmp(&other.blocks[i]) {
                        Ordering::Equal => {}
                        ord => {
                            let prefix_len = (self.blocks[i] ^ other.blocks[i]).leading_zeros();
                            ones += (self.blocks[i] & !(u64::MAX >> prefix_len)).count_ones();
                            return if ones & 1 != 0 { ord.reverse() } else { ord };
                        }
                    }
                    ones += self.blocks[i].count_ones();
                }
                Ordering::Equal
            }
            ord => ord,
        }
    }

    pub fn new() -> BitVec<'a> {
        BitVec::default()
    }

    /// Takes an iterator of `BitChunk` items as input and returns the corresponding `BitVec`.
    /// The order of `BitChunk`s doesn't matter for this function and `BitChunk` may be hitting
    /// the same block. In this case, the function will simply xor them together.
    ///
    /// The function requires that `BitChunk`s only cover `num_bits` many bits. `BitChunk`s
    /// outside of that range may result in a panic!
    pub fn from_bit_chunks<I: Iterator<Item = BitChunk>>(num_bits: usize, bits: I) -> Self {
        let mut blocks = vec![0; (num_bits + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK];
        for BitChunk(idx, bits) in bits {
            blocks[idx as usize] ^= bits;
        }
        let mut result = Self {
            num_bits,
            blocks: Cow::from(blocks),
        };
        result.clear_superfluous_bits();
        result
    }

    /// Constructs a BitVec instance with num_bits and all bits initialized to the provided default value.
    #[cfg(test)]
    pub fn from_element(num_bits: usize, value: bool) -> Self {
        let num_blocks = (num_bits + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
        let blocks = vec![if value { u64::MAX } else { u64::MIN }; num_blocks];
        let mut result = BitVec {
            blocks: Cow::from(blocks),
            num_bits,
        };
        result.clear_superfluous_bits();
        result
    }

    pub fn resize(&mut self, num_bits: usize) {
        let num_blocks = (num_bits + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
        if num_blocks != self.blocks.len() {
            self.blocks.to_mut().resize(num_blocks, 0);
        }
        self.num_bits = num_bits;
        self.clear_superfluous_bits();
    }

    fn clear_superfluous_bits(&mut self) {
        if self.num_bits > 0 && self.num_bits % BITS_PER_BLOCK != 0 {
            let blocks = self.blocks.to_mut();
            let last_block = blocks.len() - 1;
            blocks[last_block] &= u64::MAX >> (BITS_PER_BLOCK - (self.num_bits % BITS_PER_BLOCK));
        }
    }

    pub fn len(&self) -> usize {
        self.num_bits
    }

    pub fn is_empty(&self) -> bool {
        self.num_bits == 0
    }

    /// Tests the bit specified by the provided zero-based bit position.
    pub fn test_bit(&self, index: usize) -> bool {
        assert!(index < self.num_bits);
        let block_idx = index / BITS_PER_BLOCK;
        let bit_idx = index % BITS_PER_BLOCK;
        (self.blocks[block_idx] & (1 << bit_idx)) != 0
    }

    /// Toggles the bit specified by the provided zero-based bit position.
    pub fn toggle(&mut self, index: usize) {
        assert!(index < self.num_bits);
        let block_idx = index / BITS_PER_BLOCK;
        let bit_idx = index % BITS_PER_BLOCK;
        self.blocks.to_mut()[block_idx] ^= 1 << bit_idx;
    }

    /// Note this function only xor-s bits that exist in both BitVec!
    /// Especially if self has less bits than other, then self will ignore some bits from other!
    pub fn xor(&mut self, other: &BitVec) {
        let my_blocks = self.blocks.to_mut();
        let other_blocks = other.blocks.deref();
        for i in 0..my_blocks.len().min(other_blocks.len()) {
            my_blocks[i] ^= other_blocks[i];
        }
        self.clear_superfluous_bits();
    }

    /// n specifies the desired zero-based index of the most significant one.
    /// The zero-based index of the desired one bit is returned.
    pub fn nth_most_significant_one(&self, mut n: usize) -> Option<usize> {
        for (block_idx, block) in self.blocks.iter().enumerate().rev() {
            let ones = block.count_ones() as usize;
            if n < ones {
                return Some(
                    nth_one(*block, (ones - n - 1) as u32) as usize + block_idx * BITS_PER_BLOCK,
                );
            } else {
                n -= ones;
            }
        }
        None
    }

    pub fn count_ones(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| block.count_ones() as usize)
            .sum()
    }

    /// Returns an iterator that iterates through all one bits from most to least significant.
    pub fn iter_ones(&'_ self) -> impl Iterator<Item = usize> + '_ {
        let bitvec = self.blocks.deref();
        OnesIter {
            bitvec,
            block_idx: bitvec.len(),
            remaining_ones_in_block: 0,
        }
    }

    /// Returns an iterator over all blocks in reverse order.
    /// The blocks are represented as `BitChunk`s.
    pub fn bit_chunks(&self) -> impl Iterator<Item = BitChunk> + '_ {
        self.blocks
            .iter()
            .enumerate()
            .rev()
            .map(|(idx, block)| BitChunk(idx as u16, *block))
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
        let block_start = range.start / BITS_PER_BLOCK;
        if block_start >= self.blocks.len() {
            return 0;
        }
        let mut result = self.blocks[block_start] >> (range.start % BITS_PER_BLOCK);
        let block_end = (range.end - 1) / BITS_PER_BLOCK;
        if block_end < self.blocks.len() && block_end != block_start {
            result |= self.blocks[block_end] << (BITS_PER_BLOCK - (range.start % BITS_PER_BLOCK));
        }
        result & (u64::MAX >> (BITS_PER_BLOCK - range.len()))
    }
}

// n specifies the zero-based index of the desired one bit where ones are enumerate from least significant
// to most significant bits.
// The function returns the zero-based bit index of that one bit.
// If no such one bit exist, the function will return 64.
pub fn nth_one(value: u64, n: u32) -> u32 {
    unsafe { std::arch::x86_64::_pdep_u64(1 << n, value).trailing_zeros() }
}

impl<'a> Index<usize> for BitVec<'a> {
    type Output = bool;

    /// Returns the value of the bit corresponding to the provided zero-based bit position.
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.num_bits);
        let block_idx = index / BITS_PER_BLOCK;
        let bit_idx = index % BITS_PER_BLOCK;
        match (self.blocks[block_idx] & (1 << bit_idx)) != 0 {
            true => &true,
            false => &false,
        }
    }
}

struct OnesIter<'a> {
    bitvec: &'a [u64],
    block_idx: usize,
    remaining_ones_in_block: u32,
}

impl<'a> Iterator for OnesIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.remaining_ones_in_block == 0 {
            if self.block_idx == 0 {
                return None;
            }
            self.block_idx -= 1;
            self.remaining_ones_in_block = self.bitvec[self.block_idx].count_ones();
        }
        self.remaining_ones_in_block -= 1;
        Some(
            nth_one(self.bitvec[self.block_idx], self.remaining_ones_in_block) as usize
                + self.block_idx * BITS_PER_BLOCK,
        )
    }
}

impl<'a> Serializable<'a> for BitVec<'a> {
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        if self.is_empty() {
            return Ok(0);
        }
        // First serialize the number of unoccupied bits in the last block as one byte.
        let mut encoded_bytes = (63 - ((self.num_bits - 1) % 64) as u8).write(writer)?;
        encoded_bytes += self.blocks.deref().as_bytes().write(writer)?;
        Ok(encoded_bytes)
    }

    fn from_bytes(mut buf: &'a [u8]) -> Self {
        if buf.is_empty() {
            return Self::default();
        }
        assert!(
            buf[0] < 64,
            "Number of unoccupied bits should be <64, got {}",
            buf[0]
        );
        let num_bits = (buf.len() - 1) * 8 - buf[0] as usize;
        buf = &buf[1..];
        const BYTES_PER_BLOCK: usize = BITS_PER_BLOCK / 8;
        assert_eq!(
            buf.len() % BYTES_PER_BLOCK,
            0,
            "buffer should be a multiple of 8 bytes, got {}",
            buf.len()
        );
        let num_blocks = buf.len() / BYTES_PER_BLOCK;
        let blocks = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u64, num_blocks) };
        Self {
            num_bits,
            blocks: Cow::from(blocks),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nth_most_significant_bit() {
        let mut bitvec = BitVec::from_element(100, false);
        (0..10).for_each(|bit| bitvec.toggle(bit * 10));
        for i in 0..10 {
            assert_eq!(bitvec.nth_most_significant_one(i), Some(90 - i * 10));
        }
    }

    #[test]
    fn test_bitvec_serialization() {
        let bitvec = BitVec::from_element(100, true);
        let mut buffer = Vec::new();
        {
            let mut writer = std::io::Cursor::new(&mut buffer);
            bitvec
                .write(&mut writer)
                .expect("writing into a buffer should never fail!");
        }
        let new_bitvec = BitVec::from_bytes(buffer.as_bytes());
        assert_eq!(bitvec, new_bitvec);
    }

    #[test]
    fn test_bitvec_index() {
        let mut bitvec = BitVec::from_element(100, true);
        for i in 0..100 {
            assert!(bitvec[i]);
        }
        bitvec.toggle(50);
        for i in 0..100 {
            assert_eq!(bitvec[i], i != 50);
        }
    }

    #[test]
    #[should_panic]
    fn test_bitvec_index_out_of_bounds() {
        let bitvec = BitVec::from_element(100, true);
        assert!(!bitvec[100]);
    }

    #[test]
    fn test_bitvec_range() {
        let mut bitvec = BitVec::from_element(120, false);
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
            let mut bitvec = BitVec::from_element(128, false);
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
        let bitvec = BitVec::from_element(128, false);
        bitvec.bit_range(&(0..65));
    }
}
