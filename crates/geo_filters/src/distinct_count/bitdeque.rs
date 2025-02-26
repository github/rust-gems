use std::collections::VecDeque;
use std::fmt::Debug;
use std::iter::Peekable;
use std::mem::{size_of, size_of_val};
use std::ops::{Index, Range};

use crate::config::{BitChunk, IsBucketType, BITS_PER_BLOCK, BYTES_PER_BLOCK};

#[derive(Clone)]
enum DequeCow<'a> {
    Owned(VecDeque<u64>),
    Borrowed(&'a [u64]),
}

impl Debug for DequeCow<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Owned(b) => Debug::fmt(b, f),
            Self::Borrowed(b) => Debug::fmt(b, f),
        }
    }
}

impl Default for DequeCow<'_> {
    fn default() -> Self {
        Self::Borrowed(&[])
    }
}

impl PartialEq for DequeCow<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Owned(l0), Self::Owned(r0)) => l0 == r0,
            (Self::Borrowed(l0), Self::Borrowed(r0)) => l0 == r0,
            (Self::Owned(l0), Self::Borrowed(r0)) | (Self::Borrowed(r0), Self::Owned(l0)) => {
                l0 == r0
            }
        }
    }
}

impl Eq for DequeCow<'_> {}

impl DequeCow<'_> {
    fn to_mut(&mut self) -> &mut VecDeque<u64> {
        match self {
            DequeCow::Owned(o) => o,
            DequeCow::Borrowed(b) => {
                *self = Self::Owned(VecDeque::from_iter(b.iter().copied()));
                self.to_mut()
            }
        }
    }
}

impl Index<usize> for DequeCow<'_> {
    type Output = u64;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            DequeCow::Owned(o) => &o[index],
            DequeCow::Borrowed(b) => &b[index],
        }
    }
}

impl DequeCow<'_> {
    fn len(&self) -> usize {
        match self {
            DequeCow::Owned(o) => o.len(),
            DequeCow::Borrowed(b) => b.len(),
        }
    }
}

enum DequeIter<'a> {
    Slice(std::slice::Iter<'a, u64>),
    VecDeque(std::collections::vec_deque::Iter<'a, u64>),
}

impl<'a> Iterator for DequeIter<'a> {
    type Item = &'a u64;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Slice(i) => i.next(),
            Self::VecDeque(i) => i.next(),
        }
    }
}

impl DoubleEndedIterator for DequeIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            Self::Slice(i) => i.next_back(),
            Self::VecDeque(i) => i.next_back(),
        }
    }
}

impl ExactSizeIterator for DequeIter<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Slice(i) => i.len(),
            Self::VecDeque(i) => i.len(),
        }
    }
}

impl DequeCow<'_> {
    fn iter(&self) -> DequeIter<'_> {
        match self {
            DequeCow::Owned(o) => DequeIter::VecDeque(o.iter()),
            DequeCow::Borrowed(b) => DequeIter::Slice(b.iter()),
        }
    }
}

/// A bit vector where every bit occupies exactly one bit (in contrast to `Vec<bool>` where each
/// bit consumes 1 byte). It only implements the minimum number of operations that we need for our
/// GeometricFilter implementation. In particular it supports xor-ing of two bit vectors and
/// iterating through one bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitDeque<'a> {
    bit_range: Range<usize>,
    blocks: DequeCow<'a>,
    max_blocks: usize,
}

impl BitDeque<'_> {
    pub fn new(max_bytes: usize) -> Self {
        assert!(max_bytes >= BYTES_PER_BLOCK);
        Self {
            bit_range: Default::default(),
            blocks: Default::default(),
            max_blocks: max_bytes / BYTES_PER_BLOCK,
        }
    }

    /// Construct a deque from the given bitchunks, up to the given maximum number of bits.
    ///
    /// NOTE: If the bitchunks iterator is empty, the result is NOT sized to `end` but will
    ///       be EMPTY instead.
    pub fn from_bit_chunks<I: Iterator<Item = BitChunk>>(
        mut chunks: Peekable<I>,
        end: usize,
        max_bytes: usize,
    ) -> Self {
        // if there are no chunks, we keep the size zero
        let end = chunks.peek().map(|_| end).unwrap_or_default();
        let mut result = Self::new(max_bytes);
        result.resize(end);
        let blocks = result.blocks.to_mut();
        for BitChunk { index, block } in chunks {
            let start = 0.into_bucket(index);
            if start < result.bit_range.start {
                break;
            }
            let index = (start - result.bit_range.start).into_index();
            blocks[index] |= block;
        }
        result
    }

    /// Return an iterator from most to least significant bitchunks for this deque.
    pub fn bit_chunks(&self) -> impl Iterator<Item = BitChunk> + '_ {
        let start_index = self.bit_range.start.into_index();
        self.blocks
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, block)| **block != 0)
            .map(move |(index, block)| BitChunk::new(index + start_index, *block))
    }

    /// Resize this deque to ensure it covers all bits up to the given end,
    /// while limiting the size to the given maximum bits.
    pub fn resize(&mut self, end: usize) {
        assert!(end >= self.bit_range.end);
        // the bit range might end in the middle of the most-significant block,
        // so we set it to the maximum supported by the current blocks
        self.bit_range.end = self.bit_range.end.div_ceil(BITS_PER_BLOCK) * BITS_PER_BLOCK;
        let blocks = self.blocks.to_mut();
        while self.bit_range.end < end {
            if blocks.len() + 1 > self.max_blocks {
                blocks.pop_front().expect("at least one block expected");
                self.bit_range.start += BITS_PER_BLOCK;
            }
            blocks.push_back(0);
            self.bit_range.end += BITS_PER_BLOCK;
        }
        // set the bit range to the given end, which may be lower than the
        // maximum of the most-significant block
        self.bit_range.end = end;
        assert!(blocks.len() <= self.max_blocks);
        assert!(self.bit_range.len().div_ceil(BITS_PER_BLOCK) == blocks.len());
    }

    pub fn bit_range(&self) -> &Range<usize> {
        &self.bit_range
    }

    /// Tests the bit specified by the provided zero-based bit position.
    pub fn test_bit(&self, index: usize) -> bool {
        assert!(self.bit_range.contains(&index));
        let (block_idx, bit_idx) = self.block_idx(index);
        (self.blocks[block_idx] & bit_idx.into_block()) != 0
    }

    /// Sets the bit specified by the provided zero-based bit position.
    pub fn set_bit(&mut self, index: usize) {
        assert!(self.bit_range.contains(&index));
        let (block_idx, bit_idx) = self.block_idx(index);
        self.blocks.to_mut()[block_idx] |= bit_idx.into_block();
    }

    fn block_idx(&self, index: usize) -> (usize, usize) {
        let (mut block_idx, bit_idx) = index.into_index_and_bit();
        block_idx -= self.bit_range.start.into_index();
        (block_idx, bit_idx)
    }

    pub fn bytes_in_memory(&self) -> usize {
        let Self {
            bit_range,
            blocks,
            max_blocks,
        } = self;
        size_of_val(bit_range) + blocks.len() * size_of::<u64>() + size_of_val(max_blocks)
    }
}

impl Index<usize> for BitDeque<'_> {
    type Output = bool;

    /// Returns the value of the bit corresponding to the provided zero-based bit position.
    fn index(&self, index: usize) -> &Self::Output {
        match self.test_bit(index) {
            true => &true,
            false => &false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitvec_index() {
        let mut bitvec = BitDeque::new(usize::MAX);
        bitvec.resize(2 * BITS_PER_BLOCK);
        (0..10).for_each(|bit| bitvec.set_bit(bit * 10));
        for i in 0..100 {
            assert_eq!(bitvec[i], i % 10 == 0);
        }
    }

    #[test]
    #[should_panic]
    fn test_bitvec_index_out_of_bounds_tail() {
        let mut bitvec = BitDeque::new(usize::MAX);
        bitvec.resize(BITS_PER_BLOCK);
        assert!(!bitvec[64]);
    }

    #[test]
    #[should_panic]
    fn test_bitvec_index_out_of_bounds_head() {
        let mut bitvec = BitDeque::new(2 * BYTES_PER_BLOCK);
        bitvec.resize(3 * BITS_PER_BLOCK);
        assert!(!bitvec[63]);
    }

    #[test]
    fn test_bitvec_resize() {
        let mut bitvec = BitDeque::new(2 * BYTES_PER_BLOCK);
        assert_eq!(2, bitvec.max_blocks);
        assert_eq!(0..0, bitvec.bit_range);
        assert_eq!(0, bitvec.blocks.len());

        bitvec.resize(1);
        assert_eq!(0..1, bitvec.bit_range);
        assert_eq!(1, bitvec.blocks.len());

        bitvec.resize(64);
        assert_eq!(0..64, bitvec.bit_range);
        assert_eq!(1, bitvec.blocks.len());

        bitvec.resize(65);
        assert_eq!(0..65, bitvec.bit_range);
        assert_eq!(2, bitvec.blocks.len());

        bitvec.resize(128);
        assert_eq!(0..128, bitvec.bit_range);
        assert_eq!(2, bitvec.blocks.len());

        bitvec.resize(129);
        assert_eq!(64..129, bitvec.bit_range);
        assert_eq!(2, bitvec.blocks.len());
    }
}
