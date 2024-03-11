use std::fmt::Debug;
use std::iter::Peekable;
use std::mem::size_of;
use std::ops::{BitOr, BitXor};
use std::{cmp::Ordering, marker::PhantomData};

use crate::config::{take_ref, IsBucketType, BITS_PER_BLOCK, BYTES_PER_BLOCK};

#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct BitChunk {
    pub index: usize,
    pub block: u64,
}

impl BitChunk {
    #[inline]
    pub fn new(index: usize, block: u64) -> Self {
        debug_assert_ne!(0, block, "bitchunks cannot be zero");
        Self { index, block }
    }
}

pub(crate) fn iter_bit_chunks(
    leading: impl Iterator<Item = usize>,
    trailing: impl Iterator<Item = BitChunk>,
) -> impl Iterator<Item = BitChunk> {
    BitChunkIterator {
        leading: leading.peekable(),
        trailing: trailing.peekable(),
    }
}

struct BitChunkIterator<I: Iterator<Item = BitChunk>, J: Iterator<Item = usize>> {
    trailing: Peekable<I>,
    leading: Peekable<J>,
}

impl<I: Iterator<Item = BitChunk>, J: Iterator<Item = usize>> Iterator for BitChunkIterator<I, J> {
    type Item = BitChunk;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(bit) = self.leading.next() {
            // Test whether it can be merged with more leading bits.
            let (index, bit) = bit.into_index_and_bit();
            let mut block: u64 = bit.into_block();
            loop {
                match self.leading.peek() {
                    Some(other_bit) if other_bit.into_index() == index => {
                        let other_block = other_bit.into_bit().into_block();
                        debug_assert!(block & other_block == 0);
                        block |= other_block;
                        self.leading.next();
                    }
                    Some(_) => return Some(BitChunk::new(index, block)),
                    _ => break,
                }
            }
            // All leading bits were consumed, test whether it can be merged with
            // trailing bits.
            match self.trailing.peek() {
                Some(BitChunk {
                    index: other_index,
                    block: other_block,
                }) if *other_index == index => {
                    debug_assert!(
                        block & other_block == 0,
                        "leading and trailing bits may not overlap"
                    );
                    block ^= *other_block;
                    self.trailing.next();
                }
                _ => {}
            }
            Some(BitChunk::new(index, block))
        } else {
            self.trailing.next()
        }
    }
}

/// Combine-s two `BitChunk` iterators using the given operator. Both iterators have to output
/// `BitChunk`s from most to least significant and must not report duplicates!
struct BinOpBitChunksIterator<
    I: Iterator<Item = BitChunk>,
    J: Iterator<Item = BitChunk>,
    Op: Fn(u64, u64) -> u64,
> {
    a: Peekable<I>,
    b: Peekable<J>,
    op: Op,
}

impl<I: Iterator<Item = BitChunk>, J: Iterator<Item = BitChunk>, Op: Fn(u64, u64) -> u64> Iterator
    for BinOpBitChunksIterator<I, J, Op>
{
    type Item = BitChunk;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match (self.a.peek(), self.b.peek()) {
                (None, _) => return self.b.next(),
                (_, None) => return self.a.next(),
                (
                    Some(BitChunk {
                        index: x,
                        block: ma,
                    }),
                    Some(BitChunk {
                        index: y,
                        block: mb,
                    }),
                ) => match x.cmp(y) {
                    Ordering::Equal => {
                        let result = BitChunk {
                            index: *x,
                            block: (self.op)(*ma, *mb),
                        };
                        self.a.next();
                        self.b.next();
                        if result.block != 0 {
                            return Some(result);
                        }
                    }
                    Ordering::Less => return self.b.next(),
                    Ordering::Greater => return self.a.next(),
                },
            }
        }
    }
}

/// Or-s two `BitChunk` iterators. Both iterators have to output `BitChunk`s from most to least
/// significant and must not report duplicates!
pub(crate) fn or_bit_chunks(
    a: impl Iterator<Item = BitChunk>,
    b: impl Iterator<Item = BitChunk>,
) -> impl Iterator<Item = BitChunk> {
    BinOpBitChunksIterator {
        a: a.peekable(),
        b: b.peekable(),
        op: u64::bitor,
    }
}

/// Xor-s two `BitChunk` iterators. Both iterators have to output `BitChunk`s from most to least
/// significant and must not report duplicates!
pub(crate) fn xor_bit_chunks(
    a: impl Iterator<Item = BitChunk>,
    b: impl Iterator<Item = BitChunk>,
) -> impl Iterator<Item = BitChunk> {
    BinOpBitChunksIterator {
        a: a.peekable(),
        b: b.peekable(),
        op: u64::bitxor,
    }
}

/// ANDs the bit chunks with the repetitive mask without compressing the result to the one bits.
/// The result will only contain non-zero BitChunks!
pub(crate) fn mask_bit_chunks(
    iter: impl Iterator<Item = BitChunk>,
    mask: u64,
    modulus: usize,
) -> impl Iterator<Item = BitChunk> {
    MaskedBitChunksIterator::new(iter, mask, modulus)
}

struct MaskedBitChunksIterator<I: Iterator<Item = BitChunk>> {
    pub iter: I,
    pub block_mask: u64, // The mask repeated to fill the full block.
    pub modulus: usize,  // Number of bits the mask consists of.
}

impl<I: Iterator<Item = BitChunk>> MaskedBitChunksIterator<I> {
    fn new(iter: I, mask: u64, modulus: usize) -> Self {
        let mut block_mask = mask;
        let mut i = modulus;
        while i < BITS_PER_BLOCK {
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
            let index = bit_chunk.index * BITS_PER_BLOCK;
            let shift = index % self.modulus;
            let curr_mask =
                (self.block_mask >> shift) | (self.block_mask << (self.modulus - shift));
            bit_chunk.block &= curr_mask;
            if bit_chunk.block != 0 {
                return Some(bit_chunk);
            }
        }
        None
    }
}

/// Returns a pair of the number of one buckets and the lowest included bucket id within
/// the given size limits for the given bitchunks.
pub(crate) fn count_ones_from_bitchunks<T: IsBucketType>(
    chunks: Peekable<impl Iterator<Item = BitChunk>>,
    max_bytes: usize,
    max_msb_len: usize,
) -> (usize, usize) {
    let mut ones = iter_ones::<T, _>(chunks);

    let mut total = take_ref(&mut ones, max_msb_len - 1).count();
    let smallest_msb = ones
        .next()
        .map(|bucket| {
            total += 1;
            bucket
        })
        .unwrap_or_default();

    count_ones_from_msb_and_lsb(
        total,
        smallest_msb,
        ones.into_bitchunks(),
        max_bytes,
        max_msb_len,
    )
}

/// Returns a pair of the number of one buckets and the lowest included bucket id within the
/// given size limits for the given most-significant bucket count and bitchunks.
pub(crate) fn count_ones_from_msb_and_lsb<T: IsBucketType>(
    mut total: usize,
    smallest_msb: T,
    lsb_chunks: Peekable<impl Iterator<Item = BitChunk>>,
    max_bytes: usize,
    max_msb_len: usize,
) -> (usize, usize) {
    let mut smallest_index = lsb_index_lower_bound(max_bytes, max_msb_len, smallest_msb);
    total += lsb_chunks
        .take_while(|BitChunk { index, .. }| *index >= smallest_index)
        .map(|BitChunk { block, .. }| block.count_ones() as usize)
        .sum::<usize>();
    if total <= max_msb_len {
        smallest_index = 0;
    }

    (total, smallest_index * BITS_PER_BLOCK)
}

/// Compute the inclusive LSB index lower bound.
#[inline]
pub(crate) fn lsb_index_lower_bound<T: IsBucketType>(
    max_bytes: usize,
    max_msb_len: usize,
    smallest_msb: T,
) -> usize {
    lsb_index_upper_bound(smallest_msb)
        .saturating_sub(max_lsb_bytes::<T>(max_bytes, max_msb_len) / BYTES_PER_BLOCK)
}

/// Compute the exclusive LSB index upper bound.
#[inline]
pub(crate) fn lsb_index_upper_bound<T: IsBucketType>(smallest_msb: T) -> usize {
    (smallest_msb.into_usize() + BITS_PER_BLOCK - 1).into_index()
}

#[inline]
pub(crate) fn max_lsb_bytes<T: IsBucketType>(max_bytes: usize, max_msb_len: usize) -> usize {
    max_bytes - size_of::<T>() * max_msb_len
}

/// Returns indices of set bits from most significant to least significant.
pub(crate) fn iter_ones<T: IsBucketType, I: Iterator<Item = BitChunk>>(
    chunks: Peekable<I>,
) -> BitChunksOnes<T, I> {
    BitChunksOnes(chunks, PhantomData)
}

pub(crate) struct BitChunksOnes<T: IsBucketType, I: Iterator<Item = BitChunk>>(
    Peekable<I>,
    PhantomData<T>,
);

impl<T: IsBucketType, I: Iterator<Item = BitChunk>> BitChunksOnes<T, I> {
    /// Returns the remaining [`BitChunk`]s.
    pub(crate) fn into_bitchunks(self) -> Peekable<I> {
        self.0
    }
}

impl<T: IsBucketType, I: Iterator<Item = BitChunk>> Iterator for BitChunksOnes<T, I> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(BitChunk { index, block }) = self.0.peek_mut() {
            let bit = T::from_block_msb(*block);
            *block ^= bit.into_block();
            let bucket = bit.into_bucket(*index);
            if *block == 0 {
                self.0.next();
            }
            Some(bucket)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::{iter_ones, BitChunk};

    #[test]
    fn test_iter_ones() {
        let chunks = vec![
            BitChunk {
                index: 3,
                block: 0b00110001, // 200..192
            },
            BitChunk {
                index: 1,
                block: 0b10001010, // 72..64
            },
            BitChunk {
                index: 0,
                block: 0b01001110, // 8..0
            },
        ];
        let expected = vec![197, 196, 192, 71, 67, 65, 6, 3, 2, 1];
        assert_eq!(
            expected.into_iter().collect_vec(),
            iter_ones::<usize, _>(chunks.into_iter().peekable()).collect_vec()
        );
    }
}
