//! A bit-vector data structure, optimized for
//! [rank](http://bitmagic.io/rank-select.html) operations.
//!
//! See also: ["Succinct data structure"](https://en.wikipedia.org/wiki/Succinct_data_structure).

type SubblockBits = u128;

// Static sizing of the various components of the data structure.
const BITS_PER_SUB_BLOCK: usize = SubblockBits::BITS as usize;
const SUB_BLOCKS_PER_BLOCK: usize = 64;
const BITS_PER_BLOCK: usize = SUB_BLOCKS_PER_BLOCK * BITS_PER_SUB_BLOCK; // 8192 Bits = 1 kBytes

/// A container for a portion of the total bit vector and the associated indices.
/// The bits within each chunk are stored from most significant bit (msb) to least significant bit (lsb).
/// i.e. index 0 of a Chunk is at the start of visual binary representation or a value of
/// 1u128 << 127.
///
/// The actual bits are stored alongside the indices because the common case will be reading this
/// information from disk (rather than random access memory), so it is beneficial to have all of
/// the data that we need in the same page.
///
/// ```text
/// index:           [ 0, 1, 2, 3, 4, 5, 6, 7 ]
/// bits:            [ 0, 1, 0, 1, 1, 0, 1, 0 ]
/// rank(exclusive): [ 0, 0, 1, 1, 2, 3, 3, 4 ]
/// block rank:      [           0            ]
/// sub-block rank:  [     0     ][     2     ]
/// ```
#[derive(Clone, Debug)]
struct Block {
    /// Rank of the first bit in this block (that is, the number of bits set in previous blocks).
    rank: u64,
    /// Rank of the first bit (bit 0) of each subblock, relative to the start of the block.
    /// That is, `sub_blocks[i]` is the number of bits set in the `bits` representing
    /// sub-blocks `0..i`. `sub_blocks[0]` is always zero.
    sub_blocks: [u16; SUB_BLOCKS_PER_BLOCK],
    /// The bit-vector.
    bits: [SubblockBits; SUB_BLOCKS_PER_BLOCK],
}

impl Block {
    /// Set a bit without updating `self.sub_blocks`.
    ///
    /// This panics if the bit was already set, because that indicates that the original positions
    /// list is invalid/had duplicates.
    fn set(&mut self, index: usize) {
        debug_assert!(index < BITS_PER_BLOCK);
        let chunk_idx = index / BITS_PER_SUB_BLOCK;
        let bit_idx = index % BITS_PER_SUB_BLOCK;
        let mask = 1 << bit_idx;
        debug_assert_eq!(self.bits[chunk_idx] & mask, 0, "toggling bits off indicates that the original data was incorrect, most likely containing duplicate values.");
        self.bits[chunk_idx] |= mask;
    }

    /// The **total rank** of the block relative local index, and the index of the one
    /// bit that establishes that rank (aka "select") **if** it occurs within that same
    /// chunk, otherwise ['None'].  The assumption is that if you would have to look back
    /// through previous chunks it would actually be cheaper to do a lookup in the original
    /// data structure that the bit vector was created from.
    fn rank(&self, local_idx: usize) -> usize {
        let mut rank = self.rank as usize;
        let sub_block = local_idx / BITS_PER_SUB_BLOCK;
        rank += self.sub_blocks[sub_block] as usize;

        let remainder = local_idx % BITS_PER_SUB_BLOCK;

        let last_chunk = local_idx / BITS_PER_SUB_BLOCK;
        let masked = if remainder == 0 {
            0
        } else {
            self.bits[last_chunk] << (BITS_PER_SUB_BLOCK - remainder)
        };
        rank + masked.count_ones() as usize
    }

    fn total_rank(&self) -> usize {
        self.sub_blocks[SUB_BLOCKS_PER_BLOCK - 1] as usize
            + self.rank as usize
            + self.bits[SUB_BLOCKS_PER_BLOCK - 1..]
                .iter()
                .map(|c| c.count_ones() as usize)
                .sum::<usize>()
    }
}

/// Builder for creating a [`BitRank`].
///
/// # Examples
///
/// ```text
/// let mut builder = BitRankBuilder::new();
/// builder.push(17);
/// builder.push(23);
/// builder.push(102);
/// let set = builder.finish();
/// assert_eq!(set.rank(100), 2);
/// ```
#[derive(Default)]
pub struct BitRankBuilder {
    blocks: Vec<Block>,
}

impl BitRankBuilder {
    /// Returns a builder that can hold integers with values `0..cap`.
    pub fn with_capacity(cap: usize) -> Self {
        const ZERO_BLOCK: Block = Block {
            rank: 0,
            sub_blocks: [0; SUB_BLOCKS_PER_BLOCK],
            bits: [0; SUB_BLOCKS_PER_BLOCK],
        };
        Self {
            blocks: vec![ZERO_BLOCK; cap.div_ceil(BITS_PER_BLOCK)],
        }
    }

    /// Adds a bit. Bits must be added in order of increasing `position`.
    pub fn push(&mut self, position: usize) {
        let block_id = position / BITS_PER_BLOCK;
        self.blocks[block_id].set(position % BITS_PER_BLOCK);
    }

    /// Finishes the `BitRank` by writing the last block of data.
    pub fn finish(mut self) -> BitRank {
        let mut total_rank = 0;
        for block in &mut self.blocks {
            block.rank = total_rank;
            let mut local_rank = 0;
            for (i, chunk) in block.bits.iter().enumerate() {
                block.sub_blocks[i] = local_rank;
                local_rank += chunk.count_ones() as u16;
            }
            total_rank += local_rank as u64
        }
        BitRank {
            blocks: self.blocks,
        }
    }
}

/// An immutable set of unsigned integers with an efficient `rank` method.
#[derive(Clone)]
pub struct BitRank {
    blocks: Vec<Block>,
}

impl BitRank {
    /// The rank at the specified index (exclusive).
    ///
    /// The (one) rank is defined as: `rank(i) = sum(b[j] for j in 0..i)`
    /// i.e. the number of elements less than `i`.
    pub fn rank(&self, idx: usize) -> usize {
        let block_num = idx / BITS_PER_BLOCK;
        if block_num >= self.blocks.len() {
            self.max_rank() // fall back to 0 bits when the bitrank data structure is empty.
        } else {
            self.blocks[block_num].rank(idx % BITS_PER_BLOCK)
        }
    }

    /// Returns the number of elements in the set.
    pub fn max_rank(&self) -> usize {
        self.blocks
            .last()
            .map(|b| b.total_rank())
            .unwrap_or_default() // fall back to 0 when the bitrank data structure is empty.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distr::Uniform;
    use rand::prelude::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Creates a `BitRank` containing the integers in `iter` (which should be strictly
    /// increasing).
    pub fn bitrank<I: IntoIterator<Item = usize>>(capacity: usize, iter: I) -> BitRank {
        let mut builder = BitRankBuilder::with_capacity(capacity);
        for position in iter {
            builder.push(position);
        }
        builder.finish()
    }

    #[test]
    fn test_rank_zero() {
        let br = bitrank(1, [0]);
        assert_eq!(br.rank(0), 0);
        assert_eq!(br.rank(1), 1);
    }

    #[test]
    fn test_empty() {
        let br = bitrank(0, []);
        assert!(br.blocks.is_empty());
    }

    #[test]
    fn test_index_out_of_bounds() {
        let br = bitrank(BITS_PER_BLOCK, [BITS_PER_BLOCK - 1]);
        assert_eq!(br.rank(BITS_PER_BLOCK), 1);
    }

    #[test]
    #[should_panic]
    fn test_duplicate_position() {
        bitrank(91, [64, 66, 68, 68, 90]);
    }

    #[test]
    fn test_rank_exclusive() {
        let br = bitrank(133, 0..132);
        assert_eq!(br.blocks.len(), 1);
        assert_eq!(br.rank(64), 64);
        assert_eq!(br.rank(132), 132);
    }

    #[test]
    fn test_rank() {
        let mut positions: Vec<usize> = (0..132).collect();
        positions.append(&mut vec![138usize, 140, 146]);
        let br = bitrank(146, positions);
        assert_eq!(br.rank(135), 132);

        let br2 = bitrank(BITS_PER_BLOCK, 0..BITS_PER_BLOCK - 5);
        assert_eq!(br2.rank(169), 169);

        let br3 = bitrank(BITS_PER_BLOCK + 6, 0..BITS_PER_BLOCK + 5);
        assert_eq!(br3.rank(BITS_PER_BLOCK), BITS_PER_BLOCK);
    }

    #[test]
    fn test_rank_idx() {
        let mut positions: Vec<usize> = (0..132).collect();
        positions.append(&mut vec![138usize, 140, 146]);
        let br = bitrank(147, positions);
        assert_eq!(br.rank(135), 132);

        let bits2: Vec<usize> = (0..BITS_PER_BLOCK - 5).collect();
        let br2 = bitrank(BITS_PER_BLOCK, bits2);
        assert_eq!(br2.rank(169), 169);

        let bits3: Vec<usize> = (0..BITS_PER_BLOCK + 5).collect();
        let br3 = bitrank(BITS_PER_BLOCK + 6, bits3);
        assert_eq!(br3.rank(BITS_PER_BLOCK), BITS_PER_BLOCK);

        let bits4: Vec<usize> = vec![1, 1000, 9999, BITS_PER_BLOCK + 1];
        let br4 = bitrank(BITS_PER_BLOCK + 1, bits4);
        assert_eq!(br4.rank(10000), 3);

        let bits5: Vec<usize> = vec![1, 1000, 9999, BITS_PER_BLOCK + 1];
        let br5 = bitrank(BITS_PER_BLOCK + 1, bits5);
        assert_eq!(br5.rank(BITS_PER_BLOCK), 3);
    }

    #[test]
    fn test_rank_large_random() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let uniform = Uniform::new(0, 1_000_000).unwrap();
        let mut random_bits = Vec::with_capacity(100_000);
        for _ in 0..100_000 {
            random_bits.push(uniform.sample(&mut rng));
        }
        random_bits.sort_unstable();
        // This isn't strictly necessary, given that the bit would just be toggled again, but it
        // ensures that we are meeting the contract.
        random_bits.dedup();
        let br = bitrank(1_000_000, random_bits.iter().copied());
        let mut rank = 0;
        for i in 0..random_bits.capacity() {
            assert_eq!(br.rank(i), rank);
            if i == random_bits[rank] {
                rank += 1;
            }
        }
    }

    /// Test that we properly handle the case where the position is out of bounds for all
    /// potentially tricky bit positions.
    #[test]
    fn test_rank_out_of_bounds() {
        for i in 1..30 {
            let br = bitrank(BITS_PER_BLOCK * i, [BITS_PER_BLOCK * i - 1]);
            assert_eq!(br.max_rank(), 1);
            assert_eq!(br.rank(BITS_PER_BLOCK * i - 1), 0);
            for j in 0..10 {
                assert_eq!(br.rank(BITS_PER_BLOCK * (i + j)), 1);
            }
        }
    }

    #[test]
    fn test_large_gap() {
        let br = bitrank(
            BITS_PER_BLOCK * 16,
            (3..4).chain(BITS_PER_BLOCK * 15..BITS_PER_BLOCK * 15 + 17),
        );
        for i in 1..15 {
            assert_eq!(br.rank(BITS_PER_BLOCK * i), 1);
        }
        for i in 0..18 {
            assert_eq!(br.rank(BITS_PER_BLOCK * 15 + i), 1 + i);
        }
    }

    #[test]
    fn test_with_capacity() {
        let mut b = BitRankBuilder::with_capacity(BITS_PER_BLOCK * 3 - 1);
        let initial_capacity = b.blocks.capacity();
        assert!(initial_capacity >= 3);
        b.push(BITS_PER_BLOCK * 3 - 2); // should not have to grow
        assert_eq!(b.blocks.capacity(), initial_capacity);

        let mut b = BitRankBuilder::with_capacity(BITS_PER_BLOCK * 3 + 1);
        let initial_capacity = b.blocks.capacity();
        assert!(initial_capacity >= 4);
        b.push(BITS_PER_BLOCK * 3); // should not have to grow
        assert_eq!(b.blocks.capacity(), initial_capacity);
    }
}
