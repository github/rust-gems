/*
   Layout:
       Want <5% permutation table!
       Max posting lists with 2^28 documents = ~90MB
       NVMe page size is 4096 bytes
       4 bytes for block offset ==> 4 billion blocks
       counts
           4 bytes ==> leads to overflow, but NOT within one posting list!
           3 bytes only works for <16 million documents :(
           Store count into middle of block
       64 block size
           index size: 64 * 4 billion ==> 256 GB
           lookup overhead: 6.25%
           blocks per page: 64 ==> can store about 4 layers within one page! ==> "worst case" 4 pages
           count: 32 byte (AVX-2, 256-bits)
               overhead of count: 6.25%
               total overhead: 12.5% ==> not good to store within lookup table!
       128 block size
           index size: 128 * 4 billion ==> 512 GB
           lookup overhead: 3.125%
           blocks per page: 32 ==> can store at least 3 layers (of at most 14) within one page! ==> "worst case" 5 pages
           count: 64 bytes (AVX-512, 512-bits)
               overhead of count: 3.125%
               total overhead: 6.25%
               1 block = 4 * 32 counts ==> store counts inside of page?
           sub-counts: 32-bytes
               one-byte for each sub-count
               two splits: --- 1 byte --- 4 bytes --- 1 byte ---
               overhead of sub-counts: 1.5625%

   Iteration:
       basic: decode next 4 values on level L
       decode ahead: decode next 16? values down to level L
           decode on each previous level ~16 values ahead
           not quite clear how to arrange the data best
           16 values fit into one avx2 register
*/

use std::cell::RefCell;

pub(crate) type Word = u64;

const BLOCK_BYTES: usize = 64;
const BLOCK_BITS: usize = BLOCK_BYTES * 8;
const BLOCKS_PER_PAGE: usize = BLOCK_BYTES / 4;
pub(crate) const WORD_BITS: usize = WORD_BYTES * 8;
pub(crate) const WORD_BYTES: usize = std::mem::size_of::<Word>();
const WORDS_PER_BLOCK: usize = BLOCK_BYTES / WORD_BYTES;
const PAGE_BYTES: usize = BLOCKS_PER_PAGE * BLOCK_BYTES;
const PAGE_BITS: usize = PAGE_BYTES * 8;
const SUPER_PAGE_BITS: usize = 4096 * 8;

#[repr(C, align(128))]
#[derive(Default, Clone)]
struct Block {
    words: [Word; WORDS_PER_BLOCK],
}

#[derive(Default)]
pub(crate) struct VirtualBitRank {
    // In order to look up bit i, use block_mapping[i / BLOCK_BITS]
    block_mapping: Vec<u32>,
    blocks: Vec<Block>,
    // Remember which pages have been accessed.
    stats: Vec<RefCell<u64>>,
}

impl VirtualBitRank {
    pub(crate) fn new() -> Self {
        Self::with_capacity(0)
    }

    pub(crate) fn with_capacity(bits: usize) -> Self {
        let bits = (bits + BLOCK_BITS - 1) & !(BLOCK_BITS - 1);
        Self {
            block_mapping: vec![0; bits / BLOCK_BITS], // 0 means unused block!!
            blocks: Vec::with_capacity(bits / BLOCK_BITS),
            stats: Vec::new(),
        }
    }

    pub(crate) fn reset_stats(&mut self) {
        self.stats = vec![RefCell::new(0); self.blocks.len() * BLOCK_BITS / SUPER_PAGE_BITS + 1];
    }

    pub(crate) fn page_count(&self) -> (usize, usize) {
        (
            self.stats
                .iter()
                .map(|v| v.borrow().count_ones() as usize)
                .sum(),
            (self.blocks.len() + BLOCKS_PER_PAGE - 1) / BLOCKS_PER_PAGE,
        )
    }

    fn bit_to_block(&self, bit: usize) -> usize {
        let block = bit / BLOCK_BITS;
        let result2 = block + (block / (BLOCKS_PER_PAGE - 1)) + 1;
        //let result = self.block_mapping[bit / BLOCK_BITS] as usize;
        //assert_eq!(result2, result);
        //if let Some(v) = self.stats.get(result * BLOCK_BITS / SUPER_PAGE_BITS / 64) {
        //    *v.borrow_mut() += 1 << (result % 64);
        //}
        result2
    }

    fn mid_rank(&self, block: usize) -> u32 {
        let first_block = block & !(BLOCKS_PER_PAGE - 1);
        let array = self.blocks[first_block].words.as_ptr() as *const u32;
        unsafe { array.add(block & (BLOCKS_PER_PAGE - 1)).read() }
    }

    pub(crate) fn rank(&self, bit: usize) -> u32 {
        let block = self.bit_to_block(bit);
        let mut rank = self.mid_rank(block);
        let word = (bit / WORD_BITS) & (WORDS_PER_BLOCK - 1);
        let bit_in_word = bit & (WORD_BITS - 1);
        if word >= WORDS_PER_BLOCK / 2 {
            for i in WORDS_PER_BLOCK / 2..word {
                rank += self.blocks[block].words[i].count_ones();
            }
            if bit_in_word != 0 {
                rank + (self.blocks[block].words[word] << (WORD_BITS - bit_in_word)).count_ones()
            } else {
                rank
            }
        } else {
            for i in word + 1..WORDS_PER_BLOCK / 2 {
                rank -= self.blocks[block].words[i].count_ones();
            }
            rank - (self.blocks[block].words[word] >> bit_in_word).count_ones()
        }
    }

    pub(crate) fn rank_with_word(&self, bit: usize) -> (u32, Word) {
        let block = self.bit_to_block(bit);
        let mut rank = self.mid_rank(block);
        let word = (bit / WORD_BITS) & (WORDS_PER_BLOCK - 1);
        let bit_in_word = bit & (WORD_BITS - 1);
        if word >= WORDS_PER_BLOCK / 2 {
            for i in WORDS_PER_BLOCK / 2..word {
                rank += self.blocks[block].words[i].count_ones();
            }
            if bit_in_word != 0 {
                (
                    rank + (self.blocks[block].words[word] << (WORD_BITS - bit_in_word))
                        .count_ones(),
                    self.blocks[block].words[word] >> bit_in_word,
                )
            } else {
                (rank, self.blocks[block].words[word])
            }
        } else {
            for i in word + 1..WORDS_PER_BLOCK / 2 {
                rank -= self.blocks[block].words[i].count_ones();
            }
            let w = self.blocks[block].words[word] >> bit_in_word;
            (rank - w.count_ones(), w)
        }
    }

    pub(crate) fn reserve(&mut self, bits: usize) {
        assert!(self.block_mapping.is_empty());
        assert!(self.blocks.is_empty());
        // let pages = (bits + PAGE_BITS - 1) / PAGE_BITS;
        let blocks = (bits + BLOCK_BITS - 1) / BLOCK_BITS;
        for _ in 0..blocks {
            if self.blocks.len() % BLOCKS_PER_PAGE == 0 {
                self.blocks.push(Block::default());
            }
            self.block_mapping.push(self.blocks.len() as u32);
            self.blocks.push(Block::default());
        }
    }

    fn get_word_mut(&mut self, bit: usize) -> &mut Word {
        let block = bit / BLOCK_BITS;
        if block >= self.block_mapping.len() {
            self.block_mapping.resize(block + 1, 0);
        }
        if self.block_mapping[block] == 0 {
            if self.blocks.len() % BLOCKS_PER_PAGE == 0 {
                self.blocks.push(Block::default()); // Block with rank information
            }
            self.block_mapping[block] = self.blocks.len() as u32;
            self.blocks.push(Block::default());
        }
        let block = self.bit_to_block(bit);
        let word = (bit / WORD_BITS) & (WORDS_PER_BLOCK - 1);
        &mut self.blocks[block].words[word]
    }

    pub(crate) fn set(&mut self, bit: usize) {
        *self.get_word_mut(bit) |= 1 << (bit & (WORD_BITS - 1));
    }

    pub(crate) fn set_nibble(&mut self, nibble_idx: usize, nibble_value: u32) {
        let bit_idx = nibble_idx * 4;
        // clear all bits...
        // *self.get_word(bit_idx) &= !(15 << (bit_idx & (WORD_BITS - 1)));
        *self.get_word_mut(bit_idx) |= (nibble_value as Word) << (bit_idx & (WORD_BITS - 1));
    }

    pub(crate) fn build(&mut self) {
        for block in &mut self.block_mapping {
            if *block == 0 {
                if self.blocks.len() % BLOCKS_PER_PAGE == 0 {
                    self.blocks.push(Block::default()); // Block with rank information
                }
                *block = self.blocks.len() as u32;
                self.blocks.push(Block::default());
            }
        }
        let mut ones = 0;
        for block in 0..self.block_mapping.len() {
            let block = self.block_mapping[block] as usize;
            for i in 0..WORDS_PER_BLOCK / 2 {
                ones += self.blocks[block].words[i].count_ones();
            }
            unsafe {
                let first_block = block & !(BLOCKS_PER_PAGE - 1);
                let array = self.blocks[first_block].words.as_mut_ptr() as *mut u32;
                let rank = array.add(block & (BLOCKS_PER_PAGE - 1));
                rank.write(ones);
            }
            for i in WORDS_PER_BLOCK / 2..WORDS_PER_BLOCK {
                ones += self.blocks[block].words[i].count_ones();
            }
        }
    }

    pub(crate) fn get_word_suffix(&self, i: usize) -> Word {
        let block = self.bit_to_block(i);
        let word = (i / WORD_BITS) & (WORDS_PER_BLOCK - 1);
        let bit = i / WORD_BITS;
        self.blocks[block].words[word] >> bit
    }

    pub(crate) fn get_word(&self, i: usize) -> Word {
        let block = self.bit_to_block(i);
        let word = (i / WORD_BITS) & (WORDS_PER_BLOCK - 1);
        let bit = i % WORD_BITS;
        let first_part = self.blocks[block].words[word] >> bit;
        if bit == 0 {
            first_part
        } else {
            let block = self.bit_to_block(i + 63);
            let word = ((i + 63) / WORD_BITS) & (WORDS_PER_BLOCK - 1);
            first_part | (self.blocks[block].words[word] << (WORD_BITS - bit))
        }
    }

    pub(crate) fn get_bit(&self, bit: usize) -> bool {
        let block = self.bit_to_block(bit);
        let word = (bit / WORD_BITS) & (WORDS_PER_BLOCK - 1);
        let bit_in_word = bit & (WORD_BITS - 1);
        self.blocks[block].words[word] & (1 << bit_in_word) != 0
    }

    pub(crate) fn get_nibble(&self, nibble_idx: usize) -> u32 {
        let bit_idx = nibble_idx * 4;
        let block = self.bit_to_block(bit_idx);
        let word = (bit_idx / WORD_BITS) & (WORDS_PER_BLOCK - 1);
        let bit_in_word = bit_idx & (WORD_BITS - 1);
        ((self.blocks[block].words[word] >> bit_in_word) & 15) as u32
    }

    fn len(&self) -> usize {
        self.block_mapping.len() * BLOCK_BITS
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use rand::{seq::SliceRandom, thread_rng, RngCore};

    use super::{VirtualBitRank, BLOCK_BITS, WORD_BITS};

    #[test]
    fn test_bitrank() {
        let mut bitrank = VirtualBitRank::with_capacity(1 << 20);
        let mut rank = vec![];
        let mut bits = vec![];
        let mut ones = 0;
        for i in 0..bitrank.len() {
            let bit = thread_rng().next_u32() % 2 == 1;
            rank.push(ones);
            bits.push(bit);
            if bit {
                bitrank.set(i);
                ones += 1;
            }
        }
        bitrank.build();
        for (i, bit) in bits.iter().enumerate() {
            assert_eq!(bitrank.get_bit(i), *bit, "at position {i}");
        }
        for (i, r) in rank.iter().enumerate() {
            assert_eq!(bitrank.rank(i), *r, "at position {i}");
        }
    }

    /// This test emulates the reordering of blocks on disk.
    /// With a real NVMe, the performance difference should be much larger.
    /// But even this basic test shows that the access pattern matters.
    /// I.e. throughput is  15% higher when access order is non-random.
    #[test]
    fn test_random_order() {
        let mut bitrank = VirtualBitRank::with_capacity(1 << 20);
        let random_bits: Vec<_> = (0..bitrank.len() / WORD_BITS)
            .map(|_| thread_rng().next_u32())
            .flat_map(|i| {
                [i, i + 1].into_iter().map(|i| {
                    (i & !(BLOCK_BITS - 1) as u32) % bitrank.len() as u32
                    //(i & !(BLOCK_BITS - 1) as u32 + BLOCK_BITS as u32 / 2) % bitrank.len() as u32
                    // i % bitrank.len() as u32
                })
            })
            .collect();
        for i in &random_bits {
            bitrank.set(*i as usize);
        }
        bitrank.build();

        let mut sorted_bits = random_bits.clone();
        sorted_bits.shuffle(&mut thread_rng());

        for _ in 0..4 {
            let time = Instant::now();
            for i in &random_bits {
                assert!(bitrank.get_bit(*i as usize), "at position {i}");
            }
            println!(
                "time to check random bits: {:?} {:?}",
                time.elapsed(),
                time.elapsed() * 100 / random_bits.len() as u32
            );

            let time = Instant::now();
            for i in &sorted_bits {
                assert!(bitrank.get_bit(*i as usize), "at position {i}");
            }
            println!(
                "time to check sorted bits: {:?} {:?}",
                time.elapsed(),
                time.elapsed() * 100 / random_bits.len() as u32
            );
        }
    }
}
