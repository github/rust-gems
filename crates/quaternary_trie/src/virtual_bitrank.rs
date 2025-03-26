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

pub(crate) type Word = u64;

const BLOCK_BYTES: usize = 64;
const BLOCK_BITS: usize = BLOCK_BYTES * 8;
pub(crate) const WORD_BITS: usize = WORD_BYTES * 8;
pub(crate) const WORD_BYTES: usize = std::mem::size_of::<Word>();
const WORDS_PER_BLOCK: usize = BLOCK_BYTES / WORD_BYTES;

#[derive(Default)]
pub(crate) struct VirtualBitRank {
    // In order to look up bit i, use block_mapping[i / BLOCK_BITS]
    words: Vec<Word>,
    aggregated: Vec<u32>,
}

impl VirtualBitRank {
    pub(crate) fn new() -> Self {
        Self::with_capacity(0)
    }

    pub(crate) fn with_capacity(bits: usize) -> Self {
        let bits = (bits + BLOCK_BITS - 1) & !(BLOCK_BITS - 1);
        Self {
            words: vec![0; bits / WORD_BITS],
            aggregated: vec![],
        }
    }

    fn bit_to_word(&self, bit: usize) -> usize {
        bit / WORD_BITS
    }

    fn mid_rank(&self, bit: usize) -> u32 {
        self.aggregated[bit / BLOCK_BITS]
    }

    pub(crate) fn rank(&self, bit: usize) -> u32 {
        let word = self.bit_to_word(bit);
        let mut rank = self.mid_rank(bit);
        let bit_in_word = bit & (WORD_BITS - 1);
        let mid_word = (word & !(WORDS_PER_BLOCK - 1)) + WORDS_PER_BLOCK / 2;
        if word >= mid_word {
            for i in mid_word..word {
                rank += self.words[i].count_ones();
            }
            if bit_in_word != 0 {
                rank + (self.words[word] << (WORD_BITS - bit_in_word)).count_ones()
            } else {
                rank
            }
        } else {
            for i in word + 1..mid_word {
                rank -= self.words[i].count_ones();
            }
            rank - (self.words[word] >> bit_in_word).count_ones()
        }
    }

    pub(crate) fn rank_with_word(&self, bit: usize) -> (u32, Word) {
        let mut rank = self.mid_rank(bit);
        let word = self.bit_to_word(bit);
        let bit_in_word = bit & (WORD_BITS - 1);
        let mid_word = (word & !(WORDS_PER_BLOCK - 1)) + WORDS_PER_BLOCK / 2;
        if word >= mid_word {
            for i in mid_word..word {
                rank += self.words[i].count_ones();
            }
            if bit_in_word != 0 {
                (
                    rank + (self.words[word] << (WORD_BITS - bit_in_word)).count_ones(),
                    self.words[word] >> bit_in_word,
                )
            } else {
                (rank, self.words[word])
            }
        } else {
            for i in word + 1..mid_word {
                rank -= self.words[i].count_ones();
            }
            let w = self.words[word] >> bit_in_word;
            (rank - w.count_ones(), w)
        }
    }

    pub(crate) fn reserve(&mut self, bits: usize) {
        assert!(self.words.is_empty());
        // let pages = (bits + PAGE_BITS - 1) / PAGE_BITS;
        let words = (bits + BLOCK_BITS - 1) / BLOCK_BITS * WORDS_PER_BLOCK;
        self.words.resize(words, 0);
    }

    fn get_word_mut(&mut self, bit: usize) -> &mut Word {
        let word = bit / WORD_BITS;
        if word >= self.words.len() {
            self.reserve(bit + 1);
        }
        let word = self.bit_to_word(bit);
        &mut self.words[word]
    }

    pub(crate) fn set(&mut self, bit: usize) {
        *self.get_word_mut(bit) |= 1 << (bit & (WORD_BITS - 1));
    }

    pub(crate) fn set_word(&mut self, i: usize, value: Word) {
        *self.get_word_mut(i) = value;
    }

    pub(crate) fn set_nibble(&mut self, nibble_idx: usize, nibble_value: u32) {
        let bit_idx = nibble_idx * 4;
        // clear all bits...
        // *self.get_word(bit_idx) &= !(15 << (bit_idx & (WORD_BITS - 1)));
        *self.get_word_mut(bit_idx) |= (nibble_value as Word) << (bit_idx & (WORD_BITS - 1));
    }

    pub(crate) fn build(&mut self) {
        self.aggregated.clear();
        self.aggregated.reserve(self.words.len() / WORDS_PER_BLOCK);
        let mut ones = 0;
        println!("BUILD {} {}", self.words.len(), self.aggregated.capacity());
        for block in 0..self.words.len() / WORDS_PER_BLOCK {
            for i in 0..WORDS_PER_BLOCK / 2 {
                ones += self.words[i + block * WORDS_PER_BLOCK].count_ones();
            }
            self.aggregated.push(ones);
            for i in WORDS_PER_BLOCK / 2..WORDS_PER_BLOCK {
                ones += self.words[i + block * WORDS_PER_BLOCK].count_ones();
            }
        }
    }

    pub(crate) fn get_word_suffix(&self, i: usize) -> Word {
        let word = self.bit_to_word(i);
        let bit = i % WORD_BITS;
        self.words[word] >> bit
    }

    pub(crate) fn get_bits(&self, i: usize) -> Word {
        let bytes = self.words.as_ptr() as *const u8;
        let bytes = unsafe { bytes.add(i / 8) };
        unsafe { std::ptr::read_unaligned(bytes as *const Word) >> (i & 7) }
    }

    pub(crate) fn get_word(&self, i: usize) -> Word {
        let word = self.bit_to_word(i);
        let bit = i % WORD_BITS;
        let first_part = self.words[word] >> bit;
        first_part | (self.words[word + (bit != 0) as usize] << ((WORD_BITS - bit) % WORD_BITS))
    }

    pub(crate) fn get_bit(&self, bit: usize) -> bool {
        let word = self.bit_to_word(bit);
        let bit_in_word = bit & (WORD_BITS - 1);
        self.words[word] & (1 << bit_in_word) != 0
    }

    pub(crate) fn get_nibble(&self, nibble_idx: usize) -> u32 {
        let bit_idx = nibble_idx * 4;
        let word = self.bit_to_word(bit_idx);
        let bit_in_word = bit_idx & (WORD_BITS - 1);
        ((self.words[word] >> bit_in_word) & 15) as u32
    }

    fn len(&self) -> usize {
        self.words.len() * WORD_BITS
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
