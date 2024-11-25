use std::arch::x86_64::_pdep_u64;

use crate::virtual_bitrank::VirtualBitRank;

pub struct ParallelTrie {
    root: Vec<u64>,
    root_ones: usize,
    max_level: usize,
    data: VirtualBitRank,
    level_idx: Vec<usize>,
}

impl ParallelTrie {
    fn fill_bit_rank<const WRITE: bool>(
        &mut self,
        prefix: u32,
        slices: &mut [&[u32]; 64],
        level: usize,
        mask: u64,
    ) {
        for t in 0..4 {
            let mut sub_mask = 0;
            let mut mask = mask;
            while mask != 0 {
                let i = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                if let Some(&value) = slices[i].get(0) {
                    if (value ^ prefix) >> (2 * level + 8) == 0
                        && (value >> (level * 2 + 6)) & 3 == t
                    {
                        if WRITE {
                            self.data.set(self.level_idx[level]);
                        }
                        if level > 0 {
                            sub_mask |= 1 << (value & 63);
                        } else {
                            slices[i] = &slices[i][1..];
                        }
                    }
                }
                self.level_idx[level] += 1;
            }
            if sub_mask != 0 {
                self.fill_bit_rank::<WRITE>(
                    prefix + (t << (level * 2 + 6)),
                    slices,
                    level - 1,
                    sub_mask,
                );
            }
        }
    }

    fn fill<const WRITE: bool>(&mut self, mut slices: [&[u32]; 64]) {
        for prefix in 0..self.root.len() {
            let mut mask = 0;
            for i in 0..64 {
                if let Some(&value) = slices[i].get(0) {
                    if value >> (self.max_level * 2 + 6) == prefix as u32 {
                        mask |= 1 << i;
                    }
                }
            }
            if WRITE {
                self.root[prefix] = mask;
                self.root_ones += mask.count_ones() as usize;
            }
            if mask != 0 {
                self.fill_bit_rank::<WRITE>(
                    (prefix as u32) << (self.max_level * 2 + 6),
                    &mut slices,
                    self.max_level - 1,
                    mask,
                );
            }
        }
    }

    pub fn build(max_doc: usize, mut v: Vec<u32>, max_level: usize) -> Self {
        v.sort_by_key(|&v| (v % 64, v / 64));
        let mut slices = [&v[..]; 64];
        let mut i = 0;
        for j in 0..64 {
            let s = i;
            while i < v.len() && v[i] % 64 == j {
                i += 1;
            }
            slices[j as usize] = &v[s..i];
        }
        let mut s = Self {
            max_level,
            data: VirtualBitRank::default(),
            root: vec![0u64; (max_doc >> (max_level * 2 + 6)) + 1],
            root_ones: 0,
            level_idx: vec![0; max_level],
        };
        s.fill::<false>(slices.clone());
        s.data.reserve(s.level_idx.iter().sum::<usize>() + 64);
        s.level_idx
            .iter_mut()
            .rev()
            .scan(0, |acc, x| {
                let old = *acc;
                *acc = *acc + *x;
                *x = old;
                Some(old)
            })
            .skip(usize::MAX)
            .next();
        s.fill::<true>(slices);
        s.data.build();
        let trie_size = (s.level_idx[0] as f32) / v.len() as f32;
        let root_size = (s.root.len() * 64) as f32 / v.len() as f32;
        println!(
            "encoded size: {trie_size} {root_size} total: {} density: {}",
            trie_size + root_size,
            s.root_ones as f32 / s.root.len() as f32 / 64.0
        );
        s
    }

    pub fn collect(&self) -> Vec<u32> {
        let mut v = Vec::new();
        let mut rank = 0;
        for (i, word) in self.root.iter().enumerate() {
            if *word != 0 {
                self.recurse(i, *word, rank * 4, self.max_level, &mut v);
            }
            rank += word.count_ones() as usize;
        }
        v
    }

    fn recurse(&self, pos: usize, mut word: u64, mut rank: usize, level: usize, v: &mut Vec<u32>) {
        if level == 0 {
            while word != 0 {
                let bit = word.trailing_zeros();
                v.push(((pos as u32) << 6) + bit);
                word &= word - 1;
            }
        } else {
            let required_bits = word.count_ones();
            let mut new_rank = self.data.rank(rank) as usize + self.root_ones;

            if required_bits == 1 {
                // TODO: simply switch to single bit recursion here instead of checking on every level again.
                // TODO: we can also read here a single nibble which is even faster.
                let mut w = self.data.get_word(rank) & 15;
                while w != 0 {
                    let zeros = w.trailing_zeros();
                    w &= w - 1;
                    self.recurse(pos * 4 + zeros as usize, word, new_rank * 4, level - 1, v);
                    new_rank += 1;
                }
            } else if required_bits <= 16 {
                let w = self.data.get_word(rank);
                let new_word = unsafe { _pdep_u64(w, word) };
                if new_word != 0 {
                    self.recurse(pos * 4, new_word, new_rank * 4, level - 1, v);
                    new_rank += new_word.count_ones() as usize;
                }

                let w = w >> required_bits;
                let new_word = unsafe { _pdep_u64(w, word) };
                if new_word != 0 {
                    self.recurse(pos * 4 + 1, new_word, new_rank * 4, level - 1, v);
                    new_rank += new_word.count_ones() as usize;
                }

                let w = w >> required_bits;
                let new_word = unsafe { _pdep_u64(w, word) };
                if new_word != 0 {
                    self.recurse(pos * 4 + 2, new_word, new_rank * 4, level - 1, v);
                    new_rank += new_word.count_ones() as usize;
                }

                let w = w >> required_bits;
                let new_word = unsafe { _pdep_u64(w, word) };
                if new_word != 0 {
                    self.recurse(pos * 4 + 3, new_word, new_rank * 4, level - 1, v);
                }
            } else {
                let w = self.data.get_word(rank);
                let new_word = unsafe { _pdep_u64(w, word) };
                if new_word != 0 {
                    self.recurse(pos * 4, new_word, new_rank * 4, level - 1, v);
                    new_rank += new_word.count_ones() as usize;
                }

                rank += required_bits as usize;
                let w = self.data.get_word(rank);
                let new_word = unsafe { _pdep_u64(w, word) };
                if new_word != 0 {
                    self.recurse(pos * 4 + 1, new_word, new_rank * 4, level - 1, v);
                    new_rank += new_word.count_ones() as usize;
                }

                rank += required_bits as usize;
                let w = self.data.get_word(rank);
                let new_word = unsafe { _pdep_u64(w, word) };
                if new_word != 0 {
                    self.recurse(pos * 4 + 2, new_word, new_rank * 4, level - 1, v);
                    new_rank += new_word.count_ones() as usize;
                }

                rank += required_bits as usize;
                let w = self.data.get_word(rank);
                let new_word = unsafe { _pdep_u64(w, word) };
                if new_word != 0 {
                    self.recurse(pos * 4 + 3, new_word, new_rank * 4, level - 1, v);
                }
            }
        }
    }
}
/*
pub struct TrieTraversal<'a> {
    trie: &'a ParallelTrie,
    // The nibble position of the node for each level.
    pos: [u32; 16],
    // The remaining bits (nibbles) of the word covering the nibble position.
    word: [u64; 16],
    // The 1-rank up to the nibble. This information is needed
    // to determine the nibble/node position of the next level.
    rank: [u32; 16],
}

impl TrieTraversal<'_> {
    fn get(&self, level: usize) -> u64 {
        self.word[level]
    }

    fn down(&mut self, level: usize) {
        let new_pos =
            self.rank[level] + (self.word[level] & !(u64::MAX << (child + 1))).count_ones();
        let old_pos = self.pos[level - 1];
        if (new_pos ^ old_pos) & !(64 / 4 - 1) == 0 {
            // In this case, we can reuse the old rank information
            let delta = (new_pos - old_pos) * 4;
            self.rank[level - 1] += (self.word[level - 1] & !(Word::MAX << delta)).count_ones();
            self.word[level - 1] = self.word[level - 1] >> delta;
        } else {
            if level > 1 {
                // for level 0, we don't need the rank information
                // self.rank[level - 1] = self.trie.data.rank(4 * new_pos as usize);
                let (r, w) = self.trie.data.rank_with_word(4 * new_pos as usize);
                self.rank[level - 1] = r;
                self.word[level - 1] = w;
            } else {
                // TODO: Get word suffix and rank information in one go...
                self.word[level - 1] = self.trie.data.get_word_suffix(4 * new_pos as usize);
            }
        }
        self.pos[level - 1] = new_pos;
    }
}

struct ParallelTrieIterator {
    stack: [u64; 16],
    word: u64,
    rank: usize,
    level: usize,
    pos: usize,
}

impl Iterator for ParallelTrieIterator {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        loop {}
    }
}
*/
#[cfg(test)]
mod tests {
    use std::time::Instant;

    use rand::{thread_rng, Rng};

    use crate::parallel4::ParallelTrie;

    #[test]
    fn test_parallel4_large() {
        let mut values: Vec<_> = (0..100_000)
            .map(|_| thread_rng().gen_range(0..100_000_000))
            .collect();
        values.sort();
        values.dedup();

        for levels in 1..12 {
            let start = Instant::now();
            let trie = ParallelTrie::build(100_000_000, values.clone(), levels);
            println!(
                "construction {levels} {:?}",
                start.elapsed() / values.len() as u32,
            );

            let start = Instant::now();
            let result = trie.collect();
            println!(
                "collect {levels} {:?}",
                start.elapsed() / values.len() as u32,
            );
            assert_eq!(result, values);
        }
    }

    #[test]
    fn test_parallel4_small() {
        let values = vec![3, 6, 7, 10, 90, 91, 120, 128, 129, 130, 231, 321, 999];
        // let values = vec![3, 6, 7, 321, 999];

        let start = Instant::now();
        let trie = ParallelTrie::build(1024, values.clone(), 1);
        println!("construction {:?}", start.elapsed() / values.len() as u32,);

        let start = Instant::now();
        let result = trie.collect();
        println!("collect {:?}", start.elapsed() / values.len() as u32,);
        assert_eq!(result, values);
    }
}
