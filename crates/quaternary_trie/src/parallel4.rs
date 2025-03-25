use std::arch::x86_64::_pdep_u64;

use crate::virtual_bitrank::VirtualBitRank;

pub struct ParallelTrie {
    root_len: usize,
    root_zeros: usize,
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
        for prefix in 0..self.root_len {
            let mut mask = 0u64;
            for i in 0..64 {
                if let Some(&value) = slices[i].get(0) {
                    if value >> (self.max_level * 2 + 6) == prefix as u32 {
                        mask |= 1 << i;
                    }
                }
            }
            if WRITE {
                self.root_zeros += 64 - mask.count_ones() as usize;
                self.data.set_word(self.level_idx[self.max_level], mask);
            }
            self.level_idx[self.max_level] += 64;
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
            root_len: (max_doc >> (max_level * 2 + 6)) + 1,
            root_zeros: 0,
            level_idx: vec![0; max_level + 1],
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
        let root_size = (s.root_len * 64) as f32 / v.len() as f32;
        println!(
            "encoded size: {root_size} total: {trie_size} density: {}",
            s.root_zeros as f32 / s.root_len as f32 / 64.0
        );
        s
    }

    pub fn collect(&self) -> Vec<u32> {
        let mut v = Vec::new();
        let mut rank = 0;
        for i in 0..self.root_len {
            let word = self.data.get_word(i * 64);
            if word != 0 {
                self.recurse(i, word, rank * 4, self.max_level, &mut v);
            }
            rank += word.count_ones() as usize;
        }
        v
    }

    fn recurse(&self, pos: usize, mut word: u64, mut rank: usize, level: usize, v: &mut Vec<u32>) {
        rank += self.root_len * 64;
        if level == 0 {
            while word != 0 {
                let bit = word.trailing_zeros();
                v.push(((pos as u32) << 6) + bit);
                word &= word - 1;
            }
        } else {
            let required_bits = word.count_ones();
            let mut new_rank = if level > 1 {
                self.data.rank(rank) as usize
            } else {
                0
            };

            if required_bits == 1 {
                // TODO: simply switch to single bit recursion here instead of checking on every level again.
                // TODO: we can also read here a single nibble which is even faster.
                let mut w = self.data.get_bits(rank) & 15;
                while w != 0 {
                    let zeros = w.trailing_zeros();
                    w &= w - 1;
                    self.recurse(pos * 4 + zeros as usize, word, new_rank * 4, level - 1, v);
                    new_rank += 1;
                }
            } else if required_bits < 15 {
                let w = self.data.get_bits(rank);
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

pub struct TrieTraversal<'a> {
    trie: &'a ParallelTrie,
    /// One bits are the active positions at the given level.
    mask: [u64; 16],
    /// 1-rank up to the position.
    rank: [u32; 16],
}

impl TrieTraversal<'_> {
    fn get(&self, level: usize) -> u64 {
        self.mask[level]
    }

    fn first_down(&mut self, level: usize) -> u64 {
        let rank = self.rank[level];
        let mask = self.mask[level];
        let required_bits = mask.count_ones();
        let w = self.trie.data.get_word(rank as usize);
        let new_mask = unsafe { _pdep_u64(w, mask) };
        let new_rank = self.trie.data.rank(rank as usize);
        self.rank[level] += required_bits;
        self.mask[level - 1] = new_mask;
        self.rank[level - 1] = new_rank * 4 + self.trie.root_len as u32 * 64;
        new_mask
    }

    fn next_down(&mut self, level: usize) -> u64 {
        let rank = self.rank[level];
        let mask = self.mask[level];
        let required_bits = mask.count_ones();
        let w = self.trie.data.get_word(rank as usize);
        let new_mask = unsafe { _pdep_u64(w, mask) };
        self.rank[level - 1] += self.mask[level - 1].count_ones() * 4;
        self.rank[level] += required_bits;
        self.mask[level - 1] = new_mask;
        new_mask
    }
}

struct ParallelTrieIterator<'a> {
    state: TrieTraversal<'a>,
    level: usize,
    pos: u32,
}

impl<'a> ParallelTrieIterator<'a> {
    fn new(trie: &'a ParallelTrie) -> Self {
        let mut s = Self {
            state: TrieTraversal {
                trie,
                mask: [0; 16],
                rank: [0; 16],
            },
            level: trie.max_level,
            pos: 0,
        };
        s.state.mask[trie.max_level] = u64::MAX;
        s
    }
}

impl Iterator for ParallelTrieIterator<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        let mut pos = self.pos;
        loop {
            let mut level = if pos == 0 {
                self.state.trie.max_level
            } else {
                (pos.trailing_zeros().saturating_sub(6) / 2) as usize
            };
            println!("level: {level} pos: {pos}");
            if level == 0 {
                let child = pos & 63;
                let mask = self.state.get(level) >> child;
                if mask == 0 {
                    pos = (pos | 63) + 1;
                    continue;
                }
                let delta = mask.trailing_zeros();
                let res = pos + delta;
                self.pos = pos + 1;
                return Some(res);
            } else if self.state.first_down(level) != 0 {
                continue;
            } else if level > self.state.trie.max_level {
                return None;
            }
            if self.state.next_down(level) != 0 {
                loop {
                    level -= 1;
                    if level == 0 {
                        let mask = self.state.get(level);
                        let delta = mask.trailing_zeros();
                        let res = pos + delta;
                        self.pos = pos + 1;
                        return Some(res);
                    }
                    if self.state.first_down(level) == 0 {
                        break;
                    }
                }
            }
            pos += 1 << (level * 2 + 6);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use rand::{thread_rng, Rng};

    use crate::parallel4::ParallelTrie;

    use super::ParallelTrieIterator;

    #[test]
    fn test_parallel4_iterator() {
        let values = vec![3, 6, 7, 10, 90, 91, 120, 128, 129, 130, 231, 321, 999];
        let trie = ParallelTrie::build(1024, values.clone(), 1);
        let mut iter = ParallelTrieIterator::new(&trie);
        let result: Vec<_> = iter.collect();
        assert_eq!(result, values);
    }

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
