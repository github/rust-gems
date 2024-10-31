use virtual_bitrank::VirtualBitRank;

mod virtual_bitrank;

const MAX_LEVEL: usize = 14;

pub struct QuarternaryTrie {
    data: VirtualBitRank,
    /// Total number of nibbles on each level.
    level_idx: [usize; MAX_LEVEL],
}

/// Level: 0 ==> ........xx
/// Level: 1 ==> ......xx..
/// Level: 2 ==> ....xx....
///              ^^^^
///             prefix
///                  ^^
///              nibble bit
/// Van Emde Boas layout/traversal
///           1
///         /    \
///       2        3
///     /  \      /  \
///    4    7    a    d
///   / \  / \  / \  / \
///   5 6  8 9  b c  e f
///
/// Process: 123, 489, 5ab, 6cd, 7ef
/// 0xxx 00xx 01xx | STOP & Recurse
/// 000x 0000 0001
/// 001x 0010 0011
/// 010x 0100 0101
/// 011x 0110 0111
///
/// now with two bit levels
///
/// 00xxxxxx 01xxxxxx 10xxxxxx 11xxxxxx
/// 0000xxxx 0001xxxx 0010xxxx 0011xxxx
/// 0100xxxx 0101xxxx 0110xxxx 0111xxxx
/// 1000xxxx 1001xxxx 1010xxxx 1011xxxx
/// 1100xxxx 1101xxxx 1110xxxx 1111xxxx
///
/// Stop and recurse
///
/// 000000xx 000001xx 000010xx 000011xx
/// 00000000 00000001 00000010 00000011
/// 00000100 00000101 00000110 00000111
/// 00001000 00001001 00001010 00001011
/// 00001100 00001101 00001110 00001111
///
/// 000100xx 000101xx 000110xx 000111xx
/// 00010000 00010001 00010010 00010011
/// 00010100 00010101 00010110 00010111
/// 00011000 00011001 00011010 00011011
/// 00011100 00011101 00011110 00011111
///
/// 001000xx 001001xx 001010xx 001011xx
/// 00100000 00100001 00100010 00100011
/// 00100100 00100101 00100110 00100111
/// 00101000 00101001 00101010 00101011
/// 00101100 00101101 00101110 00101111
///
/// 001100xx 001101xx 001110xx 001111xx
/// 00110000 00110001 00110010 00110011
/// 00110100 00110101 00110110 00110111
/// 00111000 00111001 00111010 00111011
/// 00111100 00111101 00111110 00111111
///
/// 010000xx 010001xx 010010xx 010011xx
/// ...
///
/// Process first half:
/// xxxxxxxx--------
/// Reset and process second half
/// 00000000xxxxxxxx

fn get_prefix(value: u32, level: usize) -> u32 {
    if level == 16 {
        0
    } else {
        value >> (level * 2)
    }
}

fn gallopping_search<F: Fn(u32) -> bool>(values: &mut &[u32], f: F) {
    let mut step = 1;
    while step < values.len() {
        if f(values[step]) {
            break;
        }
        *values = &mut &values[step..];
        step *= 2;
    }
    step /= 2;
    while step > 0 {
        if step < values.len() && !f(values[step]) {
            *values = &mut &values[step..];
        }
        step /= 2;
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Layout {
    Linear,
    VanEmdeBoas,
    DepthFirst,
}

impl QuarternaryTrie {
    fn count_levels(&mut self, values: &mut &[u32], level: usize) -> bool {
        let prefix = values[0] >> (level * 2 + 2);
        let mut nibble = 0;
        let mut all_set = true;
        while !values.is_empty() && values[0] >> (level * 2 + 2) == prefix {
            nibble |= 1 << ((values[0] >> level * 2) & 3);
            if level > 0 {
                all_set &= self.count_levels(values, level - 1);
            } else {
                *values = &values[1..];
            }
        }
        all_set &= nibble == 15;
        self.level_idx[level] += 1;
        all_set
    }

    fn van_emde_boas(&mut self, values: &mut &[u32], level: usize, res: usize) {
        let prefix = get_prefix(values[0], level);
        if res == 0 {
            let mut nibble = 0;
            while !values.is_empty() && get_prefix(values[0], level) == prefix {
                let v = values[0] >> (level * 2 - 2);
                nibble |= 1 << (v & 3);
                *values = &mut &values[1..];
                gallopping_search(values, |x| x >> (level * 2 - 2) > v);
            }
            if level <= MAX_LEVEL {
                self.data.set_nibble(self.level_idx[level - 1], nibble);
                self.level_idx[level - 1] += 1;
            }
            return;
        }
        // process level .. level - res
        // This level has to be processed at half resolution
        let mut copy = &values[..];
        self.van_emde_boas(&mut copy, level, res / 2);
        // Then process level - res..level - 2 *res
        // Process all the children within this subtree.
        while !values.is_empty() && get_prefix(values[0], level) == prefix {
            self.van_emde_boas(values, level - res, res / 2);
        }
    }

    fn fill_bit_rank(&mut self, values: &mut &[u32], level: usize) -> bool {
        let prefix = values[0] >> (level * 2 + 2);
        let mut nibble = 0;
        let mut all_set = true;
        while !values.is_empty() && values[0] >> (level * 2 + 2) == prefix {
            nibble |= 1 << ((values[0] >> level * 2) & 3);
            if level > 0 {
                all_set &= self.fill_bit_rank(values, level - 1);
            } else {
                *values = &values[1..];
            }
        }
        all_set &= nibble == 15;
        self.data.set_nibble(self.level_idx[level], nibble);
        self.level_idx[level] += 1;
        all_set
    }

    pub fn new(values: &[u32], layout: Layout) -> Self {
        let mut s = Self {
            data: VirtualBitRank::new(),
            level_idx: [0; MAX_LEVEL],
        };
        let mut consumed = values;
        s.count_levels(&mut consumed, MAX_LEVEL - 1);
        if matches!(layout, Layout::Linear) {
            s.data.reserve(s.level_idx.iter().sum::<usize>() * 4);
        }
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
        let mut consumed = values;
        if matches!(layout, Layout::VanEmdeBoas | Layout::Linear) {
            s.van_emde_boas(&mut consumed, 16, 8);
        } else {
            s.fill_bit_rank(&mut consumed, MAX_LEVEL - 1);
        }
        s.data.build();
        s.reset_stats();
        println!(
            "encoded size: {}",
            4.0 * s.level_idx[0] as f32 / values.len() as f32
        );
        s
    }

    fn reset_stats(&mut self) {
        self.data.reset_stats();
    }

    fn page_count(&self) -> (usize, usize) {
        self.data.page_count()
    }

    fn recurse(&self, node: usize, level: usize, value: u32, results: &mut Vec<u32>) {
        if level == 1 {
            self.recurse2(node, value, results);
        } else {
            let mut n = self.data.get_nibble(node);
            let mut value = value * 4;
            if n == 0 {
                results.extend((0..4 << (level * 2)).map(|i| (value << (level * 2)) + i));
                return;
            }
            let mut r = self.data.rank(node * 4) as usize;
            while n > 0 {
                let delta = n.trailing_zeros();
                r += 1;
                self.recurse(r, level - 1, value + delta, results);
                value += delta + 1;
                n >>= delta + 1;
            }
        }
    }

    fn recurse2(&self, node: usize, value: u32, results: &mut Vec<u32>) {
        let mut n = self.data.get_nibble(node);
        let mut value = value * 4;
        if n == 0 {
            results.extend((0..16).map(|i| (value << 2) + i));
            return;
        }
        let mut r = self.data.rank(node * 4) as usize;
        while n > 0 {
            let delta = n.trailing_zeros();
            r += 1;
            self.recurse0(r, value + delta, results);
            value += delta + 1;
            n >>= delta + 1;
        }
    }

    fn recurse0(&self, node: usize, value: u32, results: &mut Vec<u32>) {
        let mut n = self.data.get_nibble(node);
        let mut value = value * 4;
        if n == 0 {
            results.extend((0..4).map(|i| value + i));
            return;
        }
        while n > 0 {
            let delta = n.trailing_zeros();
            results.push(value + delta);
            value += delta + 1;
            n >>= delta + 1;
        }
    }

    // This is the "slow" implementation which computes at every level the rank and extract the corresponding nibble.
    pub fn collect2(&self) -> Vec<u32> {
        let mut results = Vec::with_capacity(self.level_idx[0]);
        self.recurse(0, MAX_LEVEL - 1, 0, &mut results);
        results
    }

    // This is the "fastest" implementation, since it doesn't use rank information at all during the traversal.
    // This is possible, since it iterates through ALL nodes and thus we can simply increment the positions by 1.
    // We only need the rank information to initialize the positions.
    // The only remaining "expensive" part here is the lookup of the nibble with every iteration.
    // This lookup requires the slightly complicated conversion from position into block pointer (either via the virtual mapping or via some math).
    // The math would be trivial if we wouldn't store the counters within the same page...
    // Instead one could try to cache a u64 value and keep shifting until the end is reached. Or working with the pointer into the bitrank array.
    pub fn collect(&mut self) -> Vec<u32> {
        self.level_idx[MAX_LEVEL - 1] = 0;
        for level in (1..MAX_LEVEL).into_iter().rev() {
            self.level_idx[level - 1] = self.data.rank(self.level_idx[level] * 4) as usize + 1;
        }
        let mut results = Vec::new();
        self.fast_collect_inner(MAX_LEVEL - 1, 0, &mut results);
        results
    }

    fn fast_collect_inner(&mut self, level: usize, value: u32, results: &mut Vec<u32>) {
        let mut nibble = self.data.get_nibble(self.level_idx[level]);
        self.level_idx[level] += 1;
        if nibble == 0 {
            results.extend((0..4 << (level * 2)).map(|i| (value << (level * 2)) + i));
            return;
        }
        let mut value = value * 4;
        if level == 0 {
            while nibble > 0 {
                let delta = nibble.trailing_zeros();
                results.push(value + delta);
                value += delta + 1;
                nibble >>= delta + 1;
            }
        } else {
            while nibble > 0 {
                let delta = nibble.trailing_zeros();
                self.fast_collect_inner(level - 1, value + delta, results);
                value += delta + 1;
                nibble >>= delta + 1;
            }
        }
    }
}

pub trait TrieIteratorTrait {
    fn get(&self, level: usize) -> u32;
    fn down(&mut self, level: usize, child: u32);
}

pub struct TrieTraversal<'a> {
    trie: &'a QuarternaryTrie,
    pos: [u32; MAX_LEVEL],
}

impl<'a> TrieTraversal<'a> {
    pub fn new(bpt: &'a QuarternaryTrie) -> Self {
        Self {
            trie: bpt,
            pos: [0; MAX_LEVEL],
        }
    }
}

impl TrieIteratorTrait for TrieTraversal<'_> {
    fn get(&self, level: usize) -> u32 {
        self.trie.data.get_nibble(self.pos[level] as usize)
    }

    fn down(&mut self, level: usize, child: u32) {
        let index = self.pos[level] * 4 + child;
        let new_index = self.trie.data.rank(index as usize + 1);
        self.pos[level - 1] = new_index;
    }
}

pub struct TrieIterator<T> {
    trie: T,
    item: u32,
    nibbles: [u32; MAX_LEVEL],
}

impl<T: TrieIteratorTrait> TrieIterator<T> {
    pub fn new(trie: T) -> Self {
        Self {
            trie,
            item: 0,
            nibbles: [0; MAX_LEVEL],
        }
    }
}

impl<'a, T: TrieIteratorTrait> Iterator for TrieIterator<T> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        let mut level = if self.item == 0 {
            self.nibbles[MAX_LEVEL - 1] = self.trie.get(MAX_LEVEL - 1);
            MAX_LEVEL - 1
        } else {
            (self.item.trailing_zeros() / 2) as usize
        };
        while level < MAX_LEVEL {
            let child = (self.item >> (2 * level)) & 3;
            let nibble = self.nibbles[level] >> child;
            if nibble != 0 {
                let delta = nibble.trailing_zeros();
                if level == 0 {
                    let res = self.item + delta;
                    self.item = res + 1;
                    return Some(res);
                }
                self.item += delta << (2 * level);
                self.trie.down(level, child + delta);
                level -= 1;
                self.nibbles[level] = self.trie.get(level);
            } else {
                self.item |= 3 << (level * 2);
                self.item += 1 << (level * 2);
                level = (self.item.trailing_zeros() / 2) as usize;
            }
        }
        None
    }
}

pub struct Intersection<T> {
    left: T,
    right: T,
}

impl<T: TrieIteratorTrait> Intersection<T> {
    pub fn new(left: T, right: T) -> Self {
        Self { left, right }
    }
}

impl<T: TrieIteratorTrait> TrieIteratorTrait for Intersection<T> {
    fn get(&self, level: usize) -> u32 {
        self.left.get(level) & self.right.get(level)
    }

    fn down(&mut self, level: usize, child: u32) {
        self.left.down(level, child);
        self.right.down(level, child);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use itertools::{kmerge, Itertools};
    use rand::{thread_rng, Rng};

    use crate::{Intersection, Layout, QuarternaryTrie, TrieIterator, TrieTraversal};

    #[test]
    fn test_trie() {
        let values = vec![3, 6, 7, 10];
        let mut trie = QuarternaryTrie::new(&values, Layout::VanEmdeBoas);
        assert_eq!(trie.collect(), values);

        let values: Vec<_> = (1..63).collect();
        let mut trie = QuarternaryTrie::new(&values, Layout::VanEmdeBoas);
        assert_eq!(trie.collect(), values);
    }

    #[test]
    fn test_large() {
        let mut values: Vec<_> = (0..10000000)
            .map(|_| thread_rng().gen_range(0..100000000))
            .collect();
        values.sort();
        values.dedup();

        let start = Instant::now();
        let mut trie = QuarternaryTrie::new(&values, Layout::VanEmdeBoas);
        println!("construction {:?}", start.elapsed() / values.len() as u32);

        let start = Instant::now();
        let result = trie.collect();
        println!("reconstruction {:?}", start.elapsed() / values.len() as u32);
        assert_eq!(result, values);

        let iter = TrieIterator::new(TrieTraversal::new(&trie));
        let start = Instant::now();
        let result: Vec<_> = iter.collect();
        println!("iteration {:?}", start.elapsed() / values.len() as u32);
        assert_eq!(result, values);
    }

    #[test]
    fn test_van_emde_boas_layout() {
        let values: Vec<_> = (0..64).collect();
        let mut trie = QuarternaryTrie::new(&values, Layout::VanEmdeBoas);
        assert_eq!(trie.collect(), values);
    }

    #[test]
    fn test_intersection() {
        let mut page_counts = [0, 0, 0];
        for _ in 0..3 {
            let mut values: Vec<_> = (0..10000000)
                .map(|_| thread_rng().gen_range(0..100000000))
                .collect();
            values.sort();
            values.dedup();

            let mut values2: Vec<_> = (0..10000000)
                .map(|_| thread_rng().gen_range(0..100000000))
                .collect();
            values2.sort();
            values2.dedup();

            let start = Instant::now();
            let intersection: Vec<_> = kmerge([values.iter(), values2.iter()])
                .tuple_windows()
                .filter_map(|(a, b)| if *a == *b { Some(*a) } else { None })
                .collect();
            println!(
                "kmerge intersection {:?}",
                start.elapsed() / values.len() as u32
            );
            println!("Intersection size: {}", intersection.len());

            for (i, layout) in [Layout::VanEmdeBoas, Layout::DepthFirst, Layout::Linear]
                .into_iter()
                .enumerate()
            {
                let trie = QuarternaryTrie::new(&values, layout);
                let trie2 = QuarternaryTrie::new(&values2, layout);
                let iter = TrieIterator::new(Intersection::new(
                    TrieTraversal::new(&trie),
                    TrieTraversal::new(&trie2),
                ));
                let start = Instant::now();
                let result: Vec<_> = iter.collect();
                let count = trie.page_count();
                let count2 = trie2.page_count();
                page_counts[i] += count.0 + count2.0;
                println!(
                    "trie intersection {:?} {}",
                    start.elapsed() / values.len() as u32,
                    (count.0 + count2.0) as f32 / (count.1 + count2.1) as f32
                );
                assert_eq!(result, intersection);
            }
            println!("{page_counts:?}");
        }
    }
}
