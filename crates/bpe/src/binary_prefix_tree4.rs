use std::{fmt::Debug, time::Instant};

use itertools::Itertools;
use rand::{thread_rng, Rng};

struct BitField {
    data: Vec<u64>,
    bits: usize,
    rank: Vec<u32>,
}

impl BitField {
    fn new() -> Self {
        Self {
            data: vec![],
            bits: 0,
            rank: vec![],
        }
    }

    fn push(&mut self, value: bool) {
        if self.data.len() * 64 <= self.bits {
            self.rank.push(
                self.rank.last().copied().unwrap_or_default()
                    + self.data.last().copied().unwrap_or_default().count_ones(),
            );
            self.data.push(0);
        }
        if value {
            self.data[self.bits / 64] |= 1 << (self.bits & 63);
        }
        self.bits += 1;
    }

    fn get(&self, index: usize) -> bool {
        self.data[index / 64] & (1 << (index & 63)) != 0
    }

    fn get4(&self, index: usize) -> u32 {
        (self.data[index / 16] >> ((4 * index) & 63)) as u32 // & 15
    }

    fn rank(&self, index: usize) -> usize {
        let r = self.rank[index / 64];
        (r + (self.data[index / 64] & !(u64::MAX << (index & 63))).count_ones()) as usize
    }
}

impl Debug for BitField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitField")
            .field("data", &self.data)
            .field("bits", &self.bits)
            .finish()
    }
}

struct BinaryPrefixTree {
    data: BitField,
}

const MAX_LEVEL: usize = 30;

impl BinaryPrefixTree {
    fn new(values: &[u32]) -> Self {
        let mut data = BitField::new();
        for level in (0..MAX_LEVEL).step_by(2).rev() {
            let mut previous_prefix = None;
            let mut occurrences = 0;
            for value in values {
                let prefix = *value >> level;
                if let Some(prev) = previous_prefix {
                    if prefix >> 2 != prev {
                        data.push(occurrences & 1 != 0);
                        data.push(occurrences & 2 != 0);
                        data.push(occurrences & 4 != 0);
                        data.push(occurrences & 8 != 0);
                        occurrences = 0;
                    }
                }
                occurrences |= 1 << (prefix & 3);
                previous_prefix = Some(prefix >> 2);
            }
            data.push(occurrences & 1 != 0);
            data.push(occurrences & 2 != 0);
            data.push(occurrences & 4 != 0);
            data.push(occurrences & 8 != 0);
        }
        println!("encoded size: {}", data.bits as f32 / values.len() as f32);
        Self { data }
    }

    fn as_iter(&self) -> BinaryPrefixTreeIterator<'_> {
        BinaryPrefixTreeIterator::new(self)
    }

    fn recurse(&self, node: usize, level: usize, value: u32, results: &mut Vec<u32>) {
        if level == 2 {
            self.recurse2(node, value, results);
        } else {
            let n = self.data.get4(node);
            let value = value * 4;
            let mut r = self.data.rank(node * 4);
            if n & 1 != 0 {
                r += 1;
                self.recurse(r, level - 2, value, results);
            }
            if n & 2 != 0 {
                r += 1;
                self.recurse(r, level - 2, value + 1, results);
            }
            if n & 4 != 0 {
                r += 1;
                self.recurse(r, level - 2, value + 2, results);
            }
            if n & 8 != 0 {
                r += 1;
                self.recurse(r, level - 2, value + 3, results);
            }
        }
    }

    #[inline(always)]
    fn recurse2(&self, node: usize, value: u32, results: &mut Vec<u32>) {
        let n = self.data.get4(node);
        let value = value * 4;
        let mut r = self.data.rank(node * 4);
        if n & 1 != 0 {
            r += 1;
            self.recurse0(r, value, results);
        }
        if n & 2 != 0 {
            r += 1;
            self.recurse0(r, value + 1, results);
        }
        if n & 4 != 0 {
            r += 1;
            self.recurse0(r, value + 2, results);
        }
        if n & 8 != 0 {
            r += 1;
            self.recurse0(r, value + 3, results);
        }
    }

    #[inline(always)]
    fn recurse0(&self, node: usize, value: u32, results: &mut Vec<u32>) {
        let n = self.data.get4(node);
        let value = value * 4;
        if n & 1 != 0 {
            results.push(value);
        }
        if n & 2 != 0 {
            results.push(value + 1);
        }
        if n & 4 != 0 {
            results.push(value + 2);
        }
        if n & 8 != 0 {
            results.push(value + 3);
        }
    }
}

trait TreeIterator {
    fn get(&self, level: usize) -> u32;
    fn down(&mut self, level: usize, value: bool);
}

struct BinaryPrefixTreeIterator<'a> {
    bpt: &'a BinaryPrefixTree,
    pos: [u32; MAX_LEVEL],
}

impl<'a> BinaryPrefixTreeIterator<'a> {
    fn new(bpt: &'a BinaryPrefixTree) -> Self {
        Self {
            bpt,
            pos: [0; MAX_LEVEL],
        }
    }

    fn into_iter(self) -> RustIterator<'a> {
        RustIterator::new(self)
    }

    fn into_vec(mut self) -> Vec<u32> {
        let mut res = vec![];
        // init positions
        for level in (1..MAX_LEVEL).rev() {
            let index = self.pos[level];
            let new_index = self.bpt.data.rank(index as usize) as u32 + 1;
            self.pos[level - 1] = new_index * 2;
        }
        let mut level = MAX_LEVEL - 1;
        let mut value = 0u32;
        while level < MAX_LEVEL {
            let v = self.bpt.data.get(self.pos[level] as usize);
            // println!("{value} {level} {} {v}", self.pos[level]);
            self.pos[level] += 1;
            if v {
                if level == 0 {
                    res.push(value);
                    value += 1 << level;
                    level = value.trailing_zeros() as usize;
                } else {
                    level -= 1;
                }
            } else {
                value += 1 << level;
                level = value.trailing_zeros() as usize;
            }
        }
        res
    }
}

impl<'a> TreeIterator for BinaryPrefixTreeIterator<'a> {
    fn get(&self, level: usize) -> u32 {
        self.bpt.data.get4(self.pos[level] as usize)
    }

    fn down(&mut self, level: usize, value: bool) {
        let index = self.pos[level] * 2 + value as u32;
        let new_index = self.bpt.data.rank(index as usize + 1);
        self.pos[level - 1] = new_index as u32;
    }
}

struct RustIterator<'a> {
    inner: BinaryPrefixTreeIterator<'a>,
    level: usize,
    item: u32,
}

impl<'a> RustIterator<'a> {
    fn new(inner: BinaryPrefixTreeIterator<'a>) -> Self {
        Self {
            inner,
            level: MAX_LEVEL - 1,
            item: 0,
        }
    }
}

impl<'a> Iterator for RustIterator<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        while self.level < MAX_LEVEL {
            let bitmask = self.inner.get(self.level);
            let curr = self.item & (1 << self.level);
            if curr == 0 && bitmask & 1 != 0 {
                if self.level == 0 {
                    let res = self.item;
                    self.item += 1;
                    return Some(res);
                }
                self.inner.down(self.level, false);
                self.level -= 1;
            } else if bitmask & 2 != 0 {
                if curr == 0 {
                    self.item += 1 << self.level;
                }
                if self.level == 0 {
                    let res = self.item;
                    self.item += 1;
                    self.level = self.item.trailing_zeros() as usize + 1;
                    return Some(res);
                }
                self.inner.down(self.level, true);
                self.level -= 1;
            } else {
                self.item += 1 << self.level;
                self.level = self.item.trailing_zeros() as usize;
            }
        }
        None
    }
}

#[test]
fn test_bpt() {
    let values = &[3, 6, 7, 10];
    let bpt = BinaryPrefixTree::new(values);
    println!("{:x?} {:?}", bpt.data.data, bpt.data.rank);
    /*assert!(!bpt.check(0));
    assert!(!bpt.check(1));
    assert!(!bpt.check(2));
    assert!(bpt.check(3));
    assert!(!bpt.check(4));
    assert!(!bpt.check(5));
    assert!(bpt.check(6));
    assert!(bpt.check(7));
    assert!(!bpt.check(8));
    assert!(!bpt.check(9));
    assert!(bpt.check(10));*/

    let iter = bpt.as_iter().into_iter();
    assert_eq!(iter.collect_vec(), values.iter().copied().collect_vec());

    assert_eq!(
        bpt.as_iter().into_vec(),
        values.iter().copied().collect_vec()
    );
}

#[test]
fn test_4intersection() {
    let start = Instant::now();
    let mut values = (0..1000000)
        .map(|_| thread_rng().gen_range(0..100000000))
        .collect_vec();
    values.sort();
    values.dedup();
    println!("generation of values {:?}", start.elapsed());

    let start = Instant::now();
    let bpt = BinaryPrefixTree::new(&values);
    println!("construction of tree {:?}", start.elapsed());

    //let start = Instant::now();
    //assert_eq!(bpt.as_iter().into_iter().collect_vec(), values);
    //println!("iteration {:?}", start.elapsed());

    let start = Instant::now();
    let mut v = Vec::with_capacity(values.len());
    bpt.recurse(0, MAX_LEVEL - 2, 0, &mut v);
    assert_eq!(v, values);
    println!("recursive collect {:?}", start.elapsed());

    //let start = Instant::now();
    //assert_eq!(bpt.as_iter().into_vec(), values);
    //println!("iteration {:?}", start.elapsed());

    let start = Instant::now();
    let mut v = Vec::with_capacity(values.len());
    for i in values.iter() {
        v.push(*i);
    }
    assert_eq!(v, values);
    println!("iteration {:?}", start.elapsed());
}
