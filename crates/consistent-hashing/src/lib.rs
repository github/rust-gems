use std::hash::{Hash, Hasher};

/// A trait which behaves like a pseudo-random number generator.
/// It is used to generate consistent hashes within one bucket.
/// Note: the hasher must have been seeded with the key during construction.
pub trait HashSequence {
    fn next(&mut self) -> u64;
}

/// A trait for building a special bit mask and sequences of hashes for different bit positions.
/// Note: the hasher must have been seeded with the key during construction.
pub trait HashSeqBuilder {
    type Seq: HashSequence;

    fn bit_mask(&self) -> u64;
    /// Return a HashSequence instance which is seeded with the given bit position
    /// and the seed of this builder.
    fn hash_seq(&self, bit: u64) -> Self::Seq;
}

/// A trait for building multiple independent hash builders
/// Note: the hasher must have been seeded with the key during construction.
pub trait ManySeqBuilder {
    type Builder: HashSeqBuilder;

    /// Returns the i-th independent hash builder.
    fn seq_builder(&self, i: usize) -> Self::Builder;
}

impl<H: Hasher> HashSequence for H {
    fn next(&mut self) -> u64 {
        54387634019u64.hash(self);
        self.finish()
    }
}

impl<H: Hasher + Clone> HashSeqBuilder for H {
    type Seq = H;

    fn bit_mask(&self) -> u64 {
        self.finish()
    }

    fn hash_seq(&self, bit: u64) -> Self::Seq {
        let mut hasher = self.clone();
        bit.hash(&mut hasher);
        hasher
    }
}

impl<H: Hasher + Clone> ManySeqBuilder for H {
    type Builder = H;

    fn seq_builder(&self, i: usize) -> Self::Builder {
        let mut hasher = self.clone();
        i.hash(&mut hasher);
        hasher
    }
}

/// One building block for the consistent hashing algorithm is a consistent
/// hash iterator which enumerates all the hashes for a specific bucket.
/// A bucket covers the range `(1<<bit)..(2<<bit)`.
#[derive(Default)]
struct BucketIterator<H: HashSequence> {
    hasher: H,
    n: usize,
    is_first: bool,
    bit: u64,  // A bitmask with a single bit set.
}

impl<H: HashSequence> BucketIterator<H> {
    fn new(n: usize, bit: u64, hasher: H) -> Self {
        Self {
            hasher,
            n,
            is_first: true,
            bit,
        }
    }
}

impl<H: HashSequence> Iterator for BucketIterator<H> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bit == 0 {
            return None;
        }
        if self.is_first {
            let res = (self.hasher.next() & (self.bit - 1)) + self.bit;
            self.is_first = false;
            if res < self.n as u64 {
                self.n = res as usize;
                return Some(self.n);
            }
        }
        loop {
            let res = self.hasher.next() & (self.bit * 2 - 1);
            if res & self.bit == 0 {
                return None;
            }
            if res < self.n as u64 {
                self.n = res as usize;
                return Some(self.n);
            }
        }
    }
}

/// An iterator which enumerates all the consistent hashes for a given key
/// from largest to smallest in the range `0..n`.
pub struct ConsistentHashRevIterator<H: HashSeqBuilder> {
    builder: H,
    bits: u64,
    n: usize,
    inner: Option<BucketIterator<H::Seq>>,
}

impl<H: HashSeqBuilder> ConsistentHashRevIterator<H> {
    pub fn new(n: usize, builder: H) -> Self {
        Self {
            bits: builder.bit_mask() & (n.next_power_of_two() as u64 - 1),
            builder,
            n,
            inner: None,
        }
    }
}

impl<H: HashSeqBuilder> Iterator for ConsistentHashRevIterator<H> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.n == 0 {
            return None;
        }
        if let Some(res) = self.inner.as_mut().and_then(|inner| inner.next()) {
            return Some(res);
        }
        while self.bits > 0 {
            let bit = 1 << self.bits.ilog2();
            self.bits ^= bit;
            let seq = self.builder.hash_seq(bit);
            let mut iter = BucketIterator::new(self.n, bit, seq);
            if let Some(res) = iter.next() {
                self.inner = Some(iter);
                return Some(res);
            }
        }
        self.n = 0;
        Some(0)
    }
}

/// Same as `ConsistentHashRevIterator`, but iterates from smallest to largest
/// for the range `n..`.
pub struct ConsistentHashIterator<H: HashSeqBuilder> {
    bits: u64,
    n: usize,
    builder: H,
    stack: Vec<usize>,
}

impl<H: HashSeqBuilder> ConsistentHashIterator<H> {
    pub fn new(n: usize, builder: H) -> Self {
        Self {
            bits: builder.bit_mask() & !((n + 2).next_power_of_two() as u64 / 2 - 1),
            stack: if n == 0 { vec![0] } else { vec![] },
            builder,
            n,
        }
    }
}

impl<H: HashSeqBuilder> Iterator for ConsistentHashIterator<H> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(res) = self.stack.pop() {
            return Some(res);
        }
        while self.bits > 0 {
            let bit = self.bits & !(self.bits - 1);
            self.bits &= self.bits - 1;
            let inner = BucketIterator::new(bit as usize * 2, bit, self.builder.hash_seq(bit));
            self.stack = inner.take_while(|x| *x >= self.n).collect();
            if let Some(res) = self.stack.pop() {
                return Some(res);
            }
        }
        None
    }
}

/// Wrapper around `ConsistentHashIterator` and `ConsistentHashRevIterator` to compute
/// the next or previous consistent hash for a given key for a given number of nodes `n`.
pub struct ConsistentHasher<H: HashSeqBuilder> {
    builder: H,
}

impl<H: HashSeqBuilder + Clone> ConsistentHasher<H> {
    pub fn new(builder: H) -> Self {
        Self { builder }
    }

    pub fn prev(&self, n: usize) -> Option<usize> {
        let mut sampler = ConsistentHashRevIterator::new(n, self.builder.clone());
        sampler.next()
    }

    pub fn next(&self, n: usize) -> Option<usize> {
        let mut sampler = ConsistentHashIterator::new(n, self.builder.clone());
        sampler.next()
    }
}

/// Implementation of a consistent choose k hashing algorithm.
/// It returns k distinct consistent hashes in the range `0..n`.
/// The hashes are consistent when `n` changes and when `k` changes!
/// I.e. on average exactly `1/(n+1)` (resp. `1/(k+1)`) many hashes will change
/// when `n` (resp. `k`) increases by one. Additionally, the returned `k` tuple
/// is guaranteed to be uniformely chosen from all possible `n-choose-k` tuples.
pub struct ConsistentChooseKHasher<H: ManySeqBuilder> {
    builder: H,
    k: usize,
}

impl<H: ManySeqBuilder> ConsistentChooseKHasher<H> {
    pub fn new(builder: H, k: usize) -> Self {
        Self { builder, k }
    }

    // TODO: Implement this as an iterator!
    pub fn prev(&self, mut n: usize, samples: &mut Vec<usize>)  {
        let mut samplers: Vec<_> = (0..self.k)
            .map(|i| ConsistentHashRevIterator::new(n - i, self.builder.seq_builder(i)).peekable())
            .collect();
        samples.clear();
        for i in (0..self.k).rev() {
            let mut max = 0;
            for k in 0..=i {
                while samplers[k].peek() >= Some(&(n - k)) && n - k > 0 {
                    samplers[k].next();
                }
                max = max.max(samplers[k].peek().unwrap() + k);
            }
            samples.push(max);
            n = max;
        }
    }
}


#[cfg(test)]
mod tests {
    use std::hash::DefaultHasher;

    use super::*;

    fn hasher_for_key(key: u64) -> DefaultHasher {
        let mut hasher = DefaultHasher::default();
        key.hash(&mut hasher);
        hasher
    }

    #[test]
    fn test_uniform_1() {
        for k in 0..100 {
            let hasher = hasher_for_key(k);
            let sampler = ConsistentHasher::new(hasher.clone());
            for n in 0..1000 {
                assert!(sampler.prev(n + 1) <= sampler.prev(n + 2));
                let next = sampler.next(n).unwrap();
                assert_eq!(next, sampler.prev(next + 1).unwrap());
            }
            let mut iter_rev: Vec<_> = ConsistentHashIterator::new(0, hasher.clone())
                .take_while(|x| *x < 1000)
                .collect();
            iter_rev.reverse();
            let iter: Vec<_> = ConsistentHashRevIterator::new(1000, hasher).collect();
            assert_eq!(iter, iter_rev);
        }
        let mut stats = vec![0; 13];
        for i in 0..100000 {
            let hasher = hasher_for_key(i);
            let sampler = ConsistentHasher::new(hasher);
            let x = sampler.prev(stats.len()).unwrap();
            stats[x] += 1;
        }
        println!("{stats:?}");
    }

    #[test]
    fn test_uniform_k() {
        const K: usize = 3;
        for k in 0..100 {
            let hasher = hasher_for_key(k);
            let sampler = ConsistentChooseKHasher::new(hasher, K);
            for n in K..1000 {
                let samples = sampler.prev(n + 1);
                assert!(samples.len() == K);
                for i in 0..K - 1 {
                    assert!(samples[i] < samples[i + 1]);
                }
                let next = sampler.prev(n + 2);
                for i in 0..K {
                    assert!(samples[i] <= next[i]);
                }
                let mut merged = samples.clone();
                merged.extend(next.clone());
                merged.sort();
                merged.dedup();
                assert!(
                    merged.len() == K || merged.len() == K + 1,
                    "Unexpected {samples:?} vs. {next:?}"
                );
            }
        }
        let mut stats = vec![0; 8];
        for i in 0..32 {
            let hasher = hasher_for_key(i + 32783);
            let sampler = ConsistentChooseKHasher::new(hasher, 2);
            let samples = sampler.prev(stats.len());
            for s in samples {
                stats[s] += 1;
            }
        }
        println!("{stats:?}");
        // Test consistency when increasing k!
        for k in 1..10 {
            for n in k + 1..20 {
                for key in 0..1000 {
                    let hasher = hasher_for_key(key);
                    let sampler1 = ConsistentChooseKHasher::new(hasher.clone(), k);
                    let sampler2 = ConsistentChooseKHasher::new(hasher, k + 1);
                    let set1 = sampler1.prev(n);
                    let set2 = sampler2.prev(n);
                    assert_eq!(set1.len(), k);
                    assert_eq!(set2.len(), k + 1);
                    let mut merged = set1.clone();
                    merged.extend(set2);
                    merged.sort();
                    merged.dedup();
                    assert_eq!(merged.len(), k + 1);
                }
            }
        }
    }
}
