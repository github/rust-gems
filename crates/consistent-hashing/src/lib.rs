use std::hash::{DefaultHasher, Hash, Hasher};

/// One building block for the consistent hashing algorithm is a consistent
/// hash iterator which enumerates all the hashes for a given for a specific bucket.
/// A bucket covers the range `(1<<bit)..(2<<bit)`.
#[derive(Default)]
struct BucketIterator {
    hasher: DefaultHasher,
    n: usize,
    is_first: bool,
    bit: u64,
}

impl BucketIterator {
    fn new(key: u64, n: usize, bit: u64) -> Self {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        bit.hash(&mut hasher);
        Self {
            hasher,
            n,
            is_first: true,
            bit,
        }
    }
}

impl Iterator for BucketIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bit == 0 {
            return None;
        }
        if self.is_first {
            let res = self.hasher.finish() % self.bit + self.bit;
            if res < self.n as u64 {
                self.n = res as usize;
                return Some(self.n);
            }
            self.is_first = false;
        }
        loop {
            478392.hash(&mut self.hasher);
            let res = self.hasher.finish() % (self.bit * 2);
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
pub struct ConsistentHashRevIterator {
    bits: u64,
    key: u64,
    n: usize,
    inner: BucketIterator,
}

impl ConsistentHashRevIterator {
    pub fn new(key: u64, n: usize) -> Self {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let bits = hasher.finish() % n.next_power_of_two() as u64;
        let inner = BucketIterator::default();
        Self {
            bits,
            key,
            n,
            inner,
        }
    }
}

impl Iterator for ConsistentHashRevIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.n == 0 {
            return None;
        }
        if let Some(res) = self.inner.next() {
            return Some(res);
        }
        while self.bits > 0 {
            let bit = 1 << self.bits.ilog2();
            self.bits ^= bit;
            self.inner = BucketIterator::new(self.key, self.n, bit);
            if let Some(res) = self.inner.next() {
                return Some(res);
            }
        }
        self.n = 0;
        Some(self.n)
    }
}

/// Same as `ConsistentHashRevIterator`, but iterates from smallest to largest
/// for the range `n..`.
pub struct ConsistentHashIterator {
    bits: u64,
    key: u64,
    n: usize,
    stack: Vec<usize>,
}

impl ConsistentHashIterator {
    pub fn new(key: u64, n: usize) -> Self {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let mut bits = hasher.finish() as u64;
        bits &= !((n + 2).next_power_of_two() as u64 / 2 - 1);
        let stack = if n == 0 { vec![0] } else { vec![] };
        Self {
            bits,
            key,
            n,
            stack,
        }
    }
}

impl Iterator for ConsistentHashIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(res) = self.stack.pop() {
            return Some(res);
        }
        while self.bits > 0 {
            let bit = self.bits & !(self.bits - 1);
            self.bits &= self.bits - 1;
            let inner = BucketIterator::new(self.key, bit as usize * 2, bit);
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
pub struct ConsistentHasher {
    key: u64,
}

impl ConsistentHasher {
    pub fn new(key: u64) -> Self {
        Self { key }
    }

    pub fn prev(&self, n: usize) -> Option<usize> {
        let mut sampler = ConsistentHashRevIterator::new(self.key, n);
        sampler.next()
    }

    pub fn next(&self, n: usize) -> Option<usize> {
        let mut sampler = ConsistentHashIterator::new(self.key, n);
        sampler.next()
    }
}

/// Implementation of a consistent choose k hashing algorithm.
/// It returns k distinct consistent hashes in the range `0..n`.
/// The hashes are consistent when `n` changes and when `k` changes!
/// I.e. on average exactly `1/(n+1)` (resp. `1/(k+1)`) many hashes will change
/// when `n` (resp. `k`) increases by one. Additionally, the returned `k` tuple
/// is guaranteed to be uniformely chosen from all possible `n-choose-k` tuples.
pub struct ConsistentChooseKHasher {
    key: u64,
    k: usize,
}

impl ConsistentChooseKHasher {
    pub fn new(key: u64, k: usize) -> Self {
        Self { key, k }
    }

    // TODO: Implement this as an iterator!
    pub fn prev(&self, mut n: usize) -> Vec<usize> {
        let mut samples = Vec::with_capacity(self.k);
        let mut samplers: Vec<_> = (0..self.k)
            .map(|i| ConsistentHashRevIterator::new(self.key + 43987492 * i as u64, n - i).peekable())
            .collect();
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
        samples.sort();
        samples
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_1() {
        for k in 0..100 {
            let sampler = ConsistentHasher::new(k);
            for n in 0..1000 {
                assert!(sampler.prev(n + 1) <= sampler.prev(n + 2));
                let next = sampler.next(n).unwrap();
                assert_eq!(next, sampler.prev(next + 1).unwrap());
            }
            let mut iter_rev: Vec<_> = ConsistentHashIterator::new(k, 0)
                .take_while(|x| *x < 1000)
                .collect();
            iter_rev.reverse();
            let iter: Vec<_> = ConsistentHashRevIterator::new(k, 1000).collect();
            assert_eq!(iter, iter_rev);
        }
        let mut stats = vec![0; 13];
        for i in 0..100000 {
            let sampler = ConsistentHasher::new(i);
            let x = sampler.prev(stats.len()).unwrap();
            stats[x] += 1;
        }
        println!("{stats:?}");
    }

    #[test]
    fn test_uniform_k() {
        const K: usize = 3;
        for k in 0..100 {
            let sampler = ConsistentChooseKHasher::new(k, K);
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
            let sampler = ConsistentChooseKHasher::new(i + 32783, 2);
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
                    let sampler1 = ConsistentChooseKHasher::new(key, k);
                    let sampler2 = ConsistentChooseKHasher::new(key, k + 1);
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
