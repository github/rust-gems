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

    /// Returns a bit mask indicating which buckets have at least one hash.
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
    n: usize, // Upper bound for the hash values within the bucket.
    is_first: bool,
    bit: u64, // A bitmask with a single bit set.
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
    bits: u64,                             // Bitmask of unvisited buckets.
    n: usize,                              // Exclusive upper bound for the hash values.
    inner: Option<BucketIterator<H::Seq>>, // Iterator for the current bucket.
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
    bits: u64, // Bitmasks of unvisited buckets.
    n: usize,  // Inclusive lower bound for the hash values.
    builder: H,
    stack: Vec<usize>, // Stack of hashes in the current bucket.
}

impl<H: HashSeqBuilder> ConsistentHashIterator<H> {
    pub fn new(n: usize, builder: H) -> Self {
        // Buckets are indexed by the bit value `b` (a power of two) and
        // cover the range `[b, 2*b)`. We want every bucket whose upper
        // bound exceeds `n`, i.e. `2*b > n`, i.e.
        // `b >= next_power_of_two(n/2 + 1)`.
        let bit_min = (n / 2 + 1).next_power_of_two() as u64;
        Self {
            bits: builder.bit_mask() & !(bit_min - 1),
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

impl<H: HashSeqBuilder> ConsistentHasher<H> {
    /// Construct a new ConsistentHasher with the given builder for a specific key.
    pub fn new(builder: H) -> Self {
        Self { builder }
    }

    /// Return the largest consistent hash smaller than `n`.
    pub fn prev(&self, n: usize) -> Option<usize>
    where
        H: Clone,
    {
        let mut sampler = ConsistentHashRevIterator::new(n, self.builder.clone());
        sampler.next()
    }

    /// Return the smallest consistent hash greater than or equal to `n`.
    pub fn next(&self, n: usize) -> Option<usize>
    where
        H: Clone,
    {
        let mut sampler = ConsistentHashIterator::new(n, self.builder.clone());
        sampler.next()
    }

    /// Return the largest consistent hash smaller than `n`, consuming the hasher.
    pub fn into_prev(self, n: usize) -> Option<usize> {
        ConsistentHashRevIterator::new(n, self.builder).next()
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
        assert_eq!(
            stats,
            vec![7577, 7541, 7538, 7822, 7763, 7687, 7718, 7723, 7846, 7723, 7688, 7716, 7658]
        );
    }

    /// Regression test for an off-by-one in `ConsistentHashIterator::new`'s
    /// bucket-mask construction: starting the iterator at `n` must not skip
    /// over a value `>= n` that the full iteration from 0 would visit.
    #[test]
    fn test_iterator_start_includes_n() {
        let builder = hasher_for_key(1).seq_builder(3);
        let from_zero: Vec<usize> = ConsistentHashIterator::new(0, builder.clone())
            .take(10)
            .collect();
        for &v in &from_zero {
            assert_eq!(
                ConsistentHashIterator::new(v, builder.clone()).next(),
                Some(v),
                "iterator starting at {v} must yield {v} as the first value (full sequence: {from_zero:?})",
            );
        }
    }
}
