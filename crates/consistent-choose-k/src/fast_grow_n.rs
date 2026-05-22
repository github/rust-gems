//! Fast variant of [`crate::ConsistentChooseKHasher`] specialized for
//! repeated `grow_n` calls at fixed `k`.
//!
//! Companion to [`crate::ConsistentChooseKFastHasher`] (which is fast at
//! `shrink_n` for fixed `k`). The two specializations cannot easily share
//! a single representation: `shrink_n` keeps `samples` sorted by value and
//! tracks per-position "block counts" for each *position-bound* sequence
//! id; `grow_n` keeps `samples` in insertion order and tracks per-sample
//! "life" = `seq_id - position`.
//!
//! # Algorithm sketch
//!
//! State:
//! * `next_heap`: min-heap of `(sample, packed_seq)`. `packed_seq` is
//!   `seq_id * 2 + owner_bit`. For each seq id with at least one entry
//!   in the heap, exactly one entry — the largest — has its owner bit
//!   set. When that entry is popped, the seq's next active bucket of
//!   samples is loaded into the heap (and the new largest becomes the
//!   new owner). Each bucket of a seq is materialized as a batch by
//!   running the seq's [`BucketIterator`] to exhaustion; this avoids
//!   re-running the seq's hash sequence on every single `grow_n` call.
//! * `bits[seq]`: bitmask of buckets *not yet* pushed into the heap.
//!   Lower bits correspond to smaller value ranges (`[bit, 2*bit)`),
//!   so the lowest set bit is the next bucket to push.
//! * `builders[seq]`: cached per-seq hash builder.
//! * `samples`: a [`SampleTreap`] holding `(sample, life)` pairs in
//!   insertion order, where `life = seq_id - position`. Once `k` samples
//!   are present, an entry whose `life <= 0` is the *displaced* sample
//!   that must be evicted on the next firing.
//!
//! Per-call cost: O(log k) expected — amortized constant heap pushes
//! per `grow_n`, plus a single treap pop / push.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::consistent_hash::BucketIterator;
use crate::sample_treap::SampleTreap;
use crate::{HashSeqBuilder, ManySeqBuilder};

/// Fast variant of [`crate::ConsistentChooseKHasher`] specialized for
/// repeated `grow_n` calls at fixed `k`. See module-level documentation
/// for the algorithm.
pub struct ConsistentChooseKFastGrowHasher<H: ManySeqBuilder> {
    /// Current universe size.
    n: usize,
    /// Fixed sample count (number of sequences tracked).
    k: usize,
    /// Min-heap keyed by `(sample, packed_seq)` where
    /// `packed_seq = seq_id * 2 + owner_bit`. The owner bit is set on
    /// exactly one entry per seq present in the heap (the largest); when
    /// that entry is popped, the seq's next bucket is loaded.
    next_heap: BinaryHeap<Reverse<(usize, usize)>>,
    /// Per-seq cached hash builder, used to spin up `BucketIterator`s on
    /// refill without re-deriving the builder.
    builders: Vec<H::Builder>,
    /// Per-seq bitmask of buckets not yet pushed into the heap.
    bits: Vec<u64>,
    /// Currently-selected samples in insertion order. Each entry's `life`
    /// is `seq_id - position`; an entry with `life <= 0` is displaced and
    /// will be evicted on the next firing.
    samples: SampleTreap,
}

impl<H: ManySeqBuilder> ConsistentChooseKFastGrowHasher<H> {
    /// Create a new instance for `k` sequences with `n = k`. Seeds the
    /// life array from samples `< k` and pushes the first heap-worthy
    /// bucket (the bucket containing the first sample `>= k`) into the
    /// heap for every seq.
    ///
    /// Time: O(k).
    pub fn new(builder: H, k: usize) -> Self {
        let mut next_heap = BinaryHeap::with_capacity(k);
        let mut builders = Vec::with_capacity(k);
        let mut bits = Vec::with_capacity(k);
        let mut life = vec![0; k];
        for seq in 0..k {
            let bld = builder.seq_builder(seq);
            let mut seq_bits = bld.bit_mask();
            let mut is_owner = true;
            // Walk buckets low-bit-first. Push every sample `>= k` into
            // the heap, mark the first such (largest in its bucket, since
            // BucketIterator yields decreasing) as owner; lower samples
            // feed the life array. Stop after the first bucket that
            // contributes to the heap; later buckets are kept in
            // `seq_bits` for `grow_n` to drain via `refill`.
            while seq_bits != 0 && is_owner {
                let bit = seq_bits & seq_bits.wrapping_neg();
                seq_bits ^= bit;
                let iter = BucketIterator::new(bit as usize * 2, bit, bld.hash_seq(bit));
                for l in iter {
                    let sample = l + seq;
                    if sample >= k {
                        let owner = usize::from(is_owner);
                        next_heap.push(Reverse((sample, seq * 2 + owner)));
                        is_owner = false;
                    } else {
                        life[sample] = l.max(life[sample]);
                    }
                }
            }
            debug_assert!(
                !is_owner,
                "seq {seq} must contribute at least one sample >= k"
            );
            bits.push(seq_bits);
            builders.push(bld);
        }
        let mut samples = SampleTreap::with_capacity(k);
        for (sample, life) in life.into_iter().enumerate() {
            samples.push_back(sample, life as i32);
        }

        Self {
            n: k,
            k,
            next_heap,
            builders,
            bits,
            samples,
        }
    }

    /// Current universe size.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Target sample count (fixed at construction).
    pub fn k(&self) -> usize {
        self.k
    }

    /// Returns the currently-selected samples sorted by value.
    pub fn samples(&self) -> Vec<usize> {
        self.samples.samples()
    }

    /// Grow `n` by one and update the choose-k set accordingly. Returns
    /// `Some(new_sample)` if a sequence fired (i.e. some sample changed).
    ///
    /// Time: O(log k) expected.
    pub fn grow_n(&mut self) -> Option<usize> {
        loop {
            let Reverse((next, packed_seq)) = self
                .next_heap
                .pop()
                .expect("there are always entries in the heap!");
            let seq = packed_seq >> 1;
            // If this entry was the owner (largest in heap for `seq`),
            // load the seq's next active bucket. We do this whether or
            // not the entry is stale: owners always trigger a refill.
            if (packed_seq & 1) == 1 {
                self.refill(seq);
            }
            if next >= self.n {
                self.n = next + 1;
                let pos = self
                    .samples
                    .find_rightmost_le_zero()
                    .expect("there must be a displaced sample to evict");
                self.samples.remove_at_decrementing_suffix(pos);
                self.samples.push_back(next, self.k as i32 - seq as i32 - 1);
                break Some(next);
            }
        }
    }

    /// Push the next active bucket of `seq` into the heap. Skips
    /// buckets whose samples are all below `self.n` (would be stale on
    /// arrival) and any remaining samples within a straddling bucket
    /// that are below `self.n`. Marks the first (largest) pushed sample
    /// as the new owner for `seq`.
    fn refill(&mut self, seq: usize) {
        let bld = &self.builders[seq];
        let bits = &mut self.bits[seq];
        let bit = *bits & bits.wrapping_neg();
        *bits ^= bit;
        let iter = BucketIterator::new(bit as usize * 2, bit, bld.hash_seq(bit));
        let mut is_owner = true;
        for l in iter {
            let sample = l + seq;
            let owner = usize::from(is_owner);
            self.next_heap.push(Reverse((sample, seq * 2 + owner)));
            is_owner = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::hash::{DefaultHasher, Hash};

    use super::*;
    use crate::ConsistentChooseKHasher;

    fn hasher_for_key(key: u64) -> DefaultHasher {
        let mut h = DefaultHasher::default();
        key.hash(&mut h);
        h
    }

    #[test]
    fn grow_n_matches_new_with_k() {
        for key in 0..200 {
            for k in 1..10 {
                let mut fast = ConsistentChooseKFastGrowHasher::new(hasher_for_key(key), k);
                while fast.n() < 10000 {
                    let n = fast.n();
                    let std = ConsistentChooseKHasher::new_with_k(hasher_for_key(key), n, k);
                    let mut expected: Vec<usize> = std.samples().to_vec();
                    expected.sort();
                    let mut got = fast.samples();
                    got.sort();
                    assert_eq!(got, expected, "key={key}, k={k}, n={n}");
                    fast.grow_n();
                }
            }
        }
    }

}
