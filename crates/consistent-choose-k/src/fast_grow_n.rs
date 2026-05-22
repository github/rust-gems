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
//! * `next_heap`: min-heap of `(next_candidate_sample, seq_id)` — one
//!   entry per active sequence, keyed by the sequence's smallest sample
//!   strictly greater than its currently-selected sample (or its first
//!   sample, if it has none yet).
//! * `samples`: a [`SampleTreap`] holding `(sample, life)` pairs in
//!   insertion order, where `life = seq_id - position`. Once `k` samples
//!   are present, an entry whose `life <= 0` is the *displaced* sample
//!   that must be evicted on the next firing.
//!
//! `grow_n` (semantics: `n += 1`):
//!
//! 1. While the heap's smallest `next_candidate < n_new` (in practice
//!    zero or one iteration, since `n` grows by one):
//!    1. Pop `(s, seq_id)`.
//!    2. Push the next candidate for `seq_id` (smallest sample > `s`)
//!       back into the heap.
//!    3. Append `(s, life = seq_id - new_position)` at the end of the
//!       treap via `push_back`.
//!    4. If the treap now has more than `k` entries (i.e. we displaced
//!       a sample), find the rightmost position with `life <= 0` via
//!       `find_rightmost_le_zero`, `remove_at` it, and apply
//!       `add_life_suffix(p_dead, +1)` so the remaining entries (which
//!       shifted left by one) see their `life` increase by one.
//! 2. Set `n = n_new`.
//!
//! Per-call cost: O(log k) expected (heap pop/push + treap ops).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::consistent_hash::ConsistentHashIterator;
use crate::sample_treap::SampleTreap;
use crate::ManySeqBuilder;

/// Fast variant of [`crate::ConsistentChooseKHasher`] specialized for
/// repeated `grow_n` calls at fixed `k`. See module-level documentation
/// for the algorithm.
pub struct ConsistentChooseKFastGrowHasher<H: ManySeqBuilder> {
    /// Current universe size.
    n: usize,
    /// Fixed sample count (number of sequences tracked in `next_heap`).
    k: usize,
    /// Min-heap of `(next_candidate_sample, seq_id)` keyed by the first
    /// component. A `None` next-candidate (sequence exhausted) is *not*
    /// pushed back; the heap shrinks instead.
    next_heap: BinaryHeap<Reverse<(usize, usize)>>,
    /// One long-lived `ConsistentHashIterator` per sequence id, kept
    /// positioned just past the seq's most recently popped/pushed sample.
    /// Avoids rebuilding the iterator (and re-deriving its bucket state)
    /// on every `grow_n` event.
    iters: Vec<ConsistentHashIterator<H::Builder>>,
    /// Currently-selected samples in insertion order. Each entry's `life`
    /// is `seq_id - position`; an entry with `life <= 0` is displaced and
    /// will be evicted on the next firing.
    samples: SampleTreap,
}

impl<H: ManySeqBuilder> ConsistentChooseKFastGrowHasher<H> {
    /// Create a new instance for `k` sequences with `n = 0`. Seeds the
    /// heap with each sequence's first sample; the sample treap is empty
    /// until `grow_n` is called enough times for samples to fire.
    ///
    /// Time: O(k).
    pub fn new(builder: H, k: usize) -> Self {
        let mut next_heap = BinaryHeap::with_capacity(k);
        let mut iters = Vec::with_capacity(k);
        let mut life = vec![0; k];
        for seq in 0..k {
            let mut iter = ConsistentHashIterator::new(0, builder.seq_builder(seq));
            loop {
                let l = iter.next().expect("seq must yield a sample >= k");
                let sample = l + seq;
                if sample >= k {
                    next_heap.push(Reverse((sample, seq)));
                    break;
                }
                life[sample] = l.max(life[sample]);
            }
            iters.push(iter);
        }
        let mut samples = SampleTreap::with_capacity(k);
        for (sample, life) in life.into_iter().enumerate() {
            samples.push_back(sample, life as i32);
        }

        Self {
            n: k,
            k,
            next_heap,
            iters,
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
    /// `Some(new_sample)` if a sequence fired (i.e. some sample changed),
    /// `None` otherwise.
    ///
    /// Time: O(log k) expected.
    pub fn grow_n(&mut self) -> Option<usize> {
        loop {
            let Reverse((next, seq)) = self
                .next_heap
                .pop()
                .expect("there are always k elements in the heap!");
            // Advance this seq's cached iterator until the next candidate
            // satisfies `>= self.n` (i.e. iter yield `>= self.n - seq`).
            // Under the heap invariant the very first `.next()` already
            // satisfies the bound, but we keep the inner loop for safety.
            let threshold = self.n.max(next + 1) - seq;
            let after = loop {
                let l = self.iters[seq].next().expect("seq must yield more samples");
                if l >= threshold {
                    break l + seq;
                }
            };
            self.next_heap.push(Reverse((after, seq)));
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
