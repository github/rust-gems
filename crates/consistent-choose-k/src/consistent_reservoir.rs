//! Online consistent reservoir sampling driven by
//! [`ConsistentPermutation`].
//!
//! # What is a "consistent" reservoir?
//!
//! Given a 64-bit `master_key`, a *consistent* reservoir of size `k`
//! over a universe `[0, n)` is the set
//!
//! ```text
//! reservoir(n, k) = ConsistentPermutation::new(n, master_key).take(k)
//! ```
//!
//! — the first `k` emissions of the keyed permutation of `[0, n)`.
//! By the consistency property of [`ConsistentPermutation`]:
//!
//! * For fixed `k`, growing `n` by one inserts at most one new
//!   element into `reservoir(n, k)` (and evicts at most one). The
//!   stream of `(added, evicted)` pairs as `n` ranges over `n..`
//!   is what this iterator emits.
//! * For fixed `n`, the reservoir is a **prefix** in rank order:
//!   `reservoir(n, k) ⊂ reservoir(n, k+1)`. So the same reservoir
//!   serves every `k` — there is no need to fix `k` at construction
//!   time the way standard reservoir sampling does. The first
//!   `k'` elements of the reservoir (in rank order) are themselves
//!   a valid reservoir of size `k'`.
//!
//! # Complexity
//!
//! * `ConsistentPermutation::new(n, key).take(k)` — get the
//!   reservoir for a fixed `n` and `k` from scratch: **`O(k)`**.
//! * [`ConsistentReservoir::new(k, n, key)`](Self::new) — set up
//!   the iterator at any starting `n`: **`O(k)`** (the
//!   `O(k)` walk plus a linear-time heapification of `pending`).
//! * [`Iterator::next`] — get the next `(added, evicted)` pair:
//!   **`O(log k)` amortised** (each call pops amortized a constant number of
//!   tail entries; `grow_layer`, which costs `O(k)`, is
//!   triggered on average once per `Θ(k)` admissions).
//!
//! Note that the heapification of `pending` runs in linear $O(k)$ time, but retrieving elements
//! sequentially from the heap still carries a cost of up to $O(\log k)$ per pop. The sorting/priority-queue
//! overhead could be completely eliminated and sped up to true $O(k)$ total complexity via bucket sort!
//! Because all values generated in a layer are expected to be uniformly randomly chosen within the
//! interval $[\frac{1}{4}m, m)$, bucket-sorting them into $O(k)$ buckets achieves a linear worst-case average-case runtime.
//! With that optimization, the complexity of consistent reservoir sampling would match that of standard
//! reservoir sampling amortized ($O(1)$ per processed item).
//!
//! Contrast with the classic random-jump reservoir sampling
//! algorithms (Vitter L):
//!
//! * `O(1)` to jump to the next sample in the stream.
//! * `O(k log(n / k))` to *jump* to a target universe size `n`
//!   (since L still has to iterate through the geometric jumps).
//! * Tied to the `k` fixed at construction; changing `k` breaks
//!   consistency property.
//!
//! So a consistent reservoir is the right pick when you want to
//! random-access the reservoir at an arbitrary `n` and/or vary `k`
//! while keeping the consistency property; standard reservoir sampling wins on per-element
//! streaming cost but loses on these two flexibilities.
//!
//! # Data layout
//!
//! Everything lives in a single flat `values: Vec<u32>` plus a `pending: BinaryHeap<(Reverse<u32>, u32)>`
//! of `(Reverse(value), rank)` pairs representing candidate fresh admissions.
//!
//! After construction (or after [`Self::grow_layer`]), `values` is
//! exactly the first `k + |pending|` emissions of
//! `ConsistentPermutation::new(m, master_key)` in walk order, i.e.
//! `values[r]` is the rank-`r` emission of the current top-level
//! permutation. Descents (old-reservoir items) and non-descents
//! (fresh items) as well as active and inactive items are interleaved.
//!
//! `pending` is a min-heap storing the `(Reverse(value), rank)` of the
//! **non-descent** emissions (i.e. the new items that might enter the reservoir
//! as `n` grows). By storing reversed values, `pending.pop()` efficiently returns
//! the entry with the smallest fresh value first — the next admission as `n` grows.
//!
//! # Active reservoir via `n`
//!
//! `self.n` is the current effective universe size: `reservoir()`
//! returns the items in `values` whose value is `< n`, in walk order.
//! By [`ConsistentPermutation`] consistency this is exactly
//! `ConsistentPermutation::new(n, master_key).take(k)` — the
//! consistent top-`k` for `[0, n)`.
//!
//! Each [`Iterator::next`] call advances `n` to the next admission's
//! value `+ 1`, yielding `(added_value, evicted_value)`.
//!
//! # Initial state ([`Self::new`])
//!
//! Given the user's initial `n`, pick `j_max = smallest_j_max(n)` and
//! `m = 4^(j_max+1) >= n`. Walk `ConsistentPermutation::new(m, key)`
//! pushing each emission into `values` until we have collected
//! exactly `k` in-range values (i.e. `< n`). Every out-of-range
//! emission seen along the way has its rank and value recorded in `pending`;
//! finally `pending` is initialized in $O(k)$ time using $O(k)$ linear heapification.
//!
//! The loop is guaranteed to terminate with the last push being an
//! in-range value — that's what triggers `count == k`. So
//! `values.last()` is always a real reservoir item, which is the
//! one to evict on the first admission.
//!
//! # Growing the universe ([`Self::grow_layer`])
//!
//! Opens the next top Feistel layer (`j_max += 1`, `m *= 4`) and
//! rebuilds `values` / `pending` against the new permutation by
//! walking the new top layer's counter until all `k` old-reservoir
//! entries have been consumed by descents. Every emission is pushed
//! to `new_values` in walk order; non-descents additionally have
//! their rank and value recorded in `pending`. The walk terminates on a
//! descent so `values.last()` is again a real reservoir item.
//!
//! Non-descents at counters `>= k` are real emissions of the new
//! permutation but their rank places them outside the top-`k`. They
//! still end up in `values` and `pending`, and are filtered out
//! later when popped from the heap.
//!
//! # Invariants
//!
//! Throughout the iterator's lifetime (between [`Iterator::next`]
//! calls, and restored before each yields) four invariants hold:
//!
//! 1. **The active reservoir has size `k`.** The *active set* —
//!    the entries of `values` whose value is `< n` — always
//!    contains exactly `k` elements. This is what `reservoir()`
//!    returns.
//! 2. **Every pending value is strictly larger than every active
//!    value.** Pending entries were emitted with value `>= n` by
//!    [`ConsistentPermutation`], whereas active entries are by
//!    definition `< n`. In particular the smallest pending value
//!    — which `pending.pop()` returns, since `pending` is structured
//!    as a min-heap — is the next candidate to cross into
//!    the active window as `n` grows.
//! 3. **`pending` must contain all elements from `values` which are greater or equal to `n`.**
//!    Since there are exactly `k` elements less than `n` in `values`, `pending.len()`
//!    must be larger or equal to `values.len() - k` (note that there can be values in
//!    `pending` which we discarded while searching for the last active value).
//! 4. **The last item in `values` is always an active element.**
//!    That is, `values.last()` is guaranteed to have a value `< n`.
//!
//! From these facts the algorithm of [`Iterator::next`] reads
//! off:
//!
//! **Finding the eviction.** Since the last item in `values` is always
//! an active element (invariant 4), the last active entry in walk order
//! resides exactly at the tail of `values`. Therefore, we can immediately
//! obtain the evicted value by popping `values`.
//!
//! **Finding the admission.** Pop `added_rank` from the `pending` min-heap until it is
//! strictly less than `values.len()`. If it is out-of-bounds, it corresponds
//! to an element that was previously popped from `values` because it was pending
//! at the tail, so we safely discard it and continue. Otherwise, the corresponding
//! value is our candidate `added_value`.
//!
//! **Restoring the invariants.** Once we have `added_value`, we update `n` to
//! `added_value + 1`. This makes `added_value` active, restoring the active
//! reservoir size to `k`. Finally, we restore invariant 4 by popping any pending
//! elements (values `>= n`) from the tail of `values` until either `values.len() == k`
//! or the tail is active (value `< n`).
//!
//! The stream only terminates once the universe reaches the [`ConsistentPermutation`]
//! cap (`m = 2^30`); until then, `grow_layer` re-establishes the invariants whenever
//! `pending` empties.
//! or the tail is active (value `< n`).
//!
//! The stream only terminates once the universe reaches the [`ConsistentPermutation`]
//! cap (`m = 2^30`); until then, `grow_layer` re-establishes the invariants whenever
//! `pending` empties.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::consistent_permutation::{layer_apply, ConsistentPermutation};

/// Reservoir of size `k` whose contents track the consistent top-`k`
/// by [`ConsistentPermutation`] rank, with explicit support for
/// growing the universe one Feistel layer at a time.
pub struct ConsistentReservoir {
    /// First `k + pending.len()` emissions of
    /// `ConsistentPermutation(m, master_key)` in walk order.
    /// `values[r]` is always the rank-`r` emission of the current
    /// top-level permutation.
    values: Vec<u32>,
    /// Min-heap storing `(Reverse(value), rank)` for candidate fresh admissions.
    /// `pop()` returns the entry with the smallest value first.
    pending: BinaryHeap<(Reverse<u32>, u32)>,
    k: u32,
    /// Top Feistel layer index. The current top-level universe size
    /// is `m = 4^(j_max+1) = 1 << (2*j_max + 2)`.
    j_max: u32,
    master_key: u64,
    /// Current effective universe size. `reservoir()` returns
    /// values `< n`, which equals
    /// `ConsistentPermutation::new(n, master_key).take(k)`.
    n: u32,
}

/// Smallest `j_max` such that `4^(j_max+1) >= k`. Matches the
/// formula used by [`ConsistentPermutation::new`].
fn smallest_j_max(k: u32) -> u32 {
    if k <= 1 {
        0
    } else {
        (k - 1).ilog2() / 2
    }
}

impl ConsistentReservoir {
    /// Build a consistent reservoir of size `k` for initial universe
    /// `n`, seeded by `master_key`. The initial `reservoir()` is the
    /// consistent top-`k` for `[0, n)`.
    ///
    /// Internally we use the smallest layer boundary
    /// `m = 4^(j_max+1) >= n` and walk
    /// `ConsistentPermutation::new(m, master_key)` until we have
    /// observed exactly `k` values `< n`; out-of-range emissions
    /// become pending admissions for later iteration.
    ///
    /// `master_key` is forwarded verbatim to
    /// [`ConsistentPermutation::new`] (no avalanche).
    ///
    /// # Panics
    ///
    /// Panics if `k == 0`, `n > 2^30`, or `k > n`.
    pub fn new(k: u32, n: u32, master_key: u64) -> Self {
        assert!(k >= 1, "k must be at least 1");
        assert!(n <= 1u32 << 30, "n must be at most 2^30");
        let j_max = smallest_j_max(n);
        let m = 1u32 << (2 * j_max + 2);
        let mut values = vec![];
        let mut pending_vec = vec![];
        let mut count = 0;
        for value in ConsistentPermutation::new(m, master_key) {
            values.push(value);
            if value < n {
                count += 1;
                if count == k {
                    break;
                }
            } else {
                pending_vec.push((Reverse(value), values.len() as u32 - 1));
            }
        }
        let pending = BinaryHeap::from(pending_vec);
        Self {
            values,
            pending,
            k,
            j_max,
            master_key,
            n,
        }
    }

    /// Reservoir size.
    pub fn k(&self) -> u32 {
        self.k
    }

    /// Current effective universe size — `reservoir()` returns
    /// `ConsistentPermutation::new(n(), master_key).take(k)`.
    pub fn n(&self) -> u32 {
        self.n
    }

    /// The current reservoir contents, in rank order.
    ///
    /// Equals `ConsistentPermutation::new(n(), master_key).take(k)`
    /// by [`ConsistentPermutation`] consistency.
    pub fn reservoir(&self) -> impl Iterator<Item = u32> + '_ {
        self.values.iter().copied().filter(|&v| v < self.n())
    }

    /// Open the next top Feistel layer (extends `m` to `4 * m`) and
    /// rebuild `values`/`pending` against the new permutation.
    ///
    /// Any unconsumed entries left from the previous walk are
    /// dropped — their ranks lived in the old top layer's coordinate
    /// system and have no meaning under the new one. The new walk
    /// is guaranteed to terminate on a descent, so `values.last()`
    /// is again a genuine reservoir item ready to be evicted on the
    /// next admission.
    ///
    /// # Panics
    ///
    /// Panics if growing would push the universe past
    /// [`ConsistentPermutation`]'s `n <= 2^30` cap.
    pub fn grow_layer(&mut self) {
        assert!(
            self.j_max < 14,
            "ConsistentReservoir at universe cap (m = 2^30)"
        );
        let new_j_max = self.j_max + 1;
        let new_n_bits = 2 * new_j_max + 2;
        let new_top_shift = 2 * new_j_max;
        let k = self.k as usize;
        // Drop the stale eviction tail from the previous grow_layer
        // (if the iterator wasn't fully drained, those values are no
        // longer reachable anyway).
        self.values.truncate(k);
        let old_reservoir = std::mem::take(&mut self.values);
        debug_assert_eq!(old_reservoir.len(), k);

        let mut new_values: Vec<u32> = Vec::with_capacity(k);
        let mut pending_vec: Vec<(Reverse<u32>, u32)> = Vec::new();
        let mut next_old_idx = 0usize;
        let mut counter: u32 = 0;
        while next_old_idx < k {
            let raw = layer_apply(new_n_bits, self.master_key, counter);
            let is_descent = (raw >> new_top_shift) & 0b11 == 0;
            if is_descent {
                // Consume the next old reservoir entry; this is the
                // walk's only termination condition.
                new_values.push(old_reservoir[next_old_idx]);
                next_old_idx += 1;
            } else {
                // Push the fresh emission and remember its rank
                // (= counter). Ranks >= k are filtered out later by
                // the iterator's guards.
                new_values.push(raw);
                pending_vec.push((Reverse(raw), counter));
            }
            counter += 1;
        }
        debug_assert_eq!(new_values.len(), k + pending_vec.len());
        let pending = BinaryHeap::from(pending_vec);

        self.values = new_values;
        self.pending = pending;
        self.j_max = new_j_max;
    }
}

impl Iterator for ConsistentReservoir {
    /// `(added_value, evicted_value)`: the next item admitted to the
    /// reservoir paired with the item it displaces. After yielding,
    /// `n()` advances to `added_value + 1`.
    ///
    /// `added_value` is strictly increasing across calls
    /// (`pending` is consumed smallest-value first). The stream
    /// terminates only at the [`ConsistentPermutation`] universe
    /// cap (`m = 2^30`).
    type Item = (u32, u32);

    fn next(&mut self) -> Option<(u32, u32)> {
        // Keep growing until we are guaranteed to have a pending admission.
        while self.values.len() <= self.k as usize {
            if self.j_max >= 14 {
                return None;
            }
            self.grow_layer();
        }
        // By invariant 4, the last element is always active under the previous n.
        let evicted_value = self
            .values
            .pop()
            .expect("values has more than k entries, so pop() is safe");

        let (Reverse(added_value), _) = std::iter::from_fn(|| self.pending.pop())
            .find(|&(_, rank)| rank < self.values.len() as u32)
            .expect("pending is non-empty by our invariants");

        debug_assert!(added_value >= self.n);
        self.n = added_value + 1;
        // Restore invariant 4: any pending elements at the tail of values must be popped.
        while self.values.len() > self.k as usize && self.values.last().copied().unwrap() >= self.n
        {
            self.values.pop();
        }
        return Some((added_value, evicted_value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn splitmix64(seed: u64) -> u64 {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[test]
    fn smallest_j_max_picks_smallest_layer_geq_k() {
        let cases = [
            (1u32, 0u32, 4u32),
            (2, 0, 4),
            (3, 0, 4),
            (4, 0, 4),
            (5, 1, 16),
            (7, 1, 16),
            (10, 1, 16),
            (16, 1, 16),
            (17, 2, 64),
            (64, 2, 64),
            (65, 3, 256),
            (256, 3, 256),
            (257, 4, 1024),
        ];
        for (k, expected_j, expected_m) in cases {
            assert_eq!(smallest_j_max(k), expected_j, "k={k}");
            assert_eq!(1u32 << (2 * smallest_j_max(k) + 2), expected_m, "k={k}");
        }
    }

    #[test]
    fn test_reservoir() {
        for seed in [42, 123456789, 987654321] {
            let master_key = splitmix64(seed);
            for k in [1, 2, 3, 10, 100] {
                let mut r = ConsistentReservoir::new(k, k, master_key);
                for _ in 0.. {
                    let before: Vec<_> = r.reservoir().collect();
                    let Some((added, evicted)) = r.next() else {
                        break;
                    };
                    let after: Vec<_> = r.reservoir().collect();
                    let expected: Vec<_> = ConsistentPermutation::new(added, master_key)
                        .take(k as usize)
                        .collect();
                    assert_eq!(before, expected);
                    assert!(before.contains(&evicted));
                    assert!(!after.contains(&evicted));
                    assert!(!before.contains(&added));
                    assert!(after.contains(&added));
                }
            }
        }
    }
}
