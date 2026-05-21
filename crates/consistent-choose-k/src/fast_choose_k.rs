use crate::min_seg_tree::MinSegTree;
use crate::{ConsistentHasher, ManySeqBuilder};

/// "Block count" sentinel for slots that can never be selected (e.g. their
/// sequence is exhausted). Chosen well above any realistic true count so that
/// the lazy `-1` updates applied by `shrink_n` cannot drive it down to zero
/// in any reasonable workload, and so that padding leaves in [`MinSegTree`]
/// are never selected.
const C_INF: i64 = 1_000_000_000_000;

/// Fast variant of [`crate::ConsistentChooseKHasher`] specialized for
/// repeated `shrink_n` calls at fixed `k`.
///
/// # Invariants
///
/// For each position `i` in `0..k`:
/// * `samples[i]` is the `i`-th smallest of the `k` currently chosen samples;
///   `samples` is kept sorted ascending.
/// * `next[i]` is a candidate sample from hash sequence `i` (sequence id is
///   bound to **position**, not to a slot's contents — `next` does not shift
///   when `samples` does). Initially set to `get_sample(seq=i, n=samples[i])`,
///   it may become stale relative to `samples[i]` after subsequent shifts.
/// * `c[i]` (stored in the segment-tree leaf for position `i`) tracks the
///   number of left neighbours that block position `i` as a replacement
///   candidate. The true value is `#{ j < i : samples[j] >= next[i] }`;
///   `c[i] == 0` means `samples[i - 1] < next[i]`, i.e. `next[i]` could be
///   inserted right now between `samples[i - 1]` and `samples[i]`.
///
/// # Lazy maintenance
///
/// On `shrink_n` we insert a new sample at position `chosen_i`, shifting
/// `samples[chosen_i..k - 1]` right (dropping the old max). `next` does not
/// shift; only `next[chosen_i]` is rewritten to `get_sample(chosen_i,
/// new_sample)`. The true delta to `c[j]` (for `j > chosen_i`, with `next[j]`
/// unchanged) is `0` or `-1`: it is `-1` exactly when
/// `new_sample < next[j] <= samples_old[j - 1]`. We do not check this
/// per-slot; instead we apply a blanket `-1` to `c[chosen_i + 1..k]`. This
/// can under-count the true `c[j]` by up to `1` per shrink, but never
/// over-counts, so an OST descent always yields a candidate whose tracked
/// `c` is `<= 0`.
///
/// On descent we verify the candidate against two staleness cases:
/// * **Upward stale** (`next[i] >= samples[i]`): `samples[i]` has decreased
///   via shifts at lower positions since `next[i]` was last set, so the
///   stored candidate is no longer below `samples[i]`. We refresh
///   `next[i] = get_sample(i, samples[i])` and recompute `c[i]`.
/// * **Downward stale** (`next[i] <= samples[i - 1]`): the lazy `-1` made
///   `c[i]` look unblocked but `next[i]` would actually fit at some position
///   `< i`. We correct `c[i]` upward via a binary search over `samples[..i]`.
///
/// The standard `shrink_n` performs roughly `k / 2` `get_sample` calls on
/// average; this variant performs `O(1)` `get_sample` calls per `shrink_n`
/// plus `O(log k)` segment-tree work (amortized, plus the binary-search
/// corrections).
pub struct ConsistentChooseKFastHasher<H: ManySeqBuilder> {
    builder: H,
    n: usize,
    /// Samples sorted ascending; `samples[i]` is the `i`-th smallest sample.
    samples: Vec<usize>,
    /// `next[i]` is sequence `i`'s candidate for position `i`. The index `i`
    /// is bound to the position (= sequence id); `next` does not shift when
    /// `samples` does.
    next: Vec<Option<usize>>,
    /// Per-slot block counts `c[i]`. See struct-level docs for definition.
    /// Empty when `k == 0`.
    tree: MinSegTree,
}

impl<H: ManySeqBuilder> ConsistentChooseKFastHasher<H> {
    /// Create a new instance for `n` nodes with `k = 0` samples.
    ///
    /// Time: O(1)
    pub fn new(builder: H, n: usize) -> Self {
        Self {
            builder,
            n,
            samples: Vec::new(),
            next: Vec::new(),
            tree: MinSegTree::new(&[], C_INF),
        }
    }

    /// Create with the choose-k set for `k` out of `n` nodes pre-built.
    ///
    /// Uses the same bubble construction as
    /// [`crate::ConsistentChooseKHasher::new_with_k`] to populate `samples`,
    /// then initializes the `next` values and per-slot block counts and
    /// builds the segment tree.
    pub fn new_with_k(builder: H, n: usize, k: usize) -> Self {
        assert!(n >= k, "n must be at least k");
        let mut this = Self::new(builder, n);
        // Bubble construction (identical to the standard hasher).
        let mut samples: Vec<usize> = Vec::with_capacity(k);
        for i in 0..k {
            samples.push(this.get_sample(i, n).expect("must not fail"));
        }
        for i in (0..k).rev() {
            let s = samples[0..=i].iter().copied().max().expect("non-empty");
            samples[i] = s;
            #[allow(clippy::needless_range_loop)]
            for j in 0..i {
                if samples[j] == s {
                    samples[j] = this.get_sample(j, s).expect("must not fail");
                }
            }
        }
        this.samples = samples;
        this.rebuild_from_samples();
        this
    }

    /// Returns the `k` underlying samples in increasing order.
    pub fn samples(&self) -> &[usize] {
        &self.samples
    }

    /// Returns the current universe size.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the current sample count.
    pub fn k(&self) -> usize {
        self.samples.len()
    }

    /// Grow the sample set by one element. Returns the index at which the
    /// new element was inserted in the sorted samples list.
    ///
    /// Time: O(k).
    ///
    /// Panics if `k == n`.
    pub fn grow_k(&mut self) -> usize {
        assert!(self.samples.len() < self.n, "cannot grow: k must be less than n");
        let k = self.samples.len();
        let sk = self
            .get_sample(k, self.n)
            .expect("sample sequence must not be exhausted");
        let idx = if let Some(last) = self.samples.last().copied() {
            if last < sk {
                self.samples.push(sk);
                k
            } else {
                let i = self.grow_k_cascade(last);
                self.samples.push(last);
                i
            }
        } else {
            self.samples.push(sk);
            0
        };
        // The cascade may have touched samples in `[idx, k]`; rebuilding from
        // scratch is O(new_k) = O(k + 1), the same asymptotic cost as the
        // standard hasher's `grow_k`, so we don't bother updating in place.
        self.rebuild_from_samples();
        idx
    }

    /// Mirrors the standard hasher's `shrink_n_inner`: walks `samples` from the
    /// top down, replacing each entry `>= n` with a fresh candidate from the
    /// current sequence (chained against the smaller neighbour). Returns the
    /// index at which the new (lower) sample lands.
    fn grow_k_cascade(&mut self, mut n: usize) -> usize {
        for i in (0..self.samples.len()).rev() {
            if self.samples[i] < n {
                return i + 1;
            }
            let si = self
                .get_sample(i, n)
                .expect("sample sequence must not be exhausted");
            if i > 0 && self.samples[i - 1] > si {
                self.samples[i] = self.samples[i - 1];
            } else {
                self.samples[i] = si;
            }
            n = self.samples[i];
        }
        0
    }

    /// Recomputes `next` and the segment tree from the current `samples`.
    /// `samples` must already be sorted ascending.
    fn rebuild_from_samples(&mut self) {
        let k = self.samples.len();
        let mut next: Vec<Option<usize>> = Vec::with_capacity(k);
        let mut c: Vec<i64> = Vec::with_capacity(k);
        for i in 0..k {
            let nv = self.get_sample(i, self.samples[i]);
            next.push(nv);
            let ci = match nv {
                Some(v) => (i - self.samples[..i].partition_point(|&s| s < v)) as i64,
                None => C_INF,
            };
            c.push(ci);
        }
        self.next = next;
        self.tree = MinSegTree::new(&c, C_INF);
    }

    /// Decrements `n` to the current largest sample and replaces it with the
    /// next valid sample. Returns the index at which the new sample was
    /// inserted in the sorted samples list.
    ///
    /// Panics if `n <= k` or if no replacement can be found (i.e. every
    /// sequence whose slot would be a candidate is exhausted).
    pub fn shrink_n(&mut self) -> usize {
        let k = self.samples.len();
        assert!(self.n > k, "cannot shrink: n must be greater than k");
        assert!(k > 0, "cannot shrink: samples must not be empty");
        self.n = *self.samples.last().expect("k > 0");

        // Find the right-most slot whose tracked `c[i]` is `<= 0`, verifying
        // each candidate against the true definition and correcting on miss.
        let chosen_i = loop {
            let i = self
                .tree
                .rightmost_le_zero()
                .expect("at least one slot must be selectable");
            let next_i = match self.next[i] {
                None => {
                    // Sequence `i` is exhausted; never a valid candidate.
                    self.tree.set(i, C_INF);
                    continue;
                }
                Some(v) => v,
            };
            // Stale upward: `next[i]` may be >= `samples[i]` if `samples[i]`
            // has decreased via shifts at lower positions since `next[i]` was
            // last set. Refresh slot `i` against its current `samples[i]`.
            if next_i >= self.samples[i] {
                self.refresh_slot(i);
                continue;
            }
            if i == 0 || self.samples[i - 1] < next_i {
                break i;
            }
            // Stale (under-counted) `c[i]`: recompute the true value via a
            // binary search on `samples[..i]`.
            let lb = self.lower_bound(next_i, i);
            self.tree.set(i, (i - lb) as i64);
        };
        let new_sample = self.next[chosen_i].expect("verified Some above");
        // Lazy bulk `-1` over the suffix `(chosen_i, k)`: every slot in this
        // range gains the freshly-inserted sample as a new left neighbour. The
        // true delta is `-1` exactly when `new_sample < next[j] <= samples_old[j - 1]`
        // and `0` otherwise; applying `-1` always under-counts and is corrected
        // lazily on the next descent.
        self.tree.suffix_add(chosen_i + 1, -1);
        // Insert at `chosen_i`, then drop the old largest at position `k`.
        // Samples shift; `next` does NOT shift — `next[i]` is bound to sequence
        // id `i`, i.e. to position, not to the slot's contents. After the
        // shift, `refresh_slot(chosen_i)` re-derives `next[chosen_i]` from the
        // freshly inserted `samples[chosen_i] == new_sample` and writes the
        // matching `c[chosen_i]` into the tree.
        self.samples.pop();
        self.samples.insert(chosen_i, new_sample);
        self.refresh_slot(chosen_i);
        chosen_i
    }

    /// Re-derives `next[i]` from `samples[i]` and writes the matching `c[i]`
    /// into the segment tree.
    fn refresh_slot(&mut self, i: usize) {
        let refreshed = self.get_sample(i, self.samples[i]);
        self.next[i] = refreshed;
        let new_c = match refreshed {
            Some(v) => (i - self.lower_bound(v, i)) as i64,
            None => C_INF,
        };
        self.tree.set(i, new_c);
    }

    fn get_sample(&self, seq: usize, n: usize) -> Option<usize> {
        if n <= seq {
            return None;
        }
        ConsistentHasher::new(self.builder.seq_builder(seq))
            .into_prev(n - seq)
            .map(|pos| pos + seq)
    }

    /// First index `j` in `[0, upto)` with `samples[j] >= value`. Returns
    /// `upto` if no such index exists.
    fn lower_bound(&self, value: usize, upto: usize) -> usize {
        self.samples[..upto].partition_point(|&s| s < value)
    }
}

impl<H: ManySeqBuilder> Iterator for ConsistentChooseKFastHasher<H> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.samples.len() >= self.n {
            return None;
        }
        let idx = self.grow_k();
        Some(self.samples[idx])
    }
}

#[cfg(test)]
mod tests {
    use std::hash::{DefaultHasher, Hash};

    use super::*;
    use crate::ConsistentChooseKHasher;

    fn hasher_for_key(key: u64) -> DefaultHasher {
        let mut hasher = DefaultHasher::default();
        key.hash(&mut hasher);
        hasher
    }

    #[test]
    fn test_fast_new_with_k_matches_standard() {
        for key in 0..50 {
            for n in 1..30 {
                for k in 0..=n {
                    let h = hasher_for_key(key);
                    let expected =
                        ConsistentChooseKHasher::new_with_k(h.clone(), n, k).into_samples();
                    let actual = ConsistentChooseKFastHasher::new_with_k(h, n, k)
                        .samples()
                        .to_vec();
                    assert_eq!(actual, expected, "key={key} n={n} k={k}");
                }
            }
        }
    }

    #[test]
    fn test_fast_shrink_n_matches_standard() {
        for key in 0..50 {
            for k in 1..10 {
                for n in k + 1..30 {
                    let h = hasher_for_key(key);
                    let mut fast = ConsistentChooseKFastHasher::new_with_k(h.clone(), n, k);
                    let mut standard = ConsistentChooseKHasher::new_with_k(h, n, k);
                    while *standard.samples().last().unwrap() > k {
                        let standard_idx = standard.shrink_n();
                        let fast_idx = fast.shrink_n();
                        assert_eq!(
                            fast_idx, standard_idx,
                            "key={key} n={n} k={k}: returned index mismatch"
                        );
                        assert_eq!(
                            fast.samples(),
                            standard.samples(),
                            "key={key} n={n} k={k}: samples mismatch after shrink"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_fast_shrink_n_preserves_invariants() {
        // Even if the fast variant diverges from the standard one in the
        // future, the choose-k invariants must hold.
        for key in 0..50 {
            for k in 1..10 {
                for n in k + 1..30 {
                    let h = hasher_for_key(key);
                    let mut fast = ConsistentChooseKFastHasher::new_with_k(h, n, k);
                    while *fast.samples().last().unwrap() > k {
                        fast.shrink_n();
                        let s = fast.samples();
                        assert_eq!(s.len(), k, "k must be preserved");
                        for w in s.windows(2) {
                            assert!(w[0] < w[1], "samples must be strictly sorted: {s:?}");
                        }
                        assert!(*s.last().unwrap() < fast.n, "all samples must be < n");
                    }
                }
            }
        }
    }

    #[test]
    fn test_fast_grow_k_matches_standard() {
        for key in 0..50 {
            for n in 1..30 {
                let mut fast = ConsistentChooseKFastHasher::new(hasher_for_key(key), n);
                let mut standard = ConsistentChooseKHasher::new(hasher_for_key(key), n);
                for _ in 0..n {
                    let fast_idx = fast.grow_k();
                    let standard_idx = standard.grow_k();
                    assert_eq!(
                        fast_idx, standard_idx,
                        "key={key} n={n}: grow_k returned index mismatch"
                    );
                    assert_eq!(
                        fast.samples(),
                        standard.samples(),
                        "key={key} n={n}: samples mismatch after grow_k"
                    );
                }
            }
        }
    }

    #[test]
    fn test_fast_iterator_matches_standard() {
        for key in 0..50 {
            for n in 1..30 {
                let fast: Vec<usize> = ConsistentChooseKFastHasher::new(hasher_for_key(key), n)
                    .collect();
                let standard: Vec<usize> =
                    ConsistentChooseKHasher::new(hasher_for_key(key), n).collect();
                assert_eq!(fast, standard, "key={key} n={n}: iterator order mismatch");
                assert_eq!(fast.len(), n, "key={key} n={n}: iterator length");
            }
        }
    }

    #[test]
    fn test_fast_grow_then_shrink_roundtrip() {
        // After growing all the way and then shrinking back, the surviving
        // samples must still match the choose-k semantics of the standard
        // hasher.
        for key in 0..30 {
            for n in 2..15 {
                for k in 1..n {
                    let mut fast = ConsistentChooseKFastHasher::new(hasher_for_key(key), n);
                    for _ in 0..k {
                        fast.grow_k();
                    }
                    let standard = ConsistentChooseKHasher::new_with_k(
                        hasher_for_key(key),
                        n,
                        k,
                    );
                    assert_eq!(
                        fast.samples(),
                        standard.samples(),
                        "key={key} n={n} k={k}: grow_k built unexpected samples"
                    );
                }
            }
        }
    }
}
