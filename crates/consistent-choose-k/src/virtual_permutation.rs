//! Cycle-consistent permutations, as opposed to the survivor-list consistency
//! of `ConsistentPermutation`. See `docs/virtual-permutation.md`.

use std::iter::FusedIterator;

const WEYL: u64 = 0x9E37_79B9_7F4A_7C15;

#[inline]
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

trait Permutation {
    fn forward(&self, x: u64) -> u64;
    fn inverse(&self, x: u64) -> u64;
}

/// Alternating Feistel half updates, including unequal halves for odd widths.
/// This is a noncryptographic family, not a uniform or secure PRP.
struct WordPermutation {
    key: u64,
    bits: u32,
    right_bits: u32,
    left_mask: u64,
    right_mask: u64,
}

impl WordPermutation {
    #[inline]
    fn new(seed: u64, bits: u32) -> Self {
        debug_assert!((1..=64).contains(&bits));
        let left_bits = bits / 2;
        let right_bits = bits - left_bits;
        Self {
            key: mix(seed ^ WEYL.wrapping_mul(u64::from(bits))),
            bits,
            right_bits,
            left_mask: (1u64 << left_bits) - 1,
            right_mask: (1u64 << right_bits) - 1,
        }
    }

    #[inline]
    fn round(&self, x: u64, round: u64) -> u64 {
        mix(x ^ self.key.wrapping_add(WEYL.wrapping_mul(round + 1)))
    }

    #[inline]
    fn transform_rounds<const PAIRS: u64, const INVERSE: bool>(&self, x: u64) -> u64 {
        let mut left = x >> self.right_bits;
        let mut right = x & self.right_mask;
        for i in 0..PAIRS {
            let pair = if INVERSE { PAIRS - 1 - i } else { i };
            if INVERSE {
                right ^= self.round(left, 2 * pair + 1) & self.right_mask;
                left ^= self.round(right, 2 * pair) & self.left_mask;
            } else {
                left ^= self.round(right, 2 * pair) & self.left_mask;
                right ^= self.round(left, 2 * pair + 1) & self.right_mask;
            }
        }
        (left << self.right_bits) | right
    }

    #[inline]
    fn transform<const INVERSE: bool>(&self, x: u64) -> u64 {
        // Tiny half-domains need more mixing: eight rounds had strong
        // ordered-pair bias at width three despite clean marginals.
        match self.bits {
            1 => x ^ (self.key & 1),
            2..=4 => self.transform_rounds::<12, INVERSE>(x),
            5..=7 => self.transform_rounds::<8, INVERSE>(x),
            _ => self.transform_rounds::<4, INVERSE>(x),
        }
    }
}

impl Permutation for WordPermutation {
    #[inline]
    fn forward(&self, x: u64) -> u64 {
        self.transform::<false>(x)
    }

    #[inline]
    fn inverse(&self, x: u64) -> u64 {
        self.transform::<true>(x)
    }
}

#[inline]
fn evaluate<P: Permutation>(mut n: u64, mut x: u64, mut layer: impl FnMut(u32) -> P) -> u64 {
    let mut bits = u64::BITS - (n - 1).leading_zeros();
    while bits > 0 {
        let half = 1u64 << (bits - 1);
        let permutation = layer(bits);
        loop {
            let y = permutation.forward(x);
            if y >= n {
                x = y;
                continue;
            }
            if y >= half {
                return y;
            }
            // Find the old input at the start of this Q-chain, not its old
            // output y: this applies inverse(cycle_projection(Q)) before
            // evaluating the smaller consistent permutation.
            while x >= half {
                x = permutation.inverse(x);
            }
            break;
        }
        n = half;
        bits -= 1;
    }
    0
}

/// An allocation-free, per-key permutation of `0..n` with stable replica slots.
///
/// For fixed `(n, seed)`, taking `k` items gives `k` distinct nodes and is a
/// prefix of every longer selection. Appending node `n` changes at most one
/// existing slot, and that slot changes to `n`; removing the last node changes
/// only its slot (among slots that still exist). Membership must be a prefix
/// of consecutive IDs.
///
/// **Not a drop-in replacement for [`crate::ConsistentPermutation`]:** this
/// preserves cycle projections, not survivor list order. Removing a node from
/// the output list does not generally produce the smaller permutation.
///
/// Uniform, independent permutations per key and bit width give uniform
/// selections and expected `O(k)` evaluation with constant-cost forward/inverse
/// primitives. The implemented 64-bit seeded, 8/16/24-round Feistel family is
/// only a practical noncryptographic approximation to that ideal. Neither
/// exact uniformity, cryptographic security, nor worst-case `O(k)` is claimed.
/// State is constant size; no rings, permutation tables, or duplicate sets
/// are constructed. Walks have no artificial retry limit.
///
/// ```
/// use consistent_choose_k::VirtualPermutation;
///
/// let seed = 0x1234_5678_9abc_def0; // normally a well-mixed hash of the key
/// let selection = VirtualPermutation::new(100, seed);
/// let third = selection.replica_at(2);
/// assert_eq!(selection.clone().nth(2), Some(third));
/// let replicas: Vec<_> = selection.take(3).collect();
/// assert_eq!(replicas.len(), 3);
/// ```
#[derive(Clone, Debug)]
pub struct VirtualPermutation {
    seed: u64,
    n: u64,
    next: u64,
}

impl VirtualPermutation {
    /// Construct a permutation for `1..=u64::MAX` nodes.
    ///
    /// As with [`crate::ConsistentPermutation::new`], supply a well-mixed
    /// 64-bit hash of the key. Width/round domain separation is internal and
    /// independent of `n` and of the requested replica count.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub fn new(n: u64, seed: u64) -> Self {
        assert!(n > 0, "n must be at least 1");
        Self { seed, n, next: 0 }
    }

    /// Universe size, independent of the iterator's current position.
    pub fn n(&self) -> u64 {
        self.n
    }

    /// Evaluate an absolute zero-based replica slot without advancing the
    /// iterator. Expected constant work under ideal independent permutations;
    /// no worst-case constant-time guarantee.
    ///
    /// # Panics
    ///
    /// Panics if `slot >= self.n()`.
    pub fn replica_at(&self, slot: u64) -> u64 {
        assert!(slot < self.n, "replica slot must be less than n");
        evaluate(self.n, slot, |bits| WordPermutation::new(self.seed, bits))
    }
}

impl Iterator for VirtualPermutation {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.n {
            return None;
        }
        let value = self.replica_at(self.next);
        self.next += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match usize::try_from(self.n - self.next) {
            Ok(remaining) => (remaining, Some(remaining)),
            Err(_) => (usize::MAX, None),
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.next = self.next.saturating_add(n as u64).min(self.n);
        self.next()
    }
}

impl FusedIterator for VirtualPermutation {}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed(i: u64) -> u64 {
        mix(i.wrapping_add(WEYL))
    }

    #[test]
    fn word_round_trips_all_widths() {
        for bits in 1..=64 {
            let mask = u64::MAX >> (64 - bits);
            for key in [0, 1, u64::MAX, seed(42)] {
                let p = WordPermutation::new(key, bits);
                for x in [0, 1, mask / 2, mask / 2 + 1, mask]
                    .into_iter()
                    .chain((0..128).map(|i| seed(i) & mask))
                {
                    let y = p.forward(x);
                    assert_eq!(y & !mask, 0);
                    assert_eq!(p.inverse(y), x, "bits={bits} key={key} x={x}");
                    assert_eq!(p.forward(p.inverse(x)), x);
                }
            }
        }
    }

    #[test]
    fn word_exhaustive_small_domains() {
        for bits in 1..=10 {
            for key in 0..16 {
                let p = WordPermutation::new(seed(key), bits);
                let mut outputs: Vec<_> = (0..1 << bits)
                    .map(|x| {
                        let y = p.forward(x);
                        assert_eq!(p.inverse(y), x);
                        y
                    })
                    .collect();
                outputs.sort_unstable();
                assert_eq!(outputs, (0..1 << bits).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn full_permutations_prefixes_and_membership() {
        for key in 0..32 {
            let key = seed(key);
            let mut previous = vec![];
            for n in 1..=257 {
                let permutation = VirtualPermutation::new(n, key);
                let values: Vec<_> = permutation.clone().collect();
                let mut sorted = values.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, (0..n).collect::<Vec<_>>());
                for k in [0, 1, 2, 3, 8, n / 2, n] {
                    if k <= n {
                        assert_eq!(
                            permutation.clone().take(k as usize).collect::<Vec<_>>(),
                            values[..k as usize]
                        );
                    }
                }
                let mut changed = 0;
                for (r, old) in previous.iter().enumerate() {
                    assert_eq!(permutation.replica_at(r as u64), values[r]);
                    if *old != values[r] {
                        assert_eq!(values[r], n - 1);
                        changed += 1;
                    }
                    // Cycle-deleting the new node recovers every old slot.
                    let projected = if values[r] == n - 1 {
                        values[n as usize - 1]
                    } else {
                        values[r]
                    };
                    assert_eq!(*old, projected);
                }
                assert!(changed <= 1);
                previous = values;
            }
        }
    }

    #[test]
    fn large_domains_and_power_boundaries() {
        for bits in 1..64 {
            let half = 1u64 << bits;
            for n in [half - 1, half, half + 1, u64::MAX - 1] {
                for key in [0, 1, u64::MAX, seed(123)] {
                    let p = VirtualPermutation::new(n, key);
                    let larger = VirtualPermutation::new(n + 1, key);
                    for r in [0, n / 2, n - 1] {
                        let old = p.replica_at(r);
                        let new = larger.replica_at(r);
                        assert!(old < n && new <= n);
                        assert!(new == old || new == n);
                        assert_eq!(old, if new == n { larger.replica_at(n) } else { new });
                    }
                }
            }
        }
        let mut p = VirtualPermutation::new(u64::MAX, seed(9));
        p.next = u64::MAX - 1;
        assert!(p.next().is_some());
        assert_eq!(p.next(), None);
        assert_eq!(p.next(), None);
    }

    #[test]
    fn iterator_boundaries() {
        let mut p = VirtualPermutation::new(1, 0);
        assert_eq!(p.n(), 1);
        assert_eq!(p.size_hint(), (1, Some(1)));
        assert_eq!(p.clone().take(0).count(), 0);
        assert_eq!(p.next(), Some(0));
        assert_eq!(p.size_hint(), (0, Some(0)));
        assert_eq!(p.next(), None);
        assert_eq!(p.replica_at(0), 0);
        assert_eq!(p.nth(usize::MAX), None);
        let mut p = VirtualPermutation::new(100, seed(4));
        assert_eq!(p.nth(12), Some(p.replica_at(12)));
        assert_eq!(p.next(), Some(p.replica_at(13)));
        assert_eq!(p.nth(usize::MAX), None);
        assert_eq!(p.next(), None);
    }

    #[test]
    #[should_panic(expected = "n must be at least 1")]
    fn invalid_empty_domain() {
        VirtualPermutation::new(0, 0);
    }

    #[test]
    #[should_panic(expected = "replica slot must be less than n")]
    fn invalid_slot() {
        VirtualPermutation::new(10, 0).replica_at(10);
    }

    #[test]
    fn long_cycles_are_not_truncated() {
        use std::cell::Cell;

        struct Rotation<'a> {
            mask: u64,
            calls: &'a Cell<u64>,
        }
        impl Permutation for Rotation<'_> {
            fn forward(&self, x: u64) -> u64 {
                self.calls.set(self.calls.get() + 1);
                x.wrapping_add(1) & self.mask
            }
            fn inverse(&self, x: u64) -> u64 {
                self.calls.set(self.calls.get() + 1);
                x.wrapping_sub(1) & self.mask
            }
        }
        // Every dyadic lift of these ascending cycles is the same ascending
        // cycle. A last-slot query just above a half boundary takes long walks.
        let calls = Cell::new(0);
        let result = evaluate(2049, 2048, |bits| Rotation {
            mask: (1 << bits) - 1,
            calls: &calls,
        });
        assert_eq!(result, 0);
        assert!(calls.get() > 4096);
    }

    // Deliberately explicit tables only in the oracle, never in the evaluator.
    fn project(p: &[u64], n: usize) -> Vec<u64> {
        (0..n)
            .map(|x| {
                let mut y = p[x];
                while y >= n as u64 {
                    y = p[y as usize];
                }
                y
            })
            .collect()
    }

    fn lift(lower: &[u64], q: &[u64]) -> Vec<u64> {
        let h = lower.len();
        let r = project(q, h);
        let mut inverse = vec![0; h];
        for (x, &y) in r.iter().enumerate() {
            inverse[y as usize] = x;
        }
        q.iter()
            .map(|&y| {
                if y < h as u64 {
                    lower[inverse[y as usize]]
                } else {
                    y
                }
            })
            .collect()
    }

    #[test]
    fn matches_explicit_lift_and_cycle_projection() {
        for key in 0..32 {
            let key = seed(key);
            let mut full = vec![0];
            for bits in 1..=8 {
                let p = WordPermutation::new(key, bits);
                let q: Vec<_> = (0..1 << bits).map(|x| p.forward(x)).collect();
                full = lift(&full, &q);
                for n in (full.len() / 2 + 1)..=full.len() {
                    let actual: Vec<_> = VirtualPermutation::new(n as u64, key).collect();
                    assert_eq!(actual, project(&full, n), "bits={bits} n={n}");
                }
            }
        }
    }

    fn permutations(n: usize) -> Vec<Vec<u64>> {
        fn visit(values: &mut [u64], start: usize, out: &mut Vec<Vec<u64>>) {
            if start == values.len() {
                out.push(values.to_vec());
            } else {
                for i in start..values.len() {
                    values.swap(start, i);
                    visit(values, start + 1, out);
                    values.swap(start, i);
                }
            }
        }
        let mut out = vec![];
        visit(&mut (0..n as u64).collect::<Vec<_>>(), 0, &mut out);
        out
    }

    #[test]
    fn ideal_uniform_lift_fibers() {
        use std::collections::BTreeMap;

        // Every ideal Q_1,Q_2 combination: F_4 has equal multiplicities.
        let mut counts = BTreeMap::new();
        for lower in permutations(2) {
            for q in permutations(4) {
                *counts.entry(lift(&lower, &q)).or_insert(0) += 1;
            }
        }
        assert_eq!(counts.len(), 24);
        assert!(counts.values().all(|&count| count == 2));

        // Fix F_4; every extension through each intermediate n occurs 4! times
        // at n=8, and equally often for n=5,6,7 after cycle projection.
        let lower = vec![2, 0, 3, 1];
        let mut counts: Vec<BTreeMap<Vec<u64>, usize>> = (5..=8).map(|_| BTreeMap::new()).collect();
        for q in permutations(8) {
            let full = lift(&lower, &q);
            assert_eq!(project(&full, 4), lower);
            for (i, n) in (5..=8).enumerate() {
                *counts[i].entry(project(&full, n)).or_insert(0) += 1;
            }
        }
        for (i, count) in counts.iter().enumerate() {
            let expected_distinct: usize = (5..=i + 5).product();
            assert_eq!(count.len(), expected_distinct);
            assert!(count.values().all(|&v| v == 40320 / expected_distinct));
        }
    }

    #[test]
    #[ignore = "deterministic operation-count report; not a timing or statistical CI gate"]
    fn operation_count_diagnostics() {
        use std::{cell::Cell, rc::Rc};

        struct Counted {
            permutation: WordPermutation,
            calls: Rc<Cell<[u64; 3]>>,
        }
        impl Permutation for Counted {
            fn forward(&self, x: u64) -> u64 {
                let mut calls = self.calls.get();
                calls[0] += 1;
                self.calls.set(calls);
                self.permutation.forward(x)
            }
            fn inverse(&self, x: u64) -> u64 {
                let mut calls = self.calls.get();
                calls[1] += 1;
                self.calls.set(calls);
                self.permutation.inverse(x)
            }
        }

        println!("n,slot,mean_forward,mean_inverse,mean_levels,p50_calls,p99_calls,max_calls");
        for n in [
            1,
            7,
            8,
            9,
            255,
            256,
            257,
            1023,
            1024,
            1025,
            65535,
            65536,
            65537,
            (1 << 30) - 1,
            1 << 30,
            (1 << 63) + 1,
            u64::MAX,
        ] {
            for slot in [0, n / 2, n - 1] {
                let calls = Rc::new(Cell::new([0; 3]));
                let mut totals = [0u64; 3];
                let mut samples = vec![];
                for key in 0..10_000 {
                    calls.set([0; 3]);
                    let result = evaluate(n, slot, |bits| {
                        let mut counts = calls.get();
                        counts[2] += 1;
                        calls.set(counts);
                        Counted {
                            permutation: WordPermutation::new(seed(key), bits),
                            calls: Rc::clone(&calls),
                        }
                    });
                    assert!(result < n);
                    let counts = calls.get();
                    for i in 0..3 {
                        totals[i] += counts[i];
                    }
                    samples.push(counts[0] + counts[1]);
                }
                samples.sort_unstable();
                println!(
                    "{n},{slot},{:.4},{:.4},{:.4},{},{},{}",
                    totals[0] as f64 / 10_000.0,
                    totals[1] as f64 / 10_000.0,
                    totals[2] as f64 / 10_000.0,
                    samples[4999],
                    samples[9899],
                    samples[9999],
                );
            }
        }
    }
}
