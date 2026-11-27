//! Sentinel-rooted, single-cycle consistent permutations.
//! See `docs/virtual-permutation.md` for the construction and assumptions.

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

/// Ordinary noncryptographic Feistel Q, not itself the consistent map or cycle.
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

struct SingleCycle<Q> {
    ordinary: Q,
    mask: u64,
}

impl<Q: Permutation> Permutation for SingleCycle<Q> {
    #[inline]
    fn forward(&self, x: u64) -> u64 {
        self.ordinary
            .inverse(self.ordinary.forward(x).wrapping_add(1) & self.mask)
    }

    #[inline]
    fn inverse(&self, x: u64) -> u64 {
        self.ordinary
            .inverse(self.ordinary.forward(x).wrapping_sub(1) & self.mask)
    }
}

#[inline]
fn cycle(seed: u64, bits: u32) -> SingleCycle<WordPermutation> {
    SingleCycle {
        ordinary: WordPermutation::new(seed, bits),
        mask: u64::MAX >> (64 - bits),
    }
}

#[inline]
fn evaluate<P: Permutation>(mut count: u64, mut x: u64, mut layer: impl FnMut(u32) -> P) -> u64 {
    debug_assert!(count > 0 && x < count);
    let mut bits = u64::BITS - (count - 1).leading_zeros();
    while bits > 0 {
        let half = 1u64 << (bits - 1);
        let permutation = layer(bits);
        loop {
            let y = permutation.forward(x);
            if y >= count {
                x = y;
                continue;
            }
            if y >= half {
                return y;
            }
            // Recover the old input at the start of this chain, rather than
            // its old destination y, before evaluating the lower-level cycle.
            while x >= half {
                x = permutation.inverse(x);
            }
            break;
        }
        count = half;
        bits -= 1;
    }
    0
}

/// An allocation-free, per-key ordering of `0..n` preserving survivor order.
///
/// Every prefix contains distinct nodes. Appending a node inserts it somewhere
/// in the complete order; removing the last node deletes it without reordering
/// survivors. Membership must be consecutive IDs `0..n`. Replica ranks may
/// shift when membership changes.
///
/// The iterator traverses one consistent cycle from a permanent internal
/// sentinel. It does not evaluate independent rank inputs or cache a prefix.
/// `nth(r)` replays `r + 1` successors from the current position; there is no
/// constant-time absolute-rank API.
///
/// Under independent uniform ideal permutations per key and bit width, the
/// rooted order is uniform and sequential work is expected `O(k)` for `k`
/// outputs with constant-cost primitives. This is not a worst-case bound.
/// The finite 64-bit seeded, 8/16/24-round Feistel family is noncryptographic;
/// exact uniformity, independence and cryptographic security are not claimed.
/// State is four `u64` fields, with no ring, table, duplicate set or allocation.
///
/// ```
/// use consistent_choose_k::VirtualPermutation;
///
/// let seed = 0x1234_5678_9abc_def0; // normally a well-mixed hash of the key
/// let old: Vec<_> = VirtualPermutation::new(100, seed).collect();
/// let new: Vec<_> = VirtualPermutation::new(101, seed)
///     .filter(|&node| node != 100).collect();
/// assert_eq!(old, new);
/// assert_eq!(VirtualPermutation::new(100, seed).take(3).collect::<Vec<_>>(), old[..3]);
/// ```
#[derive(Clone, Debug)]
pub struct VirtualPermutation {
    seed: u64,
    count: u64,
    cursor: u64,
    remaining: u64,
}

impl VirtualPermutation {
    /// Construct an ordering of `1..=u64::MAX - 1` real nodes.
    ///
    /// Supply a well-mixed key hash, held fixed across membership and replica
    /// counts. Internal label zero is the sentinel; real node `i` has label
    /// `i + 1`. The extra sentinel requires representable internal count `n + 1`.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0` or `n == u64::MAX`.
    pub fn new(n: u64, seed: u64) -> Self {
        assert!(n > 0, "n must be at least 1");
        assert!(n < u64::MAX, "n must be at most u64::MAX - 1");
        Self {
            seed,
            count: n + 1,
            cursor: 0,
            remaining: n,
        }
    }

    /// Number of real nodes, independent of iterator position.
    pub fn n(&self) -> u64 {
        self.count - 1
    }
}

impl Iterator for VirtualPermutation {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.cursor = evaluate(self.count, self.cursor, |bits| cycle(self.seed, bits));
        debug_assert!(self.cursor > 0 && self.cursor < self.count);
        self.remaining -= 1;
        Some(self.cursor - 1)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match usize::try_from(self.remaining) {
            Ok(remaining) => (remaining, Some(remaining)),
            Err(_) => (usize::MAX, None),
        }
    }
}

impl FusedIterator for VirtualPermutation {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct CountedCycle<'a> {
        cycle: SingleCycle<WordPermutation>,
        calls: &'a Cell<[u64; 3]>,
    }

    impl Permutation for CountedCycle<'_> {
        fn forward(&self, x: u64) -> u64 {
            let mut v = self.calls.get();
            v[0] += 1;
            self.calls.set(v);
            self.cycle.forward(x)
        }
        fn inverse(&self, x: u64) -> u64 {
            let mut v = self.calls.get();
            v[1] += 1;
            self.calls.set(v);
            self.cycle.inverse(x)
        }
    }

    fn seed(i: u64) -> u64 {
        mix(i.wrapping_add(WEYL))
    }

    fn successor(count: u64, key: u64, x: u64) -> u64 {
        evaluate(count, x, |bits| cycle(key, bits))
    }

    #[test]
    fn ordinary_and_cycle_round_trips_all_widths() {
        for bits in 1..=64 {
            let mask = u64::MAX >> (64 - bits);
            for key in [0, 1, u64::MAX, seed(42)] {
                let q = WordPermutation::new(key, bits);
                let p = cycle(key, bits);
                for x in [0, 1, mask / 2, mask / 2 + 1, mask]
                    .into_iter()
                    .chain((0..64).map(|i| seed(i) & mask))
                {
                    assert!(q.forward(x) <= mask && p.forward(x) <= mask);
                    assert_eq!(q.inverse(q.forward(x)), x);
                    assert_eq!(q.forward(q.inverse(x)), x);
                    assert_eq!(p.inverse(p.forward(x)), x);
                    assert_eq!(p.forward(p.inverse(x)), x);
                }
            }
        }
    }

    #[test]
    fn conjugates_are_single_cycles() {
        for bits in 1..=9 {
            for key in 0..16 {
                let p = cycle(seed(key), bits);
                let mut seen = vec![false; 1 << bits];
                let mut x = 0;
                for _ in 0..1 << bits {
                    assert!(!seen[x as usize]);
                    seen[x as usize] = true;
                    assert_eq!(p.inverse(p.forward(x)), x);
                    x = p.forward(x);
                }
                assert_eq!(x, 0);
                assert!(seen.into_iter().all(|v| v));
            }
        }
    }

    fn project(p: &[u64], count: usize) -> Vec<u64> {
        (0..count)
            .map(|x| {
                let mut y = p[x];
                while y >= count as u64 {
                    y = p[y as usize];
                }
                y
            })
            .collect()
    }

    fn lift(lower: &[u64], p: &[u64]) -> Vec<u64> {
        let h = lower.len();
        let r = project(p, h);
        let mut inverse = vec![0; h];
        for (x, &y) in r.iter().enumerate() {
            inverse[y as usize] = x;
        }
        p.iter()
            .map(|&y| {
                if y < h as u64 {
                    lower[inverse[y as usize]]
                } else {
                    y
                }
            })
            .collect()
    }

    fn rooted_order(p: &[u64]) -> Vec<u64> {
        let mut x = 0;
        let mut order = vec![];
        for _ in 1..p.len() {
            x = p[x as usize];
            assert_ne!(x, 0, "sentinel reached early");
            order.push(x - 1);
        }
        assert_eq!(p[x as usize], 0, "cycle does not close at sentinel");
        order
    }

    #[test]
    fn complete_orders_prefixes_and_survivor_restriction() {
        for key in 0..16 {
            let key = seed(key);
            let mut previous_order = vec![];
            let mut previous_map = vec![0];
            for n in 1..=257 {
                let iter = VirtualPermutation::new(n, key);
                let order: Vec<_> = iter.clone().collect();
                let mut sorted = order.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, (0..n).collect::<Vec<_>>());
                assert_eq!(
                    order
                        .iter()
                        .copied()
                        .filter(|&node| node < n - 1)
                        .collect::<Vec<_>>(),
                    previous_order
                );
                for k in [0, 1, 2, 3, 8, n / 2, n] {
                    if k <= n {
                        assert_eq!(
                            iter.clone().take(k as usize).collect::<Vec<_>>(),
                            order[..k as usize]
                        );
                    }
                }
                for rank in [0, n / 2, n - 1] {
                    let mut replay = iter.clone();
                    assert_eq!(replay.nth(rank as usize), Some(order[rank as usize]));
                    assert_eq!(replay.next(), order.get(rank as usize + 1).copied());
                }
                let map: Vec<_> = (0..=n).map(|x| successor(n + 1, key, x)).collect();
                assert_eq!(rooted_order(&map), order);
                assert_eq!(project(&map, n as usize), previous_map);
                previous_map = map;
                previous_order = order;
            }
        }
    }

    #[test]
    fn matches_explicit_single_cycle_lifts() {
        for key in 0..8 {
            let key = seed(key);
            let mut full = vec![0];
            for bits in 1..=8 {
                let p = cycle(key, bits);
                let table: Vec<_> = (0..1 << bits).map(|x| p.forward(x)).collect();
                full = lift(&full, &table);
                for count in full.len() / 2 + 1..=full.len() {
                    let projected = project(&full, count);
                    assert_eq!(
                        (0..count as u64)
                            .map(|x| successor(count as u64, key, x))
                            .collect::<Vec<_>>(),
                        projected
                    );
                    assert_eq!(
                        VirtualPermutation::new(count as u64 - 1, key).collect::<Vec<_>>(),
                        rooted_order(&projected)
                    );
                }
            }
        }
    }

    #[test]
    fn large_boundaries_and_sentinel_overflow() {
        let mut sizes = vec![u64::MAX - 2, u64::MAX - 1];
        for bits in 1..64 {
            let power = 1u64 << bits;
            sizes.extend([power.saturating_sub(2).max(1), power - 1, power, power + 1]);
        }
        for n in sizes {
            for key in [0, 1, u64::MAX, seed(42)] {
                let k = n.min(16) as usize;
                let values: Vec<_> = VirtualPermutation::new(n, key).take(k).collect();
                assert!(values.iter().all(|&v| v < n));
                let mut distinct = values.clone();
                distinct.sort_unstable();
                distinct.dedup();
                assert_eq!(distinct.len(), k);
                if n < u64::MAX - 1 {
                    let restricted: Vec<_> = VirtualPermutation::new(n + 1, key)
                        .take(k + 1)
                        .filter(|&node| node < n)
                        .take(k)
                        .collect();
                    assert_eq!(values, restricted);
                }
            }
        }
    }

    #[test]
    fn sequential_level_subsequences_and_inverse_charging() {
        for key in 0..8 {
            for n in 1u64..=129 {
                let bits = u64::BITS - n.leading_zeros();
                let calls: Vec<_> = (0..=bits).map(|_| Cell::new([0; 3])).collect();
                let mut selected = vec![0; bits as usize + 1];
                let mut cursor = 0;
                for k in 1..=n {
                    cursor = evaluate(n + 1, cursor, |width| CountedCycle {
                        cycle: cycle(seed(key), width),
                        calls: &calls[width as usize],
                    });
                    for width in 1..=bits {
                        let [forward, inverse, _] = calls[width as usize].get();
                        assert!(
                            inverse <= forward,
                            "each upper chain is retraced at most once"
                        );
                        if width < bits {
                            selected[width as usize] += u64::from(cursor < 1 << width);
                            assert_eq!(forward, selected[width as usize]);
                        } else {
                            assert!(forward >= k && forward < 1 << bits);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn iterator_boundaries() {
        let mut p = VirtualPermutation::new(1, 0);
        assert_eq!(p.n(), 1);
        assert_eq!(p.size_hint(), (1, Some(1)));
        assert_eq!(p.clone().take(0).count(), 0);
        assert_eq!(p.next(), Some(0));
        assert_eq!(successor(p.count, p.seed, p.cursor), 0);
        assert_eq!(p.size_hint(), (0, Some(0)));
        assert_eq!(p.next(), None);
        assert_eq!(p.next(), None);
        assert_eq!(p.nth(usize::MAX), None);
        let p = VirtualPermutation::new(u64::MAX - 1, 0);
        let expected = usize::try_from(u64::MAX - 1).ok();
        assert_eq!(p.size_hint(), (expected.unwrap_or(usize::MAX), expected));
    }

    #[test]
    #[should_panic(expected = "n must be at least 1")]
    fn invalid_empty_domain() {
        VirtualPermutation::new(0, 0);
    }

    #[test]
    #[should_panic(expected = "n must be at most u64::MAX - 1")]
    fn invalid_sentinel_overflow() {
        VirtualPermutation::new(u64::MAX, 0);
    }

    fn permutations(mut values: Vec<u64>) -> Vec<Vec<u64>> {
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
        visit(&mut values, 0, &mut out);
        out
    }

    fn all_cycles(count: usize) -> Vec<Vec<u64>> {
        permutations((1..count as u64).collect())
            .into_iter()
            .map(|order| {
                let mut map = vec![0; count];
                let mut x = 0;
                for y in order {
                    map[x] = y;
                    x = y as usize;
                }
                map
            })
            .collect()
    }

    struct Table<'a>(&'a [u64]);

    impl Permutation for Table<'_> {
        fn forward(&self, x: u64) -> u64 {
            self.0[x as usize]
        }
        fn inverse(&self, x: u64) -> u64 {
            self.0.iter().position(|&y| y == x).expect("bijection") as u64
        }
    }

    #[test]
    fn exhaustive_ideal_conjugators_and_rooted_orders() {
        use std::collections::BTreeMap;

        let mut conjugates = BTreeMap::new();
        for q in permutations((0..4).collect()) {
            let p = SingleCycle {
                ordinary: Table(&q),
                mask: 3,
            };
            let table: Vec<_> = (0..4).map(|x| p.forward(x)).collect();
            *conjugates.entry(table).or_insert(0) += 1;
        }
        assert_eq!(conjugates.len(), 6);
        assert!(conjugates.values().all(|&v| v == 4));

        let p2 = [1, 0];
        let cycles8 = all_cycles(8);
        let mut counts: Vec<BTreeMap<Vec<u64>, usize>> = (1..=7).map(|_| BTreeMap::new()).collect();
        for p4 in all_cycles(4) {
            let f4 = lift(&p2, &p4);
            for p8 in &cycles8 {
                let f8 = lift(&f4, p8);
                let mut previous = vec![];
                for n in 1..=7 {
                    let order = rooted_order(&project(&f8, n + 1));
                    let mut actual = vec![];
                    let mut cursor = 0;
                    for _ in 0..n {
                        cursor = evaluate(n as u64 + 1, cursor, |bits| {
                            Table(match bits {
                                1 => &p2,
                                2 => &p4,
                                3 => p8,
                                _ => unreachable!(),
                            })
                        });
                        assert_ne!(cursor, 0);
                        actual.push(cursor - 1);
                    }
                    assert_eq!(actual, order);
                    assert_eq!(
                        order
                            .iter()
                            .copied()
                            .filter(|&v| v < n as u64 - 1)
                            .collect::<Vec<_>>(),
                        previous
                    );
                    previous = order.clone();
                    *counts[n - 1].entry(order).or_insert(0) += 1;
                }
            }
        }
        for (i, counts) in counts.iter().enumerate() {
            let factorial: usize = (1..=i + 1).product();
            assert_eq!(counts.len(), factorial);
            assert!(counts.values().all(|&v| v == 30240 / factorial));
        }
    }

    #[test]
    fn unlucky_sequential_walks_are_not_truncated() {
        struct ReverseCycle<'a> {
            mask: u64,
            calls: &'a Cell<u64>,
        }
        impl Permutation for ReverseCycle<'_> {
            fn forward(&self, x: u64) -> u64 {
                self.calls.set(self.calls.get() + 1);
                x.wrapping_sub(1) & self.mask
            }
            fn inverse(&self, x: u64) -> u64 {
                self.calls.set(self.calls.get() + 1);
                x.wrapping_add(1) & self.mask
            }
        }
        let calls = Cell::new(0);
        let factory = |bits| ReverseCycle {
            mask: (1u64 << bits) - 1,
            calls: &calls,
        };
        let first = evaluate(2049, 0, factory);
        assert_eq!(first, 2048);
        assert_eq!(evaluate(2049, first, factory), 2047);
        assert!(calls.get() > 4096);
    }

    #[test]
    #[ignore = "deterministic sequential operation-count report, not a timing CI gate"]
    fn operation_count_diagnostics() {
        println!("n,k,mean_forward,mean_inverse,mean_q_per_output,mean_levels,p50_calls,p99_calls,max_calls,max_step_calls");
        for n in [
            1,
            2,
            3,
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
            1 << 30,
            (1 << 63) - 1,
            u64::MAX - 1,
        ] {
            let mut ks = vec![1, 3, 16];
            if n <= 256 {
                ks.extend([n / 4, n]);
            }
            ks.retain(|&k| k > 0 && k <= n);
            ks.sort_unstable();
            ks.dedup();
            for k in ks {
                let calls = Cell::new([0; 3]);
                let mut totals = [0u64; 3];
                let mut samples = vec![];
                let mut max_step = 0;
                for key in 0..10_000 {
                    calls.set([0; 3]);
                    let mut cursor = 0;
                    for _ in 0..k {
                        let before = calls.get()[0] + calls.get()[1];
                        cursor = evaluate(n + 1, cursor, |bits| {
                            let mut v = calls.get();
                            v[2] += 1;
                            calls.set(v);
                            CountedCycle {
                                cycle: cycle(seed(key), bits),
                                calls: &calls,
                            }
                        });
                        assert!(cursor > 0 && cursor <= n);
                        max_step = max_step.max(calls.get()[0] + calls.get()[1] - before);
                    }
                    let v = calls.get();
                    for i in 0..3 {
                        totals[i] += v[i];
                    }
                    samples.push(v[0] + v[1]);
                }
                samples.sort_unstable();
                println!(
                    "{n},{k},{:.4},{:.4},{:.4},{:.4},{},{},{},{}",
                    totals[0] as f64 / 10_000.0,
                    totals[1] as f64 / 10_000.0,
                    2.0 * (totals[0] + totals[1]) as f64 / (10_000 * k) as f64,
                    totals[2] as f64 / 10_000.0,
                    samples[4999],
                    samples[9899],
                    samples[9999],
                    max_step
                );
            }
        }
    }
}
