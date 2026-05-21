//! A min segment tree with point-set, range-add, and "rightmost-leaf with
//! value `<= 0`" queries, all in O(log n).
//!
//! Uses a *relative offset* encoding so range-add does not need a separate
//! lazy-propagation array.
//!
//! # Encoding
//!
//! Implicit binary heap layout: root at `0`, children of node `v` at
//! `2 * v + 1` and `2 * v + 2`. Leaves live at indices
//! `[size - 1, 2 * size - 1)`; leaf `i` is at `size - 1 + i`. `size` is a
//! power of two (or zero, for an empty tree).
//!
//! Each `seg[v]` is an *offset*, not an absolute value. The true min of the
//! subtree rooted at `v` is the sum of `seg[u]` over all `u` on the path
//! from the root down to `v` (inclusive).
//!
//! # Invariant
//!
//! At every internal node `v`: `min(seg[2 * v + 1], seg[2 * v + 2]) == 0`.
//! The heavier child's offset is its excess over its sibling, and the
//! lighter sibling contributes zero.
//!
//! Because the offset at a node already accounts for every leaf beneath it,
//! a range-add `delta` over a node fully covered by the update is a single
//! `seg[v] += delta`. Partial-cover recursion calls [`MinSegTree::rebalance`]
//! on the way back up to restore the invariant.

/// See module docs.
pub struct MinSegTree {
    /// Offset values in implicit-heap layout.
    seg: Vec<i64>,
    /// Number of leaves (a power of two, or 0 for an empty tree).
    size: usize,
}

impl MinSegTree {
    /// Builds a tree whose leaves are `leaves`, padded up to the next power
    /// of two with `padding`.
    ///
    /// `padding` should be large enough that [`MinSegTree::rightmost_le_zero`]
    /// never selects a padding leaf for any sequence of operations the caller
    /// applies.
    ///
    /// Time: O(leaves.len().next_power_of_two()).
    pub fn new(leaves: &[i64], padding: i64) -> Self {
        if leaves.is_empty() {
            return Self {
                seg: Vec::new(),
                size: 0,
            };
        }
        let size = leaves.len().next_power_of_two();
        let mut seg = vec![padding; 2 * size - 1];
        let leaf_offset = size - 1;
        for (i, &v) in leaves.iter().enumerate() {
            seg[leaf_offset + i] = v;
        }
        // Bulk-build: pull common min from each pair of children up into
        // their parent, restoring `min(seg[l], seg[r]) == 0` bottom-up.
        for v in (0..leaf_offset).rev() {
            let l = 2 * v + 1;
            let r = 2 * v + 2;
            let m = seg[l].min(seg[r]);
            seg[l] -= m;
            seg[r] -= m;
            seg[v] = m;
        }
        Self { seg, size }
    }

    /// Number of leaves (including padding). Always a power of two, or `0`.
    #[allow(dead_code)]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Sets leaf `i` to `val`.
    ///
    /// Time: O(log size).
    pub fn set(&mut self, i: usize, val: i64) {
        debug_assert!(i < self.size, "leaf index out of range");
        // Pre-shift so that the bit selecting the child at each level is
        // always the top bit of `lo`; the recursion then peels bits off by
        // left-shifting once per level.
        let depth = self.size.trailing_zeros();
        let lo = i.checked_shl(usize::BITS - depth).unwrap_or(0);
        self.set_rec(0, lo, val);
    }

    /// Descends root -> leaf, reading the top bit of `lo` to pick a child and
    /// shifting `lo` left by one per level. `val` is the target value minus
    /// the path-sum of offsets at ancestors strictly above `v`, so at the
    /// leaf we can write it directly.
    fn set_rec(&mut self, v: usize, lo: usize, val: i64) {
        if v >= self.size - 1 {
            self.seg[v] = val;
            return;
        }
        let bit = lo >> (usize::BITS - 1);
        self.set_rec(2 * v + 1 + bit, lo << 1, val - self.seg[v]);
        self.rebalance(v);
    }

    /// Adds `delta` to every leaf in `[lo, size)`.
    ///
    /// Time: O(log size).
    pub fn suffix_add(&mut self, lo: usize, delta: i64) {
        if lo >= self.size || delta == 0 {
            return;
        }
        let depth = self.size.trailing_zeros();
        let lo = lo.checked_shl(usize::BITS - depth).unwrap_or(0);
        self.suffix_add_rec(0, lo, delta);
    }

    /// Descends root -> leaf, reading the top bit of `lo` to pick a child and
    /// shifting `lo` left by one per level; bumps the right sibling whole
    /// whenever `lo` lies in the left subtree, and rebalances on unwind.
    fn suffix_add_rec(&mut self, v: usize, lo: usize, delta: i64) {
        if v >= self.size - 1 {
            // The leaf at `lo` itself is in the suffix.
            self.seg[v] += delta;
            return;
        }
        let bit = lo >> (usize::BITS - 1);
        if bit == 0 {
            // Right sibling fully covered by the suffix.
            self.seg[2 * v + 2] += delta;
            self.suffix_add_rec(2 * v + 1, lo << 1, delta);
        } else {
            // Left sibling outside the suffix.
            self.suffix_add_rec(2 * v + 2, lo << 1, delta);
        }
        self.rebalance(v);
    }

    /// Returns the right-most leaf index `i` with current value `<= 0`, or
    /// `None` if no leaf is `<= 0`.
    ///
    /// Time: O(log size).
    pub fn rightmost_le_zero(&self) -> Option<usize> {
        if self.size == 0 {
            return None;
        }
        let mut acc = self.seg[0];
        if acc > 0 {
            return None;
        }
        let leaf_offset = self.size - 1;
        let mut v = 0;
        while v < leaf_offset {
            let r = 2 * v + 2;
            let r_min = acc + self.seg[r];
            if r_min <= 0 {
                acc = r_min;
                v = r;
            } else {
                let l = 2 * v + 1;
                acc += self.seg[l];
                v = l;
            }
        }
        Some(v - leaf_offset)
    }

    /// Restores `min(seg[2 * v + 1], seg[2 * v + 2]) == 0` at internal node
    /// `v` by pulling the children's common min into `v`. Caller must have
    /// finished updating `v`'s subtree first.
    fn rebalance(&mut self, v: usize) {
        let l = 2 * v + 1;
        let r = 2 * v + 2;
        let m = self.seg[l].min(self.seg[r]);
        self.seg[l] -= m;
        self.seg[r] -= m;
        self.seg[v] += m;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force model for cross-checking against the tree.
    #[derive(Clone)]
    struct Naive {
        values: Vec<i64>,
    }

    impl Naive {
        fn new(values: Vec<i64>) -> Self {
            Self { values }
        }
        fn set(&mut self, i: usize, val: i64) {
            self.values[i] = val;
        }
        fn suffix_add(&mut self, lo: usize, delta: i64) {
            for v in &mut self.values[lo..] {
                *v += delta;
            }
        }
        fn rightmost_le_zero(&self) -> Option<usize> {
            self.values.iter().rposition(|&v| v <= 0)
        }
    }

    #[test]
    fn empty() {
        let t = MinSegTree::new(&[], 0);
        assert_eq!(t.size(), 0);
        assert_eq!(t.rightmost_le_zero(), None);
    }

    #[test]
    fn single_leaf() {
        let mut t = MinSegTree::new(&[5], 1_000_000);
        assert_eq!(t.size(), 1);
        assert_eq!(t.rightmost_le_zero(), None);
        t.set(0, -1);
        assert_eq!(t.rightmost_le_zero(), Some(0));
        t.set(0, 7);
        assert_eq!(t.rightmost_le_zero(), None);
    }

    #[test]
    fn matches_naive_under_random_ops() {
        // Deterministic pseudo-random sequence of ops.
        let n = 13usize;
        let init: Vec<i64> = (0..n as i64).map(|i| (i * 7) % 11 - 3).collect();
        let mut t = MinSegTree::new(&init, 1_000_000_000);
        let mut naive = Naive::new(init);

        // A linear-congruential prng so the test is hermetic.
        let mut state: u64 = 0xdead_beef_cafe_f00d;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state
        };

        for _ in 0..2_000 {
            match next() % 3 {
                0 => {
                    let i = (next() as usize) % n;
                    let v = (next() as i64) % 21 - 10;
                    t.set(i, v);
                    naive.set(i, v);
                }
                1 => {
                    let lo = (next() as usize) % (n + 1);
                    let d = (next() as i64) % 9 - 4;
                    t.suffix_add(lo, d);
                    naive.suffix_add(lo, d);
                }
                _ => {
                    assert_eq!(t.rightmost_le_zero(), naive.rightmost_le_zero());
                }
            }
        }
        assert_eq!(t.rightmost_le_zero(), naive.rightmost_le_zero());
    }

    #[test]
    fn padding_is_not_selected() {
        // k = 3, padded internally to size 4 with C_INF-like padding.
        let mut t = MinSegTree::new(&[10, 20, 30], 1_000_000_000);
        assert_eq!(t.size(), 4);
        // Drive every real leaf to a non-positive value; the rightmost should
        // still always be a real index in `[0, 3)`, never the padding leaf 3.
        t.suffix_add(0, -100);
        assert_eq!(t.rightmost_le_zero(), Some(2));
        t.set(2, 1);
        assert_eq!(t.rightmost_le_zero(), Some(1));
        t.set(1, 1);
        assert_eq!(t.rightmost_le_zero(), Some(0));
        t.set(0, 1);
        assert_eq!(t.rightmost_le_zero(), None);
    }
}
