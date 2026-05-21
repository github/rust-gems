//! A compact min segment tree with the same API as [`crate::min_seg_tree::MinSegTree`]
//! but storing only `n` entries instead of `2n - 1`.
//!
//! # Idea
//!
//! In the relative-offset encoding used by [`crate::min_seg_tree::MinSegTree`],
//! every internal node `v` satisfies `min(seg[left], seg[right]) == 0`, so one
//! of the two children carries no information beyond "I am the zero side".
//! We drop the zero sibling entirely and store, per sibling pair, just
//! `(heavy_offset, side_bit)`:
//! - `heavy_offset: i64` (always `>= 0`): the offset of the heavier sibling
//!   (the lighter one is implicitly `0`).
//! - `side: bool`: `false` = left child is heavy, `true` = right child is heavy.
//!
//! Plus one extra entry for the root (which has no sibling and hence no side):
//! it holds the actual min over all leaves.
//!
//! For a tree with `n` leaves (a power of two), this gives `1 + (n - 1) = n`
//! compact entries.
//!
//! # Layout
//!
//! * `val[0]`: the root's offset (= the true min over all leaves; the only
//!   entry that can be negative).
//! * For `i in 1..n`: `val[i]` packs the sibling pair's `(heavy_offset, side)`
//!   into a single `i64`: the sign bit (bit 63) is the `side` (`0` = left
//!   heavy, `1` = right heavy), and the low 63 bits are `heavy_offset`
//!   (`>= 0`, well within `i64::MAX / 2` for any realistic workload).
//!   That pair lives under original heap-layout node `i - 1`, consisting of
//!   original heap indices `2 * (i - 1) + 1` and `2 * (i - 1) + 2`.
//!
//! Compact pair indices form their own implicit-heap binary tree: the
//! children pairs of pair `i` (covering the two original subtrees) sit at
//! `2 * i` and `2 * i + 1`. Pairs in `[n/2, n)` are *leaf pairs* — their two
//! original children are leaves of the original tree. Pairs in `[1, n/2)` are
//! *internal pairs*.
//!
//! # Decoding a leaf
//!
//! To recover `actual_min(leaf i)`, start with `val[0]` and walk down,
//! consulting one pair per level. At each level, descending to the
//! `bit`-side child adds `heavy_offset(pair)` to the running sum iff
//! `bit == side(pair)`; otherwise nothing is added (that child is on the
//! zero side).

/// See module docs.
#[allow(dead_code)]
pub struct CompactMinSegTree {
    /// `val[0]` is the root's offset (true min over all leaves; may be
    /// negative). `val[i]` for `i in 1..size` packs `(side, heavy_offset)`:
    /// the sign bit is the side flag (negative = right heavy, non-negative =
    /// left heavy), the low 63 bits are the non-negative heavy offset.
    val: Vec<i64>,
    /// Number of leaves (a power of two, or 0).
    size: usize,
}

/// Sign-bit mask used to encode the side flag inside a packed pair entry.
const SIDE_BIT: i64 = i64::MIN;
/// Mask of the low 63 bits — extracts the `heavy_offset` from a packed entry.
const OFFSET_MASK: i64 = i64::MAX;

impl CompactMinSegTree {
    /// Builds a compact tree whose leaves are `leaves`, padded up to the
    /// next power of two with `padding`.
    ///
    /// Time: O(leaves.len().next_power_of_two()).
    #[allow(dead_code)]
    pub fn new(leaves: &[i64], padding: i64) -> Self {
        if leaves.is_empty() {
            return Self {
                val: Vec::new(),
                size: 0,
            };
        }
        let size = leaves.len().next_power_of_two();
        let mut val = vec![0i64; size];
        if size == 1 {
            val[0] = leaves[0];
            return Self { val, size };
        }
        // Build the full relative-encoded segment tree, then collapse it.
        let mut seg = vec![padding; 2 * size - 1];
        let leaf_offset = size - 1;
        for (i, &v) in leaves.iter().enumerate() {
            seg[leaf_offset + i] = v;
        }
        for v in (0..leaf_offset).rev() {
            let l = 2 * v + 1;
            let r = 2 * v + 2;
            let m = seg[l].min(seg[r]);
            seg[l] -= m;
            seg[r] -= m;
            seg[v] = m;
        }
        val[0] = seg[0];
        for p in 0..size - 1 {
            let l = 2 * p + 1;
            let r = 2 * p + 2;
            val[p + 1] = if seg[l] >= seg[r] {
                seg[l]
            } else {
                seg[r] | SIDE_BIT
            };
        }
        Self { val, size }
    }

    /// Number of leaves (including padding). Always a power of two, or `0`.
    #[allow(dead_code)]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Sets leaf `i` to `val`.
    ///
    /// Time: O(log size).
    #[allow(dead_code)]
    pub fn set(&mut self, i: usize, val: i64) {
        debug_assert!(i < self.size, "leaf index out of range");
        if self.size == 1 {
            self.val[0] = val;
            return;
        }
        // Walk down computing the path-sum of offsets from the root. At each
        // internal pair, we add the pair's heavy offset to `acc` iff the
        // path bit matches the heavy side; otherwise the path lies on the
        // implicit-zero sibling and contributes nothing.
        let depth = self.size.trailing_zeros();
        let mut acc = self.val[0];
        let mut p = 1;
        for d in (1..depth).rev() {
            let bit = (i >> d) & 1;
            let packed = self.val[p];
            // `(packed < 0) == (bit == 1)`: heavy side matches the path bit.
            if (packed < 0) == (bit == 1) {
                acc += packed & OFFSET_MASK;
            }
            p = 2 * p + bit;
        }
        // Leaf pair. The "other leaf" keeps its current offset, which is 0
        // unless that other leaf was the heavy side.
        let leaf_is_right = i & 1 == 1;
        let packed = self.val[p];
        let other_off = if (packed < 0) == leaf_is_right {
            // Old heavy was on the leaf side; other leaf is implicit zero.
            0
        } else {
            packed & OFFSET_MASK
        };
        let leaf_off = val - acc;
        let (pushed, new_packed) = pack_with_side(leaf_off, other_off, leaf_is_right);
        self.val[p] = new_packed;
        // For `set`, every ancestor update only feeds `pushed` to the
        // ascending side; the moment it hits zero, every further step is a
        // no-op, so we can stop.
        self.bubble_up_set(p, pushed);
    }

    /// Adds `delta` to every leaf in `[lo, size)`.
    ///
    /// Time: O(log size).
    #[allow(dead_code)]
    pub fn suffix_add(&mut self, lo: usize, delta: i64) {
        if lo >= self.size || delta == 0 {
            return;
        }
        if self.size == 1 {
            self.val[0] += delta;
            return;
        }
        // Leaf pair index in the implicit pair-heap.
        let p = self.size / 2 + lo / 2;
        let leaf_is_right = lo & 1 == 1;
        let packed = self.val[p];
        let (mut l_off, mut r_off) = unpack(packed);
        // bit==0 → both leaves in the suffix; bit==1 → only the right one.
        if !leaf_is_right {
            l_off += delta;
        }
        r_off += delta;
        let (pushed, new_packed) = pack(l_off, r_off);
        self.val[p] = new_packed;
        // On the way up, every left-descent step has its right sibling fully
        // covered by the suffix, so we also bump the parent's right offset.
        self.bubble_up_suffix(p, pushed, delta);
    }

    /// Set-style bubble-up: at each ancestor, add `pushed` to the side we
    /// ascended from, then rebalance. Returns early when `pushed` hits 0,
    /// since every further update degenerates to a no-op.
    fn bubble_up_set(&mut self, mut p: usize, mut pushed: i64) {
        while p > 1 && pushed != 0 {
            let parent = p / 2;
            let from_right = p & 1 == 1;
            let packed = self.val[parent];
            let old_off = packed & OFFSET_MASK;
            // The side we came from currently has offset `old_off` if it was
            // the heavy side, else `0` (since the other side is implicit zero).
            let from_side_is_heavy = (packed < 0) == from_right;
            let from_off = if from_side_is_heavy { old_off } else { 0 } + pushed;
            let other_off = if from_side_is_heavy { 0 } else { old_off };
            let (m, new_packed) = pack_with_side(from_off, other_off, from_right);
            self.val[parent] = new_packed;
            pushed = m;
            p = parent;
        }
        self.val[0] += pushed;
    }

    /// Suffix-style bubble-up: like `bubble_up_set`, but on every step where
    /// we ascend from a left child we also add `right_bump` to the right
    /// sibling (since `suffix_add` fully covers that subtree).
    fn bubble_up_suffix(&mut self, mut p: usize, mut pushed: i64, right_bump: i64) {
        while p > 1 {
            let parent = p / 2;
            let from_right = p & 1 == 1;
            let packed = self.val[parent];
            let (mut l_off, mut r_off) = unpack(packed);
            if from_right {
                r_off += pushed;
            } else {
                l_off += pushed;
                r_off += right_bump;
            }
            let (m, new_packed) = pack(l_off, r_off);
            self.val[parent] = new_packed;
            pushed = m;
            p = parent;
        }
        self.val[0] += pushed;
    }

    /// Right-most leaf index `i` whose current value is `<= 0`, or `None`.
    ///
    /// Time: O(log size).
    #[allow(dead_code)]
    pub fn rightmost_le_zero(&self) -> Option<usize> {
        if self.size == 0 || self.val[0] > 0 {
            return None;
        }
        if self.size == 1 {
            // Single leaf, stored at the root; we already know val[0] <= 0.
            return Some(0);
        }
        let mut acc = self.val[0];
        let mut pair_idx = 1;
        let half = self.size / 2;
        // Descend through internal pairs; the pair tree is an implicit heap,
        // so the leaf range covered by a pair is fully determined by its
        // index. No need to track p_lo/p_hi/mid alongside.
        while pair_idx < half {
            let (l_off, r_off) = unpack(self.val[pair_idx]);
            let r_min = acc + r_off;
            if r_min <= 0 {
                acc = r_min;
                pair_idx = 2 * pair_idx + 1;
            } else {
                acc += l_off;
                pair_idx *= 2;
            }
        }
        // Leaf pair: covers leaves [p_lo, p_lo + 2) where
        // `p_lo = 2 * pair_idx - size`. Prefer the right leaf.
        let (_, r_off) = unpack(self.val[pair_idx]);
        let p_lo = 2 * pair_idx - self.size;
        if acc + r_off <= 0 {
            Some(p_lo + 1)
        } else {
            Some(p_lo)
        }
    }
}

/// Unpacks a stored entry into `(left_offset, right_offset)`. The
/// non-heavy side is always `0`.
#[inline(always)]
fn unpack(packed: i64) -> (i64, i64) {
    let off = packed & OFFSET_MASK;
    if packed < 0 { (0, off) } else { (off, 0) }
}

/// Packs `(l, r)` by pulling out the common min `m = min(l, r)` and
/// returning `(m, packed)`. The packed word stores `|l - r|` with the
/// sign bit set iff the right side is the heavier (post-subtraction).
///
/// `l` and `r` may be transiently negative (a suffix_add with `delta < 0`
/// can push one side below zero before its parent pulls the min back up);
/// what matters is that `m` is the common min and `|l - r|` is the residual.
#[inline(always)]
fn pack(l: i64, r: i64) -> (i64, i64) {
    let m = l.min(r);
    // `diff = l - r`: positive iff left is heavier, negative iff right is.
    let diff = l - r;
    let off = diff.unsigned_abs() as i64;
    // Sign bit of `diff` shifted into the MSB: `SIDE_BIT` iff right heavy.
    let side = ((diff as u64) >> 63 << 63) as i64;
    (m, off | side)
}

/// Like [`pack`] but tagged: `from_side_is_right` says which input side is
/// `from`; result side is computed correspondingly. Used by the leaf-pair
/// and bubble-up paths where we logically have a "this side / other side"
/// view rather than "left / right".
#[inline(always)]
fn pack_with_side(from_off: i64, other_off: i64, from_side_is_right: bool) -> (i64, i64) {
    if from_side_is_right {
        pack(other_off, from_off)
    } else {
        pack(from_off, other_off)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::min_seg_tree::MinSegTree;

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
        let t = CompactMinSegTree::new(&[], 0);
        assert_eq!(t.size(), 0);
        assert_eq!(t.rightmost_le_zero(), None);
    }

    #[test]
    fn single_leaf() {
        let mut t = CompactMinSegTree::new(&[5], 1_000_000);
        assert_eq!(t.size(), 1);
        assert_eq!(t.rightmost_le_zero(), None);
        t.set(0, -1);
        assert_eq!(t.rightmost_le_zero(), Some(0));
        t.set(0, 7);
        assert_eq!(t.rightmost_le_zero(), None);
        t.suffix_add(0, -10);
        assert_eq!(t.rightmost_le_zero(), Some(0));
    }

    #[test]
    fn worked_example_from_design_doc() {
        // n = 4, leaves [3, 1, 5, 2]. Verify decoding round-trips.
        let mut t = CompactMinSegTree::new(&[3, 1, 5, 2], 1_000_000);
        // No leaf <= 0 yet.
        assert_eq!(t.rightmost_le_zero(), None);
        // Bring each leaf to zero in turn and check rightmost_le_zero.
        t.set(2, 0);
        assert_eq!(t.rightmost_le_zero(), Some(2));
        t.set(2, 5);
        t.set(0, 0);
        assert_eq!(t.rightmost_le_zero(), Some(0));
        t.set(0, 3);
        // Push every leaf below zero via a suffix_add.
        t.suffix_add(0, -100);
        // Right-most should be index 3.
        assert_eq!(t.rightmost_le_zero(), Some(3));
        // Restore leaf 3 to a positive value.
        t.set(3, 1);
        assert_eq!(t.rightmost_le_zero(), Some(2));
    }

    fn lcg_rng() -> impl FnMut() -> u64 {
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        }
    }

    #[test]
    fn matches_naive_under_random_ops() {
        for &n in &[1usize, 2, 3, 4, 5, 7, 8, 13, 16, 32] {
            let init: Vec<i64> = (0..n as i64).map(|i| (i * 13) % 17 - 5).collect();
            let mut t = CompactMinSegTree::new(&init, 1_000_000_000);
            let mut naive = Naive::new(init);
            let mut next = lcg_rng();

            for _ in 0..3_000 {
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
                        assert_eq!(
                            t.rightmost_le_zero(),
                            naive.rightmost_le_zero(),
                            "mismatch at n={n}"
                        );
                    }
                }
            }
            assert_eq!(t.rightmost_le_zero(), naive.rightmost_le_zero());
        }
    }

    /// Cross-check: every operation on the compact tree must produce the same
    /// `rightmost_le_zero` result as the same op on the non-compact tree.
    #[test]
    fn matches_min_seg_tree_under_random_ops() {
        for &n in &[1usize, 2, 4, 7, 8, 13, 16, 32] {
            let init: Vec<i64> = (0..n as i64).map(|i| (i * 11) % 19 - 7).collect();
            let mut compact = CompactMinSegTree::new(&init, 1_000_000_000);
            let mut full = MinSegTree::new(&init, 1_000_000_000);
            let mut next = lcg_rng();

            for _ in 0..3_000 {
                match next() % 3 {
                    0 => {
                        let i = (next() as usize) % n;
                        let v = (next() as i64) % 25 - 12;
                        compact.set(i, v);
                        full.set(i, v);
                    }
                    1 => {
                        let lo = (next() as usize) % (n + 1);
                        let d = (next() as i64) % 11 - 5;
                        compact.suffix_add(lo, d);
                        full.suffix_add(lo, d);
                    }
                    _ => {
                        assert_eq!(
                            compact.rightmost_le_zero(),
                            full.rightmost_le_zero(),
                            "compact vs full mismatch at n={n}"
                        );
                    }
                }
            }
            assert_eq!(compact.rightmost_le_zero(), full.rightmost_le_zero());
        }
    }

    #[test]
    fn padding_is_not_selected() {
        // n = 3 → padded internally to size 4.
        let mut t = CompactMinSegTree::new(&[10, 20, 30], 1_000_000_000);
        assert_eq!(t.size(), 4);
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
