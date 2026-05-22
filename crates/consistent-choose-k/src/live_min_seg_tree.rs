//! A compact min segment tree with order-statistics support over LIVE
//! leaves only, plus O(log size) tombstoning of arbitrary live leaves.
//!
//! Same compact (pair-heap) packing as
//! [`crate::compact_min_seg_tree::CompactMinSegTree`] for the life-min tree,
//! and a parallel per-pair "alive count" array tracking the number of live
//! leaves in each pair's LEFT subtree. Memory: 12 bytes per leaf
//! (`i64` for the life offset, `u32` for the alive count).
//!
//! All public leaf indices are LOGICAL (= rank among live leaves). The
//! structure internally maps logical -> physical leaf using `alive` for
//! rank-select during descent.
//!
//! Tombstoned leaves are simply set to `padding` (which must be `>` every
//! real life value, so they are never selected by [`Self::find_dead`]) and
//! their entry in the `alive` array is decremented along the path to the
//! root. The caller is responsible for compacting (rebuilding) the tree
//! when the dead-leaf fraction grows too large.

const SIDE_BIT: i64 = i64::MIN;
const OFFSET_MASK: i64 = i64::MAX;

/// See module docs.
pub struct LiveMinSegTree {
    /// Pair-heap of life offsets. Same layout as `CompactMinSegTree::val`.
    val: Vec<i64>,
    /// Pair-heap of alive counts.
    ///
    /// * `alive[0]` is the total number of live leaves (mirrors `val[0]`
    ///   being the global min).
    /// * For `p` in `1..size`, `alive[p]` is the number of live leaves in
    ///   the LEFT subtree of the pair whose offset is stored in `val[p]`.
    ///   The right subtree's alive count is `parent_total - alive[p]`.
    alive: Vec<u32>,
    /// Power-of-two physical capacity (number of leaves, real + tombstone),
    /// or `0` when the tree is empty.
    size: usize,
    /// Number of leaves ever pushed (i.e. number of physical slots in use,
    /// counting both live leaves and tombstones).
    physical_len: usize,
    /// Number of currently live (non-tombstoned) leaves.
    n_live: usize,
    /// Padding value used for unused slots and tombstones. Must be `>` every
    /// real life value the caller will ever push.
    padding: i64,
}

impl LiveMinSegTree {
    /// Builds an empty tree, preallocating the underlying buffers to fit up
    /// to `capacity` leaves without reallocation.
    ///
    /// Time: O(1).
    pub fn with_capacity(capacity: usize, padding: i64) -> Self {
        let cap = if capacity == 0 {
            0
        } else {
            capacity.next_power_of_two()
        };
        Self {
            val: Vec::with_capacity(cap),
            alive: Vec::with_capacity(cap),
            size: 0,
            physical_len: 0,
            n_live: 0,
            padding,
        }
    }

    /// Number of live leaves currently in the tree.
    pub fn len(&self) -> usize {
        self.n_live
    }

    /// True iff no live leaves are stored.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.n_live == 0
    }

    /// Number of physical leaves (live + tombstoned). Useful for caller-side
    /// compaction heuristics.
    #[allow(dead_code)]
    pub fn physical_len(&self) -> usize {
        self.physical_len
    }

    /// Number of tombstoned leaves.
    #[allow(dead_code)]
    pub fn n_dead(&self) -> usize {
        self.physical_len - self.n_live
    }

    /// Physical capacity (power of two, or `0` when empty).
    #[allow(dead_code)]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Appends `life` as a new live leaf at the rightmost physical slot.
    /// The new leaf's logical index is `len() - 1` after this call.
    ///
    /// Time: amortized O(log size); worst-case O(size) on the doubling step.
    pub fn push(&mut self, life: i64) {
        if self.physical_len == self.size {
            self.grow();
        }
        let p_phys = self.physical_len;
        self.physical_len = p_phys + 1;
        self.n_live += 1;
        self.set_physical(p_phys, life);
        self.bump_alive_path(p_phys, 1);
    }

    /// Overwrites the life of the live leaf at logical position `p`.
    ///
    /// Time: O(log size).
    #[allow(dead_code)]
    pub fn set(&mut self, p: usize, life: i64) {
        debug_assert!(p < self.n_live, "logical index out of range");
        let p_phys = self.physical_at_rank(p);
        self.set_physical(p_phys, life);
    }

    /// Returns the logical index of the rightmost live leaf whose life is
    /// `<= 0`, or `None` if no such leaf exists.
    ///
    /// Time: O(log size).
    pub fn find_dead(&self) -> Option<usize> {
        if self.size == 0 || self.n_live == 0 || self.val[0] > 0 {
            return None;
        }
        if self.size == 1 {
            // Single leaf, which is live (n_live > 0) and val[0] <= 0.
            return Some(0);
        }
        let mut acc = self.val[0];
        let mut pair_idx = 1;
        let mut logical = 0u32;
        let half = self.size / 2;
        while pair_idx < half {
            let (l_off, r_off) = unpack(self.val[pair_idx]);
            let r_min = acc + r_off;
            if r_min <= 0 {
                // Skip the entire left subtree → it contributes alive[pair_idx]
                // live leaves to the left of our target.
                logical += self.alive[pair_idx];
                acc = r_min;
                pair_idx = 2 * pair_idx + 1;
            } else {
                acc += l_off;
                pair_idx *= 2;
            }
        }
        // Leaf pair. Tombstones sit at `>= padding > 0`, so any leaf with
        // value `<= 0` is necessarily live; no need to consult `alive`.
        let (_, r_off) = unpack(self.val[pair_idx]);
        if acc + r_off <= 0 {
            // Right leaf is dead; skip the left leaf's alive bit.
            logical += self.alive[pair_idx];
            Some(logical as usize)
        } else {
            Some(logical as usize)
        }
    }

    /// Tombstones the live leaf at logical position `p`.
    ///
    /// After this call:
    /// * `len()` decreases by 1, `n_dead()` increases by 1.
    /// * Every live leaf strictly to the right of the killed slot has its
    ///   life incremented by 1 (matching the "shift-left" semantics where
    ///   each surviving leaf inherits the rank of its right neighbour).
    ///
    /// Time: O(log size).
    pub fn kill(&mut self, p: usize) {
        debug_assert!(p < self.n_live, "logical index out of range");
        let p_phys = self.physical_at_rank(p);
        // Tombstone the leaf and bump all physically-later leaves (real or
        // tombstone) by +1. Tombstones sit at `>= padding`, so they remain
        // unselectable.
        self.set_physical(p_phys, self.padding);
        self.suffix_add(p_phys + 1, 1);
        self.bump_alive_path(p_phys, -1);
        self.n_live -= 1;
    }

    // ---- Internal helpers (all in physical-leaf space) ------------------

    /// Doubles `size`, preserving the existing tree as the left subtree of a
    /// fresh root and filling the right subtree with all-padding leaves.
    /// The `alive` array is grown in lockstep with all-zero right subtree.
    fn grow(&mut self) {
        if self.size == 0 {
            self.val.push(self.padding);
            self.alive.push(0);
            self.size = 1;
            return;
        }
        let old_size = self.size;
        let new_size = old_size * 2;
        let old_root_min = self.val[0];
        let old_root_alive = self.alive[0];
        self.val.resize(new_size, 0);
        self.alive.resize(new_size, 0);
        // Shift each level of the pair-heap one step deeper. Process from
        // the deepest level upward; each step is a contiguous slice copy
        // followed by zeroing the now-vacated source.
        let mut level = old_size / 2;
        while level > 0 {
            self.val.copy_within(level..2 * level, 2 * level);
            self.val[level..2 * level].fill(0);
            self.alive.copy_within(level..2 * level, 2 * level);
            self.alive[level..2 * level].fill(0);
            level /= 2;
        }
        // Right subtree is all-padding (life) and all-dead (alive=0). The
        // new root pair's right child reaches that padding subtree (min =
        // padding, alive = 0); its left child reaches the relocated old
        // root (min = old_root_min, alive = old_root_alive).
        let r_off = self.padding - old_root_min;
        debug_assert!(r_off >= 0, "padding must be >= every real leaf");
        self.val[1] = SIDE_BIT | r_off;
        self.alive[1] = old_root_alive;
        self.size = new_size;
    }

    /// Translates a logical rank to a physical leaf index.
    fn physical_at_rank(&self, mut rank: usize) -> usize {
        debug_assert!(rank < self.n_live);
        if self.size == 1 {
            return 0;
        }
        let mut pair_idx = 1;
        let half = self.size / 2;
        while pair_idx < half {
            let left_alive = self.alive[pair_idx] as usize;
            if rank < left_alive {
                pair_idx *= 2;
            } else {
                rank -= left_alive;
                pair_idx = 2 * pair_idx + 1;
            }
        }
        // Leaf pair: `alive[pair_idx]` is 0 or 1 (alive bit of the left leaf).
        let left_alive = self.alive[pair_idx] as usize;
        let p_lo = 2 * pair_idx - self.size;
        if rank < left_alive { p_lo } else { p_lo + 1 }
    }

    /// Sets physical leaf `i` to `val` (mirrors `CompactMinSegTree::set`).
    fn set_physical(&mut self, i: usize, val: i64) {
        debug_assert!(i < self.size, "leaf index out of range");
        if self.size == 1 {
            self.val[0] = val;
            return;
        }
        let depth = self.size.trailing_zeros();
        let mut acc = self.val[0];
        let mut p = 1;
        for d in (1..depth).rev() {
            let bit = (i >> d) & 1;
            let packed = self.val[p];
            if (packed < 0) == (bit == 1) {
                acc += packed & OFFSET_MASK;
            }
            p = 2 * p + bit;
        }
        let leaf_is_right = i & 1 == 1;
        let packed = self.val[p];
        let other_off = if (packed < 0) == leaf_is_right {
            0
        } else {
            packed & OFFSET_MASK
        };
        let leaf_off = val - acc;
        let (pushed, new_packed) = pack_with_side(leaf_off, other_off, leaf_is_right);
        self.val[p] = new_packed;
        self.bubble_up_set(p, pushed);
    }

    /// Adds `delta` to every physical leaf in `[lo, size)` (mirrors
    /// `CompactMinSegTree::suffix_add`).
    fn suffix_add(&mut self, lo: usize, delta: i64) {
        if lo >= self.size || delta == 0 {
            return;
        }
        if self.size == 1 {
            self.val[0] += delta;
            return;
        }
        let p = self.size / 2 + lo / 2;
        let leaf_is_right = lo & 1 == 1;
        let packed = self.val[p];
        let (mut l_off, mut r_off) = unpack(packed);
        if !leaf_is_right {
            l_off += delta;
        }
        r_off += delta;
        let (pushed, new_packed) = pack(l_off, r_off);
        self.val[p] = new_packed;
        self.bubble_up_suffix(p, pushed, delta);
    }

    /// Adjusts the `alive` array along the path from physical leaf `p_phys`
    /// to the root by `delta` (typically `+1` for `push`, `-1` for `kill`).
    /// `alive[parent]` is bumped iff we ascend from the LEFT child.
    fn bump_alive_path(&mut self, p_phys: usize, delta: i32) {
        if self.size == 1 {
            apply_delta(&mut self.alive[0], delta);
            return;
        }
        let mut pair_idx = self.size / 2 + p_phys / 2;
        // Leaf pair: bump alive[pair_idx] iff the leaf is on the left side
        // (i.e. p_phys is even). The leaf-pair entry stores the left leaf's
        // alive bit only.
        if p_phys & 1 == 0 {
            apply_delta(&mut self.alive[pair_idx], delta);
        }
        // Walk up the pair-heap.
        while pair_idx > 1 {
            let parent = pair_idx / 2;
            let from_left = pair_idx & 1 == 0;
            if from_left {
                apply_delta(&mut self.alive[parent], delta);
            }
            pair_idx = parent;
        }
        // Root total.
        apply_delta(&mut self.alive[0], delta);
    }

    fn bubble_up_set(&mut self, mut p: usize, mut pushed: i64) {
        while p > 1 && pushed != 0 {
            let parent = p / 2;
            let from_right = p & 1 == 1;
            let packed = self.val[parent];
            let old_off = packed & OFFSET_MASK;
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
}

#[inline(always)]
fn apply_delta(slot: &mut u32, delta: i32) {
    if delta >= 0 {
        *slot += delta as u32;
    } else {
        *slot -= (-delta) as u32;
    }
}

#[inline(always)]
fn unpack(packed: i64) -> (i64, i64) {
    let off = packed & OFFSET_MASK;
    if packed < 0 { (0, off) } else { (off, 0) }
}

#[inline(always)]
fn pack(l: i64, r: i64) -> (i64, i64) {
    let m = l.min(r);
    let diff = l - r;
    let off = diff.unsigned_abs() as i64;
    let side = ((diff as u64) >> 63 << 63) as i64;
    (m, off | side)
}

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

    const PAD: i64 = 1_000_000_000;

    fn lcg_rng() -> impl FnMut() -> u64 {
        let mut state: u64 = 0xC0FFEE_DEADBEEF;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        }
    }

    /// Naive reference: stores the live lives in their logical order.
    struct Naive {
        lives: Vec<i64>,
    }
    impl Naive {
        fn new() -> Self {
            Self { lives: Vec::new() }
        }
        fn push(&mut self, life: i64) {
            self.lives.push(life);
        }
        fn find_dead(&self) -> Option<usize> {
            self.lives.iter().rposition(|&v| v <= 0)
        }
        fn kill(&mut self, p: usize) {
            self.lives.remove(p);
            for v in &mut self.lives[p..] {
                *v += 1;
            }
        }
        fn set(&mut self, p: usize, life: i64) {
            self.lives[p] = life;
        }
        fn len(&self) -> usize {
            self.lives.len()
        }
    }

    #[test]
    fn empty() {
        let t = LiveMinSegTree::with_capacity(0, PAD);
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
        assert_eq!(t.find_dead(), None);
        assert_eq!(t.physical_len(), 0);
        assert_eq!(t.n_dead(), 0);
    }

    #[test]
    fn single_leaf() {
        let mut t = LiveMinSegTree::with_capacity(1, PAD);
        t.push(5);
        assert_eq!(t.len(), 1);
        assert_eq!(t.find_dead(), None);
        t.set(0, -1);
        assert_eq!(t.find_dead(), Some(0));
        t.kill(0);
        assert_eq!(t.len(), 0);
        assert_eq!(t.n_dead(), 1);
        assert_eq!(t.find_dead(), None);
    }

    #[test]
    fn push_then_find_dead_basic() {
        let mut t = LiveMinSegTree::with_capacity(4, PAD);
        // Push 4 lives; none <= 0.
        for &v in &[3i64, 7, 5, 2] {
            t.push(v);
        }
        assert_eq!(t.len(), 4);
        assert_eq!(t.find_dead(), None);
        // Make logical index 2 dead.
        t.set(2, -1);
        assert_eq!(t.find_dead(), Some(2));
        // Make logical index 3 dead too; rightmost is now 3.
        t.set(3, 0);
        assert_eq!(t.find_dead(), Some(3));
    }

    #[test]
    fn kill_shifts_lives_correctly() {
        let mut t = LiveMinSegTree::with_capacity(4, PAD);
        for &v in &[5i64, -1, 7, -2] {
            t.push(v);
        }
        // Rightmost dead is logical 3 (life -2).
        assert_eq!(t.find_dead(), Some(3));
        t.kill(3);
        // After kill: remaining logical lives are [5, -1, 7]. No bump (nothing right of 3).
        // Rightmost dead is now logical 1.
        assert_eq!(t.len(), 3);
        assert_eq!(t.find_dead(), Some(1));
        t.kill(1);
        // After kill: remaining live lives in logical order: [5, 7+1] = [5, 8].
        assert_eq!(t.len(), 2);
        assert_eq!(t.find_dead(), None);
    }

    #[test]
    fn find_dead_returns_rightmost() {
        let mut t = LiveMinSegTree::with_capacity(8, PAD);
        for &v in &[-1i64, -2, -3, -4, -5, -6, -7, -8] {
            t.push(v);
        }
        assert_eq!(t.find_dead(), Some(7));
    }

    #[test]
    fn padding_is_not_selected() {
        // Tombstones (= padding) must never be returned by find_dead, even
        // after suffix bumps from neighbouring kills.
        let mut t = LiveMinSegTree::with_capacity(4, PAD);
        for &v in &[1i64, 2, 3, 4] {
            t.push(v);
        }
        // Tombstone everything by repeatedly killing the dead leaf, but
        // first artificially make every leaf dead so kills cascade.
        for i in 0..4 {
            t.set(i, -100);
        }
        for _ in 0..4 {
            let p = t.find_dead().unwrap();
            t.kill(p);
        }
        assert_eq!(t.find_dead(), None);
        assert_eq!(t.len(), 0);
        assert_eq!(t.n_dead(), 4);
    }

    #[test]
    fn doubling_preserves_state() {
        let mut t = LiveMinSegTree::with_capacity(0, PAD);
        // Pushes that span several doublings.
        let lives: Vec<i64> = (1..=9).map(|i| i as i64).collect();
        for &v in &lives {
            t.push(v);
        }
        assert_eq!(t.len(), 9);
        assert_eq!(t.find_dead(), None);
        // Force a dead at logical 4 (life 5 + 0 bumps so far).
        t.set(4, -1);
        assert_eq!(t.find_dead(), Some(4));
    }

    /// Cross-check against the naive reference with a random sequence of
    /// pushes / kills / sets.
    #[test]
    fn matches_naive_under_random_ops() {
        let mut next = lcg_rng();
        for trial in 0..50 {
            let mut t = LiveMinSegTree::with_capacity(0, PAD);
            let mut n = Naive::new();
            for _ in 0..200 {
                let op = next() % 10;
                if op < 6 || n.len() == 0 {
                    // Push a life in a wide range so we get both positive
                    // and negative ones.
                    let life: i64 = (next() as i64).rem_euclid(26) - 5;
                    t.push(life);
                    n.push(life);
                } else if op < 9 {
                    // Kill (prefer find_dead's choice, else random index).
                    let p = match n.find_dead() {
                        Some(p) => p,
                        None => (next() as usize) % n.len(),
                    };
                    let tp = t.find_dead();
                    let np = n.find_dead();
                    assert_eq!(tp, np, "trial {trial}");
                    t.kill(p);
                    n.kill(p);
                } else {
                    // Overwrite the life of a random logical leaf.
                    let p = (next() as usize) % n.len();
                    let life: i64 = (next() as i64).rem_euclid(26) - 5;
                    t.set(p, life);
                    n.set(p, life);
                }
                assert_eq!(t.len(), n.len(), "trial {trial}");
                assert_eq!(t.find_dead(), n.find_dead(), "trial {trial}");
            }
        }
    }
}
