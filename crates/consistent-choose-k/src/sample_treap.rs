//! Dynamic order-statistics tree for the sorted samples list of the fast
//! choose-k hasher. Implemented as an **implicit treap** (in-order index =
//! key) with a *relative-offset* encoding of the per-node `life` value.
//! Vec-backed: every node lives in `nodes` at a u32 index, freed slots are
//! reused.
//!
//! Each node carries one element of the in-order sequence; internal nodes
//! and "leaves" are the same thing — internal nodes are also data.
//!
//! All operations are expressed via two primitives:
//! * `merge(a, b)`  — concatenate the in-order sequences of two treaps.
//! * `split(root, k)` — split the sequence into the first `k` and the rest.
//!
//! Both run in O(log n) expected. Public ops compose them.
//!
//! # Relative-offset encoding
//!
//! Each node stores `min_off: i32` and `life: i32` such that:
//! * `actual_subtree_min(v) == sum of min_off` along the path from the tree
//!   root down to `v` (inclusive).
//! * `actual_life(v) == actual_subtree_min(v) + v.life`, with `v.life >= 0`.
//!
//! Invariant at every (non-NIL) node `v`:
//! `min(v.life, v.left.min_off, v.right.min_off) == 0`, where NIL children
//! contribute `+infinity`. Combined with `life >= 0`, this also gives
//! `child.min_off >= 0` at every internal edge.
//!
//! This is the same trick `min_seg_tree.rs` uses: a range-add of `delta` to
//! the entire subtree at `v` is just `nodes[v].min_off += delta` (the
//! reference frame for everything inside `v` shifts together). No lazy
//! tags are needed.
//!
//! # Reference frames
//!
//! Each subtree's *root* `min_off` is interpreted relative to the parent
//! subtree's actual min — i.e., when a subtree is detached (e.g. between a
//! `split` and the following `merge`), its root's `min_off` equals its
//! absolute subtree min. When a subtree is reattached as the child of some
//! node `p`, its root's `min_off` is shifted by `-p.min_off` (and by
//! `+p.min_off` when extracted back out). `split`/`merge` perform exactly
//! these adjustments at every recursive boundary.
//!
//! Index `0` is the **NIL sentinel** representing the empty subtree. It
//! lives permanently in `nodes[0]` so every index dereference is valid
//! without a null check, and is never freed or returned to callers. Its
//! fields are all zero **except** `min_off = i32::MAX` — that value acts
//! as `+infinity` in the `min(life, l.min_off, r.min_off)` computations
//! used by `update`, so a NIL child never "wins" the min (and therefore
//! never spuriously satisfies the invariant). NIL is also never written
//! to, so its sentinel value is preserved: writes inside `update` are
//! guarded so a non-zero rebase never clobbers it.

const NIL: u32 = 0;

#[derive(Clone, Copy)]
struct Node {
    sample: usize,
    /// Relative-offset encoding: see module docs. For the NIL sentinel
    /// this is `i32::MAX`, acting as `+infinity` in min comparisons.
    min_off: i32,
    /// `actual_life - actual_subtree_min`. Always `>= 0` (except sentinel).
    life: i32,
    size: u32,
    left: u32,
    right: u32,
    priority: u32,
}

impl Node {
    /// The permanent NIL sentinel stored at `nodes[0]`. All zero fields
    /// except `min_off`, which is `+infinity` so NIL is invisible in min
    /// comparisons.
    const SENTINEL: Node = Node {
        sample: 0,
        min_off: i32::MAX,
        life: 0,
        size: 0,
        left: NIL,
        right: NIL,
        priority: 0,
    };
}

/// Implicit-key treap with relative-offset encoding on the `life` field.
/// See module docs.
pub struct SampleTreap {
    nodes: Vec<Node>,
    /// Free list of `nodes` indices that can be reused on the next insert.
    /// Never contains `NIL`.
    free: Vec<u32>,
    root: u32,
    /// Internal RNG state for generating node priorities. Deterministic and
    /// instance-local so tests are reproducible.
    rng_state: u64,
}

impl SampleTreap {
    /// New empty treap, preallocating space for `capacity` nodes.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut nodes = Vec::with_capacity(capacity + 1);
        nodes.push(Node::SENTINEL);
        Self {
            nodes,
            free: Vec::new(),
            root: NIL,
            rng_state: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// Number of elements currently in the treap.
    pub fn len(&self) -> usize {
        self.nodes[self.root as usize].size as usize
    }

    /// True iff the treap holds no elements.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.root == NIL
    }

    /// Reads the sample value at in-order position `p`. Panics if out of
    /// range. Time: O(log n).
    #[allow(dead_code)]
    pub fn get_sample(&self, p: usize) -> usize {
        debug_assert!(p < self.len(), "index out of range");
        self.read_at(p).0
    }

    /// Reads `(sample, life)` at in-order position `p`. Panics if out of
    /// range. Time: O(log n).
    #[allow(dead_code)]
    pub fn get(&self, p: usize) -> (usize, i32) {
        debug_assert!(p < self.len(), "index out of range");
        self.read_at(p)
    }

    /// Appends `(sample, life)` at the end of the in-order sequence.
    ///
    /// Time: O(log n) expected.
    pub fn push_back(&mut self, sample: usize, life: i32) {
        let new_node = self.alloc(sample, life);
        let root = self.root;
        // Skip the redundant `split(root, len)` that an `insert_at(len, ...)`
        // would do (it walks the entire right spine for nothing): just
        // merge the fresh singleton onto the right.
        self.root = self.merge(root, new_node);
    }

    /// Inserts `(sample, life)` at in-order position `p` (i.e. the new
    /// node's index becomes `p`; everything previously at `>= p` shifts
    /// right by one).
    ///
    /// Time: O(log n) expected.
    #[allow(dead_code)]
    pub fn insert_at(&mut self, p: usize, sample: usize, life: i32) {
        debug_assert!(p <= self.len(), "insertion position out of range");
        let new_node = self.alloc(sample, life);
        let root = self.root;
        let (left, right) = self.split(root, p);
        let m = self.merge(left, new_node);
        self.root = self.merge(m, right);
    }

    /// Removes the node at in-order position `p` and returns its
    /// `(sample, life)` pair (with the absolute life value).
    ///
    /// Time: O(log n) expected.
    #[allow(dead_code)]
    pub fn remove_at(&mut self, p: usize) -> (usize, i32) {
        debug_assert!(p < self.len(), "removal position out of range");
        let root = self.root;
        let (left, right) = self.split(root, p);
        let (mid, right) = self.split(right, 1);
        debug_assert!(mid != NIL && self.nodes[mid as usize].size == 1);
        // After split, `mid` is a singleton at top-level reference frame.
        // For a singleton: actual_subtree_min == actual_life, and `life`
        // (the relative offset) is 0; so `mid.min_off` is the absolute life.
        let m = &self.nodes[mid as usize];
        let result = (m.sample, m.min_off + m.life);
        self.free.push(mid);
        self.root = self.merge(left, right);
        result
    }

    /// Removes the node at in-order position `p` and decrements the life of
    /// every remaining node at positions `[p, len)` (i.e. the original
    /// suffix `[p + 1, len)`, which has just shifted left by one) by `1`.
    ///
    /// Semantically equivalent to:
    /// ```ignore
    /// let removed = self.remove_at(p);
    /// self.add_life_suffix(p, -1);
    /// removed
    /// ```
    /// but it folds the suffix decrement into the same split/merge pair as
    /// the removal, avoiding a second descent down the same spine.
    ///
    /// Time: O(log n) expected.
    pub fn remove_at_decrementing_suffix(&mut self, p: usize) -> (usize, i32) {
        debug_assert!(p < self.len(), "removal position out of range");
        let root = self.root;
        let (left, right) = self.split(root, p);
        let (mid, right) = self.split(right, 1);
        debug_assert!(mid != NIL && self.nodes[mid as usize].size == 1);
        let m = &self.nodes[mid as usize];
        let result = (m.sample, m.min_off + m.life);
        self.free.push(mid);
        // The detached `right` subtree holds the original positions
        // `[p + 1, len)`, which become `[p, len - 1)` in the merged
        // result. Shift its reference frame by `-1` in place — a single
        // field bump on the subtree root (cf. `add_life_suffix`).
        if right != NIL {
            self.nodes[right as usize].min_off -= 1;
        }
        self.root = self.merge(left, right);
        result
    }

    /// Adds `delta` to the `life` of every node at in-order positions
    /// `[p, len)` (the suffix starting at `p`).
    ///
    /// Time: O(log n) expected.
    #[allow(dead_code)]
    pub fn add_life_suffix(&mut self, p: usize, delta: i32) {
        debug_assert!(p <= self.len(), "position out of range");
        if p == self.len() || delta == 0 {
            return;
        }
        let root = self.root;
        let (left, right) = self.split(root, p);
        // Apply `+delta` to the detached `right` subtree. With the
        // relative-offset encoding the entire subtree's reference frame
        // shifts by `delta` via a single field bump. `right` is non-NIL
        // here because `p < len()` guarantees the split leaves at least
        // one element on the right.
        debug_assert!(right != NIL);
        self.nodes[right as usize].min_off += delta;
        self.root = self.merge(left, right);
    }

    /// Returns the in-order position of the **rightmost** node whose `life`
    /// is `<= 0`, or `None` if no such node exists.
    ///
    /// Time: O(log n) expected.
    pub fn find_rightmost_le_zero(&self) -> Option<usize> {
        let root = self.root;
        if root == NIL {
            return None;
        }
        let root_min = self.nodes[root as usize].min_off;
        if root_min > 0 {
            return None;
        }
        Some(self.descend_rightmost_le_zero(root, 0, root_min))
    }

    /// In-order list of `sample` values. Useful for tests and the hasher's
    /// `samples()` getter.
    #[allow(dead_code)]
    pub fn samples(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.len());
        self.collect_in_order(self.root, &mut out);
        out
    }

    // ----- Internals: layout, allocation, aggregates ---------------------

    /// Allocate a singleton at the "top-level" reference frame: its
    /// `min_off` equals its absolute `life`, and `life`-delta is 0. When
    /// merged into the tree, `merge` will lower it relative to its new
    /// parent.
    fn alloc(&mut self, sample: usize, life: i32) -> u32 {
        // SplitMix64-style RNG step; high 32 bits are well-mixed.
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let priority = (self.rng_state >> 32) as u32;
        let node = Node {
            sample,
            min_off: life,
            life: 0,
            size: 1,
            left: NIL,
            right: NIL,
            priority,
        };
        if let Some(idx) = self.free.pop() {
            self.nodes[idx as usize] = node;
            idx
        } else {
            let idx = self.nodes.len() as u32;
            self.nodes.push(node);
            idx
        }
    }

    /// Restore the local invariant at `idx`:
    ///   `min(idx.life, left.min_off, right.min_off) == 0`
    /// and recompute `size`. Any common excess `dM` shared by all three is
    /// "lifted up" into `idx.min_off` (so the subtree-min path-sum stays
    /// correct, the contained `life`-deltas stay non-negative, and the
    /// child offsets stay non-negative).
    ///
    /// NIL children read as `min_off = i32::MAX` so they cannot win the
    /// `min()` — that's safe to use directly without a conditional. The
    /// writes are still guarded against NIL: if `dM != 0` and one child is
    /// NIL, writing `-= dM` would corrupt the sentinel.
    fn update(&mut self, idx: u32) {
        debug_assert!(idx != NIL, "update called on NIL");
        let (left, right, life) = {
            let n = &self.nodes[idx as usize];
            (n.left, n.right, n.life)
        };
        let l_min = self.nodes[left as usize].min_off;
        let r_min = self.nodes[right as usize].min_off;
        let d_m = life.min(l_min).min(r_min);
        if d_m != 0 {
            // Shift everything inside `idx`'s subtree down by `d_m` (so the
            // residual minimum becomes 0) and pay `d_m` into `idx.min_off`.
            // Actual subtree values are unchanged; only the relative
            // encoding rebases.
            let n = &mut self.nodes[idx as usize];
            n.life -= d_m;
            n.min_off += d_m;
            if left != NIL {
                self.nodes[left as usize].min_off -= d_m;
            }
            if right != NIL {
                self.nodes[right as usize].min_off -= d_m;
            }
        }
        let l_size = self.nodes[left as usize].size;
        let r_size = self.nodes[right as usize].size;
        self.nodes[idx as usize].size = 1 + l_size + r_size;
    }

    // ----- Internals: split / merge --------------------------------------

    /// Split the subtree rooted at `idx` into `(left, right)` where `left`
    /// contains the first `k` in-order elements and `right` contains the
    /// rest. Both outputs are at the same reference frame as the caller's
    /// `idx`.
    fn split(&mut self, idx: u32, k: usize) -> (u32, u32) {
        if idx == NIL {
            return (NIL, NIL);
        }
        let l_size = self.nodes[self.nodes[idx as usize].left as usize].size as usize;
        // Capture `idx.min_off` BEFORE `update(idx)` runs so we lift any
        // promoted sibling using the pre-update offset. The promoted
        // sibling is no longer a descendant of `idx`, so its reference
        // frame is `idx`'s old parent — which corresponds to old
        // `idx.min_off`.
        let idx_min_off = self.nodes[idx as usize].min_off;
        if k <= l_size {
            // Recurse into the left child; the returned `l1` is detached
            // and rises to be `idx`'s sibling (in the caller's frame).
            let (l1, l2) = self.split(self.nodes[idx as usize].left, k);
            if l1 != NIL {
                self.nodes[l1 as usize].min_off += idx_min_off;
            }
            self.nodes[idx as usize].left = l2;
            self.update(idx);
            (l1, idx)
        } else {
            let (r1, r2) = self.split(self.nodes[idx as usize].right, k - l_size - 1);
            if r2 != NIL {
                self.nodes[r2 as usize].min_off += idx_min_off;
            }
            self.nodes[idx as usize].right = r1;
            self.update(idx);
            (idx, r2)
        }
    }

    /// Merge two treaps whose in-order sequences are concatenated as
    /// `left` then `right`. Both inputs share the same reference frame;
    /// the output is at that same frame.
    fn merge(&mut self, left: u32, right: u32) -> u32 {
        if left == NIL {
            return right;
        }
        if right == NIL {
            return left;
        }
        let lp = self.nodes[left as usize].priority;
        let rp = self.nodes[right as usize].priority;
        if lp > rp {
            // `left` stays root; `right` descends into `left`'s right
            // subtree. Lower `right` to `left`'s subtree-min frame.
            self.nodes[right as usize].min_off -= self.nodes[left as usize].min_off;
            self.nodes[left as usize].right = self.merge(self.nodes[left as usize].right, right);
            self.update(left);
            left
        } else {
            self.nodes[left as usize].min_off -= self.nodes[right as usize].min_off;
            self.nodes[right as usize].left = self.merge(left, self.nodes[right as usize].left);
            self.update(right);
            right
        }
    }

    // ----- Internals: descent helpers ------------------------------------

    /// Read at position `p`, accumulating `min_off` along the descent path
    /// to recover the absolute life value at the target.
    fn read_at(&self, mut p: usize) -> (usize, i32) {
        let mut idx = self.root;
        let mut accum: i32 = 0;
        loop {
            debug_assert!(idx != NIL, "position out of range");
            accum += self.nodes[idx as usize].min_off;
            let n = &self.nodes[idx as usize];
            let l_size = self.nodes[n.left as usize].size as usize;
            if p < l_size {
                idx = n.left;
            } else if p == l_size {
                return (n.sample, accum + n.life);
            } else {
                p -= l_size + 1;
                idx = n.right;
            }
        }
    }

    /// Descend to the rightmost node with `actual_life <= 0`. Caller has
    /// verified the subtree at `idx` contains such a node (i.e.
    /// `accum_at_idx == actual_subtree_min(idx) <= 0`).
    fn descend_rightmost_le_zero(&self, mut idx: u32, mut base: usize, mut accum: i32) -> usize {
        loop {
            debug_assert!(idx != NIL);
            debug_assert!(accum <= 0);
            let n = &self.nodes[idx as usize];
            let l_size = self.nodes[n.left as usize].size as usize;
            // Prefer the right subtree; then this node; then the left
            // subtree. Guard each min_off arithmetic against NIL (whose
            // sentinel min_off = i32::MAX must never be added to a
            // possibly-negative accum).
            if n.right != NIL {
                let r_accum = accum + self.nodes[n.right as usize].min_off;
                if r_accum <= 0 {
                    base += l_size + 1;
                    accum = r_accum;
                    idx = n.right;
                    continue;
                }
            }
            if accum + n.life <= 0 {
                return base + l_size;
            }
            debug_assert!(n.left != NIL);
            let l_accum = accum + self.nodes[n.left as usize].min_off;
            debug_assert!(l_accum <= 0);
            accum = l_accum;
            idx = n.left;
        }
    }

    fn collect_in_order(&self, idx: u32, out: &mut Vec<usize>) {
        if idx == NIL {
            return;
        }
        let n = &self.nodes[idx as usize];
        self.collect_in_order(n.left, out);
        out.push(n.sample);
        self.collect_in_order(n.right, out);
    }

    /// Returns the in-order list of `(sample, life)` pairs.
    pub fn lifetimes(&self) -> Vec<(usize, i32)> {
        let mut out = Vec::with_capacity(self.len());
        self.collect_lifetimes(self.root, 0, &mut out);
        out
    }

    fn collect_lifetimes(&self, idx: u32, accum: i32, out: &mut Vec<(usize, i32)>) {
        if idx == NIL {
            return;
        }
        let n = &self.nodes[idx as usize];
        let current_accum = accum + n.min_off;
        self.collect_lifetimes(n.left, current_accum, out);
        out.push((n.sample, current_accum + n.life));
        self.collect_lifetimes(n.right, current_accum, out);
    }
}

impl std::fmt::Debug for SampleTreap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.lifetimes().into_iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_rng() -> impl FnMut() -> u64 {
        let mut state: u64 = 0xC0FFEE_DEADBEEF;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        }
    }

    impl SampleTreap {
        /// Recursive invariant checker used by tests. Verifies:
        /// * sizes are correct,
        /// * `life >= 0` and (non-root) `min_off >= 0`,
        /// * `min(life, left.min_off, right.min_off) == 0`,
        /// * actual lives accumulated from root equal the values returned
        ///   by `get`.
        fn check_invariants(&self) {
            fn rec(t: &SampleTreap, idx: u32, accum: i32, is_root: bool, pos: &mut usize) {
                if idx == NIL {
                    return;
                }
                let n = &t.nodes[idx as usize];
                if !is_root {
                    assert!(n.min_off >= 0, "non-root min_off should be >= 0");
                }
                assert!(n.life >= 0, "life-delta should be >= 0");
                let l_min = t.nodes[n.left as usize].min_off;
                let r_min = t.nodes[n.right as usize].min_off;
                assert_eq!(
                    n.life.min(l_min).min(r_min),
                    0,
                    "invariant min(life, l, r) == 0 violated",
                );
                let l_size = t.nodes[n.left as usize].size;
                let r_size = t.nodes[n.right as usize].size;
                assert_eq!(n.size, 1 + l_size + r_size, "size mismatch");
                let sub_accum = accum + n.min_off;
                rec(t, n.left, sub_accum, false, pos);
                // Verify get(pos) returns the same life as our accumulated
                // walk would produce.
                let got = t.get(*pos);
                assert_eq!(got.0, n.sample);
                assert_eq!(got.1, sub_accum + n.life);
                *pos += 1;
                rec(t, n.right, sub_accum, false, pos);
            }
            let mut pos = 0usize;
            rec(self, self.root, 0, true, &mut pos);
            assert_eq!(pos, self.len());
        }
    }

    /// Naive reference for the same operations, backed by a Vec.
    struct Naive {
        v: Vec<(usize, i32)>,
    }
    impl Naive {
        fn new() -> Self {
            Self { v: Vec::new() }
        }
        fn len(&self) -> usize {
            self.v.len()
        }
        fn insert_at(&mut self, p: usize, sample: usize, life: i32) {
            self.v.insert(p, (sample, life));
        }
        fn remove_at(&mut self, p: usize) -> (usize, i32) {
            self.v.remove(p)
        }
        fn remove_at_decrementing_suffix(&mut self, p: usize) -> (usize, i32) {
            let removed = self.v.remove(p);
            for entry in &mut self.v[p..] {
                entry.1 -= 1;
            }
            removed
        }
        fn add_life_suffix(&mut self, p: usize, d: i32) {
            for i in p..self.v.len() {
                self.v[i].1 += d;
            }
        }
        fn find_rightmost_le_zero(&self) -> Option<usize> {
            self.v.iter().rposition(|&(_, l)| l <= 0)
        }
        fn samples(&self) -> Vec<usize> {
            self.v.iter().map(|&(s, _)| s).collect()
        }
        fn get(&self, p: usize) -> (usize, i32) {
            self.v[p]
        }
    }

    #[test]
    fn empty() {
        let t = SampleTreap::with_capacity(0);
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
        assert_eq!(t.find_rightmost_le_zero(), None);
        assert!(t.samples().is_empty());
    }

    #[test]
    fn push_back_basic() {
        let mut t = SampleTreap::with_capacity(4);
        t.push_back(10, 3);
        t.push_back(20, 7);
        t.push_back(30, 0);
        t.push_back(40, 5);
        assert_eq!(t.len(), 4);
        assert_eq!(t.samples(), vec![10, 20, 30, 40]);
        assert_eq!(t.find_rightmost_le_zero(), Some(2));
        assert_eq!(t.get(0), (10, 3));
        assert_eq!(t.get(3), (40, 5));
        t.check_invariants();
    }

    #[test]
    fn insert_at_middle() {
        let mut t = SampleTreap::with_capacity(0);
        t.push_back(10, 3);
        t.push_back(30, 5);
        t.insert_at(1, 20, 4);
        assert_eq!(t.samples(), vec![10, 20, 30]);
        assert_eq!(t.get(1), (20, 4));
        t.check_invariants();
    }

    #[test]
    fn remove_at_returns_value() {
        let mut t = SampleTreap::with_capacity(0);
        for (s, l) in [(10, 3), (20, -1), (30, 5)] {
            t.push_back(s, l);
        }
        let removed = t.remove_at(1);
        assert_eq!(removed, (20, -1));
        assert_eq!(t.samples(), vec![10, 30]);
        t.check_invariants();
    }

    #[test]
    fn add_life_suffix_basic() {
        let mut t = SampleTreap::with_capacity(0);
        for (s, l) in [(10, 1), (20, 1), (30, 1), (40, 1)] {
            t.push_back(s, l);
        }
        t.add_life_suffix(2, -3);
        // Lives are now [1, 1, -2, -2].
        assert_eq!(t.find_rightmost_le_zero(), Some(3));
        assert_eq!(t.get(0).1, 1);
        assert_eq!(t.get(1).1, 1);
        assert_eq!(t.get(2).1, -2);
        assert_eq!(t.get(3).1, -2);
        t.check_invariants();
    }

    #[test]
    fn find_rightmost_picks_rightmost() {
        let mut t = SampleTreap::with_capacity(0);
        for (s, l) in [(10, -1), (20, -2), (30, -3), (40, -4)] {
            t.push_back(s, l);
        }
        assert_eq!(t.find_rightmost_le_zero(), Some(3));
        t.check_invariants();
    }

    #[test]
    fn find_rightmost_none() {
        let mut t = SampleTreap::with_capacity(0);
        for (s, l) in [(10, 1), (20, 2), (30, 3)] {
            t.push_back(s, l);
        }
        assert_eq!(t.find_rightmost_le_zero(), None);
        t.check_invariants();
    }

    #[test]
    fn nested_suffix_adds_get_returns_cumulative() {
        let mut t = SampleTreap::with_capacity(0);
        for i in 0..16i32 {
            t.push_back(i as usize, i);
        }
        // Initial life at p is p. After add_life_suffix(0, -5): p - 5.
        t.add_life_suffix(0, -5);
        // After add_life_suffix(5, -10): p - 5 if p < 5 else p - 15.
        t.add_life_suffix(5, -10);
        assert_eq!(t.get(4).1, -1);
        assert_eq!(t.get(5).1, -10);
        assert_eq!(t.get(7).1, -8);
        assert_eq!(t.get(11).1, -4);
        assert_eq!(t.get(12).1, -3);
        assert_eq!(t.get(15).1, 0);
        t.check_invariants();
    }

    #[test]
    fn matches_naive_under_random_ops() {
        let mut rng = lcg_rng();
        for _trial in 0..50 {
            let mut t = SampleTreap::with_capacity(0);
            let mut n = Naive::new();
            for _ in 0..200 {
                let op = rng() % 12;
                if op < 4 || n.len() == 0 {
                    let sample = (rng() % 1000) as usize;
                    let life = (rng() as i32).rem_euclid(20) - 5;
                    let p = (rng() as usize) % (n.len() + 1);
                    t.insert_at(p, sample, life);
                    n.insert_at(p, sample, life);
                } else if op < 6 {
                    let p = (rng() as usize) % n.len();
                    assert_eq!(t.remove_at(p), n.remove_at(p));
                } else if op < 8 {
                    let p = (rng() as usize) % n.len();
                    assert_eq!(
                        t.remove_at_decrementing_suffix(p),
                        n.remove_at_decrementing_suffix(p),
                    );
                } else if op < 10 {
                    let p = (rng() as usize) % (n.len() + 1);
                    let d = (rng() as i32).rem_euclid(11) - 5;
                    t.add_life_suffix(p, d);
                    n.add_life_suffix(p, d);
                } else if !n.v.is_empty() {
                    let p = (rng() as usize) % n.len();
                    assert_eq!(t.get(p), n.get(p));
                }
                assert_eq!(t.len(), n.len());
                assert_eq!(t.find_rightmost_le_zero(), n.find_rightmost_le_zero());
                assert_eq!(t.samples(), n.samples());
                // Compare every (sample, life) — stronger than samples/find
                // alone, since wrong positive lives would otherwise hide.
                for p in 0..n.len() {
                    assert_eq!(t.get(p), n.get(p), "mismatch at position {p}");
                }
                t.check_invariants();
            }
        }
    }
}
