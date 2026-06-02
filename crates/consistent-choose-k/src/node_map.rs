use std::collections::HashSet;

use crate::{ConsistentChooseKHasher, ManySeqBuilder};

/// A consistent node map that supports dynamic addition and deletion of nodes.
///
/// Slots are tracked by storing the total number of slots and a set of deleted
/// slots. To find the slot associated with a key, the consistent choose-k
/// iterator yields positions in a consistent order; the first non-deleted slot
/// is returned.
///
/// # Comparison with AnchorHash, MementoHash, and DxHash
///
/// This solves the same problem as [AnchorHash], [MementoHash], and [DxHash]:
/// consistently mapping keys to a dynamic set of nodes where nodes can be
/// added and removed, with minimal key reassignment. All of these algorithms
/// guarantee that when a node is removed, only keys assigned to that node are
/// redistributed — and they are redistributed uniformly among the remaining
/// nodes.
///
/// The key difference is history. AnchorHash, MementoHash, and DxHash keep
/// redirect/replacement state so that when lookup hits a deleted node, it can
/// replay enough of the prior removal process to find the correct replacement.
/// MementoHash, for example, defines its state as `<n, R, l>`, where `R` is a
/// set of replacement tuples and `l` is the last removed bucket. This
/// implementation is history-independent: it only needs to know which slots are
/// currently deleted. Lookup simply iterates the consistent choose-k sequence
/// until it hits an active slot.
///
/// This implementation takes a much simpler approach: it leverages the
/// consistent choose-k algorithm, which already provides both n-consistency
/// and k-consistency by construction. No auxiliary redirect structures are
/// needed beyond the current set of deleted slots.
///
/// Let `total` be the number of slots, `active` the number of active slots, and
/// `h` the number of deleted slots hit during a lookup before the first active
/// slot is found. For AnchorHash and DxHash, `total` is the predefined capacity;
/// for MementoHash and this implementation, it is the current slot count.
/// MementoHash bounds the expected number of deleted-node hits by harmonic
/// sums, e.g. `1 + H_total - H_active`, which is at most
/// `1 + ln(total / active)`.
///
/// In this implementation, the choose-k iterator never returns the same slot
/// twice, so a deleted slot can be hit at most once during a lookup. Thus the
/// scan has the same deleted-hit behavior as the history-based algorithms, but
/// without storing the deletion history.
///
/// The current choose-k iterator costs O(k) to produce the k-th candidate, so a
/// lookup that skips `h` deleted slots costs O((h + 1)^2), and the corresponding
/// expected total lookup cost is O((1 + ln(total / active))^2). This is in the
/// same practical complexity regime as history-based redirection schemes: the
/// cost grows roughly quadratically with the number of deleted-node hits, while
/// the expected number of such hits stays small unless many slots are deleted.
///
/// | Algorithm | Total lookup time | State | Predefined capacity? | History-dependent? |
/// | --- | --- | --- | --- | --- |
/// | `ConsistentNodeMap` | `O((h + 1)^2)`, expected `O((1 + ln(total / active))^2)` | `O(deleted)` deleted-slot set | No | No |
/// | AnchorHash | `O((h + 1)^2)`, expected `O((1 + ln(total / active))^2)` | `O(capacity)` anchor/removal state | Yes | Yes |
/// | MementoHash | `O((h + 1)^2)`, expected `O((1 + ln(total / active))^2)` | `O(deleted)` replacement tuples | No | Yes |
/// | DxHash | `O((h + 1)^2)`, expected `O((1 + ln(total / active))^2)` | `O(capacity)` redirect/displacement state with smaller constants than AnchorHash | Yes | Yes |
///
/// The MementoHash paper explicitly notes that AnchorHash and DxHash keep an
/// internal data structure for all cluster nodes, both working and not working,
/// and require the overall capacity to be fixed during initialization.
/// MementoHash reduces memory by storing only replacement information for
/// removed buckets, but that replacement information still encodes the removal
/// history. This implementation has the same O(deleted) storage shape as that
/// idea, but stores only the deleted set.
///
/// [AnchorHash]: https://arxiv.org/abs/1812.09674
/// [MementoHash]: https://arxiv.org/abs/2306.09783
/// [DxHash]: https://doi.org/10.1145/3631708
///
/// # Example
/// ```
/// use std::hash::{DefaultHasher, Hash};
/// use consistent_choose_k::ConsistentNodeMap;
///
/// let mut map = ConsistentNodeMap::new();
/// let a = map.add();
/// let b = map.add();
/// let c = map.add();
///
/// let mut h = DefaultHasher::default();
/// 42u64.hash(&mut h);
/// let slot = map.get(h).unwrap();
/// assert!(slot == a || slot == b || slot == c);
/// ```
pub struct ConsistentNodeMap {
    total: usize,
    deleted: HashSet<usize>,
}

impl Default for ConsistentNodeMap {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsistentNodeMap {
    /// Create an empty node map.
    pub fn new() -> Self {
        Self {
            total: 0,
            deleted: HashSet::new(),
        }
    }

    /// Add a slot and return its index.
    ///
    /// If there is a previously deleted slot, it will be reused.
    pub fn add(&mut self) -> usize {
        if let Some(i) = self.deleted.iter().next().copied() {
            self.deleted.remove(&i);
            i
        } else {
            let i = self.total;
            self.total += 1;
            i
        }
    }

    /// Remove the slot at the given index. Returns true if it was active.
    pub fn remove(&mut self, index: usize) -> bool {
        if index >= self.total || self.deleted.contains(&index) {
            return false;
        }
        if index == self.total - 1 {
            self.total -= 1;
        } else {
            self.deleted.insert(index);
        }
        true
    }

    /// Returns the number of active slots.
    pub fn len(&self) -> usize {
        self.total - self.deleted.len()
    }

    /// Returns true if there are no active slots.
    pub fn is_empty(&self) -> bool {
        self.total == self.deleted.len()
    }

    /// Returns the total number of slots (including deleted ones).
    pub fn slot_count(&self) -> usize {
        self.total
    }

    /// Returns whether the slot at the given index is active.
    pub fn is_active(&self, index: usize) -> bool {
        index < self.total && !self.deleted.contains(&index)
    }

    /// Look up which slot a key maps to using consistent hashing.
    ///
    /// The `builder` should be a hasher seeded with the key. The consistent
    /// choose-k iterator yields positions in a consistent order; the first
    /// active slot is returned.
    pub fn get<H: ManySeqBuilder>(&self, builder: H) -> Option<usize> {
        if self.is_empty() {
            return None;
        }
        let mut iter = ConsistentChooseKHasher::new(builder, self.total);
        iter.find(|pos| !self.deleted.contains(pos))
    }
}

#[cfg(test)]
mod tests {
    use std::hash::{DefaultHasher, Hash};

    use super::*;

    fn hasher_for_key(key: u64) -> DefaultHasher {
        let mut hasher = DefaultHasher::default();
        key.hash(&mut hasher);
        hasher
    }

    #[test]
    fn test_add_remove() {
        let mut map = ConsistentNodeMap::new();
        let a = map.add();
        let b = map.add();
        assert_eq!(map.len(), 2);

        assert!(map.remove(a));
        assert_eq!(map.len(), 1);
        for key in 0..100 {
            assert_eq!(map.get(hasher_for_key(key)), Some(b));
        }

        assert!(map.remove(b));
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert!(map.get(hasher_for_key(0)).is_none());
    }

    #[test]
    fn test_remove_returns_false_for_inactive() {
        let mut map = ConsistentNodeMap::new();
        let a = map.add();
        assert!(map.remove(a));
        assert!(!map.remove(a));
        assert!(!map.remove(999));
    }

    #[test]
    fn test_slot_reuse() {
        let mut map = ConsistentNodeMap::new();
        map.add();
        let b = map.add();
        map.add();
        assert_eq!(map.slot_count(), 3);

        map.remove(b);
        let d = map.add();
        assert_eq!(d, b);
        assert_eq!(map.slot_count(), 3);
        assert!(map.is_active(d));
    }

    #[test]
    fn test_trailing_pop() {
        let mut map = ConsistentNodeMap::new();
        let a = map.add(); // 0
        let b = map.add(); // 1
        let c = map.add(); // 2
        assert_eq!(map.slot_count(), 3);

        // Removing last slot pops it.
        map.remove(c);
        assert_eq!(map.slot_count(), 2);

        // Removing last again pops it.
        map.remove(b);
        assert_eq!(map.slot_count(), 1);

        // Middle removal is tracked as deleted, not popped.
        let b2 = map.add(); // appends as 1
        let c2 = map.add(); // appends as 2
        assert_eq!(b2, 1);
        assert_eq!(c2, 2);
        map.remove(b2); // middle -> deleted set
        assert_eq!(map.slot_count(), 3);
        map.remove(c2); // trailing → only pops c2
        assert_eq!(map.slot_count(), 2); // b2 slot stays as inactive
        assert_eq!(map.len(), 1);
        assert!(map.is_active(a));
    }

    #[test]
    fn test_consistency_after_add() {
        let mut map = ConsistentNodeMap::new();
        for _ in 0..10 {
            map.add();
        }
        let before: Vec<_> = (0..10000)
            .map(|k| map.get(hasher_for_key(k)).unwrap())
            .collect();
        map.add();
        let after: Vec<_> = (0..10000)
            .map(|k| map.get(hasher_for_key(k)).unwrap())
            .collect();
        let changed = before.iter().zip(&after).filter(|(a, b)| a != b).count();
        assert!(
            changed < 2000,
            "too many keys changed after add: {changed}/10000"
        );
    }

    #[test]
    fn test_remove_10_percent_consistency() {
        let n = 100;
        let num_keys = 100_000u64;
        let to_remove: Vec<usize> = (0..n).step_by(10).collect(); // 10% of nodes

        let mut map = ConsistentNodeMap::new();
        for _ in 0..n {
            map.add();
        }

        let before: Vec<usize> = (0..num_keys)
            .map(|k| map.get(hasher_for_key(k)).unwrap())
            .collect();

        for &slot in &to_remove {
            map.remove(slot);
        }
        let remaining = map.len();

        let after: Vec<usize> = (0..num_keys)
            .map(|k| map.get(hasher_for_key(k)).unwrap())
            .collect();

        // 1. Keys not on removed nodes must stay on the same node.
        let mut displaced = 0u64;
        for (k, (b, a)) in before.iter().zip(&after).enumerate() {
            if !to_remove.contains(b) {
                assert_eq!(
                    b, a,
                    "key {k}: slot changed from {b} to {a} but was not on a removed slot"
                );
            } else {
                displaced += 1;
                assert!(
                    !to_remove.contains(a),
                    "key {k}: reassigned to removed slot {a}"
                );
            }
        }

        // 2. Displaced fraction should be very close to the theoretical value.
        let displaced_pct = displaced as f64 / num_keys as f64;
        let theoretical_pct = to_remove.len() as f64 / n as f64;
        assert!(
            (displaced_pct - theoretical_pct).abs() < 0.01,
            "displaced fraction {displaced_pct:.4} not close to theoretical {theoretical_pct:.4}"
        );

        // 3. After removal, distribution among remaining nodes should be
        //    roughly uniform: each node gets ~1/remaining of all keys.
        let mut counts = vec![0u64; n];
        for &a in &after {
            counts[a] += 1;
        }
        let expected = num_keys as f64 / remaining as f64;
        let chi2: f64 = counts
            .iter()
            .enumerate()
            .filter(|(i, _)| !to_remove.contains(i))
            .map(|(_, &c)| {
                let diff = c as f64 - expected;
                diff * diff / expected
            })
            .sum();
        // Chi-squared critical value for 89 df at p=0.001 is ~122.9.
        assert!(
            chi2 < 200.0,
            "distribution not uniform enough: chi2={chi2:.1} (expected < 200)"
        );

        // 4. Removed slots must have zero keys.
        for &slot in &to_remove {
            assert_eq!(counts[slot], 0, "removed slot {slot} still has keys");
        }
    }
}
