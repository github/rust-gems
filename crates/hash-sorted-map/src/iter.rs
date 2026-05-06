use std::marker::PhantomData;
use std::mem::ManuallyDrop;

use super::group_ops::{self};
use super::hash_sorted_map::{Group, HashSortedMap, NO_OVERFLOW};

/// State shared by `Iter`, `IterMut`, and `IntoIter`: tracks which primary
/// group we're visiting and where we are within that group's overflow chain.
struct IterCursor {
    /// Index of the next primary group to visit (0..num_primary).
    primary: u32,
    /// Number of primary groups (1 << n_bits).
    num_primary: u32,
    /// Current position within the group we're scanning: group index in the
    /// groups array, and a SIMD bitmask of remaining occupied slots.
    current_group: u32,
    current_mask: group_ops::Mask,
}

impl IterCursor {
    fn new<K, V, S>(map: &HashSortedMap<K, V, S>) -> Self {
        let num_primary = 1u32 << map.n_bits;
        Self {
            primary: 0,
            num_primary,
            // Start past all allocated groups so the first call falls through to
            // "move to next primary group" rather than checking overflow on an
            // un-scanned group 0.
            current_group: map.groups.len() as u32,
            current_mask: 0,
        }
    }

    /// Advance to the next occupied slot, returning `(group_index, slot)`.
    /// Visits primary groups 0..num_primary in order; for each, follows the
    /// overflow chain. Within each group, yields occupied slots via bitmask.
    fn next_slot<K, V>(&mut self, groups: &[Group<K, V>]) -> Option<(usize, usize)> {
        loop {
            if let Some(slot) = group_ops::next_match(&mut self.current_mask) {
                return Some((self.current_group as usize, slot));
            }
            // Current group exhausted — try overflow chain.
            let gi = self.current_group as usize;
            if gi < groups.len() && groups[gi].overflow != NO_OVERFLOW {
                let next = groups[gi].overflow;
                self.current_group = next;
                self.current_mask = group_ops::match_full(&groups[next as usize].ctrl);
                continue;
            }
            // No more overflow — move to next primary group.
            if self.primary >= self.num_primary {
                return None;
            }
            let gi = self.primary as usize;
            self.primary += 1;
            self.current_group = gi as u32;
            self.current_mask = group_ops::match_full(&groups[gi].ctrl);
        }
    }
}

/// Immutable iterator over `(&K, &V)` pairs.
pub struct Iter<'a, K, V> {
    groups: &'a [Group<K, V>],
    cursor: IterCursor,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        let (gi, slot) = self.cursor.next_slot(self.groups)?;
        let group = &self.groups[gi];
        // SAFETY: slot is occupied (bitmask guarantees ctrl byte has high bit set).
        unsafe {
            Some((
                group.keys[slot].assume_init_ref(),
                group.values[slot].assume_init_ref(),
            ))
        }
    }
}

/// Mutable iterator over `(&K, &mut V)` pairs.
pub struct IterMut<'a, K, V> {
    groups: *mut [Group<K, V>],
    cursor: IterCursor,
    _marker: PhantomData<&'a mut HashSortedMap<K, V>>,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        // SAFETY: we use raw pointer to avoid holding multiple &mut borrows.
        // The cursor guarantees each slot is yielded at most once.
        let groups = unsafe { &mut *self.groups };
        let (gi, slot) = self.cursor.next_slot(groups)?;
        let group = &mut groups[gi];
        unsafe {
            Some((
                group.keys[slot].assume_init_ref(),
                group.values[slot].assume_init_mut(),
            ))
        }
    }
}

/// Owning iterator that yields `(K, V)` pairs and consumes the map.
pub struct IntoIter<K, V, S> {
    inner: ManuallyDrop<HashSortedMap<K, V, S>>,
    cursor: IterCursor,
}

impl<K, V, S> Iterator for IntoIter<K, V, S> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        let (gi, slot) = self.cursor.next_slot(&self.inner.groups)?;
        let group = &self.inner.groups[gi];
        // SAFETY: slot is occupied (bitmask guarantees ctrl byte has high bit set).
        unsafe {
            Some((
                group.keys[slot].assume_init_read(),
                group.values[slot].assume_init_read(),
            ))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.inner.len))
    }
}

impl<K, V, S> Drop for IntoIter<K, V, S> {
    fn drop(&mut self) {
        // Continue iterating to drop remaining entries one by one.
        while let Some((gi, slot)) = self.cursor.next_slot(&self.inner.groups) {
            unsafe {
                self.inner.groups[gi].keys[slot].assume_init_drop();
                self.inner.groups[gi].values[slot].assume_init_drop();
            }
        }
        // All entries consumed or dropped above. Set num_groups to 0 so the
        // map's Drop won't try to drop them again, then let it run to free
        // the groups allocation and drop hash_builder.
        self.inner.num_groups = 0;
        unsafe { ManuallyDrop::drop(&mut self.inner) };
    }
}

impl<K, V, S> HashSortedMap<K, V, S> {
    /// Returns an iterator over `(&K, &V)` pairs.
    ///
    /// Entries are visited in group-index order (primary groups in order of
    /// hash prefix, each followed by its overflow chain). Within each group,
    /// occupied slots are visited in slot order.
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            groups: &self.groups,
            cursor: IterCursor::new(self),
        }
    }

    /// Returns a mutable iterator over `(&K, &mut V)` pairs.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        let cursor = IterCursor::new(self);
        IterMut {
            groups: &mut *self.groups as *mut [Group<K, V>],
            cursor,
            _marker: PhantomData,
        }
    }

    /// Consumes the map and returns an iterator over `(K, V)` pairs.
    pub fn into_iter(self) -> IntoIter<K, V, S> {
        let cursor = IterCursor::new(&self);
        IntoIter {
            inner: ManuallyDrop::new(self),
            cursor,
        }
    }
}

impl<K, V, S> IntoIterator for HashSortedMap<K, V, S> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, S>;
    fn into_iter(self) -> Self::IntoIter {
        self.into_iter()
    }
}

impl<'a, K, V, S> IntoIterator for &'a HashSortedMap<K, V, S> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K, V, S> IntoIterator for &'a mut HashSortedMap<K, V, S> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use std::hash::{BuildHasher, Hasher};

    use super::*;

    /// Degenerate hasher that returns a fixed hash code, for forcing collisions.
    struct FixedHasher(u64);

    impl Hasher for FixedHasher {
        fn finish(&self) -> u64 {
            self.0
        }
        fn write(&mut self, _bytes: &[u8]) {}
    }

    #[derive(Clone)]
    struct FixedState(u64);

    impl BuildHasher for FixedState {
        type Hasher = FixedHasher;
        fn build_hasher(&self) -> FixedHasher {
            FixedHasher(self.0)
        }
    }

    #[test]
    fn iter_empty() {
        let map: HashSortedMap<u32, u32> = HashSortedMap::new();
        assert_eq!(map.iter().count(), 0);
    }

    #[test]
    fn iter_yields_all_entries() {
        let mut map = HashSortedMap::new();
        for i in 0..100u32 {
            map.insert(i, i * 10);
        }
        let mut collected: Vec<(u32, u32)> = map.iter().map(|(&k, &v)| (k, v)).collect();
        collected.sort();
        assert_eq!(collected.len(), 100);
        for i in 0..100u32 {
            assert_eq!(collected[i as usize], (i, i * 10));
        }
    }

    #[test]
    fn iter_with_overflow_chains() {
        let mut map = HashSortedMap::with_capacity_and_hasher(1, FixedState(0xABCD));
        for i in 0..50u32 {
            map.insert(i, i);
        }
        let collected: Vec<u32> = map.iter().map(|(&k, _)| k).collect();
        assert_eq!(collected.len(), 50);
        let mut sorted = collected.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 50);
    }

    #[test]
    fn iter_mut_mutates_values() {
        let mut map = HashSortedMap::new();
        for i in 0..20u32 {
            map.insert(i, i);
        }
        for (_, v) in map.iter_mut() {
            *v *= 2;
        }
        for i in 0..20u32 {
            assert_eq!(map.get(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn into_iter_yields_all() {
        let mut map = HashSortedMap::new();
        for i in 0..100u32 {
            map.insert(i, i * 3);
        }
        let mut collected: Vec<(u32, u32)> = map.into_iter().collect();
        collected.sort();
        assert_eq!(collected.len(), 100);
        for i in 0..100u32 {
            assert_eq!(collected[i as usize], (i, i * 3));
        }
    }

    #[test]
    fn into_iter_partial_consume_then_drop() {
        let mut map: HashSortedMap<String, String> = HashSortedMap::new();
        for i in 0..50u32 {
            map.insert(format!("key-{i}"), format!("val-{i}"));
        }
        let mut iter = map.into_iter();
        for _ in 0..10 {
            let _ = iter.next();
        }
        drop(iter);
    }

    #[test]
    fn into_iter_empty() {
        let map: HashSortedMap<u32, u32> = HashSortedMap::new();
        assert_eq!(map.into_iter().count(), 0);
    }

    #[test]
    fn into_iter_with_overflow() {
        let mut map = HashSortedMap::with_capacity_and_hasher(1, FixedState(0));
        for i in 0..80u32 {
            map.insert(i, i);
        }
        let collected: Vec<(u32, u32)> = map.into_iter().collect();
        assert_eq!(collected.len(), 80);
        let mut keys: Vec<u32> = collected.into_iter().map(|(k, _)| k).collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), 80);
    }

    #[test]
    fn into_iter_after_grow() {
        let mut map = HashSortedMap::with_capacity(1);
        for i in 0..500u32 {
            map.insert(i, i);
        }
        let collected: Vec<(u32, u32)> = map.into_iter().collect();
        assert_eq!(collected.len(), 500);
    }

    /// Track drops to verify no leaks or double-drops.
    #[test]
    fn into_iter_drop_count() {
        use std::cell::Cell;
        use std::rc::Rc;

        #[derive(Clone)]
        struct Tracked(Rc<Cell<usize>>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let counter = Rc::new(Cell::new(0usize));
        let n = 100;
        {
            let mut map = HashSortedMap::new();
            for i in 0..n {
                map.insert(i, Tracked(counter.clone()));
            }
            let mut iter = map.into_iter();
            for _ in 0..n / 2 {
                let _ = iter.next();
            }
        }
        assert_eq!(counter.get(), n);
    }

    #[test]
    fn for_loop_ref() {
        let mut map = HashSortedMap::new();
        map.insert(1, "a");
        map.insert(2, "b");
        let mut count = 0;
        for (_k, _v) in &map {
            count += 1;
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn for_loop_mut() {
        let mut map = HashSortedMap::new();
        map.insert(1u32, 10u32);
        map.insert(2, 20);
        for (_, v) in &mut map {
            *v += 1;
        }
        assert_eq!(map.get(&1), Some(&11));
        assert_eq!(map.get(&2), Some(&21));
    }

    #[test]
    fn for_loop_owned() {
        let mut map = HashSortedMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        let mut sum = 0;
        for (_k, v) in map {
            sum += v;
        }
        assert_eq!(sum, 30);
    }
}
