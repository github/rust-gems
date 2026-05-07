use core::mem::MaybeUninit;
use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

use smallvec::SmallVec;

use super::container::HashSortedContainer;
use super::group::Group;
use super::group_ops::{self, CTRL_EMPTY, GROUP_SIZE};

pub(crate) use super::group::NO_OVERFLOW;

// ── Helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn tag(hash: u64) -> u8 {
    (hash as u8) | 0x80
}

#[inline]
fn slot_hint(hash: u64) -> usize {
    ((hash >> 7) & (GROUP_SIZE as u64 - 1)) as usize
}

// ────────────────────────────────────────────────────────────────────────
// HashSortedMap — wraps a container with a hash builder
// ────────────────────────────────────────────────────────────────────────

/// Insertion-only hash map with SIMD group scanning.
///
/// Uses NEON on aarch64, SSE2 on x86_64, scalar fallback elsewhere.
/// Generic over key type `K`, value type `V`, and hash builder `S`.
pub struct HashSortedMap<K, V, S = RandomState> {
    pub(crate) container: HashSortedContainer<K, V>,
    hash_builder: S,
}

impl<K: Hash + Eq, V> Default for HashSortedMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq, V> HashSortedMap<K, V> {
    pub fn new() -> Self {
        Self::with_capacity_and_hasher(0, RandomState::new())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, RandomState::new())
    }
}

impl<K, V, S> HashSortedMap<K, V, S> {
    pub fn with_hasher(hash_builder: S) -> Self {
        Self::with_capacity_and_hasher(0, hash_builder)
    }

    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        let adjusted = (capacity as f64 / group_ops::MAX_FILL).ceil() as usize;
        let min_groups = (adjusted.div_ceil(GROUP_SIZE)).max(1).next_power_of_two();
        let n_bits = min_groups.trailing_zeros().max(1);
        Self {
            container: HashSortedContainer::new(n_bits),
            hash_builder,
        }
    }

    pub fn len(&self) -> usize {
        self.container.len
    }

    pub fn is_empty(&self) -> bool {
        self.container.len == 0
    }

    /// Consume the map, returning the underlying container and hash builder.
    pub fn into_parts(self) -> (HashSortedContainer<K, V>, S) {
        // Prevent Drop from running on self — we're moving fields out.
        let this = std::mem::ManuallyDrop::new(self);
        unsafe {
            let container = std::ptr::read(&this.container);
            let hash_builder = std::ptr::read(&this.hash_builder);
            (container, hash_builder)
        }
    }
}

impl<K: Hash + Eq, V, S: BuildHasher> HashSortedMap<K, V, S> {
    /// Sort all entries within each primary group chain by their hash value
    /// and return the underlying container.
    ///
    /// After sorting, iteration visits entries in hash order within each
    /// primary group (and since primary groups are visited in group-index
    /// order, the overall iteration is in full hash order).
    ///
    /// Consumes the map because lookups via `get()` won't work correctly
    /// after sorting (the preferred `slot_hint` position might now be empty,
    /// breaking an invariant).
    ///
    /// # Complexity
    ///
    /// Each of `n` elements hashes uniformly into one of `m` primary groups,
    /// so chain lengths follow `X_i ~ Binomial(n, 1/m)` with `E[X_i] = n/m`.
    /// With a quadratic sort per chain the total expected cost is:
    ///
    /// ```text
    /// Σ E[X_i²] = m · (Var[X_i] + E[X_i]²)
    ///           = m · (n/m · (1 − 1/m) + n²/m²)
    ///           = n · (1 − 1/m) + n²/m
    /// ```
    ///
    /// Dividing by `n` gives the expected cost per element: `1 + n/m` (for
    /// `m ≫ 1`). Since `n/m` is the average chain length, bounded by
    /// `GROUP_SIZE / MAX_FILL ≈ 16`, the per-element cost stays constant.
    pub fn sort_by_hash(mut self) -> HashSortedContainer<K, V> {
        let num_primary = 1usize << self.container.n_bits;
        let mut chain: SmallVec<[u32; 4]> = SmallVec::new();
        let mut hashes: SmallVec<[u64; 16]> = SmallVec::new();

        for primary_gi in 0..num_primary {
            chain.clear();
            hashes.clear();

            // Collect group indices in this chain.
            let mut gi = primary_gi;
            loop {
                chain.push(gi as u32);
                let overflow = self.container.groups[gi].overflow;
                if overflow == NO_OVERFLOW {
                    break;
                }
                gi = overflow as usize;
            }
            // All groups before the last are fully packed (overflow is only
            // allocated when the previous group is full). Compute hashes for
            // those directly.
            for &cgi in &chain[..chain.len() - 1] {
                let g = &self.container.groups[cgi as usize];
                for slot in 0..GROUP_SIZE {
                    let hash = self
                        .hash_builder
                        .hash_one(unsafe { g.keys[slot].assume_init_ref() });
                    hashes.push(hash);
                }
            }
            // The last group may have gaps — compact it to the front.
            let last_gi = *chain.last().unwrap() as usize;
            compact_last_group(&mut self.container.groups[last_gi], &self.hash_builder, &mut hashes);
            let n = hashes.len();
            // Insertion sort by hash.
            for i in 1..n {
                let mut j = i;
                while j > 0 && hashes[j - 1] > hashes[j] {
                    hashes.swap(j - 1, j);
                    swap_chain_slots(&mut self.container.groups, &chain, j - 1, j);
                    j -= 1;
                }
            }

        }
        self.container
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let hash = self.hash_builder.hash_one(&key);
        self.insert_hashed(hash, key, value)
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hash_builder.hash_one(key);
        self.get_hashed(hash, key)
    }

    /// Returns a mutable reference to the value for `key`, inserting `f()` if absent.
    #[inline]
    pub fn get_or_insert_with<F: FnOnce() -> V>(&mut self, key: K, f: F) -> &mut V {
        self.entry(key).or_insert_with(f)
    }

    /// Returns a mutable reference to the value for `key`, inserting `V::default()` if absent.
    pub fn get_or_default(&mut self, key: K) -> &mut V
    where
        V: Default,
    {
        self.get_or_insert_with(key, V::default)
    }

    /// Returns an [`Entry`] for `key`, providing in-place access to its value
    /// (insertion, mutation, or read). The lookup chain is walked exactly once;
    /// the resulting `VacantEntry` already knows where to write.
    #[inline]
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, S> {
        let hash = self.hash_builder.hash_one(&key);
        match self.find_or_insertion_slot(hash, &key) {
            FindResult::Found(ptr) => Entry::Occupied(OccupiedEntry {
                // SAFETY: pointer is valid for `'_` (bounded by `&mut self`).
                value: unsafe { &mut *ptr },
            }),
            FindResult::Vacant(insertion) => Entry::Vacant(VacantEntry {
                phantom: PhantomData,
                map: self,
                hash,
                key,
                insertion,
            }),
        }
    }

    fn insert_hashed(&mut self, hash: u64, key: K, value: V) -> Option<V> {
        let tag = tag(hash);
        let mut gi = self.container.group_index(hash);
        loop {
            let group = &mut self.container.groups[gi];
            // SIMD scan group for tag match.
            let mut tag_mask = group_ops::match_tag(&group.ctrl, tag);
            while let Some(i) = group_ops::next_match(&mut tag_mask) {
                if unsafe { group.keys[i].assume_init_ref() } == &key {
                    let old =
                        std::mem::replace(unsafe { group.values[i].assume_init_mut() }, value);
                    return Some(old);
                }
            }
            // Check for empty slot in this group.
            let empty_mask = group_ops::match_empty(&group.ctrl);
            if empty_mask != 0 {
                let i = group_ops::lowest(empty_mask);
                group.ctrl[i] = tag;
                group.keys[i] = MaybeUninit::new(key);
                group.values[i] = MaybeUninit::new(value);
                self.container.len += 1;
                return None;
            }
            // Group full — follow or create overflow chain.
            let overflow = group.overflow;
            if overflow != NO_OVERFLOW {
                gi = overflow as usize;
            } else {
                if self.container.num_groups as usize == self.container.groups.len() {
                    self.grow();
                    // n_bits changed; recompute the primary group and retry.
                    gi = self.container.group_index(hash);
                    continue;
                }
                let new_gi = self.container.num_groups as usize;
                self.container.num_groups += 1;
                self.container.groups[gi].overflow = new_gi as u32;
                let group = &mut self.container.groups[new_gi];
                group.ctrl[0] = tag;
                group.keys[0] = MaybeUninit::new(key);
                group.values[0] = MaybeUninit::new(value);
                self.container.len += 1;
                return None;
            }
        }
    }

    fn get_hashed<Q>(&self, hash: u64, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let tag = tag(hash);
        let mut gi = self.container.group_index(hash);

        loop {
            let group = &self.container.groups[gi];
            // SIMD scan group for tag match.
            let mut tag_mask = group_ops::match_tag(&group.ctrl, tag);
            while let Some(i) = group_ops::next_match(&mut tag_mask) {
                if unsafe { group.keys[i].assume_init_ref() }.borrow() == key {
                    return Some(unsafe { group.values[i].assume_init_ref() });
                }
            }
            if group_ops::match_empty(&group.ctrl) != 0 {
                return None;
            }
            if group.overflow == NO_OVERFLOW {
                return None;
            }
            gi = group.overflow as usize;
        }
    }

    /// Single-walk variant that returns either the found slot or precise
    /// information about where to insert. Used by [`entry`].
    ///
    /// Returns raw pointers (instead of indices) so the caller can write
    /// directly without re-indexing. Pointers remain valid for the lifetime
    /// of `&mut self` until any reallocation (`grow`).
    fn find_or_insertion_slot(&mut self, hash: u64, key: &K) -> FindResult<K, V> {
        let tag = tag(hash);
        let mut gi = self.container.group_index(hash);

        loop {
            let group = &mut self.container.groups[gi];

            // SIMD scan group for tag match.
            let mut tag_mask = group_ops::match_tag(&group.ctrl, tag);
            while let Some(i) = group_ops::next_match(&mut tag_mask) {
                if unsafe { group.keys[i].assume_init_ref() } == key {
                    return FindResult::Found(group.values[i].as_mut_ptr());
                }
            }
            // Check for empty slot in this group.
            let empty_mask = group_ops::match_empty(&group.ctrl);
            if empty_mask != 0 {
                let i = group_ops::lowest(empty_mask);
                return FindResult::Vacant(Insertion::Empty {
                    group: group as *mut _,
                    slot: i,
                });
            }
            // Group full — follow or report end of chain.
            if group.overflow == NO_OVERFLOW {
                return FindResult::Vacant(Insertion::NeedsOverflow {
                    tail: group as *mut _,
                });
            }
            gi = group.overflow as usize;
        }
    }

    fn grow(&mut self) {
        let old_groups = std::mem::replace(
            &mut self.container.groups,
            Vec::<Group<K, V>>::new().into_boxed_slice(),
        );
        let old_num_groups = self.container.num_groups as usize;
        let old_len = self.container.len;

        self.container.n_bits += 1;
        let (new_groups, num_primary) = HashSortedContainer::alloc_groups(self.container.n_bits);
        self.container.groups = new_groups;
        self.container.num_groups = num_primary;
        self.container.len = 0;

        for group in &old_groups[..old_num_groups] {
            let mut full_mask = group_ops::match_full(&group.ctrl);
            while let Some(i) = group_ops::next_match(&mut full_mask) {
                let hash = self
                    .hash_builder
                    .hash_one(unsafe { group.keys[i].assume_init_ref() });
                self.insert_for_grow(hash, group.keys[i].as_ptr(), group.values[i].as_ptr());
            }
        }
        // Group<K, V> has no Drop (keys/values are MaybeUninit), so dropping
        // old_groups runs no destructors but does free the backing buffer.
        drop(old_groups);

        debug_assert_eq!(self.container.len, old_len);
    }

    fn insert_for_grow(&mut self, hash: u64, key_src: *const K, value_src: *const V) {
        let tag = tag(hash);
        let mut hint = slot_hint(hash);
        let gi = self.container.group_index(hash);
        let mut group = &mut self.container.groups[gi];

        loop {
            if group.ctrl[hint] == CTRL_EMPTY {
                break;
            }
            let empty_mask = group_ops::match_empty(&group.ctrl);
            if empty_mask != 0 {
                hint = group_ops::lowest(empty_mask);
                break;
            }
            let overflow = group.overflow;
            if overflow != NO_OVERFLOW {
                group = &mut self.container.groups[overflow as usize];
            } else {
                let new_gi = self.container.num_groups as usize;
                group.overflow = new_gi as u32;
                self.container.num_groups += 1;
                group = &mut self.container.groups[new_gi];
                break;
            }
        }
        group.ctrl[hint] = tag;
        unsafe {
            group.keys[hint]
                .as_mut_ptr()
                .copy_from_nonoverlapping(key_src, 1);
            group.values[hint]
                .as_mut_ptr()
                .copy_from_nonoverlapping(value_src, 1);
        }
        self.container.len += 1;
    }
}

// ── Chain-slot helpers for sort_by_hash ─────────────────────────────────

/// Map a flat position (0..chain.len()*GROUP_SIZE) to a (group_index, slot).
#[inline]
fn chain_slot(chain: &[u32], pos: usize) -> (usize, usize) {
    (chain[pos / GROUP_SIZE] as usize, pos % GROUP_SIZE)
}

/// Compact the last group in a chain: move all occupied entries to slots
/// 0..n and clear the rest. Computes hashes for each occupied entry and
/// appends them to `hashes`.
fn compact_last_group<K: Hash, V, S: BuildHasher>(
    group: &mut Group<K, V>,
    hash_builder: &S,
    hashes: &mut SmallVec<[u64; 16]>,
) {
    let mut write = 0usize;
    let mut full_mask = group_ops::match_full(&group.ctrl);
    while let Some(read) = group_ops::next_match(&mut full_mask) {
        let hash = hash_builder.hash_one(unsafe { group.keys[read].assume_init_ref() });
        hashes.push(hash);
        if read != write {
            unsafe {
                group.keys[write] = std::ptr::read(&group.keys[read]);
                group.values[write] = std::ptr::read(&group.values[read]);
            }
        }
        write += 1;
    }
    // Fix ctrl bytes: only the top bit matters (full vs empty).
    for slot in 0..write {
        group.ctrl[slot] = 0x80;
    }
    for slot in write..GROUP_SIZE {
        group.ctrl[slot] = CTRL_EMPTY;
    }
}

/// Swap the ctrl byte, key, and value between two flat positions in a chain.
fn swap_chain_slots<K, V>(
    groups: &mut [Group<K, V>],
    chain: &[u32],
    a: usize,
    b: usize,
) {
    let (gi_a, slot_a) = chain_slot(chain, a);
    let (gi_b, slot_b) = chain_slot(chain, b);
    if gi_a == gi_b {
        let g = &mut groups[gi_a];
        g.keys.swap(slot_a, slot_b);
        g.values.swap(slot_a, slot_b);
    } else {
        let (ga, gb) = if gi_a < gi_b {
            let (left, right) = groups.split_at_mut(gi_b);
            (&mut left[gi_a], &mut right[0])
        } else {
            let (left, right) = groups.split_at_mut(gi_a);
            (&mut right[0], &mut left[gi_b])
        };
        std::mem::swap(&mut ga.keys[slot_a], &mut gb.keys[slot_b]);
        std::mem::swap(&mut ga.values[slot_a], &mut gb.values[slot_b]);
    }
}

// ────────────────────────────────────────────────────────────────────────
// Entry API
// ────────────────────────────────────────────────────────────────────────

/// Result of a single chain walk during `entry()`: either the existing slot
/// for the key or a pre-computed insertion location for a vacant entry.
enum FindResult<K, V> {
    /// Pointer to the existing value.
    Found(*mut V),
    /// Where to insert if the caller decides to add a new entry.
    Vacant(Insertion<K, V>),
}

/// Pre-computed insertion location stashed inside [`VacantEntry`] so that
/// `insert()` doesn't need to re-walk the chain. Pointers remain valid as
/// long as no reallocation occurs (the grow path re-walks via the slow path).
enum Insertion<K, V> {
    /// An empty slot is waiting at `(group, slot)`.
    Empty {
        group: *mut Group<K, V>,
        slot: usize,
    },
    /// The chain is full; allocate a new overflow group and link via `tail`.
    NeedsOverflow { tail: *mut Group<K, V> },
}

/// View into a single entry in a [`HashSortedMap`], either occupied or vacant.
pub enum Entry<'a, K, V, S> {
    Occupied(OccupiedEntry<'a, V>),
    Vacant(VacantEntry<'a, K, V, S>),
}

/// View into an occupied entry.
pub struct OccupiedEntry<'a, V> {
    value: &'a mut V,
}

/// View into a vacant entry. Holds the borrow of the map plus the hash, key,
/// and pre-computed insertion slot.
pub struct VacantEntry<'a, K, V, S> {
    phantom: PhantomData<&'a mut HashSortedMap<K, V, S>>,
    map: *mut HashSortedMap<K, V, S>,
    hash: u64,
    key: K,
    insertion: Insertion<K, V>,
}

impl<'a, K: Hash + Eq, V, S: BuildHasher> Entry<'a, K, V, S> {
    /// Insert `default` if vacant; return a mutable reference to the value either way.
    #[inline]
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => v.insert(default),
        }
    }

    /// Insert `f()` if vacant; `f` runs only on the vacant branch.
    #[inline]
    pub fn or_insert_with<F: FnOnce() -> V>(self, f: F) -> &'a mut V {
        match self {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => v.insert(f()),
        }
    }

    /// Insert `V::default()` if vacant.
    #[inline]
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }

    /// Apply `f` to the value if occupied; pass through unchanged otherwise.
    #[inline]
    pub fn and_modify<F: FnOnce(&mut V)>(self, f: F) -> Self {
        match self {
            Entry::Occupied(mut o) => {
                f(o.get_mut());
                Entry::Occupied(o)
            }
            v @ Entry::Vacant(_) => v,
        }
    }
}

impl<'a, V> OccupiedEntry<'a, V> {
    /// Get a shared reference to the value.
    #[inline]
    pub fn get(&self) -> &V {
        &*self.value
    }

    /// Get a mutable reference to the value.
    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        self.value
    }

    /// Consume the entry, returning the mutable reference with the entry's lifetime.
    #[inline]
    pub fn into_mut(self) -> &'a mut V {
        self.value
    }
}

impl<'a, K: Hash + Eq, V, S: BuildHasher> VacantEntry<'a, K, V, S> {
    /// Insert `value` and return a mutable reference to it.
    /// Writes directly to the slot pre-computed during `entry()`; only re-walks
    /// the chain on the rare grow path (where the pre-computed pointers become
    /// stale because grow re-allocates the groups buffer).
    #[inline]
    pub fn insert(self, value: V) -> &'a mut V {
        let map = self.map;
        let hash = self.hash;
        let key = self.key;

        let (group_ptr, slot) = match self.insertion {
            Insertion::Empty { group, slot } => (group, slot),
            Insertion::NeedsOverflow { tail } => {
                let (new_gi, new_group) = unsafe {
                    let map = &mut *map;
                    if map.container.num_groups as usize == map.container.groups.len() {
                        return insert_after_grow(map, hash, key, value);
                    }
                    let new_gi = map.container.num_groups as usize;
                    map.container.num_groups += 1;
                    let new_group: *mut Group<K, V> = &mut map.container.groups[new_gi];
                    (new_gi, new_group)
                };
                unsafe {
                    // SAFETY: `tail` was obtained from `&mut self.container.groups[..]` and
                    // remains valid because no reallocation occurred between
                    // `entry()` and now (we hold the only `&mut self`).
                    (*tail).overflow = new_gi as u32;
                }
                (new_group, 0)
            }
        };

        let tag = tag(hash);
        unsafe {
            (*map).container.len += 1;
            // SAFETY: `group_ptr` points into `map.container.groups` and is valid for `'a`.
            let group = &mut *group_ptr;
            group.ctrl[slot] = tag;
            group.keys[slot] = MaybeUninit::new(key);
            group.values[slot] = MaybeUninit::new(value);
            group.values[slot].assume_init_mut()
        }
    }
}

/// Cold path: the chain was full, the table is at capacity, and we need to
/// grow before inserting. Grows the map, then re-walks via `entry()` to find
/// the new insertion slot.
#[cold]
#[inline(never)]
fn insert_after_grow<K: Hash + Eq, V, S: BuildHasher>(
    map: &mut HashSortedMap<K, V, S>,
    _hash: u64,
    key: K,
    value: V,
) -> &mut V {
    map.grow();
    map.entry(key).or_insert(value)
}

// No custom Drop needed for HashSortedMap — dropping `container` handles entries.

#[cfg(test)]
mod tests {
    use std::hash::{BuildHasher, Hasher};

    use super::*;

    #[test]
    fn insert_and_get() {
        let mut map = HashSortedMap::new();
        map.insert(100, "hello");
        map.insert(200, "world");
        assert_eq!(map.get(&100), Some(&"hello"));
        assert_eq!(map.get(&200), Some(&"world"));
        assert_eq!(map.get(&999), None);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn insert_overwrite() {
        let mut map = HashSortedMap::new();
        map.insert(42, "a");
        assert_eq!(map.insert(42, "b"), Some("a"));
        assert_eq!(map.get(&42), Some(&"b"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn grow_preserves_entries() {
        let mut map = HashSortedMap::new();
        for i in 0..200u32 {
            map.insert(i, i * 10);
        }
        assert_eq!(map.len(), 200);
        for i in 0..200u32 {
            assert_eq!(map.get(&i), Some(&(i * 10)), "missing key {i}");
        }
    }

    #[test]
    fn many_entries() {
        let mut map = HashSortedMap::with_capacity(2000);
        for i in 0..2000u32 {
            map.insert(i.wrapping_mul(2654435761), i);
        }
        assert_eq!(map.len(), 2000);
        for i in 0..2000u32 {
            assert_eq!(map.get(&i.wrapping_mul(2654435761)), Some(&i));
        }
    }

    #[test]
    fn overflow_chain() {
        let mut map = HashSortedMap::with_capacity(8);
        for i in 0..20u32 {
            let key = i | 0xAB000000;
            map.insert(key, i);
        }
        assert_eq!(map.len(), 20);
        for i in 0..20u32 {
            let key = i | 0xAB000000;
            assert_eq!(map.get(&key), Some(&i), "missing key {key:#x}");
        }
    }

    #[test]
    fn grow_on_overflow_exhaustion() {
        let mut map = HashSortedMap::with_capacity(1);
        let old_n_bits = map.container.n_bits;
        for i in 0..100u32 {
            let key = i | 0xFF000000;
            map.insert(key, i);
        }
        assert!(map.container.n_bits > old_n_bits, "should have grown");
        assert_eq!(map.len(), 100);
        for i in 0..100u32 {
            let key = i | 0xFF000000;
            assert_eq!(map.get(&key), Some(&i), "missing key {key:#x} after grow");
        }
    }

    #[test]
    fn string_keys() {
        let mut map = HashSortedMap::new();
        map.insert("hello".to_string(), 1);
        map.insert("world".to_string(), 2);
        assert_eq!(map.get("hello"), Some(&1));
        assert_eq!(map.get("world"), Some(&2));
        assert_eq!(map.get("missing"), None);
        assert_eq!(map.len(), 2);

        assert_eq!(map.insert("hello".to_string(), 3), Some(1));
        assert_eq!(map.get("hello"), Some(&3));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn get_or_default_basics() {
        let mut map: HashSortedMap<&str, i32> = HashSortedMap::new();
        // Inserts default (0), then mutates.
        *map.get_or_default("a") += 5;
        *map.get_or_default("b") += 7;
        // Subsequent calls return the existing value.
        *map.get_or_default("a") += 3;
        assert_eq!(map.get(&"a"), Some(&8));
        assert_eq!(map.get(&"b"), Some(&7));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn get_or_insert_with_lazy() {
        let mut map: HashSortedMap<u32, String> = HashSortedMap::new();
        let mut call_count = 0;
        let mut make = |s: &str| {
            call_count += 1;
            s.to_string()
        };
        // First call: f runs, inserts "first".
        assert_eq!(
            map.get_or_insert_with(1, || make("first")),
            &mut "first".to_string()
        );
        // Second call with same key: f does NOT run; returns existing.
        assert_eq!(
            map.get_or_insert_with(1, || make("second")),
            &mut "first".to_string()
        );
        // New key: f runs.
        assert_eq!(
            map.get_or_insert_with(2, || make("third")),
            &mut "third".to_string()
        );
        assert_eq!(call_count, 2);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn get_or_default_survives_grow() {
        let mut map: HashSortedMap<u32, u32> = HashSortedMap::with_capacity(1);
        for i in 0..500u32 {
            *map.get_or_default(i) = i * 2;
        }
        assert_eq!(map.len(), 500);
        for i in 0..500u32 {
            assert_eq!(map.get(&i), Some(&(i * 2)), "missing key {i}");
        }
    }

    #[test]
    fn entry_or_default_counting() {
        // Classic counting workload via Entry API.
        let mut map: HashSortedMap<&str, u32> = HashSortedMap::new();
        for word in ["a", "b", "a", "c", "b", "a"] {
            *map.entry(word).or_default() += 1;
        }
        assert_eq!(map.get(&"a"), Some(&3));
        assert_eq!(map.get(&"b"), Some(&2));
        assert_eq!(map.get(&"c"), Some(&1));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn entry_or_insert_lazy() {
        let mut map: HashSortedMap<u32, String> = HashSortedMap::new();
        let mut call_count = 0;
        let mut make = |s: &str| {
            call_count += 1;
            s.to_string()
        };
        // First call: f runs, inserts.
        let v = map.entry(1).or_insert_with(|| make("first"));
        assert_eq!(v, "first");
        // Second call with same key: f does NOT run.
        let v = map.entry(1).or_insert_with(|| make("second"));
        assert_eq!(v, "first");
        assert_eq!(call_count, 1);
    }

    #[test]
    fn entry_and_modify() {
        let mut map: HashSortedMap<u32, u32> = HashSortedMap::new();
        // Vacant: and_modify is a no-op, then or_insert(0) runs.
        *map.entry(7).and_modify(|v| *v *= 10).or_insert(1) += 100;
        assert_eq!(map.get(&7), Some(&101));
        // Occupied: and_modify runs, or_insert is skipped.
        *map.entry(7).and_modify(|v| *v *= 2).or_insert(99) += 1;
        assert_eq!(map.get(&7), Some(&203));
    }

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
    fn test_collisions() {
        // Tiny initial capacity + all collisions
        let mut m = HashSortedMap::with_capacity_and_hasher(1, FixedState(0));
        for i in 0..200u32 {
            m.insert(i, i);
        }
        assert_eq!(m.len(), 200);
        for i in 0..200u32 {
            assert_eq!(m.get(&i), Some(&i));
        }
    }

    // ── sort_by_hash tests ──────────────────────────────────────────────

    #[test]
    fn sort_by_hash_empty() {
        let map: HashSortedMap<u32, u32> = HashSortedMap::new();
        let container = map.sort_by_hash();
        assert_eq!(container.len, 0);
    }

    #[test]
    fn sort_by_hash_single() {
        let mut map = HashSortedMap::new();
        map.insert(42u32, "hello");
        let container = map.sort_by_hash();
        assert_eq!(container.len, 1);
        let entries: Vec<_> = container.into_iter().collect();
        assert_eq!(entries, vec![(42, "hello")]);
    }

    #[test]
    fn sort_by_hash_preserves_entries() {
        let mut map = HashSortedMap::new();
        for i in 0..200u32 {
            map.insert(i, i * 10);
        }
        let container = map.sort_by_hash();
        assert_eq!(container.len, 200);
        let mut entries: Vec<_> = container.into_iter().collect();
        entries.sort_by_key(|&(k, _)| k);
        for i in 0..200u32 {
            assert_eq!(entries[i as usize], (i, i * 10), "missing key {i}");
        }
    }

    #[test]
    fn sort_by_hash_produces_hash_order() {
        use std::collections::hash_map::RandomState;

        let hasher = RandomState::new();
        let mut map = HashSortedMap::with_hasher(hasher.clone());
        for i in 0..500u32 {
            map.insert(i, i);
        }
        let container = map.sort_by_hash();
        // Iteration should now yield entries in hash order.
        let mut prev_hash = 0u64;
        let mut first = true;
        for (&k, _) in &container {
            let h = hasher.hash_one(&k);
            if !first {
                assert!(h > prev_hash, "hash order violated: {prev_hash:#x} > {h:#x}");
            }
            prev_hash = h;
            first = false;
        }
    }

    #[test]
    fn sort_by_hash_with_overflow() {
        // Force overflow chains via fixed hash, then sort.
        let mut map = HashSortedMap::with_capacity_and_hasher(1, FixedState(0));
        for i in 0..50u32 {
            map.insert(i, i);
        }
        let container = map.sort_by_hash();
        assert_eq!(container.len, 50);
        let mut entries: Vec<_> = container.into_iter().collect();
        entries.sort_by_key(|&(k, _)| k);
        for i in 0..50u32 {
            assert_eq!(entries[i as usize], (i, i), "missing key {i}");
        }
    }

    #[test]
    fn sort_by_hash_with_strings() {
        use std::collections::hash_map::RandomState;

        let hasher = RandomState::new();
        let mut map = HashSortedMap::with_hasher(hasher.clone());
        for i in 0..100u32 {
            map.insert(format!("key-{i}"), format!("val-{i}"));
        }
        let container = map.sort_by_hash();
        assert_eq!(container.len, 100);
        let mut prev_hash = 0u64;
        let mut first = true;
        for (k, _) in &container {
            let h = hasher.hash_one(k);
            if !first {
                assert!(h > prev_hash, "hash order violated: {prev_hash:#x} > {h:#x}");
            }
            prev_hash = h;
            first = false;
        }
    }
}
