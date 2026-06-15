use core::mem::MaybeUninit;
use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

use super::group::Group;
use super::group_ops::{self, CTRL_EMPTY, GROUP_SIZE};

pub(crate) use super::group::NO_OVERFLOW;

// ── Helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn tag(hash: u64) -> u8 {
    (hash as u8) | 0x80
}

// ────────────────────────────────────────────────────────────────────────
// SortingHash
// ────────────────────────────────────────────────────────────────────────

/// Maps a key to the 64-bit hash that determines its position in the map.
///
/// The high bits select the primary group, so visiting groups in index order
/// yields entries in ascending hash order — the property [`HashSortedMap`]
/// relies on for sorted iteration and linear-time merging. The hash should
/// therefore be well distributed in its high bits.
///
/// Every [`BuildHasher`] (the default [`RandomState`], `foldhash`, `ahash`,
/// `fnv`, …) implements this trait automatically through a blanket impl, so
/// it can be used as a drop-in. For full control — including keys that do not
/// implement [`Hash`] — implement this single method directly instead of the
/// streaming [`Hasher`](std::hash::Hasher) interface:
///
/// ```
/// use hash_sorted_map::{HashSortedMap, SortingHash};
///
/// #[derive(Default)]
/// struct Identity;
/// impl SortingHash<u32> for Identity {
///     fn hash(&self, &key: &u32) -> u64 {
///         (key as u64) | ((key as u64) << 32)
///     }
/// }
///
/// let mut map = HashSortedMap::with_hasher(Identity);
/// map.insert(42u32, "answer");
/// assert_eq!(map.get(&42), Some(&"answer"));
/// ```
pub trait SortingHash<K: ?Sized> {
    /// Returns the hash of `key`.
    fn hash(&self, key: &K) -> u64;
}

/// Bridges the standard library's [`BuildHasher`] to [`SortingHash`], so any
/// existing hasher keeps working unchanged.
impl<K: Hash + ?Sized, S: BuildHasher> SortingHash<K> for S {
    #[inline]
    fn hash(&self, key: &K) -> u64 {
        self.hash_one(key)
    }
}

// ────────────────────────────────────────────────────────────────────────
// HashSortedMap
// ────────────────────────────────────────────────────────────────────────

/// Insertion-only hash map with SIMD group scanning.
///
/// Uses NEON on aarch64, SSE2 on x86_64, scalar fallback elsewhere.
/// Generic over key type `K`, value type `V`, and hashing strategy `S`
/// (any [`SortingHash<K>`](SortingHash), which every [`BuildHasher`] satisfies).
pub struct HashSortedMap<K, V, S = RandomState> {
    pub(crate) groups: Box<[Group<K, V>]>,
    pub(crate) num_groups: u32,
    pub(crate) n_bits: u32,
    pub(crate) len: usize,
    hasher: S,
}

impl<K: Hash + Eq, V> Default for HashSortedMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq, V> HashSortedMap<K, V> {
    /// Creates an empty map using the default [`RandomState`] hasher.
    pub fn new() -> Self {
        Self::with_capacity_and_hasher(0, RandomState::new())
    }

    /// Creates an empty map that can hold at least `capacity` entries without
    /// growing, using the default [`RandomState`] hasher.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, RandomState::new())
    }
}

impl<K, V, S> HashSortedMap<K, V, S> {
    /// Creates an empty map that hashes keys with `hasher`.
    pub fn with_hasher(hasher: S) -> Self {
        Self::with_capacity_and_hasher(0, hasher)
    }

    /// Creates an empty map that hashes keys with `hasher` and can hold at
    /// least `capacity` entries without growing.
    pub fn with_capacity_and_hasher(capacity: usize, hasher: S) -> Self {
        let adjusted = (capacity as f64 / group_ops::MAX_FILL).ceil() as usize;
        let min_groups = (adjusted.div_ceil(GROUP_SIZE)).max(1).next_power_of_two();
        let n_bits = min_groups.trailing_zeros().max(1);
        let (groups, num_groups) = Self::alloc_groups(n_bits);
        Self {
            groups,
            num_groups,
            n_bits,
            len: 0,
            hasher,
        }
    }

    /// Returns the number of entries in the map.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the map contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn alloc_groups(n_bits: u32) -> (Box<[Group<K, V>]>, u32) {
        let num_primary = 1usize << n_bits;
        let total = num_primary + num_primary / 8 + 1;
        let mut groups: Vec<Group<K, V>> = Vec::with_capacity(total);
        groups.resize_with(total, Group::new);
        (groups.into_boxed_slice(), num_primary as u32)
    }

    #[inline]
    pub(crate) fn group_index(&self, hash: u64) -> usize {
        (hash >> (64 - self.n_bits)) as usize
    }
}

impl<K: Eq + Ord, V, S: SortingHash<K>> HashSortedMap<K, V, S> {
    /// Sort all entries within each primary group chain by their hash value,
    /// breaking ties by key.
    ///
    /// After sorting, iteration visits entries in hash order within each
    /// primary group (and since primary groups are visited in group-index
    /// order, the overall iteration is in full hash order).
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
    /// `GROUP_SIZE / MAX_FILL`, the per-element cost stays constant.
    pub fn sort_by_hash(&mut self) {
        let num_primary = 1usize << self.n_bits;
        let mut chain: Vec<u32> = Vec::new();
        let mut hashes: Vec<u64> = Vec::new();

        for primary_gi in 0..num_primary {
            chain.clear();
            hashes.clear();

            // Collect group indices in this chain.
            let mut gi = primary_gi;
            loop {
                chain.push(gi as u32);
                let overflow = self.groups[gi].overflow;
                if overflow == NO_OVERFLOW {
                    break;
                }
                gi = overflow as usize;
            }
            // All groups before the last are fully packed (overflow is only
            // allocated when the previous group is full). Compute hashes for
            // those directly.
            for &cgi in &chain[..chain.len() - 1] {
                let g = &self.groups[cgi as usize];
                for slot in 0..GROUP_SIZE {
                    let hash = self.hasher.hash(unsafe { g.keys[slot].assume_init_ref() });
                    hashes.push(hash);
                }
            }
            let g =
                &self.groups[*chain.last().expect("chain should have at least one group") as usize];
            for slot in 0..GROUP_SIZE {
                if g.ctrl[slot] == CTRL_EMPTY {
                    break;
                }
                let hash = self.hasher.hash(unsafe { g.keys[slot].assume_init_ref() });
                hashes.push(hash);
            }

            let n = hashes.len();
            // Insertion sort by (hash, key).
            for i in 1..n {
                // Extract element at position i.
                let cur_hash = hashes[i];
                let (gi, si) = chain_slot(&chain, i);
                let cur_key = unsafe { self.groups[gi].keys[si].assume_init_read() };
                let cur_val = unsafe { self.groups[gi].values[si].assume_init_read() };
                // Find insertion point via linear scan backward.
                let mut j = i;
                while j > 0 {
                    let (gj, sj) = chain_slot(&chain, j - 1);
                    let prev_key = unsafe { self.groups[gj].keys[sj].assume_init_ref() };
                    if (hashes[j - 1], prev_key) <= (cur_hash, &cur_key) {
                        break;
                    }
                    j -= 1;
                }
                if j < i {
                    // Shift positions j..i up by one.
                    hashes.copy_within(j..i, j + 1);
                    for pos in (j..i).rev() {
                        let (src_g, src_s) = chain_slot(&chain, pos);
                        let (dst_g, dst_s) = chain_slot(&chain, pos + 1);
                        unsafe {
                            let k = std::ptr::read(&self.groups[src_g].keys[src_s]);
                            let v = std::ptr::read(&self.groups[src_g].values[src_s]);
                            self.groups[dst_g].keys[dst_s] = k;
                            self.groups[dst_g].values[dst_s] = v;
                        }
                    }
                }
                // Insert at position j (or write back to i if already in place).
                hashes[j] = cur_hash;
                let (gj, sj) = chain_slot(&chain, j);
                self.groups[gj].keys[sj] = MaybeUninit::new(cur_key);
                self.groups[gj].values[sj] = MaybeUninit::new(cur_val);
            }
            // Rebuild ctrl/tag bytes from the sorted hashes so that
            // get/insert/entry still work after sorting.
            // This adds a small performance penalty of maybe 6%.
            for (pos, &h) in hashes.iter().enumerate() {
                let (gi, si) = chain_slot(&chain, pos);
                self.groups[gi].ctrl[si] = tag(h);
            }
        }
    }
}

impl<K: Eq, V, S: SortingHash<K>> HashSortedMap<K, V, S> {
    /// Inserts a key/value pair, returning the previous value for `key` if it
    /// was already present (otherwise `None`).
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let hash = self.hasher.hash(&key);
        self.insert_hashed(hash, key, value)
    }

    /// Returns a reference to the value for `key`, or `None` if it is absent.
    ///
    /// The key may be any borrowed form of `K`, as long as the borrowed value
    /// hashes and compares equal to the owned key.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
        S: SortingHash<Q>,
    {
        let hash = self.hasher.hash(key);
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
        let hash = self.hasher.hash(&key);
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
        let mut gi = self.group_index(hash);
        loop {
            let group = &mut self.groups[gi];
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
            let occupied_slots = group_ops::count_occupied(&group.ctrl);
            if occupied_slots != GROUP_SIZE {
                group.ctrl[occupied_slots] = tag;
                group.keys[occupied_slots] = MaybeUninit::new(key);
                group.values[occupied_slots] = MaybeUninit::new(value);
                self.len += 1;
                return None;
            }
            // Group full — follow or create overflow chain.
            let overflow = group.overflow;
            if overflow != NO_OVERFLOW {
                gi = overflow as usize;
            } else {
                if self.num_groups as usize == self.groups.len() {
                    self.grow();
                    // n_bits changed; recompute the primary group and retry.
                    gi = self.group_index(hash);
                    continue;
                }
                let new_gi = self.num_groups as usize;
                self.num_groups += 1;
                self.groups[gi].overflow = new_gi as u32;
                let group = &mut self.groups[new_gi];
                group.ctrl[0] = tag;
                group.keys[0] = MaybeUninit::new(key);
                group.values[0] = MaybeUninit::new(value);
                self.len += 1;
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
        let mut gi = self.group_index(hash);

        loop {
            let group = &self.groups[gi];
            // SIMD scan group for tag match.
            let mut tag_mask = group_ops::match_tag(&group.ctrl, tag);
            while let Some(i) = group_ops::next_match(&mut tag_mask) {
                if unsafe { group.keys[i].assume_init_ref() }.borrow() == key {
                    return Some(unsafe { group.values[i].assume_init_ref() });
                }
            }
            if group.ctrl[GROUP_SIZE - 1] == CTRL_EMPTY {
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
        let mut gi = self.group_index(hash);

        loop {
            let group = &mut self.groups[gi];

            // SIMD scan group for tag match.
            let mut tag_mask = group_ops::match_tag(&group.ctrl, tag);
            while let Some(i) = group_ops::next_match(&mut tag_mask) {
                if unsafe { group.keys[i].assume_init_ref() } == key {
                    return FindResult::Found(group.values[i].as_mut_ptr());
                }
            }
            // Check for empty slot in this group.
            let occupied_slots = group_ops::count_occupied(&group.ctrl);
            if occupied_slots != GROUP_SIZE {
                return FindResult::Vacant(Insertion::Empty {
                    group: group as *mut _,
                    slot: occupied_slots,
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
            &mut self.groups,
            Vec::<Group<K, V>>::new().into_boxed_slice(),
        );
        let old_num_groups = self.num_groups as usize;
        let old_len = self.len;

        self.n_bits += 1;
        let (new_groups, num_primary) = Self::alloc_groups(self.n_bits);
        self.groups = new_groups;
        self.num_groups = num_primary;
        self.len = 0;

        for group in &old_groups[..old_num_groups] {
            for i in 0..group_ops::count_occupied(&group.ctrl) {
                let hash = self.hasher.hash(unsafe { group.keys[i].assume_init_ref() });
                self.insert_for_grow(hash, group.keys[i].as_ptr(), group.values[i].as_ptr());
            }
        }
        // Group<K, V> has no Drop (keys/values are MaybeUninit), so dropping
        // old_groups runs no destructors but does free the backing buffer.
        drop(old_groups);

        debug_assert_eq!(self.len, old_len);
    }

    fn insert_for_grow(&mut self, hash: u64, key_src: *const K, value_src: *const V) {
        let tag = tag(hash);
        let gi = self.group_index(hash);
        let mut group = &mut self.groups[gi];

        let slot = loop {
            let occupied = group_ops::count_occupied(&group.ctrl);
            if occupied != GROUP_SIZE {
                break occupied;
            }
            let overflow = group.overflow;
            if overflow != NO_OVERFLOW {
                group = &mut self.groups[overflow as usize];
            } else {
                let new_gi = self.num_groups as usize;
                group.overflow = new_gi as u32;
                self.num_groups += 1;
                group = &mut self.groups[new_gi];
                break 0;
            }
        };
        group.ctrl[slot] = tag;
        unsafe {
            group.keys[slot]
                .as_mut_ptr()
                .copy_from_nonoverlapping(key_src, 1);
            group.values[slot]
                .as_mut_ptr()
                .copy_from_nonoverlapping(value_src, 1);
        }
        self.len += 1;
    }
}

// ── Chain-slot helpers for sort_by_hash ─────────────────────────────────

/// Map a flat position (0..chain.len()*GROUP_SIZE) to a (group_index, slot).
#[inline]
fn chain_slot(chain: &[u32], pos: usize) -> (usize, usize) {
    (chain[pos / GROUP_SIZE] as usize, pos % GROUP_SIZE)
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
    /// An occupied entry whose key already exists in the map.
    Occupied(OccupiedEntry<'a, V>),
    /// A vacant entry whose key is absent from the map.
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

impl<'a, K: Eq, V, S: SortingHash<K>> Entry<'a, K, V, S> {
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

impl<'a, K: Eq, V, S: SortingHash<K>> VacantEntry<'a, K, V, S> {
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
                    if map.num_groups as usize == map.groups.len() {
                        return insert_after_grow(map, key, value);
                    }
                    let new_gi = map.num_groups as usize;
                    map.num_groups += 1;
                    let new_group: *mut Group<K, V> = &mut map.groups[new_gi];
                    (new_gi, new_group)
                };
                unsafe {
                    // SAFETY: `tail` was obtained from `&mut groups[..]` and
                    // remains valid because no reallocation occurred between
                    // `entry()` and now (we hold the only `&mut self`).
                    (*tail).overflow = new_gi as u32;
                }
                (new_group, 0)
            }
        };

        let tag = tag(hash);
        unsafe {
            (*map).len += 1;
            // SAFETY: `group_ptr` points into `map.groups` and is valid for `'a`.
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
fn insert_after_grow<K: Eq, V, S: SortingHash<K>>(
    map: &mut HashSortedMap<K, V, S>,
    key: K,
    value: V,
) -> &mut V {
    map.grow();
    map.entry(key).or_insert(value)
}

impl<K, V, S> Drop for HashSortedMap<K, V, S> {
    fn drop(&mut self) {
        for group in &mut self.groups[..self.num_groups as usize] {
            for i in 0..GROUP_SIZE {
                if group.ctrl[i] != CTRL_EMPTY {
                    unsafe { group.keys[i].assume_init_drop() };
                    unsafe { group.values[i].assume_init_drop() };
                }
            }
        }
    }
}

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
        let old_n_bits = map.n_bits;
        for i in 0..100u32 {
            let key = i | 0xFF000000;
            map.insert(key, i);
        }
        assert!(map.n_bits > old_n_bits, "should have grown");
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

    #[test]
    fn custom_sorting_hash_without_hash_key() {
        // A key type that intentionally does NOT implement `Hash`, proving the
        // map only requires `SortingHash` (not `std::hash::Hash`) when a custom
        // hasher is supplied.
        #[derive(PartialEq, Eq)]
        struct Key(u32);

        struct ByValue;
        impl SortingHash<Key> for ByValue {
            fn hash(&self, key: &Key) -> u64 {
                (key.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            }
        }

        let mut map = HashSortedMap::with_hasher(ByValue);
        for i in 0..200u32 {
            assert_eq!(map.insert(Key(i), i), None);
        }
        assert_eq!(map.len(), 200);
        for i in 0..200u32 {
            assert_eq!(map.get(&Key(i)), Some(&i));
        }
        assert_eq!(map.get(&Key(999)), None);
    }

    // ── sort_by_hash tests ──────────────────────────────────────────────

    #[test]
    fn sort_by_hash_empty() {
        let mut map: HashSortedMap<u32, u32> = HashSortedMap::new();
        map.sort_by_hash();
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn sort_by_hash_single() {
        let mut map = HashSortedMap::new();
        map.insert(42u32, "hello");
        map.sort_by_hash();
        assert_eq!(map.len(), 1);
        let entries: Vec<_> = map.into_iter().collect();
        assert_eq!(entries, vec![(42, "hello")]);
    }

    #[test]
    fn sort_by_hash_preserves_entries() {
        let mut map = HashSortedMap::new();
        for i in 0..200u32 {
            map.insert(i, i * 10);
        }
        map.sort_by_hash();
        assert_eq!(map.len(), 200);
        // Lookups must still work after sorting.
        for i in 0..200u32 {
            assert_eq!(map.get(&i), Some(&(i * 10)), "get failed for key {i}");
        }
        let mut entries: Vec<_> = map.into_iter().collect();
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
        map.sort_by_hash();
        // Iteration should now yield entries in (hash, key) order.
        let mut prev_hash = 0u64;
        let mut prev_key = 0u32;
        let mut first = true;
        for (&k, _) in &map {
            let h = hasher.hash_one(k);
            if !first {
                assert!(
                    (h, k) >= (prev_hash, prev_key),
                    "(hash, key) order violated: ({prev_hash:#x}, {prev_key}) > ({h:#x}, {k})"
                );
            }
            prev_hash = h;
            prev_key = k;
            first = false;
        }
    }

    #[test]
    fn sort_by_hash_with_overflow() {
        // Force overflow chains via fixed hash — all keys collide, so sort
        // should produce key order as tie-breaker.
        let mut map = HashSortedMap::with_capacity_and_hasher(1, FixedState(0));
        for i in 0..50u32 {
            map.insert(i, i);
        }
        map.sort_by_hash();
        assert_eq!(map.len(), 50);
        // All hashes are equal, so entries should be in key order.
        let entries: Vec<_> = map.into_iter().collect();
        for i in 0..50u32 {
            assert_eq!(entries[i as usize], (i, i), "key order violated at {i}");
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
        map.sort_by_hash();
        assert_eq!(map.len(), 100);
        let mut prev_hash = 0u64;
        let mut prev_key = String::new();
        let mut first = true;
        for (k, _) in &map {
            let h = hasher.hash_one(k);
            if !first {
                assert!(
                    (h, k) >= (prev_hash, &prev_key),
                    "(hash, key) order violated"
                );
            }
            prev_hash = h;
            prev_key = k.clone();
            first = false;
        }
    }
}
