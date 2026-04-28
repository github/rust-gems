use core::mem::MaybeUninit;
use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};

// Platform-dependent group size: 16 on x86_64 (SSE2), 8 everywhere else.
#[cfg(target_arch = "x86_64")]
const GROUP_SIZE: usize = 16;
#[cfg(not(target_arch = "x86_64"))]
const GROUP_SIZE: usize = 8;

const CTRL_EMPTY: u8 = 0x00;
const NO_OVERFLOW: u32 = u32::MAX;

#[cfg(target_arch = "x86_64")]
type Mask = u32;
#[cfg(not(target_arch = "x86_64"))]
type Mask = u64;

// ── SIMD group operations ───────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod group_ops {
    #[cfg(target_arch = "x86")]
    use core::arch::x86 as x86;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64 as x86;

    use super::{Mask, GROUP_SIZE};

    #[inline(always)]
    pub fn match_tag(ctrl: &[u8; GROUP_SIZE], tag: u8) -> Mask {
        unsafe {
            let group = x86::_mm_loadu_si128(ctrl.as_ptr() as *const x86::__m128i);
            let cmp = x86::_mm_cmpeq_epi8(group, x86::_mm_set1_epi8(tag as i8));
            x86::_mm_movemask_epi8(cmp) as u32
        }
    }

    #[inline(always)]
    pub fn match_empty(ctrl: &[u8; GROUP_SIZE]) -> Mask {
        match_tag(ctrl, super::CTRL_EMPTY)
    }

    /// Mask of slots whose ctrl byte has the high bit set (occupied).
    /// Uses SSE2 `_mm_movemask_epi8` which extracts the top bit of each byte.
    #[inline(always)]
    pub fn match_full(ctrl: &[u8; GROUP_SIZE]) -> Mask {
        unsafe {
            let group = x86::_mm_loadu_si128(ctrl.as_ptr() as *const x86::__m128i);
            x86::_mm_movemask_epi8(group) as u32
        }
    }

    #[inline(always)]
    pub fn lowest(mask: Mask) -> usize {
        mask.trailing_zeros() as usize
    }

    #[inline(always)]
    pub fn clear_slot(mask: Mask, slot: usize) -> Mask {
        mask & !(1u32 << slot)
    }

    #[inline(always)]
    pub fn next_match(mask: &mut Mask) -> Option<usize> {
        if *mask == 0 {
            return None;
        }
        let i = lowest(*mask);
        *mask &= *mask - 1;
        Some(i)
    }
}

#[cfg(target_arch = "aarch64")]
mod group_ops {
    use core::arch::aarch64 as neon;

    use super::{Mask, GROUP_SIZE};

    #[inline(always)]
    pub fn match_tag(ctrl: &[u8; GROUP_SIZE], tag: u8) -> Mask {
        unsafe {
            let group = neon::vld1_u8(ctrl.as_ptr());
            let cmp = neon::vceq_u8(group, neon::vdup_n_u8(tag));
            neon::vget_lane_u64(neon::vreinterpret_u64_u8(cmp), 0) & 0x8080808080808080
        }
    }

    #[inline(always)]
    pub fn match_empty(ctrl: &[u8; GROUP_SIZE]) -> Mask {
        unsafe {
            let group = neon::vld1_u8(ctrl.as_ptr());
            let cmp = neon::vceq_u8(group, neon::vdup_n_u8(0));
            neon::vget_lane_u64(neon::vreinterpret_u64_u8(cmp), 0) & 0x8080808080808080
        }
    }

    /// Mask of slots whose ctrl byte has the high bit set (occupied).
    #[inline(always)]
    pub fn match_full(ctrl: &[u8; GROUP_SIZE]) -> Mask {
        unsafe {
            let group = neon::vld1_u8(ctrl.as_ptr());
            neon::vget_lane_u64(neon::vreinterpret_u64_u8(group), 0) & 0x8080808080808080
        }
    }

    #[inline(always)]
    pub fn lowest(mask: Mask) -> usize {
        (mask.trailing_zeros() >> 3) as usize
    }

    #[inline(always)]
    pub fn clear_slot(mask: Mask, slot: usize) -> Mask {
        mask & !(0x80u64 << (slot * 8))
    }

    #[inline(always)]
    pub fn next_match(mask: &mut Mask) -> Option<usize> {
        if *mask == 0 {
            return None;
        }
        let i = lowest(*mask);
        *mask &= *mask - 1;
        Some(i)
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod group_ops {
    use super::{Mask, GROUP_SIZE};

    #[inline(always)]
    pub fn match_tag(ctrl: &[u8; GROUP_SIZE], tag: u8) -> Mask {
        let word = u64::from_ne_bytes(*ctrl);
        let broadcast = 0x0101010101010101u64 * (tag as u64);
        let xor = word ^ broadcast;
        (xor.wrapping_sub(0x0101010101010101)) & !xor & 0x8080808080808080
    }

    #[inline(always)]
    pub fn match_empty(ctrl: &[u8; GROUP_SIZE]) -> Mask {
        let word = u64::from_ne_bytes(*ctrl);
        !word & 0x8080808080808080
    }

    /// Mask of slots whose ctrl byte has the high bit set (occupied).
    #[inline(always)]
    pub fn match_full(ctrl: &[u8; GROUP_SIZE]) -> Mask {
        let word = u64::from_ne_bytes(*ctrl);
        word & 0x8080808080808080
    }

    #[inline(always)]
    pub fn lowest(mask: Mask) -> usize {
        (mask.trailing_zeros() >> 3) as usize
    }

    #[inline(always)]
    pub fn clear_slot(mask: Mask, slot: usize) -> Mask {
        mask & !(0x80u64 << (slot * 8))
    }

    #[inline(always)]
    pub fn next_match(mask: &mut Mask) -> Option<usize> {
        if *mask == 0 {
            return None;
        }
        let i = lowest(*mask);
        *mask &= *mask - 1;
        Some(i)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn tag(hash: u64) -> u8 {
    (hash as u8) | 0x80
}

#[inline]
fn slot_hint(hash: u64) -> usize {
    ((hash >> 7) & (GROUP_SIZE as u64 - 1)) as usize
}

struct Group<K, V> {
    ctrl: [u8; GROUP_SIZE],
    keys: [MaybeUninit<K>; GROUP_SIZE],
    values: [MaybeUninit<V>; GROUP_SIZE],
    overflow: u32,
}

impl<K, V> Group<K, V> {
    fn new() -> Self {
        Self {
            ctrl: [CTRL_EMPTY; GROUP_SIZE],
            keys: [const { MaybeUninit::uninit() }; GROUP_SIZE],
            values: [const { MaybeUninit::uninit() }; GROUP_SIZE],
            overflow: NO_OVERFLOW,
        }
    }
}

/// Insertion-only hash map with SIMD group scanning.
///
/// Uses NEON on aarch64, SSE2 on x86_64, scalar fallback elsewhere.
/// Generic over key type `K`, value type `V`, and hash builder `S`.
pub struct SimdPrefixHashMap<K, V, S = RandomState> {
    groups: Box<[Group<K, V>]>,
    num_groups: u32,
    n_bits: u32,
    len: usize,
    hash_builder: S,
}

impl<K: Hash + Eq, V> SimdPrefixHashMap<K, V> {
    pub fn new() -> Self {
        Self::with_capacity_and_hasher(0, RandomState::new())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, RandomState::new())
    }
}

impl<K, V, S> SimdPrefixHashMap<K, V, S> {
    pub fn with_hasher(hash_builder: S) -> Self {
        Self::with_capacity_and_hasher(0, hash_builder)
    }

    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        let adjusted = capacity.checked_mul(8).unwrap_or(usize::MAX) / 7;
        let min_groups = (adjusted / GROUP_SIZE).max(1).next_power_of_two();
        let n_bits = min_groups.trailing_zeros().max(1);
        let (groups, num_primary) = Self::alloc_groups(n_bits);
        Self {
            groups,
            num_groups: num_primary,
            n_bits,
            len: 0,
            hash_builder,
        }
    }

    /// Allocate a fully default-initialized boxed slice sized for `n_bits` primary groups
    /// plus the standard 12.5% overflow reserve. Returns the slice and the number of
    /// primary groups (which is also the initial in-use count).
    fn alloc_groups(n_bits: u32) -> (Box<[Group<K, V>]>, u32) {
        let num_primary = 1usize << n_bits;
        let total = num_primary + num_primary / 8 + 1;
        let mut groups: Vec<Group<K, V>> = Vec::with_capacity(total);
        groups.resize_with(total, Group::new);
        (groups.into_boxed_slice(), num_primary as u32)
    }

    #[inline]
    fn group_index(&self, hash: u64) -> usize {
        (hash >> (64 - self.n_bits)) as usize
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<K: Hash + Eq, V, S: BuildHasher> SimdPrefixHashMap<K, V, S> {
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
            FindResult::Empty { group, slot } => Entry::Vacant(VacantEntry {
                map: self,
                hash,
                key,
                insertion: Insertion::Empty { group, slot },
            }),
            FindResult::NeedsOverflow { tail } => Entry::Vacant(VacantEntry {
                map: self,
                hash,
                key,
                insertion: Insertion::NeedsOverflow { tail },
            }),
        }
    }

    fn insert_hashed(&mut self, hash: u64, key: K, value: V) -> Option<V> {
        let tag = tag(hash);
        let hint = slot_hint(hash);
        let mut gi = self.group_index(hash);

        loop {
            let group = &mut self.groups[gi];

            // Fast path: check preferred slot.
            let c = group.ctrl[hint];
            if c == CTRL_EMPTY {
                group.ctrl[hint] = tag;
                group.keys[hint] = MaybeUninit::new(key);
                group.values[hint] = MaybeUninit::new(value);
                self.len += 1;
                return None;
            }
            if c == tag && unsafe { group.keys[hint].assume_init_ref() } == &key {
                let old = std::mem::replace(
                    unsafe { group.values[hint].assume_init_mut() },
                    value,
                );
                return Some(old);
            }

            // Slow path: SIMD scan group for tag match.
            let mut tag_mask = group_ops::match_tag(&group.ctrl, tag);
            tag_mask = group_ops::clear_slot(tag_mask, hint);
            while let Some(i) = group_ops::next_match(&mut tag_mask) {
                if unsafe { group.keys[i].assume_init_ref() } == &key {
                    let old = std::mem::replace(
                        unsafe { group.values[i].assume_init_mut() },
                        value,
                    );
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
                group.ctrl[hint] = tag;
                group.keys[hint] = MaybeUninit::new(key);
                group.values[hint] = MaybeUninit::new(value);
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
        let (gi, slot) = self.find_slot(hash, key)?;
        Some(unsafe { self.groups[gi].values[slot].assume_init_ref() })
    }

    /// Look up `key` and return its `(group_index, slot)` if present.
    /// Pure read-only lookup — does not allocate or modify the table.
    fn find_slot<Q>(&self, hash: u64, key: &Q) -> Option<(usize, usize)>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let tag = tag(hash);
        let hint = slot_hint(hash);
        let mut gi = self.group_index(hash);

        loop {
            let group = &self.groups[gi];

            // Fast path: preferred slot.
            let c = group.ctrl[hint];
            if c == tag
                && unsafe { group.keys[hint].assume_init_ref() }.borrow() == key
            {
                return Some((gi, hint));
            }

            // Slow path: SIMD scan group.
            let mut tag_mask = group_ops::match_tag(&group.ctrl, tag);
            tag_mask = group_ops::clear_slot(tag_mask, hint);
            while let Some(i) = group_ops::next_match(&mut tag_mask) {
                if unsafe { group.keys[i].assume_init_ref() }.borrow() == key {
                    return Some((gi, i));
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
        let hint = slot_hint(hash);
        let mut gi = self.group_index(hash);

        loop {
            let group = &mut self.groups[gi];

            // Fast path: preferred slot.
            let c = group.ctrl[hint];
            if c == CTRL_EMPTY {
                return FindResult::Empty { group: group as *mut _, slot: hint };
            }
            if c == tag && unsafe { group.keys[hint].assume_init_ref() } == key {
                return FindResult::Found(group.values[hint].as_mut_ptr());
            }

            // Slow path: SIMD scan group for tag match.
            let mut tag_mask = group_ops::match_tag(&group.ctrl, tag);
            tag_mask = group_ops::clear_slot(tag_mask, hint);
            while let Some(i) = group_ops::next_match(&mut tag_mask) {
                if unsafe { group.keys[i].assume_init_ref() } == key {
                    return FindResult::Found(group.values[i].as_mut_ptr());
                }
            }

            // Check for empty slot in this group.
            let empty_mask = group_ops::match_empty(&group.ctrl);
            if empty_mask != 0 {
                let i = group_ops::lowest(empty_mask);
                return FindResult::Empty { group: group as *mut _, slot: i };
            }

            // Group full — follow or report end of chain.
            if group.overflow == NO_OVERFLOW {
                return FindResult::NeedsOverflow { tail: group as *mut _ };
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
            let mut full_mask = group_ops::match_full(&group.ctrl);
            while let Some(i) = group_ops::next_match(&mut full_mask) {
                let hash = self.hash_builder.hash_one(unsafe {
                    group.keys[i].assume_init_ref()
                });
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
        let mut hint = slot_hint(hash);
        let gi = self.group_index(hash);
        let mut group = &mut self.groups[gi];

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
                group = &mut self.groups[overflow as usize];
            } else {
                let new_gi = self.num_groups as usize;
                self.groups[gi].overflow = new_gi as u32;
                self.num_groups += 1;
                group = &mut self.groups[new_gi];
                break;
            }
        }
        group.ctrl[hint] = tag;
        unsafe { group.keys[hint].as_mut_ptr().copy_from_nonoverlapping(key_src, 1) };
        unsafe { group.values[hint].as_mut_ptr().copy_from_nonoverlapping(value_src, 1) };
        self.len += 1;
    }
}

// ────────────────────────────────────────────────────────────────────────
// Entry API
// ────────────────────────────────────────────────────────────────────────

/// Result of a single chain walk during `entry()`: either the existing slot
/// for the key, an empty slot ready for insertion, or end-of-chain when no
/// empty slot exists (and a new overflow group must be allocated).
enum FindResult<K, V> {
    /// Pointer to the existing value.
    Found(*mut V),
    /// Pointer to the group with an empty slot at index `slot`.
    Empty { group: *mut Group<K, V>, slot: usize },
    /// End of chain — the caller must allocate an overflow group and link it
    /// via `tail`'s overflow field.
    NeedsOverflow { tail: *mut Group<K, V> },
}

/// Pre-computed insertion location stashed inside [`VacantEntry`] so that
/// `insert()` doesn't need to re-walk the chain. Pointers remain valid as
/// long as no reallocation occurs (the grow path re-walks via the slow path).
enum Insertion<K, V> {
    /// An empty slot is waiting at `(group, slot)`.
    Empty { group: *mut Group<K, V>, slot: usize },
    /// The chain is full; allocate a new overflow group and link via `tail`.
    NeedsOverflow { tail: *mut Group<K, V> },
}

/// View into a single entry in a [`SimdPrefixHashMap`], either occupied or vacant.
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
    map: &'a mut SimdPrefixHashMap<K, V, S>,
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
                if map.num_groups as usize == map.groups.len() {
                    return insert_after_grow(map, hash, key, value);
                }
                let new_gi = map.num_groups as usize;
                map.num_groups += 1;
                // SAFETY: `tail` was obtained from `&mut self.groups[..]` and
                // remains valid because no reallocation occurred between
                // `entry()` and now (we hold the only `&mut self`).
                unsafe {
                    (*tail).overflow = new_gi as u32;
                }
                let new_group: *mut Group<K, V> = &mut map.groups[new_gi];
                (new_group, slot_hint(hash))
            }
        };

        let tag = tag(hash);
        // SAFETY: `group_ptr` points into `map.groups` and is valid for `'a`.
        unsafe {
            let group = &mut *group_ptr;
            group.ctrl[slot] = tag;
            group.keys[slot] = MaybeUninit::new(key);
            group.values[slot] = MaybeUninit::new(value);
            map.len += 1;
            group.values[slot].assume_init_mut()
        }
    }
}

/// Cold path: the chain was full, the table is at capacity, and we need to
/// grow before inserting. Re-walks via the slow path after grow.
///
/// After `grow()` doubles `num_primary` (`n_bits += 1`), our key's new
/// primary group can have at most ~half the old chain's keys, so hitting
/// `NeedsOverflow` again would require `GROUP_SIZE` keys to all collide on
/// one extra bit of hash — essentially impossible for any reasonable hash.
/// (`insert_for_grow` relies on the same assumption to skip its own
/// capacity check.)
#[cold]
#[inline(never)]
fn insert_after_grow<'a, K: Hash + Eq, V, S: BuildHasher>(
    map: &'a mut SimdPrefixHashMap<K, V, S>,
    hash: u64,
    key: K,
    value: V,
) -> &'a mut V {
    map.grow();
    match map.find_or_insertion_slot(hash, &key) {
        FindResult::Empty { group, slot } => {
            let tag = tag(hash);
            // SAFETY: `group` points into `map.groups` and is valid for `'a`.
            unsafe {
                let g = &mut *group;
                g.ctrl[slot] = tag;
                g.keys[slot] = MaybeUninit::new(key);
                g.values[slot] = MaybeUninit::new(value);
                map.len += 1;
                g.values[slot].assume_init_mut()
            }
        }
        // After grow, the new primary group for `key` cannot be full (see
        // function docs), and the key wasn't in the table before grow.
        FindResult::NeedsOverflow { .. } | FindResult::Found(_) => {
            unreachable!("post-grow walk must hit an empty slot")
        }
    }
}

impl<K, V, S> Drop for SimdPrefixHashMap<K, V, S> {
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
    use super::*;

    #[test]
    fn insert_and_get() {
        let mut map = SimdPrefixHashMap::new();
        map.insert(100, "hello");
        map.insert(200, "world");
        assert_eq!(map.get(&100), Some(&"hello"));
        assert_eq!(map.get(&200), Some(&"world"));
        assert_eq!(map.get(&999), None);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn insert_overwrite() {
        let mut map = SimdPrefixHashMap::new();
        map.insert(42, "a");
        assert_eq!(map.insert(42, "b"), Some("a"));
        assert_eq!(map.get(&42), Some(&"b"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn grow_preserves_entries() {
        let mut map = SimdPrefixHashMap::new();
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
        let mut map = SimdPrefixHashMap::with_capacity(2000);
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
        let mut map = SimdPrefixHashMap::with_capacity(8);
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
        let mut map = SimdPrefixHashMap::with_capacity(1);
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
        let mut map = SimdPrefixHashMap::new();
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
        let mut map: SimdPrefixHashMap<&str, i32> = SimdPrefixHashMap::new();
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
        let mut map: SimdPrefixHashMap<u32, String> = SimdPrefixHashMap::new();
        let mut call_count = 0;
        let mut make = |s: &str| {
            call_count += 1;
            s.to_string()
        };
        // First call: f runs, inserts "first".
        assert_eq!(map.get_or_insert_with(1, || make("first")), &mut "first".to_string());
        // Second call with same key: f does NOT run; returns existing.
        assert_eq!(map.get_or_insert_with(1, || make("second")), &mut "first".to_string());
        // New key: f runs.
        assert_eq!(map.get_or_insert_with(2, || make("third")), &mut "third".to_string());
        assert_eq!(call_count, 2);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn get_or_default_survives_grow() {
        let mut map: SimdPrefixHashMap<u32, u32> = SimdPrefixHashMap::with_capacity(1);
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
        let mut map: SimdPrefixHashMap<&str, u32> = SimdPrefixHashMap::new();
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
        let mut map: SimdPrefixHashMap<u32, String> = SimdPrefixHashMap::new();
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
        let mut map: SimdPrefixHashMap<u32, u32> = SimdPrefixHashMap::new();
        // Vacant: and_modify is a no-op, then or_insert(0) runs.
        *map.entry(7).and_modify(|v| *v *= 10).or_insert(1) += 100;
        assert_eq!(map.get(&7), Some(&101));
        // Occupied: and_modify runs, or_insert is skipped.
        *map.entry(7).and_modify(|v| *v *= 2).or_insert(99) += 1;
        assert_eq!(map.get(&7), Some(&203));
    }
}
