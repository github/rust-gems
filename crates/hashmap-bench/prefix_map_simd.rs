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
    groups: Vec<Group<K, V>>,
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
        let num_primary = 1usize << n_bits;
        let total = num_primary + num_primary / 8 + 1;
        let mut groups = Vec::with_capacity(total);
        groups.resize_with(num_primary, Group::new);
        Self {
            groups,
            n_bits,
            len: 0,
            hash_builder,
        }
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

    fn insert_hashed(&mut self, hash: u64, key: K, value: V) -> Option<V> {
        let tag = tag(hash);
        let hint = slot_hint(hash);
        let mut gi = self.group_index(hash);

        loop {
            let group = &self.groups[gi];

            // Fast path: check preferred slot.
            let c = group.ctrl[hint];
            if c == CTRL_EMPTY {
                let group = &mut self.groups[gi];
                group.ctrl[hint] = tag;
                group.keys[hint] = MaybeUninit::new(key);
                group.values[hint] = MaybeUninit::new(value);
                self.len += 1;
                return None;
            }
            if c == tag && unsafe { group.keys[hint].assume_init_ref() } == &key {
                let old = std::mem::replace(
                    unsafe { self.groups[gi].values[hint].assume_init_mut() },
                    value,
                );
                drop(key);
                return Some(old);
            }

            // Slow path: SIMD scan group for tag match.
            let mut tag_mask = group_ops::match_tag(&group.ctrl, tag);
            tag_mask = group_ops::clear_slot(tag_mask, hint);
            while let Some(i) = group_ops::next_match(&mut tag_mask) {
                if unsafe { group.keys[i].assume_init_ref() } == &key {
                    let old = std::mem::replace(
                        unsafe { self.groups[gi].values[i].assume_init_mut() },
                        value,
                    );
                    drop(key);
                    return Some(old);
                }
            }

            // Check for empty slot in this group.
            let empty_mask = group_ops::match_empty(&group.ctrl);
            if empty_mask != 0 {
                let i = group_ops::lowest(empty_mask);
                let group = &mut self.groups[gi];
                group.ctrl[i] = tag;
                group.keys[i] = MaybeUninit::new(key);
                group.values[i] = MaybeUninit::new(value);
                self.len += 1;
                return None;
            }

            // Group full — follow or create overflow chain.
            let overflow = self.groups[gi].overflow;
            if overflow != NO_OVERFLOW {
                gi = overflow as usize;
            } else {
                if self.groups.len() == self.groups.capacity() {
                    self.grow();
                    return self.insert_hashed(hash, key, value);
                }
                let new_gi = self.groups.len();
                self.groups.push(Group::new());
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
                return Some(unsafe { group.values[hint].assume_init_ref() });
            }

            // Slow path: SIMD scan group.
            let mut tag_mask = group_ops::match_tag(&group.ctrl, tag);
            tag_mask = group_ops::clear_slot(tag_mask, hint);
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

    fn grow(&mut self) {
        let old_groups = std::mem::take(&mut self.groups);
        let old_len = self.len;

        self.n_bits += 1;
        let num_primary = 1usize << self.n_bits;
        let total = num_primary + num_primary / 8 + 1;
        self.groups = Vec::with_capacity(total);
        self.groups.resize_with(num_primary, Group::new);
        self.len = 0;

        for group in &old_groups {
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
        let hint = slot_hint(hash);
        let mut gi = self.group_index(hash);

        loop {
            let group = &self.groups[gi];

            if group.ctrl[hint] == CTRL_EMPTY {
                let group = &mut self.groups[gi];
                group.ctrl[hint] = tag;
                unsafe { group.keys[hint].as_mut_ptr().copy_from_nonoverlapping(key_src, 1) };
                unsafe { group.values[hint].as_mut_ptr().copy_from_nonoverlapping(value_src, 1) };
                self.len += 1;
                return;
            }

            let empty_mask = group_ops::match_empty(&group.ctrl);
            if empty_mask != 0 {
                let i = group_ops::lowest(empty_mask);
                let group = &mut self.groups[gi];
                group.ctrl[i] = tag;
                unsafe { group.keys[i].as_mut_ptr().copy_from_nonoverlapping(key_src, 1) };
                unsafe { group.values[i].as_mut_ptr().copy_from_nonoverlapping(value_src, 1) };
                self.len += 1;
                return;
            }

            let overflow = self.groups[gi].overflow;
            if overflow != NO_OVERFLOW {
                gi = overflow as usize;
            } else {
                let new_gi = self.groups.len();
                self.groups.push(Group::new());
                self.groups[gi].overflow = new_gi as u32;
                let group = &mut self.groups[new_gi];
                group.ctrl[hint] = tag;
                unsafe { group.keys[hint].as_mut_ptr().copy_from_nonoverlapping(key_src, 1) };
                unsafe { group.values[hint].as_mut_ptr().copy_from_nonoverlapping(value_src, 1) };
                self.len += 1;
                return;
            }
        }
    }
}

impl<K, V, S> Drop for SimdPrefixHashMap<K, V, S> {
    fn drop(&mut self) {
        for group in &mut self.groups {
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
}
