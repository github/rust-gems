use core::mem::MaybeUninit;
use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};

const GROUP_SIZE: usize = 8;
const CTRL_EMPTY: u8 = 0x00;
const NO_OVERFLOW: u32 = u32::MAX;

#[inline(always)]
fn likely(b: bool) -> bool {
    if !b { cold_path() }
    b
}

#[inline(always)]
fn unlikely(b: bool) -> bool {
    if b { cold_path() }
    b
}

#[cold]
#[inline(never)]
fn cold_path() {}

#[inline]
fn tag(hash: u64) -> u8 {
    (hash as u8) | 0x80
}

#[inline]
fn slot_hint(hash: u64) -> usize {
    ((hash >> 7) & 0x7) as usize
}

#[inline]
fn match_byte(ctrl: &[u8; GROUP_SIZE], byte: u8) -> u64 {
    let word = u64::from_ne_bytes(*ctrl);
    let broadcast = 0x0101010101010101u64 * (byte as u64);
    let xor = word ^ broadcast;
    (xor.wrapping_sub(0x0101010101010101)) & !xor & 0x8080808080808080
}

#[inline]
fn match_empty(ctrl: &[u8; GROUP_SIZE]) -> u64 {
    let word = u64::from_ne_bytes(*ctrl);
    !word & 0x8080808080808080
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

/// Insertion-only hash map with overflow chaining and slot-hint fast path.
///
/// Generic over key type `K`, value type `V`, and hash builder `S`.
pub struct PrefixHashMap<K, V, S = RandomState> {
    groups: Vec<Group<K, V>>,
    n_bits: u32,
    len: usize,
    hash_builder: S,
}

impl<K: Hash + Eq, V> PrefixHashMap<K, V> {
    pub fn new() -> Self {
        Self::with_capacity_and_hasher(0, RandomState::new())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, RandomState::new())
    }
}

impl<K, V, S> PrefixHashMap<K, V, S> {
    pub fn with_hasher(hash_builder: S) -> Self {
        Self::with_capacity_and_hasher(0, hash_builder)
    }

    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        // Target ≤87.5% load (7/8), matching hashbrown's load factor.
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

impl<K: Hash + Eq, V, S: BuildHasher> PrefixHashMap<K, V, S> {
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
            if likely(c == CTRL_EMPTY) {
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
                // Drop the incoming key since we're keeping the stored one.
                drop(key);
                return Some(old);
            }

            // Slow path: scan group for tag match.
            let mut tag_mask = match_byte(&group.ctrl, tag);
            tag_mask &= !(0x80u64 << (hint * 8));
            while tag_mask != 0 {
                let i = (tag_mask.trailing_zeros() >> 3) as usize;
                tag_mask &= tag_mask - 1;
                if unlikely(unsafe { group.keys[i].assume_init_ref() } == &key) {
                    let old = std::mem::replace(
                        unsafe { self.groups[gi].values[i].assume_init_mut() },
                        value,
                    );
                    drop(key);
                    return Some(old);
                }
            }

            // Check for empty slot in this group.
            let empty_mask = match_empty(&group.ctrl);
            if likely(empty_mask != 0) {
                let i = (empty_mask.trailing_zeros() >> 3) as usize;
                let group = &mut self.groups[gi];
                group.ctrl[i] = tag;
                group.keys[i] = MaybeUninit::new(key);
                group.values[i] = MaybeUninit::new(value);
                self.len += 1;
                return None;
            }

            // Group full — follow or create overflow chain.
            let overflow = self.groups[gi].overflow;
            if unlikely(overflow == NO_OVERFLOW) {
                return self.insert_overflow(gi, hash, key, value);
            }
            gi = overflow as usize;
        }
    }

    #[cold]
    #[inline(never)]
    fn insert_overflow(&mut self, gi: usize, hash: u64, key: K, value: V) -> Option<V> {
        if self.groups.len() == self.groups.capacity() {
            self.grow();
            return self.insert_hashed(hash, key, value);
        }
        let hint = slot_hint(hash);
        let tag = tag(hash);
        let new_gi = self.groups.len();
        self.groups.push(Group::new());
        self.groups[gi].overflow = new_gi as u32;
        let group = &mut self.groups[new_gi];
        group.ctrl[hint] = tag;
        group.keys[hint] = MaybeUninit::new(key);
        group.values[hint] = MaybeUninit::new(value);
        self.len += 1;
        None
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
            if likely(c == tag)
                && unsafe { group.keys[hint].assume_init_ref() }.borrow() == key
            {
                return Some(unsafe { group.values[hint].assume_init_ref() });
            }

            // Slow path: scan group.
            let mut tag_mask = match_byte(&group.ctrl, tag);
            tag_mask &= !(0x80u64 << (hint * 8));
            while tag_mask != 0 {
                let i = (tag_mask.trailing_zeros() >> 3) as usize;
                tag_mask &= tag_mask - 1;
                if likely(
                    unsafe { group.keys[i].assume_init_ref() }.borrow() == key,
                ) {
                    return Some(unsafe { group.values[i].assume_init_ref() });
                }
            }

            // If group has empty slots, key is not present.
            if likely(match_empty(&group.ctrl) != 0) {
                return None;
            }

            // Follow overflow chain.
            if unlikely(group.overflow == NO_OVERFLOW) {
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
            let ctrl_word = u64::from_ne_bytes(group.ctrl);
            if ctrl_word == 0 {
                continue;
            }
            let mut full_mask = ctrl_word & 0x8080808080808080;
            while full_mask != 0 {
                let i = (full_mask.trailing_zeros() >> 3) as usize;
                full_mask &= full_mask - 1;
                let hash = self.hash_builder.hash_one(unsafe {
                    group.keys[i].assume_init_ref()
                });
                self.insert_for_grow(hash, group.keys[i].as_ptr(), group.values[i].as_ptr());
            }
        }
        // Prevent double-drop — keys/values were copied out via raw pointers.
        std::mem::forget(old_groups);

        debug_assert_eq!(self.len, old_len);
    }

    /// Fast insert for grow: no duplicate check, raw pointer copy.
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

            let empty_mask = match_empty(&group.ctrl);
            if empty_mask != 0 {
                let i = (empty_mask.trailing_zeros() >> 3) as usize;
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

impl<K, V, S> Drop for PrefixHashMap<K, V, S> {
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
        let mut map = PrefixHashMap::new();
        map.insert(100, "hello");
        map.insert(200, "world");
        assert_eq!(map.get(&100), Some(&"hello"));
        assert_eq!(map.get(&200), Some(&"world"));
        assert_eq!(map.get(&999), None);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn insert_overwrite() {
        let mut map = PrefixHashMap::new();
        map.insert(42, "a");
        assert_eq!(map.insert(42, "b"), Some("a"));
        assert_eq!(map.get(&42), Some(&"b"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn grow_preserves_entries() {
        let mut map = PrefixHashMap::new();
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
        let mut map = PrefixHashMap::with_capacity(2000);
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
        let mut map = PrefixHashMap::with_capacity(8);
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
        let mut map = PrefixHashMap::with_capacity(1);
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
        let mut map = PrefixHashMap::new();
        map.insert("hello".to_string(), 1);
        map.insert("world".to_string(), 2);
        assert_eq!(map.get("hello"), Some(&1));
        assert_eq!(map.get("world"), Some(&2));
        assert_eq!(map.get("missing"), None);
        assert_eq!(map.len(), 2);

        // Overwrite
        assert_eq!(map.insert("hello".to_string(), 3), Some(1));
        assert_eq!(map.get("hello"), Some(&3));
        assert_eq!(map.len(), 2);
    }
}
