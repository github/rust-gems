use core::mem::MaybeUninit;

const GROUP_SIZE: usize = 8;
const CTRL_EMPTY: u8 = 0x00;
const NO_OVERFLOW: u32 = u32::MAX;

/// A single group: 8 slots with control bytes, keys, values, and an overflow pointer.
struct Group<V> {
    ctrl: [u8; GROUP_SIZE],
    keys: [u32; GROUP_SIZE],
    values: [MaybeUninit<V>; GROUP_SIZE],
    overflow: u32, // index into groups vec, or NO_OVERFLOW
}

impl<V> Group<V> {
    fn new() -> Self {
        Self {
            ctrl: [CTRL_EMPTY; GROUP_SIZE],
            keys: [0; GROUP_SIZE],
            values: [const { MaybeUninit::uninit() }; GROUP_SIZE],
            overflow: NO_OVERFLOW,
        }
    }
}

/// Insertion-only hash map where the key IS a hash (`u32`).
///
/// Groups are stored in a single `Vec<Group>`. The first `2^n_bits` groups
/// are primary buckets (addressed by key prefix). When a primary group is
/// full, an overflow group is allocated from the end of the vec and linked
/// via `overflow`.
pub struct PrefixHashMap<V> {
    groups: Vec<Group<V>>,
    n_bits: u32,
    num_primary: u32,
    len: usize,
}

#[inline]
fn tag(key: u32) -> u8 {
    (key as u8) | 0x80
}

#[inline]
fn slot_hint(key: u32) -> usize {
    ((key >> 7) & 0x7) as usize
}

#[inline]
fn match_byte(ctrl: &[u8; GROUP_SIZE], byte: u8) -> u64 {
    let word = u64::from_ne_bytes(*ctrl);
    let broadcast = 0x0101010101010101u64 * (byte as u64);
    let xor = word ^ broadcast;
    // Zero bytes in xor → matches. Use: (v - 0x01..01) & !v & 0x80..80
    (xor.wrapping_sub(0x0101010101010101)) & !xor & 0x8080808080808080
}

#[inline]
fn match_empty(ctrl: &[u8; GROUP_SIZE]) -> u64 {
    let word = u64::from_ne_bytes(*ctrl);
    !word & 0x8080808080808080
}

impl<V> PrefixHashMap<V> {
    #[inline]
    fn group_index(&self, key: u32) -> usize {
        (key >> (32 - self.n_bits)) as usize
    }

    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let min_groups = (capacity / GROUP_SIZE).max(1).next_power_of_two();
        let n_bits = min_groups.trailing_zeros().max(1);
        let num_primary = 1usize << n_bits;
        // Reserve ~12.5% extra groups for overflow.
        let total = num_primary + num_primary / 8 + 1;
        let mut groups = Vec::with_capacity(total);
        groups.resize_with(num_primary, Group::new);
        Self {
            groups,
            n_bits,
            num_primary: num_primary as u32,
            len: 0,
        }
    }

    pub fn insert(&mut self, key: u32, value: V) -> Option<V> {
        let tag = tag(key);
        let hint = slot_hint(key);
        let mut gi = self.group_index(key);

        loop {
            let group = &self.groups[gi];

            // Fast path: check preferred slot.
            let c = group.ctrl[hint];
            if c == CTRL_EMPTY {
                let group = &mut self.groups[gi];
                group.ctrl[hint] = tag;
                group.keys[hint] = key;
                group.values[hint] = MaybeUninit::new(value);
                self.len += 1;
                return None;
            }
            if c == tag && group.keys[hint] == key {
                let old = std::mem::replace(
                    unsafe { self.groups[gi].values[hint].assume_init_mut() },
                    value,
                );
                return Some(old);
            }

            // Slow path: scan group for tag match.
            let mut tag_mask = match_byte(&group.ctrl, tag);
            tag_mask &= !(0x80u64 << (hint * 8)); // clear hint slot
            while tag_mask != 0 {
                let i = (tag_mask.trailing_zeros() >> 3) as usize;
                tag_mask &= tag_mask - 1;
                if group.keys[i] == key {
                    let old = std::mem::replace(
                        unsafe { self.groups[gi].values[i].assume_init_mut() },
                        value,
                    );
                    return Some(old);
                }
            }

            // Check for empty slot in this group.
            let empty_mask = match_empty(&group.ctrl);
            if empty_mask != 0 {
                let i = (empty_mask.trailing_zeros() >> 3) as usize;
                let group = &mut self.groups[gi];
                group.ctrl[i] = tag;
                group.keys[i] = key;
                group.values[i] = MaybeUninit::new(value);
                self.len += 1;
                return None;
            }

            // Group full — follow or create overflow chain.
            let overflow = self.groups[gi].overflow;
            if overflow != NO_OVERFLOW {
                gi = overflow as usize;
            } else {
                let max_overflow = self.num_primary / 8 + 1;
                let num_overflow = self.groups.len() as u32 - self.num_primary;
                if num_overflow >= max_overflow {
                    // Overflow exhausted — grow and retry.
                    self.grow();
                    return self.insert(key, value);
                }
                // Allocate a new overflow group.
                let new_gi = self.groups.len();
                self.groups.push(Group::new());
                self.groups[gi].overflow = new_gi as u32;
                // Insert into the new group's preferred slot.
                let group = &mut self.groups[new_gi];
                group.ctrl[hint] = tag;
                group.keys[hint] = key;
                group.values[hint] = MaybeUninit::new(value);
                self.len += 1;
                return None;
            }
        }
    }

    pub fn get(&self, key: u32) -> Option<&V> {
        let tag = tag(key);
        let hint = slot_hint(key);
        let mut gi = self.group_index(key);

        loop {
            let group = &self.groups[gi];

            // Fast path: preferred slot.
            let c = group.ctrl[hint];
            if c == tag && group.keys[hint] == key {
                return Some(unsafe { group.values[hint].assume_init_ref() });
            }
            if c == CTRL_EMPTY {
                // Preferred slot empty and no overflow means not found
                // (only if no overflow — check below)
            }

            // Slow path: scan group.
            let mut tag_mask = match_byte(&group.ctrl, tag);
            tag_mask &= !(0x80u64 << (hint * 8)); // clear hint slot
            while tag_mask != 0 {
                let i = (tag_mask.trailing_zeros() >> 3) as usize;
                tag_mask &= tag_mask - 1;
                if group.keys[i] == key {
                    return Some(unsafe { group.values[i].assume_init_ref() });
                }
            }

            // If group has empty slots, key is not present.
            if match_empty(&group.ctrl) != 0 {
                return None;
            }

            // Follow overflow chain.
            if group.overflow == NO_OVERFLOW {
                return None;
            }
            gi = group.overflow as usize;
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    fn grow(&mut self) {
        let old_groups = std::mem::take(&mut self.groups);
        let old_len = self.len;

        self.n_bits += 1;
        let num_primary = 1usize << self.n_bits;
        let total = num_primary + num_primary / 8 + 1;
        self.num_primary = num_primary as u32;
        self.groups = Vec::with_capacity(total);
        self.groups.resize_with(num_primary, Group::new);
        self.len = 0;

        for group in old_groups {
            for i in 0..GROUP_SIZE {
                if group.ctrl[i] != CTRL_EMPTY {
                    let key = group.keys[i];
                    let value = unsafe { group.values[i].assume_init_read() };
                    self.insert(key, value);
                }
            }
            // Don't drop values — we moved them out with assume_init_read.
            std::mem::forget(group);
        }

        debug_assert_eq!(self.len, old_len);
    }
}

impl<V> Drop for PrefixHashMap<V> {
    fn drop(&mut self) {
        for group in &mut self.groups {
            for i in 0..GROUP_SIZE {
                if group.ctrl[i] != CTRL_EMPTY {
                    unsafe { group.values[i].assume_init_drop() };
                }
            }
        }
    }
}

// ── NoHintScalarPrefixHashMap ───────────────────────────────────────────────
// Same scalar match functions, but no slot_hint fast path — always does a
// full group scan. Used to isolate the impact of slot_hint vs SIMD.

pub struct NoHintScalarPrefixHashMap<V> {
    groups: Vec<Group<V>>,
    n_bits: u32,
    num_primary: u32,
    len: usize,
}

impl<V> NoHintScalarPrefixHashMap<V> {
    #[inline]
    fn group_index(&self, key: u32) -> usize {
        (key >> (32 - self.n_bits)) as usize
    }

    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let min_groups = (capacity / GROUP_SIZE).max(1).next_power_of_two();
        let n_bits = min_groups.trailing_zeros().max(1);
        let num_primary = 1usize << n_bits;
        let total = num_primary + num_primary / 8 + 1;
        let mut groups = Vec::with_capacity(total);
        groups.resize_with(num_primary, Group::new);
        Self {
            groups,
            n_bits,
            num_primary: num_primary as u32,
            len: 0,
        }
    }

    pub fn insert(&mut self, key: u32, value: V) -> Option<V> {
        let tag = tag(key);
        let mut gi = self.group_index(key);

        loop {
            let group = &self.groups[gi];

            // Scan group for tag match (no slot_hint fast path).
            let mut tag_mask = match_byte(&group.ctrl, tag);
            while tag_mask != 0 {
                let i = (tag_mask.trailing_zeros() >> 3) as usize;
                tag_mask &= tag_mask - 1;
                if group.keys[i] == key {
                    let old = std::mem::replace(
                        unsafe { self.groups[gi].values[i].assume_init_mut() },
                        value,
                    );
                    return Some(old);
                }
            }

            // Check for empty slot in this group.
            let empty_mask = match_empty(&group.ctrl);
            if empty_mask != 0 {
                let i = (empty_mask.trailing_zeros() >> 3) as usize;
                let group = &mut self.groups[gi];
                group.ctrl[i] = tag;
                group.keys[i] = key;
                group.values[i] = MaybeUninit::new(value);
                self.len += 1;
                return None;
            }

            // Group full — follow or create overflow chain.
            let overflow = self.groups[gi].overflow;
            if overflow != NO_OVERFLOW {
                gi = overflow as usize;
            } else {
                let max_overflow = self.num_primary / 8 + 1;
                let num_overflow = self.groups.len() as u32 - self.num_primary;
                if num_overflow >= max_overflow {
                    self.grow();
                    return self.insert(key, value);
                }
                let new_gi = self.groups.len();
                self.groups.push(Group::new());
                self.groups[gi].overflow = new_gi as u32;
                let group = &mut self.groups[new_gi];
                group.ctrl[0] = tag;
                group.keys[0] = key;
                group.values[0] = MaybeUninit::new(value);
                self.len += 1;
                return None;
            }
        }
    }

    pub fn get(&self, key: u32) -> Option<&V> {
        let tag = tag(key);
        let mut gi = self.group_index(key);

        loop {
            let group = &self.groups[gi];

            // Scan group for tag match (no slot_hint fast path).
            let mut tag_mask = match_byte(&group.ctrl, tag);
            while tag_mask != 0 {
                let i = (tag_mask.trailing_zeros() >> 3) as usize;
                tag_mask &= tag_mask - 1;
                if group.keys[i] == key {
                    return Some(unsafe { group.values[i].assume_init_ref() });
                }
            }

            // If group has empty slots, key is not present.
            if match_empty(&group.ctrl) != 0 {
                return None;
            }

            // Follow overflow chain.
            if group.overflow == NO_OVERFLOW {
                return None;
            }
            gi = group.overflow as usize;
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    fn grow(&mut self) {
        let old_groups = std::mem::take(&mut self.groups);
        let old_len = self.len;

        self.n_bits += 1;
        let num_primary = 1usize << self.n_bits;
        let total = num_primary + num_primary / 8 + 1;
        self.num_primary = num_primary as u32;
        self.groups = Vec::with_capacity(total);
        self.groups.resize_with(num_primary, Group::new);
        self.len = 0;

        for group in old_groups {
            for i in 0..GROUP_SIZE {
                if group.ctrl[i] != CTRL_EMPTY {
                    let key = group.keys[i];
                    let value = unsafe { group.values[i].assume_init_read() };
                    self.insert(key, value);
                }
            }
            std::mem::forget(group);
        }

        debug_assert_eq!(self.len, old_len);
    }
}

impl<V> Drop for NoHintScalarPrefixHashMap<V> {
    fn drop(&mut self) {
        for group in &mut self.groups {
            for i in 0..GROUP_SIZE {
                if group.ctrl[i] != CTRL_EMPTY {
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
        assert_eq!(map.get(100), Some(&"hello"));
        assert_eq!(map.get(200), Some(&"world"));
        assert_eq!(map.get(999), None);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn insert_overwrite() {
        let mut map = PrefixHashMap::new();
        map.insert(42, "a");
        assert_eq!(map.insert(42, "b"), Some("a"));
        assert_eq!(map.get(42), Some(&"b"));
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
            assert_eq!(map.get(i), Some(&(i * 10)), "missing key {i}");
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
            assert_eq!(map.get(i.wrapping_mul(2654435761)), Some(&i));
        }
    }

    #[test]
    fn overflow_chain() {
        // Force overflow by inserting many keys with same prefix.
        let mut map = PrefixHashMap::with_capacity(8);
        for i in 0..20u32 {
            // All keys have same top bits → same group → forces overflow.
            let key = i | 0xAB000000;
            map.insert(key, i);
        }
        assert_eq!(map.len(), 20);
        for i in 0..20u32 {
            let key = i | 0xAB000000;
            assert_eq!(map.get(key), Some(&i), "missing key {key:#x}");
        }
    }

    #[test]
    fn grow_on_overflow_exhaustion() {
        // Start tiny (2 primary groups), force enough collisions to exhaust overflow.
        let mut map = PrefixHashMap::with_capacity(1);
        let old_n_bits = map.n_bits;
        for i in 0..100u32 {
            let key = i | 0xFF000000; // all same prefix → single group chain
            map.insert(key, i);
        }
        assert!(map.n_bits > old_n_bits, "should have grown");
        assert_eq!(map.len(), 100);
        for i in 0..100u32 {
            let key = i | 0xFF000000;
            assert_eq!(map.get(key), Some(&i), "missing key {key:#x} after grow");
        }
    }
}
