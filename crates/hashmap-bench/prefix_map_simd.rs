use core::mem::MaybeUninit;

// Platform-dependent group size: 16 on x86_64 (SSE2), 8 everywhere else.
#[cfg(target_arch = "x86_64")]
const GROUP_SIZE: usize = 16;
#[cfg(not(target_arch = "x86_64"))]
const GROUP_SIZE: usize = 8;

const CTRL_EMPTY: u8 = 0x00;
const NO_OVERFLOW: u32 = u32::MAX;

// ── Match‑mask abstraction ──────────────────────────────────────────────────
// Each platform returns a different mask type from group scans. We unify the
// interface via a Mask type alias and free functions.

#[cfg(target_arch = "x86_64")]
type Mask = u32; // movemask: one bit per slot, bottom 16 used
#[cfg(not(target_arch = "x86_64"))]
type Mask = u64; // one byte per slot, high bit indicates match

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

    /// Index of the lowest matching slot.
    #[inline(always)]
    pub fn lowest(mask: Mask) -> usize {
        mask.trailing_zeros() as usize
    }

    /// Clear a single slot from the mask.
    #[inline(always)]
    pub fn clear_slot(mask: Mask, slot: usize) -> Mask {
        mask & !(1u32 << slot)
    }

    /// Advance to next match, returning slot index.
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

// ── Group struct ────────────────────────────────────────────────────────────

struct Group<V> {
    ctrl: [u8; GROUP_SIZE],
    keys: [u32; GROUP_SIZE],
    values: [MaybeUninit<V>; GROUP_SIZE],
    overflow: u32,
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

// ── Helper functions ────────────────────────────────────────────────────────

#[inline]
fn tag(key: u32) -> u8 {
    (key as u8) | 0x80
}

#[inline]
fn slot_hint(key: u32) -> usize {
    ((key >> 7) & (GROUP_SIZE as u32 - 1)) as usize
}

// ── SimdPrefixHashMap ───────────────────────────────────────────────────────

/// Insertion-only hash map where the key IS a hash (`u32`).
///
/// Same algorithm as `PrefixHashMap` but with platform-specific SIMD
/// group scanning (SSE2 on x86_64, NEON on aarch64, scalar fallback elsewhere).
/// On x86_64 the group size is widened to 16 slots to exploit SSE2.
pub struct SimdPrefixHashMap<V> {
    groups: Vec<Group<V>>,
    n_bits: u32,
    num_primary: u32,
    len: usize,
}

impl<V> SimdPrefixHashMap<V> {
    #[inline]
    fn group_index(&self, key: u32) -> usize {
        (key >> (32 - self.n_bits)) as usize
    }

    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(capacity: usize) -> Self {
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

            // Slow path: SIMD scan group for tag match.
            let mut tag_mask = group_ops::match_tag(&group.ctrl, tag);
            tag_mask = group_ops::clear_slot(tag_mask, hint);
            while let Some(i) = group_ops::next_match(&mut tag_mask) {
                if group.keys[i] == key {
                    let old = std::mem::replace(
                        unsafe { self.groups[gi].values[i].assume_init_mut() },
                        value,
                    );
                    return Some(old);
                }
            }

            // Check for empty slot in this group.
            let empty_mask = group_ops::match_empty(&group.ctrl);
            if empty_mask != 0 {
                let i = group_ops::lowest(empty_mask);
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
                if self.groups.len() == self.groups.capacity() {
                    self.grow();
                    return self.insert(key, value);
                }
                let new_gi = self.groups.len();
                self.groups.push(Group::new());
                self.groups[gi].overflow = new_gi as u32;
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

            // Slow path: SIMD scan group.
            let mut tag_mask = group_ops::match_tag(&group.ctrl, tag);
            tag_mask = group_ops::clear_slot(tag_mask, hint);
            while let Some(i) = group_ops::next_match(&mut tag_mask) {
                if group.keys[i] == key {
                    return Some(unsafe { group.values[i].assume_init_ref() });
                }
            }

            // If group has empty slots, key is not present.
            if group_ops::match_empty(&group.ctrl) != 0 {
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

        for group in &old_groups {
            let ctrl_word = u64::from_ne_bytes(group.ctrl);
            if ctrl_word == 0 {
                continue;
            }
            let mut full_mask = ctrl_word & 0x8080808080808080;
            while full_mask != 0 {
                let i = (full_mask.trailing_zeros() >> 3) as usize;
                full_mask &= full_mask - 1;
                let key = group.keys[i];
                self.insert_for_grow(key, group.values[i].as_ptr());
            }
        }
        std::mem::forget(old_groups);

        debug_assert_eq!(self.len, old_len);
    }

    /// Fast insert for grow: no duplicate check, raw pointer copy.
    fn insert_for_grow(&mut self, key: u32, value_src: *const V) {
        let tag = tag(key);
        let hint = slot_hint(key);
        let mut gi = self.group_index(key);

        loop {
            let group = &self.groups[gi];

            if group.ctrl[hint] == CTRL_EMPTY {
                let group = &mut self.groups[gi];
                group.ctrl[hint] = tag;
                group.keys[hint] = key;
                unsafe { group.values[hint].as_mut_ptr().copy_from_nonoverlapping(value_src, 1) };
                self.len += 1;
                return;
            }

            let empty_mask = group_ops::match_empty(&group.ctrl);
            if empty_mask != 0 {
                let i = group_ops::lowest(empty_mask);
                let group = &mut self.groups[gi];
                group.ctrl[i] = tag;
                group.keys[i] = key;
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
                group.keys[hint] = key;
                unsafe { group.values[hint].as_mut_ptr().copy_from_nonoverlapping(value_src, 1) };
                self.len += 1;
                return;
            }
        }
    }
}

impl<V> Drop for SimdPrefixHashMap<V> {
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

// ── NoHintPrefixHashMap (SIMD, no slot_hint) ────────────────────────────────
// Same as SimdPrefixHashMap but always does a full group scan — no preferred
// slot fast path. This isolates the pure SIMD scan cost.

pub struct NoHintPrefixHashMap<V> {
    groups: Vec<Group<V>>,
    n_bits: u32,
    num_primary: u32,
    len: usize,
}

impl<V> NoHintPrefixHashMap<V> {
    #[inline]
    fn group_index(&self, key: u32) -> usize {
        (key >> (32 - self.n_bits)) as usize
    }

    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(capacity: usize) -> Self {
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
            num_primary: num_primary as u32,
            len: 0,
        }
    }

    pub fn insert(&mut self, key: u32, value: V) -> Option<V> {
        let tag = tag(key);
        let mut gi = self.group_index(key);

        loop {
            let group = &self.groups[gi];

            // Scan group for tag match.
            let mut tag_mask = group_ops::match_tag(&group.ctrl, tag);
            while let Some(i) = group_ops::next_match(&mut tag_mask) {
                if group.keys[i] == key {
                    let old = std::mem::replace(
                        unsafe { self.groups[gi].values[i].assume_init_mut() },
                        value,
                    );
                    return Some(old);
                }
            }

            // Check for empty slot in this group.
            let empty_mask = group_ops::match_empty(&group.ctrl);
            if empty_mask != 0 {
                let i = group_ops::lowest(empty_mask);
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
                if self.groups.len() == self.groups.capacity() {
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

            // Scan group for tag match.
            let mut tag_mask = group_ops::match_tag(&group.ctrl, tag);
            while let Some(i) = group_ops::next_match(&mut tag_mask) {
                if group.keys[i] == key {
                    return Some(unsafe { group.values[i].assume_init_ref() });
                }
            }

            // If group has empty slots, key is not present.
            if group_ops::match_empty(&group.ctrl) != 0 {
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

impl<V> Drop for NoHintPrefixHashMap<V> {
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
        let mut map = SimdPrefixHashMap::new();
        map.insert(100, "hello");
        map.insert(200, "world");
        assert_eq!(map.get(100), Some(&"hello"));
        assert_eq!(map.get(200), Some(&"world"));
        assert_eq!(map.get(999), None);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn insert_overwrite() {
        let mut map = SimdPrefixHashMap::new();
        map.insert(42, "a");
        assert_eq!(map.insert(42, "b"), Some("a"));
        assert_eq!(map.get(42), Some(&"b"));
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
            assert_eq!(map.get(i), Some(&(i * 10)), "missing key {i}");
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
            assert_eq!(map.get(i.wrapping_mul(2654435761)), Some(&i));
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
            assert_eq!(map.get(key), Some(&i), "missing key {key:#x}");
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
            assert_eq!(map.get(key), Some(&i), "missing key {key:#x} after grow");
        }
    }

    /// Verify SIMD match functions produce identical results to the scalar versions.
    #[test]
    fn simd_matches_scalar() {
        // Scalar reference implementations
        fn scalar_match_tag(ctrl: &[u8; GROUP_SIZE], tag: u8) -> Vec<usize> {
            ctrl.iter()
                .enumerate()
                .filter(|(_, &c)| c == tag)
                .map(|(i, _)| i)
                .collect()
        }
        fn scalar_match_empty(ctrl: &[u8; GROUP_SIZE]) -> Vec<usize> {
            ctrl.iter()
                .enumerate()
                .filter(|(_, &c)| c == CTRL_EMPTY)
                .map(|(i, _)| i)
                .collect()
        }

        // Decode a SIMD mask into sorted slot indices
        fn decode_mask(mut mask: Mask) -> Vec<usize> {
            let mut out = vec![];
            while let Some(i) = group_ops::next_match(&mut mask) {
                out.push(i);
            }
            out
        }

        // Test with various control byte patterns
        let patterns: Vec<[u8; GROUP_SIZE]> = vec![
            [CTRL_EMPTY; GROUP_SIZE],
            [0x80; GROUP_SIZE],
            {
                let mut p = [CTRL_EMPTY; GROUP_SIZE];
                p[0] = 0xAB;
                p[GROUP_SIZE - 1] = 0xAB;
                p
            },
            {
                let mut p = [CTRL_EMPTY; GROUP_SIZE];
                for (i, b) in p.iter_mut().enumerate() {
                    *b = if i % 2 == 0 { 0x80 | (i as u8) } else { CTRL_EMPTY };
                }
                p
            },
            {
                let mut p = [0u8; GROUP_SIZE];
                for (i, b) in p.iter_mut().enumerate() {
                    *b = 0x80 | (i as u8);
                }
                p
            },
        ];

        for ctrl in &patterns {
            // Test match_empty
            let simd_empty = decode_mask(group_ops::match_empty(ctrl));
            let scalar_empty = scalar_match_empty(ctrl);
            assert_eq!(simd_empty, scalar_empty, "match_empty mismatch for {ctrl:?}");

            // Test match_tag with various tags
            for &tag in &[0x80, 0x81, 0xAB, 0xFF] {
                let simd_tag = decode_mask(group_ops::match_tag(ctrl, tag));
                let scalar_tag = scalar_match_tag(ctrl, tag);
                assert_eq!(
                    simd_tag, scalar_tag,
                    "match_tag mismatch for tag={tag:#x}, ctrl={ctrl:?}"
                );
            }
        }
    }
}
