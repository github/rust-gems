use super::group::Group;
use super::group_ops::{CTRL_EMPTY, GROUP_SIZE};

/// Core storage for a hash-sorted map. Owns the group array and supports
/// iteration and drop. Does not contain a hasher — use [`HashSortedMap`]
/// for insertion and lookup.
pub struct HashSortedContainer<K, V> {
    pub(crate) groups: Box<[Group<K, V>]>,
    pub(crate) num_groups: u32,
    pub(crate) n_bits: u32,
    pub(crate) len: usize,
}

impl<K, V> HashSortedContainer<K, V> {
    pub(crate) fn alloc_groups(n_bits: u32) -> (Box<[Group<K, V>]>, u32) {
        let num_primary = 1usize << n_bits;
        let total = num_primary + num_primary / 8 + 1;
        let mut groups: Vec<Group<K, V>> = Vec::with_capacity(total);
        groups.resize_with(total, Group::new);
        (groups.into_boxed_slice(), num_primary as u32)
    }

    pub(crate) fn new(n_bits: u32) -> Self {
        let (groups, num_primary) = Self::alloc_groups(n_bits);
        Self {
            groups,
            num_groups: num_primary,
            n_bits,
            len: 0,
        }
    }

    #[inline]
    pub(crate) fn group_index(&self, hash: u64) -> usize {
        (hash >> (64 - self.n_bits)) as usize
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<K, V> Drop for HashSortedContainer<K, V> {
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
