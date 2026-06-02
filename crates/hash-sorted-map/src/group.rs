use core::mem::MaybeUninit;

use super::group_ops::{CTRL_EMPTY, GROUP_SIZE};

pub(crate) const NO_OVERFLOW: u32 = u32::MAX;

pub(crate) struct Group<K, V> {
    pub(crate) ctrl: [u8; GROUP_SIZE],
    pub(crate) keys: [MaybeUninit<K>; GROUP_SIZE],
    pub(crate) values: [MaybeUninit<V>; GROUP_SIZE],
    pub(crate) overflow: u32,
}

impl<K, V> Group<K, V> {
    pub(crate) fn new() -> Self {
        Self {
            ctrl: [CTRL_EMPTY; GROUP_SIZE],
            keys: [const { MaybeUninit::uninit() }; GROUP_SIZE],
            values: [const { MaybeUninit::uninit() }; GROUP_SIZE],
            overflow: NO_OVERFLOW,
        }
    }
}
