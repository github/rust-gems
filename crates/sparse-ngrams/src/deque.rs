//! Stack-allocated circular buffer (monotone deque).

use std::mem::MaybeUninit;

/// Deque element representing two neighboring bytes in the input.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PosStateBytes {
    /// Absolute index position between the two bigram characters.
    /// I.e. 1 references the very first bigram.
    pub index: u32,
    pub value: u8,
}

/// Stack-allocated circular buffer holding up to `CAP` elements.
/// Replaces `VecDeque<PosStateBytes>` — avoids heap allocation and fits in a
/// single cache line for small CAP values.
pub(crate) struct FixedDeque<const CAP: usize> {
    data: [MaybeUninit<PosStateBytes>; CAP],
    start: u8,
    len: u8,
}

impl<const CAP: usize> FixedDeque<CAP> {
    pub fn new() -> Self {
        Self {
            data: [MaybeUninit::uninit(); CAP],
            start: 0,
            len: 0,
        }
    }

    #[inline]
    pub fn front(&self) -> Option<&PosStateBytes> {
        if self.len == 0 {
            None
        } else {
            Some(unsafe { self.data[self.start as usize].assume_init_ref() })
        }
    }

    #[inline]
    pub fn back(&self) -> Option<&PosStateBytes> {
        if self.len == 0 {
            None
        } else {
            let idx = (self.start + self.len - 1) as usize % CAP;
            Some(unsafe { self.data[idx].assume_init_ref() })
        }
    }

    #[inline]
    pub fn pop_front(&mut self) {
        debug_assert!(self.len > 0);
        self.start = (self.start + 1) % CAP as u8;
        self.len -= 1;
    }

    #[inline]
    pub fn pop_back(&mut self) {
        debug_assert!(self.len > 0);
        self.len -= 1;
    }

    #[inline]
    pub fn push_back(&mut self, val: PosStateBytes) {
        debug_assert!((self.len as usize) < CAP);
        let idx = (self.start + self.len) as usize % CAP;
        self.data[idx] = MaybeUninit::new(val);
        self.len += 1;
    }
}
