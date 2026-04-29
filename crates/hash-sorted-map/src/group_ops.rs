// Platform-dependent group size: 16 on x86_64 (SSE2), 8 everywhere else.
#[cfg(target_arch = "x86_64")]
pub const GROUP_SIZE: usize = 16;
#[cfg(not(target_arch = "x86_64"))]
pub const GROUP_SIZE: usize = 8;

pub const CTRL_EMPTY: u8 = 0x00;

#[cfg(target_arch = "x86_64")]
pub type Mask = u32;
#[cfg(not(target_arch = "x86_64"))]
pub type Mask = u64;

// ── SIMD group operations ───────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod arch {
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
mod arch {
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
mod arch {
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

pub use arch::*;
