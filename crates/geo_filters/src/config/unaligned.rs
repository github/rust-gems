use std::cmp::Ordering;
use std::fmt::{self, Debug, Formatter};
use std::mem::{align_of, size_of, size_of_val};
use std::ops::{Add, BitAnd, BitAndAssign, BitXor, BitXorAssign, Rem, Shl, Shr};

use crate::config::IsBucketType;

/// A wrapper that stores a value of type `T` with 1-byte alignment (`#[repr(C, packed)]`),
/// allowing zero-copy borrowing of unaligned byte slices while supporting value comparisons
/// and bitwise/bucket operations.
#[derive(Copy, Default)]
#[repr(C, packed)]
pub(crate) struct Unaligned<T>(T);

impl<T: Copy> Clone for Unaligned<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Copy> Unaligned<T> {
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    #[inline]
    pub const fn get(self) -> T {
        self.0
    }
}

impl<T: IsBucketType> Unaligned<T> {
    #[inline]
    pub fn slice_from_bytes(bytes: &[u8]) -> (&[Self], &[u8]) {
        const {
            assert!(align_of::<Self>() == 1);
            assert!(size_of::<Self>() == size_of::<T>());
        }
        let elem_size = size_of::<Self>();
        let len = bytes.len() / elem_size;
        let (head, tail) = bytes.split_at(len * elem_size);
        // SAFETY:
        // - `Unaligned<T>` is `#[repr(C, packed)]` with alignment 1 and size `size_of::<T>()`.
        // - `T: IsBucketType` is only implemented for primitive unsigned integers (`u8`, `u16`,
        //   `u32`, `u64`, `usize`), for which all initialized byte patterns are valid.
        // - `head` has exact byte length `len * size_of::<Self>()`.
        let slice = unsafe { std::slice::from_raw_parts(head.as_ptr().cast::<Self>(), len) };
        (slice, tail)
    }

    #[inline]
    pub fn slice_as_bytes(slice: &[Self]) -> &[u8] {
        // SAFETY: `Unaligned<T>` is `#[repr(C, packed)]` wrapping a primitive unsigned integer `T`
        // with no padding, so all bytes in `slice` are initialized.
        unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), size_of_val(slice)) }
    }
}

impl<T: Copy> From<T> for Unaligned<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T: Copy + Debug> Debug for Unaligned<T> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.get(), f)
    }
}

impl<T: Copy + PartialEq> PartialEq for Unaligned<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl<T: Copy + PartialEq> PartialEq<T> for Unaligned<T> {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.get() == *other
    }
}

macro_rules! impl_partial_eq_for_primitive {
    ($($t:ty),* $(,)?) => {
        $(
            impl PartialEq<Unaligned<$t>> for $t {
                #[inline]
                fn eq(&self, other: &Unaligned<$t>) -> bool {
                    *self == other.get()
                }
            }
        )*
    };
}

impl_partial_eq_for_primitive!(u8, u16, u32, u64, usize);

impl<T: Copy + Eq> Eq for Unaligned<T> {}

impl<T: Copy + PartialOrd> PartialOrd for Unaligned<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.get().partial_cmp(&other.get())
    }
}

impl<T: Copy + PartialOrd> PartialOrd<T> for Unaligned<T> {
    #[inline]
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        self.get().partial_cmp(other)
    }
}

impl<T: Copy + Ord> Ord for Unaligned<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.get().cmp(&other.get())
    }
}

impl<T: Copy + Add<Output = T>> Add for Unaligned<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.get() + rhs.get())
    }
}

impl<T: Copy + Rem<Output = T>> Rem for Unaligned<T> {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        Self::new(self.get() % rhs.get())
    }
}

impl<T: Copy + BitAnd<Output = T>> BitAnd<T> for Unaligned<T> {
    type Output = T;

    #[inline]
    fn bitand(self, rhs: T) -> Self::Output {
        self.get() & rhs
    }
}

impl<T: Copy + BitAnd<Output = T>> BitAnd<T> for &Unaligned<T> {
    type Output = T;

    #[inline]
    fn bitand(self, rhs: T) -> Self::Output {
        self.get() & rhs
    }
}

impl<T: Copy + BitAnd<Output = T>> BitAndAssign<T> for Unaligned<T> {
    #[inline]
    fn bitand_assign(&mut self, rhs: T) {
        self.0 = self.get() & rhs;
    }
}

impl<T: Copy + BitXor<Output = T>> BitXor for Unaligned<T> {
    type Output = T;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        self.get() ^ rhs.get()
    }
}

impl<T: Copy + BitXor<Output = T>> BitXor<T> for Unaligned<T> {
    type Output = T;

    #[inline]
    fn bitxor(self, rhs: T) -> Self::Output {
        self.get() ^ rhs
    }
}

impl<T: Copy + BitXor<Output = T>> BitXorAssign<T> for Unaligned<T> {
    #[inline]
    fn bitxor_assign(&mut self, rhs: T) {
        self.0 = self.get() ^ rhs;
    }
}

impl<T: Copy + Shl<usize, Output = T>> Shl<usize> for Unaligned<T> {
    type Output = T;

    #[inline]
    fn shl(self, rhs: usize) -> Self::Output {
        self.get() << rhs
    }
}

impl<T: Copy + Shr<usize, Output = T>> Shr<usize> for Unaligned<T> {
    type Output = T;

    #[inline]
    fn shr(self, rhs: usize) -> Self::Output {
        self.get() >> rhs
    }
}

impl Unaligned<u64> {
    #[inline]
    pub const fn count_ones(self) -> u32 {
        self.get().count_ones()
    }
}

impl<T: IsBucketType> IsBucketType for Unaligned<T> {
    const BITS: u32 = T::BITS;

    #[inline]
    fn from_usize(v: usize) -> Self {
        Self::new(T::from_usize(v))
    }

    #[inline]
    fn into_usize(self) -> usize {
        self.get().into_usize()
    }
}
