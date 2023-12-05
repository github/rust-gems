//! Traits for arena data structures.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use super::Arena;

/// Trait for types that live inside an arena, and can be cloned, as long as the clone does not
/// outlive the arena.
///
/// Arena data structures are full of pointers. Pointers are dangerous. `AVec<T>`, for example,
/// contains a pointer to a buffer in the same arena where its elements are stored. If an `AVec<T>`
/// were somehow *removed* from its arena, and the arena dropped, that would become a dangling
/// pointer.
///
/// This trait provides a clone-like method for such types. It is the caller's responsibilty to
/// clone responsibly, i.e. to make sure the clone doesn't outlive the arena.
pub trait CloneWithinArena {
    /// Make a copy of `self`, for use within the same arena as the original.
    ///
    /// # Safety
    ///
    /// The caller must ensure the copy does not outlive the arena where the original lives.
    /// Usually this is done by storing the clone in the same arena.
    unsafe fn clone_within_arena(&self) -> Self;
}

impl CloneWithinArena for AtomicU32 {
    unsafe fn clone_within_arena(&self) -> Self {
        AtomicU32::new(self.load(Ordering::Relaxed))
    }
}

impl CloneWithinArena for AtomicU64 {
    unsafe fn clone_within_arena(&self) -> Self {
        AtomicU64::new(self.load(Ordering::Relaxed))
    }
}

/// Basic types are self contained, i.e. they don't contain pointers to other types.
/// They can be safely cloned from outside the arena.
pub trait SelfContained: Copy {}

impl SelfContained for u8 {}
impl SelfContained for u16 {}
impl SelfContained for u32 {}
impl SelfContained for u64 {}
impl SelfContained for usize {}

impl SelfContained for i64 {}

impl SelfContained for &'static str {}

/// Our lock free data structures live completely inside an arena. Once they got created, they can be modified as needed.
/// But we need to create them somehow at the beginning. That's what this trait enables.
pub trait SelfContainedDefault: Default {}

/// This trait is implemented for types whose data can be written into a specific arena.
/// This means either the data is self contained or the data has pointers to data within the same arena.
/// A function call may allocate memory within the arena. Therefore, it is highly discouraged to
/// use this trait when the returned instance may be dropped.
pub trait IntoArena<T> {
    /// Convert `self` to the type that will be stored in the arena.
    ///
    /// # Panics
    ///
    /// If `self` is not safe to store in `arena`, for example because it contains pointers
    /// into another arena.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the resulting `T` value does not outlive `arena`.
    /// Usually this is done by storing it in the same arena.
    unsafe fn into_arena(self, arena: &Arena) -> T;
}

impl<T: SelfContained> IntoArena<T> for T {
    /// # Safety
    ///
    /// This is safe because the data is self contained.
    unsafe fn into_arena(self, _arena: &Arena) -> T {
        self
    }
}

impl<T: IntoArena<T>> IntoArena<Option<T>> for Option<T> {
    /// # Safety
    ///
    /// A type wrapped into an `Option` inherits the `IntoArena` trait from its base type.
    unsafe fn into_arena(self, _arena: &Arena) -> Option<T> {
        self
    }
}

impl<T: SelfContained> CloneWithinArena for T {
    unsafe fn clone_within_arena(&self) -> Self {
        *self
    }
}
