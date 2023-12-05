//! Arena handles.

use std::fmt::{self, Debug, Formatter};
use std::ops::Deref;
use std::sync::Arc;

use bumpalo::Bump;

use super::traits::{SelfContained, SelfContainedDefault};
use super::Writer;

/// Handle to an arena where only one thread may allocate, but other threads can safely use
/// allocated memory. The arena stays around until all threads are done using the memory.
///
/// **Critical safety rule:** Only one thread may allocate memory from a Bump arena at a time.
/// Otherwise, multiple threads, racing to allocate memory at the same time, could both get the
/// same memory and crash horribly after writing all over one another's data.
///
/// To enforce this rule, only the bearer of `Arena` can use the allocator. Other threads get
/// opaque `KeepAliveHandle`s which do nothing except keep the arena from being freed.
#[derive(Debug, Clone)]
pub struct Arena {
    arena: Arc<Bump>,
}

impl Deref for Arena {
    type Target = Bump;

    fn deref(&self) -> &Bump {
        &self.arena
    }
}

impl Arena {
    /// Create an arena.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        // Note: Clippy complains about us putting a `Bump` into an `Arc`, because that is usually
        // a mistake; but here it's on purpose. Other threads don't need to access the `Bump`, only
        // keep it alive (or drop it).
        #[allow(clippy::arc_with_non_send_sync)]
        Arena {
            arena: Arc::new(Bump::new()),
        }
    }

    /// Just as alloc, but the caller has to ensure that the object doesn't reference
    /// data outside of the arena!
    ///
    /// # Safety
    ///
    /// Caller has to ensure that references don't leave the arena.
    pub unsafe fn alloc_unsafe<T>(&self, value: T) -> Writer<T> {
        let target: &mut T = self.arena.alloc(value);
        unsafe { Writer::new(self, target) }
    }

    /// Allocate an object in the arena.
    pub fn alloc<T>(&self, value: T) -> Writer<T>
    where
        T: SelfContained,
    {
        let target: &mut T = self.arena.alloc(value);
        unsafe { Writer::new(self, target) }
    }

    /// Allocate a default object in the arena.
    pub fn alloc_default<T>(&self) -> Writer<T>
    where
        T: SelfContainedDefault,
    {
        let target: &mut T = self.arena.alloc(T::default());
        unsafe { Writer::new(self, target) }
    }

    pub(super) fn keep_alive(&self) -> KeepAliveHandle {
        KeepAliveHandle {
            handle: self.arena.clone(),
        }
    }
}

/// Opaque handle to an arena that does nothing but keep it from being freed. Unlike `Arena` this
/// can be cloned and shared across threads.
///
/// Private. An implementation detail of `ABox`.
#[derive(Clone)]
pub(super) struct KeepAliveHandle {
    #[allow(dead_code)] // field exists only for behavior on drop
    handle: Arc<Bump>,
}

/// Mark handles as safe to send across threads.
///
/// This is more liberal than a plain `Arc<T>`, which requires `T: Send + Sync`. It is safe (even
/// though `Bump` is not `Sync`) because `KeepAliveHandle` never actually constructs a reference to
/// the underlying `Bump` (until we are dropping it).
///
/// We will drop the `Bump` on whichever thread happens to drop its handle last. That is OK because
/// `Bump` is `Send`.
unsafe impl Send for KeepAliveHandle {}

/// Referencing a KeepAliveHandle across threads is harmless, as it does not provide
/// access to the underlying `Bump` value.
unsafe impl Sync for KeepAliveHandle {}

impl Debug for KeepAliveHandle {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // Don't delegate to self.handle, as the overall safety of `KeepAliveHandle` relies on
        // *not* accessing the underlying `Bump` value!
        write!(f, "KeepAliveHandle")
    }
}
