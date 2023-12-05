//! Smart pointers for lock-free data structures.

use std::fmt::{self, Debug, Formatter};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};

use super::arena::KeepAliveHandle;
use super::hash::StoreRelaxed;
use super::traits::{IntoArena, SelfContained, SelfContainedDefault};
use super::{AHashKey, Arena, CloneWithinArena};

// --- Writer -------------------------------------------------------------------------------------

/// Like a `mut` reference for lock-free data structures.
///
/// Like a `&mut T`, a `Writer<T>` gives you access to all `T`'s ordinary (`&self`) methods,
/// and also access to modify the value `T`.
///
/// Unlike a `&mut T`, it doesn't promise exclusive access:
/// -   Other threads may be reading the `T` concurrently.
/// -   Other writers may exist that point to the same `T` (but only in the same thread).
///
/// It would be undefined behavior to make a real `&mut` pointing to shared data.
///
/// Note: There's not a `Reader` type because it's OK to use plain old references, `&'r T`, to
/// refer to arena data.
pub struct Writer<'w, T> {
    /// Value to which we have shared-write access.
    target: &'w T,

    /// Invariant: `*self.target` lives in `self.arena`.
    arena: &'w Arena,

    /// Hack to make writers not be `Sync` or `Send`.
    ///
    /// `Sync`: Sharing writers across threads would defeat the whole point. `Send`: The writer
    /// thread can have many writers at once, so sending one to another thread would break the
    /// invariant.
    _marker: PhantomData<*mut T>,
}

/// Implement `Copy` and `Clone` explicitly, since `derive` only works if `T` is `Copy` and
/// `Clone`.
impl<'a, T> Copy for Writer<'a, T> {}

impl<'a, T> Clone for Writer<'a, T> {
    fn clone(&self) -> Self {
        // Safety: If `self` satisfies the invariant, so will the clone.
        *self
    }
}

impl<'w, T> Deref for Writer<'w, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.target
    }
}

impl<'w, T> Debug for Writer<'w, T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        <T as Debug>::fmt(&**self, f)
    }
}

impl<'w, T> Writer<'w, T> {
    /// Creates a new `Writer`.
    ///
    /// # Safety
    ///
    /// This is safe if `target` lives in `arena`. Otherwise `target` may just be a short-lived
    /// borrow of an object that will outlive `arena`, leading to dangling pointers.
    pub unsafe fn new(arena: &'w Arena, target: &'w T) -> Self {
        Writer {
            target,
            arena,
            _marker: PhantomData,
        }
    }

    /// Get the arena this writer's `target` lives in.
    pub fn arena(&self) -> &'w Arena {
        self.arena
    }

    /// Get a direct (but read-only) Rust reference to the target of this writer.
    pub fn target(&self) -> &'w T {
        self.target
    }

    /// Make a writer that refers to another value in the same arena.
    ///
    /// # Safety
    ///
    /// This is safe as long as `*child` actually lives in the same arena as `self.target`.
    pub unsafe fn make_child_writer<U>(&self, child: &'w U) -> Writer<'w, U> {
        Writer {
            target: child,
            arena: self.arena,
            _marker: PhantomData,
        }
    }

    /// Converts a `Writer` into a `WriteHandle` which increments the reference count to the arena.
    pub fn write_handle(&self) -> WriteHandle<T> {
        WriteHandle {
            arena: self.arena.clone(),
            ptr: self.target as *const T as *mut T,
        }
    }

    /// Converts a `Writer` into a `ReadHandle` which can be shared across threads.
    pub fn read_handle(&self) -> ReadHandle<T> {
        ReadHandle {
            arena: self.arena.keep_alive(),
            ptr: self.target as *const T as *mut T,
        }
    }
}

impl<'a, T: CloneWithinArena> IntoArena<T> for Writer<'a, T> {
    /// # Safety
    ///
    /// The instance that we copy lives already within an arena. Copying it within the same arena
    /// is safe.
    unsafe fn into_arena(self, arena: &Arena) -> T {
        assert!(ptr::eq(self.arena(), arena));
        unsafe { self.target.clone_within_arena() }
    }
}

// --- ABox ---------------------------------------------------------------------------------------

/// Atomic pointer from arena data to arena data, a low-level building block.
///
/// This is the type that's actually stored in arena lock-free data structures.
///
/// -   Unlike a `Box<T>`, an `ABox<T>` isn't necessarily the unique pointer to the `T`. The graph
///     of pointers in the arena is an arbitrary graph, not a tree.
///
/// -   Unlike an `Arc<T>`, there's no work to do when cloning or dropping an `ABox<T>`.
///
/// -   Like an `Arc<T>`, an `ABox<T>` inherits shared (`&self`) methods of the underlying `T`.
///     (It implements `Deref`.)
///
/// -   `ABox<T>` confers read access to a `T` value. A single writer thread may concurrently
///     modify the value.
pub struct ABox<T> {
    /// Invariant: `self.ptr` is non-null and points to a valid `T`.
    ///
    /// This `ABox` must not outlive the target and become a dangling pointer. (This is usually
    /// done by storing the `T` and the `ABox<T>` in the same arena.)
    ///
    /// If `self.ptr` is changed to point to something else, the old `T` *still* must be retained.
    /// Another reader thread might still be accessing it.
    ptr: AtomicPtr<T>,

    /// Make `ABox` not be `Send` or `Sync`, so Rust enforces thread safety.
    _marker: PhantomData<*const T>,
}

impl<T> ABox<T> {
    /// Get a reference to the data. On x86 this compiles to a single `mov` instruction.
    pub fn get(&self) -> &T {
        let p = self.load_raw();
        assert!(!p.is_null());
        // Safety: struct invariant ensures this points to a valid `T` in the same arena.
        // All other operations that could break those invariants are themselves unsafe.
        unsafe { &*p }
    }

    /// Make an `ABox` from a reference.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the resulting `ABox` does not outlive `*target`. (This is
    /// usually done by storing the `ABox` in the same arena where the `T` already lives.)
    unsafe fn from_ref(target: &T) -> Self {
        ABox {
            ptr: AtomicPtr::new(target as *const T as *mut T),
            _marker: PhantomData,
        }
    }

    /// Get the raw pointer value.
    fn load_raw(&self) -> *const T {
        self.ptr.load(Ordering::Acquire)
    }

    /// Set the raw pointer value.
    ///
    /// # Safety
    ///
    /// This is safe if `p` points to a `T` that will live as long as this `ABox`.
    unsafe fn store_raw(&self, p: *const T) {
        self.ptr.store(p as *mut T, Ordering::Release);
    }
}

impl<'w, T> Writer<'w, ABox<T>> {
    /// Change the pointer value.
    ///
    /// Note: The `IntoArena` trait might allocate the instance within the arena (e.g. for basic types).
    pub fn store<X: IntoArena<ABox<T>>>(&self, value: X) {
        unsafe {
            // Safety: We're about to store this new Ptr in the arena.
            let ptr = value.into_arena(self.arena());
            let p = ptr.load_raw();
            // Safety: We're the sole writer, and `into_arena` asserted that `value` lives in
            // `self.arena()`.
            self.target().store_raw(p);
        }
    }

    /// Gets a "mutable" reference to the data being pointed to.
    pub fn get_writer(&self) -> Writer<'w, T> {
        // Safety: Struct invariant guarantees the referent will live at least as long as this
        // ABox, which will live at least for lifetime `'w`.
        unsafe { self.make_child_writer(&**self.target) }
    }
}

impl<T: SelfContained + Eq + Hash> AHashKey for ABox<T> {
    type KeyRef<'a> = &'a T where T: 'a;

    fn empty() -> Self {
        ABox {
            ptr: AtomicPtr::new(ptr::null_mut()),
            _marker: PhantomData,
        }
    }

    // Views are never null.
    fn is_empty(_key: &T) -> bool {
        false
    }

    fn load_acquire(&self) -> Option<&T> {
        let p = self.load_raw();
        if p.is_null() {
            None
        } else {
            // Safety: struct invariant ensures this points to a valid `T` in the same arena.
            // All other operations that could break those invariants are themselves unsafe.
            Some(unsafe { &*p })
        }
    }

    unsafe fn key_into_arena(arena: &Arena, key: &T) -> Self {
        // TODO: make Format::View implement a Serializable trait without a lifetime so this extra work isn't
        // necessary here.
        ABox {
            ptr: AtomicPtr::new(arena.alloc(*key).target() as *const _ as *mut _),
            _marker: PhantomData,
        }
    }

    unsafe fn store_release(&self, key: Self) {
        // Safety: The trait passes responsibility on to the caller.
        unsafe { self.store_raw(key.load_raw()) };
    }

    fn reborrow<'short, 'long: 'short>(key: &'long T) -> &'short T {
        key
    }
}

impl<'a, T> From<Writer<'a, T>> for ABox<T> {
    fn from(value: Writer<'a, T>) -> Self {
        ABox {
            ptr: AtomicPtr::new(value.target as *const _ as *mut _),
            _marker: Default::default(),
        }
    }
}

impl<'a, T> StoreRelaxed<Writer<'a, T>> for ABox<T> {
    fn store_relaxed(&self, value: Writer<'a, T>) {
        self.ptr
            .store(value.target as *const _ as *mut _, Ordering::Relaxed);
    }
}

/// Note: Unlike `Box`, cloning an `ABox<T>` does not clone the `T`.
impl<T> CloneWithinArena for ABox<T> {
    unsafe fn clone_within_arena(&self) -> Self {
        // Safety: Caller will ensure the clone does not outlive the target.
        ABox {
            ptr: AtomicPtr::new(self.load_raw() as *mut T),
            _marker: PhantomData,
        }
    }
}

impl<T> Deref for ABox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        // Safety: Struct invariant ensures this is safe for the lifetime of the borrow.
        unsafe { &*self.load_raw() }
    }
}

impl<'a, T> IntoArena<ABox<T>> for Writer<'a, T> {
    unsafe fn into_arena(self, arena: &Arena) -> ABox<T> {
        assert!(ptr::eq(self.arena(), arena));
        // Safety: We just asserted that `arena` is the right arena. Caller will ensure the result
        // doesn't outlive that arena.
        unsafe { ABox::from_ref(self.target()) }
    }
}

impl<T: SelfContained> IntoArena<ABox<T>> for T {
    unsafe fn into_arena(self, arena: &Arena) -> ABox<T> {
        let t = arena.alloc(self);
        // Safety: Caller will ensure the result doesn't outlive the arena.
        unsafe { ABox::from_ref(&t) }
    }
}

impl<T> Debug for ABox<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        <T as Debug>::fmt(self, f)
    }
}

// --- Ptr ----------------------------------------------------------------------------------------

/// Atomic pointer from arena data to arena data, a low-level building block.
///
/// This is the type that's actually stored in arena lock-free data structures.
///
/// -   Unlike a `Box<T>`, a `Ptr<T>` isn't necessarily the unique pointer to the `T`.
///     The graph of pointers in the arena is an arbitrary graph, not a tree.
///
/// -   Unlike an `Arc<T>`, there's no work to do when cloning or dropping a `Ptr<T>`.
///
/// -   `Ptr<T>` confers read access to a `T` value. A single writer thread may concurrently
///     modify the value.
///
/// Like any pointer, practically all operations on a `Ptr` are unsafe.
pub struct Ptr<T> {
    /// Invariant: `self.ptr` is either null or points to a valid `T`.
    ///
    /// This `Ptr` must not outlive the target and become a dangling pointer. (This is usually done
    /// by storing the `T` and the `Ptr<T>` in the same arena.)
    ///
    /// If `self.ptr` is changed to point to something else, the old `T` *still* must be retained.
    /// Another reader thread might still be accessing it.
    ptr: AtomicPtr<T>,

    _marker: PhantomData<Option<T>>,
}

/// Cloning a `Ptr<T>` does not clone the `T`, so caller must ensure the clone does not outlive the
/// target.
impl<T> CloneWithinArena for Ptr<T> {
    unsafe fn clone_within_arena(&self) -> Self {
        // Safety: If the original instance points to an address within the arena or null, then the
        // clone will do the same and thus can be stored inside the same arena.
        Ptr {
            ptr: AtomicPtr::new(self.ptr.load(Ordering::Acquire)),
            _marker: PhantomData,
        }
    }
}

impl<T> SelfContainedDefault for Ptr<T> {}

impl<T> Default for Ptr<T> {
    fn default() -> Self {
        Ptr {
            ptr: AtomicPtr::default(),
            _marker: PhantomData,
        }
    }
}

impl<T> Ptr<T> {
    /// Return a null pointer.
    pub fn null() -> Self {
        Self::default()
    }

    /// Make a `Ptr` from a reference.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the resulting `Ptr` does not outlive `*target`. (This is
    /// usually done by storing the `Ptr` in the same arena where the `T` already lives.)
    pub unsafe fn from_ref(target: &T) -> Self {
        Ptr {
            ptr: AtomicPtr::new(target as *const T as *mut T),
            _marker: PhantomData,
        }
    }

    /// Get the raw pointer value.
    pub fn load_raw(&self) -> *const T {
        self.ptr.load(Ordering::Acquire)
    }

    /// Change the pointer value.
    ///
    /// # Safety
    ///
    /// Caller must be the sole writer. `ptr` must be null or point to a valid `T` that will live
    /// as long as this `Ptr` value (typically, a pointer to a value allocated in the same arena).
    pub unsafe fn store_raw(&self, ptr: *const T) {
        self.ptr.store(ptr as *mut T, Ordering::Release);
    }

    /// Get a reference to the data. On x86 this compiles to a single `mov` instruction.
    pub fn get(&self) -> Option<&T> {
        let p = self.load_raw();
        if p.is_null() {
            None
        } else {
            // Safety: struct invariant ensures this points to a valid `T` in the same arena.
            // All other operations that could break those invariants are themselves unsafe.
            Some(unsafe { &*p })
        }
    }
}

impl<'w, T> Writer<'w, Ptr<T>> {
    /// Change the pointer value.
    ///
    /// Note: The `IntoArena` trait might allocate the instance within the arena (e.g. for basic types).
    pub fn store<X: IntoArena<Ptr<T>>>(&self, value: X) {
        unsafe {
            // Safety: We're about to store this new Ptr in the arena.
            let ptr = value.into_arena(self.arena());
            let p = ptr.load_raw();
            // Safety: We're the sole writer, and `into_arena` asserted that `value` lives in
            // `self.arena()`.
            self.store_raw(p);
        }
    }

    /// Change the pointer value to null. Calling `ptr.store(None)` requires a complicated type annotation
    /// which this function avoids.
    pub fn clear(&self) {
        self.target.ptr.store(ptr::null_mut(), Ordering::Release);
    }

    /// Gets a "mutable" reference to the data being pointed to.
    pub fn get_writer(&self) -> Option<Writer<'w, T>> {
        self.target
            .get()
            .map(|ptr| unsafe { self.make_child_writer(ptr) })
    }
}

impl<'a, T> IntoArena<Ptr<T>> for Option<Writer<'a, T>> {
    /// # Safety
    ///
    /// Per definition of the `Writer` type the instance lives within an arena.
    /// We have to ensure that it is the correct arena and then it's safe to convert the target into a `Ptr`.
    unsafe fn into_arena(self, arena: &Arena) -> Ptr<T> {
        if let Some(writer) = self {
            assert!(ptr::eq(writer.arena(), arena));
            unsafe { Ptr::from_ref(writer.target()) }
        } else {
            Ptr::null()
        }
    }
}

impl<T: SelfContained> IntoArena<Ptr<T>> for T {
    /// # Safety
    ///
    /// Per definition of the `SelfContained` trait it is safe to allocate this type inside of the arena.
    /// As a result, Ptr<T> is now a valid instance for the provided arena as well.
    unsafe fn into_arena(self, arena: &Arena) -> Ptr<T> {
        unsafe { Ptr::from_ref(&arena.alloc(self)) }
    }
}

impl<T> Debug for Ptr<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        <Option<&T> as Debug>::fmt(&self.get(), f)
    }
}

impl<T> From<ABox<T>> for Ptr<T> {
    fn from(b: ABox<T>) -> Self {
        Ptr {
            ptr: b.ptr,
            _marker: PhantomData,
        }
    }
}

// --- Read handles -------------------------------------------------------------------------------

/// Kind of like Arc for arena data. Shared ownership, read-only access.
///
/// Smart pointer from normal Rust code to a value that lives in an arena. The handle keeps the
/// whole arena alive via reference counting.
///
pub struct ReadHandle<T: ?Sized> {
    /// Invariant: If `ptr` points into any arena, directly or indirectly, `arena` is that arena.
    #[allow(dead_code)] // field exists only for behavior on drop
    arena: KeepAliveHandle,
    ptr: *const T,
}

impl<T> Clone for ReadHandle<T> {
    fn clone(&self) -> Self {
        Self {
            arena: self.arena.clone(),
            ptr: self.ptr,
        }
    }
}

unsafe impl<T> Send for ReadHandle<T> where T: Sync {}

unsafe impl<T> Sync for ReadHandle<T> where T: Sync {}

impl<T: ?Sized> Deref for ReadHandle<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.ptr }
    }
}

impl<T> Debug for ReadHandle<T>
where
    T: Debug + ?Sized,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        <T as Debug>::fmt(&**self, f)
    }
}

// --- Write handles ------------------------------------------------------------------------------

/// Like ReadHandle, but also providing write access. Confined to the sole writer thread.
///
/// Smart pointer from normal Rust code to a value that lives in an arena, conferring read and
/// write access. The handle keeps the whole arena alive via reference counting.
#[derive(Clone)]
pub struct WriteHandle<T: ?Sized> {
    /// Invariant: If `ptr` points into any arena, directly or indirectly, `arena` is that arena.
    #[allow(dead_code)] // field exists only for behavior on drop
    pub(super) arena: Arena,
    pub(super) ptr: *const T,
}

impl<T> Debug for WriteHandle<T>
where
    T: Debug + ?Sized,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        <T as Debug>::fmt(&**self, f)
    }
}

impl<T: ?Sized> Deref for WriteHandle<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.ptr }
    }
}

impl<T> WriteHandle<T> {
    /// Makes a `Writer` for this object. The writer is where you'll find safe methods to modify
    /// data structures, like `.push()` on `Writer<AVec<T>>` or `.get_or_insert_default()` on
    /// `Writer<AHashMap<K, V>>`.
    pub fn writer(&self) -> Writer<'_, T> {
        Writer {
            target: unsafe { &*self.ptr },
            arena: &self.arena,
            _marker: PhantomData,
        }
    }

    /// Makes a read handle to this object. This is useful because a `ReadHandle` is `Send`.
    pub fn read_handle(&self) -> ReadHandle<T> {
        ReadHandle {
            arena: self.arena.keep_alive(),
            ptr: self.ptr,
        }
    }
}
