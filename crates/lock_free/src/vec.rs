//! Concurrent append-only vectors.

use std::alloc::Layout;
use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::atomic::{AtomicU32, Ordering};
use std::{mem, ptr, slice};

use bumpalo::Bump;

use super::traits::{IntoArena, SelfContainedDefault};
use super::{CloneWithinArena, Ptr, Writer};

/// Fixed-capacity concurent append-only vector.
///
/// Invariants:
/// -   `len <= capacity`
/// -   `len` monotonically increases.
/// -   `capacity` never changes.
/// -   A buffer `[T; capacity]` immediately follows the struct.
/// -   Elements `0..len` of the buffer are initialized and won't be accessed mutably for the
///     lifetime of the vector.
struct AFixedVec<T> {
    len: AtomicU32,
    capacity: u32,
    _marker: PhantomData<[T]>,
}

/// Concurrent append-only vector.
///
/// Both a single writing thread and multiple reading threads can access an `AVec` at the same
/// time, lock-free.
///
/// This isn't normally possible in Rust. To make it safe, this vector has many constraints:
///
/// -   The elements of an `AVec` are immutable once they are inserted and cannot be removed.
/// -   Only one thread may push values to an `AVec` at a time.
/// -   The memory for an `AVec` must be managed by an arena allocator.
/// -   When `.push()` is called, if the vector is already full, a new buffer is allocated and
///     the old buffer *is not freed*.
/// -   `AVec`s can only live in arenas. You can make one (an empty one) outside of an arena, but
///     you can't push to it -- you can't get a `Writer` to it. This restriction is necessary
///     for safety because an `AVec` is a pointer to arena memory. Without the arena, the
///     pointer could be dangling.
pub struct AVec<T> {
    /// Invariant: ptr is null or points to a valid `AFixedVec`. The only practical way to maintain
    /// this is to store the `AVec` and the `AFixedVec` in the same arena.
    ptr: Ptr<AFixedVec<T>>,
}

impl<T> CloneWithinArena for AVec<T> {
    unsafe fn clone_within_arena(&self) -> Self {
        AVec {
            // Safety: Transitive.
            ptr: unsafe { self.ptr.clone_within_arena() },
        }
    }
}

impl<T> AFixedVec<T> {
    /// Allocate a new vector.
    fn new<'arena>(arena: &'arena Bump, capacity: usize, elements: &[T]) -> &'arena Self
    where
        T: CloneWithinArena,
    {
        let len = elements.len();
        assert!(len <= capacity);
        assert_eq!(
            capacity as u32 as usize, capacity,
            "capacity must fit in 32 bits"
        );

        let buffer_layout =
            Layout::array::<T>(capacity).expect("overflow computing AFixedVec buffer size");
        let (layout, buffer_offset) = Layout::new::<Self>()
            .extend(buffer_layout)
            .expect("overflow computing AFixedVec buffer size");
        assert_eq!(
            buffer_offset,
            Self::BUFFER_OFFSET,
            "AFixedVec::buffer assumes this"
        );
        let header_ptr = arena.alloc_layout(layout.pad_to_align()).as_ptr() as *mut AFixedVec<T>;

        unsafe {
            // Safety: `header` is nonzero and properly aligned and sized for this write;
            // and the memory is newly allocated, so only the current thread is accessing it.
            std::ptr::write(
                header_ptr,
                AFixedVec {
                    len: AtomicU32::new(len as u32),
                    capacity: capacity as u32,
                    _marker: PhantomData,
                },
            );

            // Safety: The memory is allocated from `arena` so it'll be good for the lifetime
            // `'arena`.
            let header = &mut *header_ptr;

            // Safety: The memory is newly allocated. We checked `elements.len()` above, so the
            // writes are in range. It's safe to `clone_within_arena` because the new values are
            // stored in the same arena as the original ones.
            let buf_ptr = header.buffer_mut();
            for (i, item) in elements.iter().enumerate() {
                std::ptr::write(buf_ptr.add(i), item.clone_within_arena());
            }
            header
        }
    }

    fn capacity(&self) -> usize {
        self.capacity as usize
    }

    fn len(&self) -> usize {
        self.len.load(Ordering::Acquire) as usize
    }

    /// Offset of the `[T]`, in bytes, from the start of `self`.
    /// This is just `sizeof Self` rounded up to a multiple of `alignof T`.
    const BUFFER_OFFSET: usize = (mem::size_of::<Self>() + mem::align_of::<T>() - 1)
        / mem::align_of::<T>()
        * mem::align_of::<T>();

    fn buffer(&self) -> *const T {
        // Safety: Only `AFixedVec::new` creates `AFixedVec`s. It allocates a buffer of type `[T;
        // capacity]` immediately after the struct and asserts that `BUFFER_OFFSET` is correct.
        unsafe { (self as *const Self as *const u8).add(Self::BUFFER_OFFSET) as *const T }
    }

    fn buffer_mut(&mut self) -> *mut T {
        self.buffer() as *mut T
    }

    /// Returns the current contents of the vector as a slice.
    fn as_slice(&self) -> &[T] {
        // Safety: `buffer` is non-null and aligned, as required. It's OK if `len` and/or `capacity`
        // is 0, though that will not normally be the case.
        unsafe { std::slice::from_raw_parts(self.buffer(), self.len()) }
    }

    /// Returns an iterator over the elements already stored in the vector.
    ///
    /// It's safe to call this method while another thread is appending to the vector. This method
    /// gets the number of elements already in the vector, and the iterator will observe only that
    /// many elements.
    fn iter(&self) -> std::slice::Iter<T> {
        self.as_slice().iter()
    }

    /// Atomically adds an element to the end of the vector, growing the buffer if necessary.
    /// If the vector is already full, this uses `arena` to allocate a new buffer, copies the
    /// elements over, and returns a reference to the new buffer.
    ///
    /// # Safety
    ///
    /// Caller must be the exclusive writer of the vector.
    unsafe fn push<'arena>(&'arena self, arena: &'arena Bump, value: T) -> &'arena Self
    where
        T: CloneWithinArena,
    {
        let len = self.len();
        let capacity = self.capacity as usize;
        let vec = if len == capacity {
            // Reallocate.
            let new_capacity = capacity * 2;
            let old_slice = unsafe { slice::from_raw_parts((*self).buffer(), len) };
            AFixedVec::new(arena, new_capacity, old_slice)
        } else {
            self
        };

        // Safety: We have room. No other thread is looking at this memory because we have not
        // bumped `vec.len` yet. The cast from `*const T` to `*mut T` is OK because the caller
        // guarantees we're the only writer.
        unsafe {
            ptr::write(vec.buffer().add(len) as *mut T, value);
        }
        vec.len.store((len + 1) as u32, Ordering::Release);
        vec
    }
}

impl<'a, T> IntoIterator for &'a AFixedVec<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> Debug for AFixedVec<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

impl<'w, T> Writer<'w, AFixedVec<T>> {
    /// Like the standard `Vec`'s `get_mut()` method, except get a writer, not a `mut` reference.
    fn get_writer_impl(&self, i: usize) -> Option<Writer<'w, T>> {
        // Safety: elem is in the same arena as self.target().
        self.target()
            .as_slice()
            .get(i)
            .map(|elem| unsafe { self.make_child_writer(elem) })
    }
}

impl<T> AVec<T> {
    /// Creates a new empty vector.
    pub fn new() -> Self {
        Self::default()
    }

    fn storage(&self) -> Option<&AFixedVec<T>> {
        self.ptr.get()
    }

    /// Returns the number of elements the current buffer can hold.
    pub fn capacity(&self) -> usize {
        match self.storage() {
            None => 0,
            Some(vec) => vec.capacity(),
        }
    }

    /// Returns the number of elements in the vector.
    pub fn len(&self) -> usize {
        match self.storage() {
            None => 0,
            Some(vec) => vec.len(),
        }
    }

    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a slice containing all the elements of the vector.
    pub fn as_slice(&self) -> &[T] {
        match self.storage() {
            None => &[],
            Some(vec) => vec.as_slice(),
        }
    }
}

impl<T> SelfContainedDefault for AVec<T> {}

impl<T> Default for AVec<T> {
    fn default() -> Self {
        AVec {
            ptr: Ptr::default(),
        }
    }
}

impl<T> Deref for AVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T> IntoIterator for &'a AVec<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> Debug for AVec<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

impl<'w, T> Writer<'w, AVec<T>> {
    /// Atomically adds a default element to the end of the vector.
    pub fn push_default(&self)
    where
        T: CloneWithinArena + SelfContainedDefault,
    {
        self.push_inner(T::default());
    }

    /// Atomically adds an element to the end of the vector.
    pub fn push<X: IntoArena<T>>(&self, item: X)
    where
        T: CloneWithinArena,
    {
        // Safety: Just checked that the item is in the same arena.
        self.push_inner(unsafe { item.into_arena(self.arena()) });
    }

    /// Atomically adds an element to the end of the vector.
    fn push_inner(&self, value: T)
    where
        T: CloneWithinArena,
    {
        let v = self.target();
        let arena = self.arena();
        match v.storage() {
            None => {
                let storage = AFixedVec::new(arena, 8, slice::from_ref(&value));
                unsafe {
                    v.ptr.store_raw(storage);
                }
            }
            Some(storage) => {
                let new_storage = unsafe { storage.push(arena, value) };
                if !ptr::eq(storage, new_storage) {
                    unsafe {
                        v.ptr.store_raw(new_storage);
                    }
                }
            }
        }
    }

    fn storage_writer(&self) -> Option<Writer<'w, AFixedVec<T>>> {
        self.target()
            .storage()
            .map(|fixed_vec| unsafe { self.make_child_writer(fixed_vec) })
    }

    /// Gets a writer for an element of the target vector.
    ///
    /// This is like `Vec::get_mut`, but returns a writer instead of a `mut` reference.
    pub fn get_writer(&self, i: usize) -> Option<Writer<'w, T>> {
        self.storage_writer()
            .and_then(|storage| storage.get_writer_impl(i))
    }
}
