//! Serialization.

use std::alloc::Layout;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::{ptr, slice};

use github_pspack::Serializable;

use super::{
    ptr::Ptr,
    traits::{IntoArena, SelfContainedDefault},
    Arena, CloneWithinArena, Writer,
};

/// Pointer to serialized data stored in the arena.
pub struct ASerialized<F: Format> {
    ptr: Ptr<APackBuffer>,
    _marker: PhantomData<*const F>,
}

/// A slice whose size cannot be changed.
#[repr(C)] // make sure `data` is actually laid out at the end
pub struct APackBuffer {
    len: usize,
    data: [u8; 0],
}

/// Dummy type that represents a binary data format.
///
/// Every `Format` has a `View` type for parsing the various fields etc. out of
/// serialized bytes. To get a view, call `F::View::from_bytes(bytes)`.
pub trait Format {
    /// Type that can get individual fields out of the serialized bytes.
    /// This is lightweight, like a reference to the underlying byte slice.
    // TODO: Make a borrowed-TreePath type that's Copy so this can have a Copy
    // bound instead of Clone.
    type View<'a>: Clone + Serializable<'a> + Debug;

    /// View must be something we can "reborrow" with a shorter lifetime (like a normal
    /// reference).
    fn reborrow<'short, 'long: 'short>(view: Self::View<'long>) -> Self::View<'short>;
}

/// Marker trait for types that implement `Serializable` in such a way that the
/// output bytes are compatible with a particular data format, `Self::Format`.
pub trait SerializableAs: Debug {
    /// The format that this type is compatible with.
    type Format: Format;

    /// Convert `self` to bytes in the format `Self::Format`.
    fn to_bytes(&self) -> Vec<u8>;
}

impl<F: Format> ASerialized<F> {
    pub(super) fn null() -> Self {
        ASerialized {
            ptr: Ptr::null(),
            _marker: PhantomData,
        }
    }

    /// # Safety
    ///
    /// Caller must ensure the return value does not outlive the arena.
    pub(super) unsafe fn from_vec(arena: &Arena, bytes: Vec<u8>) -> ASerialized<F> {
        let ptr = APackBuffer::allocate(arena, &bytes);
        let abox = unsafe { ptr.into_arena(arena) };
        ASerialized {
            ptr: abox.into(),
            _marker: PhantomData,
        }
    }

    /// # Safety
    ///
    /// Caller must ensure the return value does not outlive the arena.
    unsafe fn allocate(arena: &Arena, data: impl SerializableAs<Format = F>) -> ASerialized<F> {
        let bytes = data.to_bytes();
        // Safety: We pass responsibility on to the caller.
        unsafe { Self::from_vec(arena, bytes) }
    }

    /// Returns true if this is a null pointer. Apart from this method, a null
    /// pointer behaves exactly like a pointer to a zero-length buffer.
    pub fn is_null(&self) -> bool {
        self.ptr.get().is_none()
    }

    /// Get a view of the serialized data.
    pub fn get(&self) -> F::View<'_> {
        F::View::from_bytes(self.as_slice())
    }

    fn as_slice(&self) -> &[u8] {
        self.ptr.get().map(|x| x.as_slice()).unwrap_or(&[])
    }

    /// Switch this ASerialized to point to a different buffer.
    ///
    /// # Safety
    ///
    /// `new_value` must either be null or point to a buffer that lives in the
    /// same arena as `self`.
    pub(super) unsafe fn store(&self, new_value: Self) {
        let new_ptr = new_value.ptr.load_raw();
        // Safety: We pass responsibility on to the caller.
        unsafe {
            self.ptr.store_raw(new_ptr);
        }
    }
}

impl<F: Format> CloneWithinArena for ASerialized<F> {
    unsafe fn clone_within_arena(&self) -> Self {
        Self {
            ptr: unsafe { self.ptr.clone_within_arena() },
            _marker: PhantomData,
        }
    }
}

impl<F: Format> SelfContainedDefault for ASerialized<F> {}

impl<F: Format> Default for ASerialized<F> {
    fn default() -> Self {
        Self {
            ptr: Ptr::default(),
            _marker: Default::default(),
        }
    }
}

impl<F, T> IntoArena<ASerialized<F>> for T
where
    F: Format,
    T: SerializableAs<Format = F>,
{
    unsafe fn into_arena(self, arena: &Arena) -> ASerialized<F> {
        unsafe { ASerialized::allocate(arena, self) }
    }
}

impl<F: Format> std::fmt::Debug for ASerialized<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.get())
    }
}

impl APackBuffer {
    /// Returns the buffer content as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(&self.data as *const u8, self.len) }
    }

    /// FIXME
    pub fn allocate<'w>(arena: &'w Arena, src: &[u8]) -> Writer<'w, APackBuffer> {
        let buffer_layout =
            Layout::array::<u8>(src.len()).expect("overflow computing APackBuffer buffer size");
        let (layout, _) = Layout::new::<Self>()
            .extend(buffer_layout)
            .expect("overflow computing APackBuffer buffer size");
        let buffer_ptr = arena.alloc_layout(layout.pad_to_align()).as_ptr() as *mut APackBuffer;
        unsafe {
            // Safety: `buffer_ptr` is nonzero and properly aligned and sized for this write;
            // and the memory is newly allocated, so only the current thread is accessing it.
            std::ptr::write(
                buffer_ptr,
                APackBuffer {
                    len: src.len(),
                    data: [],
                },
            );
            ptr::copy_nonoverlapping(
                src.as_ptr(),
                ptr::addr_of_mut!((*buffer_ptr).data) as *mut u8,
                src.len(),
            );

            // Safety: We just allocated this buffer in `arena`. This thread will be the sole
            // writer until `'w` expires because we've borrowed the arena for that long.
            Writer::new(arena, &*buffer_ptr)
        }
    }
}

/// A type that represents a `str` instance.
/// It implements `SerializableAs`, such that we don't have to implement yet another wrapper to a buffer.
pub struct StrFormat {}

impl Format for StrFormat {
    type View<'a> = &'a str;

    fn reborrow<'short, 'long: 'short>(view: Self::View<'long>) -> Self::View<'short> {
        view
    }
}

impl<'a> SerializableAs for &'a str {
    type Format = StrFormat;

    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = vec![];
        self.write(&mut buf).expect("failed to write serializable");
        buf
    }
}
