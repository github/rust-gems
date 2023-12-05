//! Hash keys.

use std::alloc::Layout;
use std::cell::UnsafeCell;
use std::cmp::Eq;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::{mem, ptr};

use super::traits::{IntoArena, SelfContainedDefault};
use super::{ASerialized, Arena, CloneWithinArena, Format, Ptr, Writer};

/// Type that can store AHashMap keys. Must support atomic load and store operations.
pub trait AHashKey: Sized + CloneWithinArena {
    /// Type that is always used for hash computation and comparison.
    /// This is the type passed in to `AHashMap::get()` and friends.
    /// (This isn't necessarily a reference type, but it can be.)
    // TODO: Make a borrowed-TreePath type that's Copy so this bound can be
    // Copy instead of Clone. Cloning keys is basically free, whether they're
    // integers or references.
    type KeyRef<'a>: Clone + Hash + Eq
    where
        Self: 'a;

    /// Returns the empty value (indicating an empty hash table entry).
    /// The entry value must not be used as an actual key.
    fn empty() -> Self;

    /// Returns true if the given query key is empty.
    fn is_empty(key: Self::KeyRef<'_>) -> bool;

    /// Atomically loads the key with `Acquire` ordering. Returns None if the key is empty.
    fn load_acquire(&self) -> Option<Self::KeyRef<'_>>;

    /// Moves a key into the given arena.
    ///
    /// # Safety
    ///
    /// If this type of key lives in arenas, then the caller must ensure the resulting value does
    /// not outlive the arena, usually by storing it in the same arena. (Some key types, like
    /// `u32`, don't care about arenas.)
    unsafe fn key_into_arena(arena: &Arena, key: Self::KeyRef<'_>) -> Self;

    /// Atomically stores a non-empty key value with `Release` ordering.
    ///
    /// # Safety
    ///
    /// If this type of key lives in arenas, then the caller must ensure that `self` and `key` live
    /// in the same arena.
    unsafe fn store_release(&self, key: Self);

    /// KeyRef must be something we can "reborrow" with a shorter lifetime (like a normal
    /// reference). We only need this in one place to make the compiler happy.
    fn reborrow<'short, 'long: 'short>(key: Self::KeyRef<'long>) -> Self::KeyRef<'short>;
}

impl AHashKey for AtomicU32 {
    type KeyRef<'a> = u32;

    fn empty() -> Self {
        AtomicU32::new(0)
    }

    fn is_empty(key: u32) -> bool {
        key == 0
    }

    fn load_acquire(&self) -> Option<u32> {
        match self.load(Ordering::Acquire) {
            0 => None,
            x => Some(x),
        }
    }

    unsafe fn key_into_arena(_arena: &Arena, key: u32) -> AtomicU32 {
        AtomicU32::new(key)
    }

    unsafe fn store_release(&self, val: AtomicU32) {
        self.store(val.into_inner(), Ordering::Release);
    }

    fn reborrow<'short, 'long: 'short>(key: u32) -> u32 {
        key
    }
}

impl AHashKey for AtomicU64 {
    type KeyRef<'a> = u64;

    fn empty() -> Self {
        AtomicU64::new(0)
    }

    fn is_empty(key: u64) -> bool {
        key == 0
    }

    fn load_acquire(&self) -> Option<u64> {
        match self.load(Ordering::Acquire) {
            0 => None,
            x => Some(x),
        }
    }

    unsafe fn key_into_arena(_arena: &Arena, key: u64) -> AtomicU64 {
        AtomicU64::new(key)
    }

    unsafe fn store_release(&self, val: AtomicU64) {
        self.store(val.into_inner(), Ordering::Release);
    }

    fn reborrow<'short, 'long: 'short>(key: u64) -> u64 {
        key
    }
}

impl<F> AHashKey for ASerialized<F>
where
    F: Format + 'static,
    for<'a> F::View<'a>: Hash + Eq,
{
    type KeyRef<'a> = F::View<'a> where Self: 'a;

    fn empty() -> Self {
        ASerialized::null()
    }

    // Views are never null.
    fn is_empty(_key: F::View<'_>) -> bool {
        false
    }

    fn load_acquire(&self) -> Option<F::View<'_>> {
        if self.is_null() {
            None
        } else {
            Some(self.get())
        }
    }

    unsafe fn key_into_arena(arena: &Arena, key: F::View<'_>) -> Self {
        // TODO: make Format::View implement a Serializable trait without a lifetime so this extra work isn't
        // necessary here.
        use github_pspack::Serializable;
        let mut buf = Vec::new();
        key.write(&mut buf)
            .expect("writing to an in-memory buffer can't fail");
        // Safety: The trait passes responsibility on to the caller.
        unsafe { ASerialized::from_vec(arena, buf) }
    }

    unsafe fn store_release(&self, key: Self) {
        // Safety: The trait passes responsibility on to the caller.
        unsafe {
            self.store(key);
        }
    }

    fn reborrow<'short, 'long: 'short>(key: Self::KeyRef<'long>) -> Self::KeyRef<'short> {
        <F as Format>::reborrow(key)
    }
}

/// Trait for types that support atomic assignment to `Self` from type `V`.
pub trait StoreRelaxed<V>: From<V> {
    /// Atomically updates the value in `self`, with `Ordering::Relaxed` semantics.
    fn store_relaxed(&self, value: V);
}

/// Fixed-size hash table used by AHashMap.
struct AHashTable<K, V, S> {
    len: AtomicU32,
    table_len: u32,
    hasher_builder: S,
    _marker: PhantomData<[Entry<K, V>]>, // TODO: use the `data: [Entry<K, V>; 0]` trick here
}

/// A hash map.
pub struct AHashMap<K, V, S = RandomState> {
    ptr: Ptr<AHashTable<K, V, S>>,
}

/// The type of hash table slots. This is like `Option<(K, V)>` because a slot is either empty, or
/// filled with a key and value. But unlike `Option`, it supports an atomic "fill" operation, so
/// that entries can be added to a shared hash table, lock-free.
pub struct Entry<K, V> {
    key: K,
    value: UnsafeCell<MaybeUninit<V>>,
}

impl<K, V, S> SelfContainedDefault for AHashMap<K, V, S> {}

impl<K, V> Entry<K, V>
where
    K: AHashKey,
{
    /// A new empty entry.
    fn empty() -> Self {
        Entry {
            key: K::empty(),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    /// Atomically loads the key from this entry.
    ///
    /// Returns `None` if this slot is empty, and `Some` key otherwise.
    fn load_key(&self) -> Option<K::KeyRef<'_>> {
        self.key.load_acquire()
    }

    /// Gets a reference to the value.
    ///
    /// # Safety
    ///
    /// The slot must not be empty.
    unsafe fn value(&self) -> &V {
        // Safety: OK because the caller guarantees store was called.
        unsafe { (*(self.value.get() as *const MaybeUninit<V>)).assume_init_ref() }
    }

    /// Atomically fills in this entry.
    ///
    /// # Safety
    ///
    /// This slot must be empty. The caller must be the sole writer thread for this entry. `arena`
    /// must be the right arena. `key` must not be a null key.
    unsafe fn fill(&self, key: K, value: V) {
        debug_assert!(self.key.load_acquire().is_none());
        let value_ptr = self.value.get();
        unsafe {
            // Safety: Caller ensures we are the only writer and this slot is empty. Because it's
            // empty, no other thread will call `value()` until after we do `key.store()` below.
            (*value_ptr).write(value);
            // Safety: Caller vouches that `arena` is the right arena.
            self.key.store_release(key);
        }
        debug_assert!(self.key.load_acquire().is_some());
    }
}

impl<K, V, S> AHashTable<K, V, S>
where
    K: AHashKey,
    S: BuildHasher + Default,
{
    /// Allocate a buffer and header.
    fn new(arena: &Arena, table_len: usize) -> &Self {
        assert_eq!(
            table_len as u32 as usize, table_len,
            "table_len must fit in 32 bits"
        );

        let data_layout =
            Layout::array::<Entry<K, V>>(table_len).expect("overflow computing AVec buffer size");
        let layout = Layout::new::<Self>()
            .extend(data_layout) // data array
            .expect("overflow computing AVec buffer size")
            .0
            .pad_to_align();
        let header = arena.alloc_layout(layout).as_ptr() as *mut Self;

        unsafe {
            // Safety: `new_data` is nonzero and properly aligned and sized for these writes;
            // and the memory is newly allocated, so only the current thread is accessing it.
            ptr::write(
                header,
                AHashTable {
                    len: AtomicU32::new(0),
                    table_len: table_len as u32,
                    hasher_builder: S::default(),
                    _marker: PhantomData,
                },
            );
            let table_ptr = (*header).data_mut();
            for i in 0..table_len {
                // Safety: Same as above.
                ptr::write(table_ptr.add(i), Entry::empty());
            }
            // Safety: The memory is allocated from `arena`.
            &*header
        }
    }

    /// Returns the number of key-value pairs in the table.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire) as usize
    }

    /// Returns the number of slots in the table. The actual number of values that should be stored
    /// in the table is some fraction of this.
    pub fn table_len(&self) -> usize {
        self.table_len as usize
    }

    fn data(&self) -> *const Entry<K, V> {
        assert_eq!(mem::size_of::<Self>() % mem::align_of::<Entry<K, V>>(), 0);
        // Safety: All code that allocates `AHashTable`s ensures that a buffer of type `[T;
        // table_len]` immediately follows the struct. The assertions above check that no padding is
        // necessary.
        unsafe { (self as *const Self).add(1) as *const Entry<K, V> }
    }

    fn data_mut(&mut self) -> *mut Entry<K, V> {
        self.data() as *mut Entry<K, V>
    }

    fn as_slice(&self) -> &[Entry<K, V>] {
        // Safety: `data` is non-null and aligned, as required. It's OK if `len` and/or `table_len`
        // is 0, though that will not normally be the case.
        unsafe { std::slice::from_raw_parts(self.data(), self.table_len()) }
    }

    fn is_full(&self) -> bool {
        const LOAD_FACTOR: f64 = 0.5;
        let len = self.len.load(Ordering::Acquire);
        let table_len = self.table_len;
        len + 1 >= table_len || len as f64 >= LOAD_FACTOR * table_len as f64
    }
}

impl<K, V, S> AHashTable<K, V, S>
where
    K: AHashKey,
    V: CloneWithinArena,
    S: BuildHasher + Default,
{
    fn hash(&self, key: K::KeyRef<'_>) -> u64 {
        let mut hasher = self.hasher_builder.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    fn find<'map>(&'map self, key: K::KeyRef<'_>) -> Result<&'map Entry<K, V>, &'map Entry<K, V>> {
        debug_assert!(self.len() < self.table_len as usize);
        debug_assert!(self.table_len.is_power_of_two());

        let slots = self.as_slice();
        let mask = self.table_len as usize - 1;
        let mut i = self.hash(key.clone()) as usize & mask;
        loop {
            let slot = &slots[i];
            match slot.load_key() {
                None => break Err(slot),
                Some(slot_key) if K::reborrow(key.clone()) == slot_key => break Ok(slot),
                _ => {}
            }
            i = i.wrapping_add(1) & mask;
        }
    }

    fn get(&self, key: K::KeyRef<'_>) -> Option<&V> {
        match self.find(key) {
            Ok(slot) => Some(unsafe { slot.value() }),
            Err(_slot) => None,
        }
    }

    /// Double the size of the table if needed to perform an insert.
    ///
    /// # Safety
    ///
    /// Caller must be the sole writer. `arena` must be the arena where `self` lives.
    unsafe fn prepare_for_insert<'arena>(&'arena self, arena: &'arena Arena) -> &'arena Self {
        if self.is_full() {
            // Resize the table.
            let old_slice = self.as_slice();
            let new_table = AHashTable::<K, V, S>::new(arena, self.table_len as usize * 2);
            let mut len = 0u32;
            for slot in old_slice {
                if let Some(k) = slot.load_key() {
                    match new_table.find(k) {
                        Ok(_slot) => {
                            panic!("unexpected collision rehashing supposedly unique keys")
                        }
                        Err(target) => unsafe {
                            // Safety: `.value()` is OK because we just checked that the slot is non-empty.
                            // `clone_within_arena()` is OK because we are immediately writing the value
                            // back into the same arena.
                            //
                            // Separately, we don't know exactly what these trait methods do, so
                            // it's distantly possible they could somehow re-enter the hash table
                            // and cause some other value to be inserted. This would likely end in
                            // stack overflow, so don't do that; but even if not, it will not cause
                            // unsafety -- just odd AHashMap behavior, as our table will clobber
                            // the nested work.
                            let value = slot.value().clone_within_arena();
                            let key = slot.key.clone_within_arena();

                            // Safety: We just checked the slot is empty. Caller ensures we're the
                            // sole writer and that `arena` is correct.
                            target.fill(key, value);
                        },
                    }
                    assert!(len != u32::MAX);
                    len += 1;
                }
            }
            debug_assert!(len + 1 < new_table.table_len);
            new_table.len.store(len, Ordering::Release);
            new_table
        } else {
            self
        }
    }

    /// # Safety
    ///
    /// Caller must be the sole writer. `arena` must be the arena where `self` lives.
    unsafe fn insert<'arena, U>(
        &'arena self,
        arena: &'arena Arena,
        key: K::KeyRef<'_>,
        value: U,
    ) -> &'arena Self
    where
        V: StoreRelaxed<U>,
    {
        let header = unsafe { self.prepare_for_insert(arena) };
        let len = header.len.load(Ordering::Relaxed);
        match header.find(key.clone()) {
            Ok(slot) => unsafe {
                // Safety: We just checked that the slot is not vacant. Caller ensures we're the
                // sole writer.
                (*slot.value.get()).assume_init_ref().store_relaxed(value)
            },
            Err(slot) => {
                assert!(len != u32::MAX);
                debug_assert!(len + 1 < header.table_len);
                // Safety: Caller ensures we're the sole writer.
                unsafe { slot.fill(K::key_into_arena(arena, key), value.into()) };
            }
        }
        header.len.store(len + 1, Ordering::Release);
        header
    }

    /// # Safety
    ///
    /// This is safe if `arena` is the right arena.
    unsafe fn get_or_insert_with<'arena, F>(
        &'arena self,
        arena: &'arena Arena,
        key: K::KeyRef<'_>,
        make_value: F,
    ) -> (&'arena Self, &'arena V)
    where
        F: FnOnce() -> V,
    {
        let (table, slot) = match self.find(key.clone()) {
            Ok(slot) => (self, slot),
            Err(slot) => {
                // Safety: We will store this in the arena (or drop it).
                let new_key = unsafe { K::key_into_arena(arena, key.clone()) };
                let new_value = make_value();
                // Safety: Caller affirms it's the right arena.
                let new_table = unsafe { self.prepare_for_insert(arena) };
                if ptr::eq(new_table, self) && slot.load_key().is_none() {
                    unsafe {
                        // Safety: We just checked that the slot is empty.
                        slot.fill(new_key, new_value);
                    }
                    let len = self.len.load(Ordering::Acquire);
                    self.len.store(len + 1, Ordering::Release);
                    (self, slot)
                } else {
                    // Something changed! We resized; or `K::key_into_arena` or `make_value`
                    // inserted a key. Or both. Have to redo the lookup.
                    match new_table.find(key) {
                        Ok(slot) => (new_table, slot),
                        Err(slot) => {
                            // Safety: find() says the slot is empty.
                            unsafe {
                                slot.fill(new_key, new_value);
                            }
                            let len = new_table.len.load(Ordering::Acquire);
                            new_table.len.store(len + 1, Ordering::Release);
                            (new_table, slot)
                        }
                    }
                }
            }
        };
        // Safety: Above code ensures the slot is not empty.
        let value = unsafe { slot.value() };
        (table, value)
    }
}

impl<'iter, K, V, S> IntoIterator for &'iter AHashTable<K, V, S>
where
    K: AHashKey,
    S: BuildHasher + Default,
{
    type IntoIter = Iter<'iter, K, V>;
    type Item = (&'iter K, &'iter V);
    fn into_iter(self) -> Self::IntoIter {
        Iter {
            table: self.as_slice(),
            index: 0,
        }
    }
}

/// Hash table iterator.
///
/// Note: Iterators are sometimes affected by concurrent writes and sometimes not. Unless you're
/// sure no one is modifying the hash table, don't count on consistent output!
pub struct Iter<'iter, K, V> {
    table: &'iter [Entry<K, V>],
    index: usize,
}

impl<'iter, K, V> Iterator for Iter<'iter, K, V>
where
    K: AHashKey,
{
    type Item = (&'iter K, &'iter V);

    fn next(&mut self) -> Option<Self::Item> {
        for i in self.index..self.table.len() {
            if self.table[i].load_key().is_some() {
                let key = &self.table[i].key;
                // Safety: We just checked the slot is non-empty.
                let value = unsafe { self.table[i].value() };
                self.index = i + 1;
                return Some((key, value));
            }
        }
        self.index = self.table.len();
        None
    }
}

impl<K, V, S> Default for AHashMap<K, V, S> {
    fn default() -> Self {
        AHashMap { ptr: Ptr::null() }
    }
}

impl<K, V, S> CloneWithinArena for AHashMap<K, V, S> {
    unsafe fn clone_within_arena(&self) -> Self {
        // Safety: Caller's responsibility.
        AHashMap {
            ptr: unsafe { self.ptr.clone_within_arena() },
        }
    }
}

impl<K, V, S> AHashMap<K, V, S>
where
    K: AHashKey,
    V: CloneWithinArena,
    S: BuildHasher + Default,
{
    /// Creates a new empty vector.
    pub fn new() -> Self {
        Self::default()
    }

    fn table(&self) -> Option<&AHashTable<K, V, S>> {
        self.ptr.get()
    }

    /// # Safety
    ///
    /// Caller must be the sole writer. `arena` must be the arena where `self` lives.
    unsafe fn force_table<'arena>(
        &'arena self,
        arena: &'arena Arena,
    ) -> &'arena AHashTable<K, V, S> {
        self.table().unwrap_or_else(|| {
            const MIN_TABLE_SIZE: usize = 8;
            let new_table = AHashTable::new(arena, MIN_TABLE_SIZE);
            unsafe {
                self.ptr.store_raw(new_table);
            }
            new_table
        })
    }

    /// Returns the number of slots in the table. The actual number of values that should be stored
    /// in the table is some fraction of this.
    pub fn table_len(&self) -> usize {
        match self.table() {
            None => 0,
            Some(h) => h.table_len as usize,
        }
    }

    /// Returns the number of key-value pairs in the map.
    pub fn len(&self) -> usize {
        match self.table() {
            None => 0,
            Some(h) => h.len(),
        }
    }

    /// Returns `true` if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// `key` may be any borrowed form of the map’s key type `K`, but `Hash` and `Eq` on the
    /// borrowed form must match those for the key type.
    pub fn get<'map>(&'map self, key: K::KeyRef<'_>) -> Option<&'map V> {
        self.table().and_then(|table| table.get(key))
    }

    /// Insert a key-value pair or atomically overwrite the value of an existing entry.
    ///
    /// # Safety
    ///
    /// Caller must be the sole writer. `arena` must be the arena `self` lives in.
    pub unsafe fn insert<U>(&self, arena: &Arena, key: K::KeyRef<'_>, value: U)
    where
        V: StoreRelaxed<U>,
    {
        // Safety: Caller's responsibility.
        let table = unsafe { self.force_table(arena) };
        let new_table = unsafe { table.insert(arena, key, value) };
        if !ptr::eq(table, new_table) {
            // Safety: We just allocated table in the same arena as table.
            unsafe {
                self.ptr.store_raw(new_table);
            }
        }
    }

    /// Get a reference to the value associated with `key`, inserting a new entry if none exists.
    ///
    /// # Safety
    ///
    /// Caller must be the sole writer. `arena` must be the arena `self` lives in.
    unsafe fn get_or_insert_with<'arena, F>(
        &'arena self,
        arena: &'arena Arena,
        key: K::KeyRef<'_>,
        f: F,
    ) -> &'arena V
    where
        F: FnOnce() -> V,
    {
        // Safety: Caller's responsibility.
        let table = unsafe { self.force_table(arena) };
        let (new_table, value) = unsafe { table.get_or_insert_with(arena, key, f) };
        if !ptr::eq(table, new_table) {
            // Safety: We just allocated table in the same arena as table.
            unsafe {
                self.ptr.store_raw(new_table);
            }
        }
        value
    }

    /// Returns an iterator over the key-value pairs in the map.
    /// Keys are returned by value (since they are stored in atomics).
    /// Values are returned by reference.
    pub fn iter(&self) -> Iter<K, V> {
        IntoIterator::into_iter(self)
    }
}

impl<'iter, K, V, S> IntoIterator for &'iter AHashMap<K, V, S>
where
    K: AHashKey,
    V: CloneWithinArena,
    S: BuildHasher + Default,
{
    type IntoIter = Iter<'iter, K, V>;
    type Item = (&'iter K, &'iter V);

    fn into_iter(self) -> Self::IntoIter {
        match self.table() {
            None => Iter {
                table: &[],
                index: 0,
            },
            Some(table) => table.into_iter(),
        }
    }
}

impl<'w, K, V, S> Writer<'w, AHashMap<K, V, S>>
where
    K: AHashKey,
    V: CloneWithinArena,
    S: BuildHasher + Default,
{
    /// Like `std::collections::HashMap::get_mut`, except this returns a writer, not a `mut`
    /// reference.
    pub fn get_writer<'key>(&self, key: K::KeyRef<'key>) -> Option<Writer<'w, V>> {
        assert!(!K::is_empty(key.clone()));
        // Safety: If `r` exists, it's in the same arena as target.
        self.target()
            .get(key)
            .map(|r| unsafe { self.make_child_writer(r) })
    }

    /// Get a writer to the value associated with `key`, inserting a new entry if none exists.
    ///
    /// If there's not already an entry in the table for `key`, this calls `value()` and inserts an
    /// entry associating `key` with the value it returns.
    pub fn get_or_insert_with<'key, F>(&self, key: K::KeyRef<'key>, f: F) -> Writer<'w, V>
    where
        F: FnOnce() -> V,
    {
        assert!(!K::is_empty(key.clone()));
        // Safety: Writer invariants guarantee we're the only writer and have the right arena.
        let r = unsafe { self.target().get_or_insert_with(self.arena(), key, f) };
        // Safety: `r` is in the same arena as target.
        unsafe { self.make_child_writer(r) }
    }

    /// Get a writer to the value associated with `key`, inserting a new entry if none exists.
    ///
    /// If there's not already an entry in the table for `key`, this inserts an entry associating
    /// `key` with `V::default()`.
    pub fn get_or_insert_default<'key>(&self, key: K::KeyRef<'key>) -> Writer<'w, V>
    where
        V: Default,
    {
        assert!(!K::is_empty(key.clone()));
        self.get_or_insert_with(key, || V::default())
    }

    /// Get a writer to the value assocaited with `key`, inserting a new entry if none exists.
    pub fn get_or_insert<'key>(
        &self,
        key: K::KeyRef<'key>,
        value: impl IntoArena<V>,
    ) -> Writer<'w, V> {
        assert!(!K::is_empty(key.clone()));
        self.get_or_insert_with(key, move || unsafe { value.into_arena(self.arena()) })
    }

    /// Insert a key-value pair or atomically overwrite the value of an existing entry.
    pub fn insert<U>(&self, key: K::KeyRef<'_>, value: U)
    where
        V: StoreRelaxed<U>,
    {
        assert!(!K::is_empty(key.clone()));
        // Safety: Writer invariants guarantee we're the only writer and have the right arena.
        unsafe {
            self.target().insert(self.arena(), key, value);
        }
    }
}
