//! A hash map whose groups are ordered by hash prefix, enabling efficient
//! sorted-order iteration and linear-time merging of two maps.
//!
//! [`HashSortedMap`] is a Swiss-table-inspired, insertion-only hash map. Its
//! groups are laid out by hash prefix, so visiting them in order yields entries
//! sorted by hashed key. This makes merging two maps a single linear scan and
//! lets serialization in hash-key order happen without an extra sort.
//!
//! Hashing is customized through the single-method [`SortingHash`] trait. Any
//! standard [`BuildHasher`](std::hash::BuildHasher) works out of the box via a
//! blanket implementation.
//!
//! ```
//! use hash_sorted_map::HashSortedMap;
//!
//! let mut map = HashSortedMap::new();
//! map.insert("hello", 1);
//! map.insert("world", 2);
//! assert_eq!(map.get("hello"), Some(&1));
//!
//! // Iterate in ascending hash order.
//! map.sort_by_hash();
//! let entries: Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
//! assert_eq!(entries.len(), 2);
//! ```
#![warn(missing_docs)]

mod group;
mod group_ops;
mod hash_sorted_map;
mod iter;

pub use hash_sorted_map::{Entry, HashSortedMap, OccupiedEntry, SortingHash, VacantEntry};
pub use iter::{IntoIter, Iter, IterMut};
