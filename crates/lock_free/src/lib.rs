//! Arena-based lock-free concurrent data structures with a single writer thread.
//!
//! This is a small framework for data structures with the following characteristics:
//! -   Both a single mutating thread **and** any number of read-only threads can race freely;
//! -   There are no locks and therefore no waiting. Readers never block waiting for a writer or
//!     vice versa.
//!
//! To get the above features, we accept the following limitations:
//! -   Memory is allocated from a bump-allocating arena (to avoid GC). Since this is so integral
//!     to memory safety, the data structure types (`ABox`, `AVec`, `AHashMap`) can't even be
//!     constructed outside of an arena.
//! -   Since we do not free memory until we are done with a whole arena, the growable data
//!     structures use up to 2x as much memory.
//! -   Data is immutable once written. This means the vectors are append-only and the hash tables
//!     don't support deletions.

#![deny(missing_docs)]

pub mod arena;
pub mod hash;
pub mod ptr;
pub mod serializable;
pub mod traits;
pub mod vec;

#[cfg(test)]
mod tests;

pub use arena::Arena;
pub use hash::{AHashKey, AHashMap};
pub use ptr::{ABox, Ptr, ReadHandle, WriteHandle, Writer};
pub use serializable::{ASerialized, Format, SerializableAs, StrFormat};
pub use traits::{CloneWithinArena, SelfContained, SelfContainedDefault};
pub use vec::AVec;
