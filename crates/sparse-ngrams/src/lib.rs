//! Sparse n-gram extraction from byte slices.
//!
//! Sparse grams are a way of selecting variable-length n-grams (longer than 2 bytes) without
//! extracting all possible n-grams. The algorithm is deterministic: the same extraction logic
//! works for every substring, so that substring searches are supported.
//!
//! # How it works
//!
//! Each consecutive byte pair (bigram) is assigned a priority based on how frequently it occurs
//! in a large code corpus. A monotone deque tracks potential n-gram boundaries: an n-gram
//! boundary occurs wherever a bigram has lower priority than all bigrams between it and the
//! previous boundary.
//!
//! For a document of N bytes, this produces roughly 2N n-grams: all bigrams plus algorithmically
//! selected longer n-grams (up to [`MAX_SPARSE_GRAM_SIZE`] bytes).
//!
//! # Example
//!
//! ```
//! use sparse_ngrams::{NGram, collect_sparse_grams, MAX_SPARSE_GRAM_SIZE};
//!
//! let input = b"hello world";
//! let grams = collect_sparse_grams(input);
//! assert!(grams.len() > input.len() - 1);
//! for gram in &grams {
//!     assert!(gram.len() >= 2);
//!     assert!(gram.len() <= MAX_SPARSE_GRAM_SIZE as usize);
//! }
//! ```

mod deque;
mod extract;
mod ngram;
mod table;

pub use ngram::NGram;

/// Number of high-frequency bigrams used to build the priority table.
pub const NUM_FREQUENT_BIGRAMS: usize = 65534;

/// Maximum length (in bytes) of a sparse n-gram.
pub const MAX_SPARSE_GRAM_SIZE: u32 = 8;

pub use extract::{collect_sparse_grams, collect_sparse_grams_deque, collect_sparse_grams_scan, max_sparse_grams};

