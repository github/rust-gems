//! Sparse n-gram extraction from byte slices.
//!
//! Sparse grams are a way of selecting variable-length n-grams (longer than 2 bytes) without
//! extracting all possible n-grams. The algorithm is deterministic: the same extraction logic
//! works for every substring, so that substring searches are supported.
//!
//! # How it works
//!
//! Each consecutive byte pair (bigram) is assigned a priority based on how frequently it occurs
//! in a large code corpus (see [`bigram_priority`]). A monotone deque tracks potential n-gram
//! boundaries: an n-gram boundary occurs wherever a bigram has lower priority than the bigrams
//! between it and the previous boundary.
//!
//! A substring of length 3..=[`MAX_SPARSE_GRAM_SIZE`] is emitted as a sparse n-gram when both its
//! left and right boundary bigrams have a priority strictly below every interior bigram. All
//! bigrams are always emitted.
//!
//! For a document of N bytes, this produces at most 3(N-1) n-grams: all bigrams plus algorithmically
//! selected longer n-grams (up to [`MAX_SPARSE_GRAM_SIZE`] bytes).
//!
//! # Normalization
//!
//! The bigram priority model only scores ASCII byte pairs; any byte with the high bit set resolves
//! to priority `0`. Callers building a case-insensitive index should normalize input first (fold
//! uppercase to lowercase, map multi-byte UTF-8 to high-bit-set bytes) before extraction.
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
//!     assert!(gram.len() <= MAX_SPARSE_GRAM_SIZE);
//! }
//! ```

mod deque;
mod extract;
mod ngram;
mod table;

pub use ngram::NGram;
pub use table::bigram_priority;

/// Maximum length (in bytes) of a sparse n-gram.
pub const MAX_SPARSE_GRAM_SIZE: usize = 8;

pub use extract::{
    collect_sparse_grams, collect_sparse_grams_deque, collect_sparse_grams_scan, max_sparse_grams,
};
