//! Compact n-gram representation using a polynomial rolling hash.
//!
//! An [`NGram`] packs both a hash and the byte length into a single `u32`:
//! the upper 24 bits hold the rolling hash and the lower 8 bits hold the length.
//! This makes it suitable as a cheap, fixed-size key for hash maps and sets.

use std::fmt;

use crate::MAX_SPARSE_GRAM_SIZE;

/// Prime for the polynomial rolling hash.
pub(crate) const POLY_HASH_PRIME: u32 = 2_654_435_761;

/// Precomputed powers of [`POLY_HASH_PRIME`] for rolling-hash range queries.
/// `POLY_POWERS[i] = POLY_HASH_PRIME.pow(i)` (wrapping `u32`).
pub(crate) const POLY_POWERS: [u32; MAX_SPARSE_GRAM_SIZE as usize + 1] = {
    let mut p = [0u32; MAX_SPARSE_GRAM_SIZE as usize + 1];
    p[0] = 1;
    let mut i = 1;
    while i < p.len() {
        p[i] = (p[i - 1] as u64 * POLY_HASH_PRIME as u64) as u32;
        i += 1;
    }
    p
};

/// A compact n-gram identifier: upper 24 bits are a polynomial rolling hash,
/// lower 8 bits are the byte length of the n-gram.
///
/// Two `NGram` values are equal iff both their hash and length match, which
/// greatly reduces collision probability compared to a bare hash.
///
/// # Construction
///
/// Use [`NGram::from_bytes`] for one-off hashing, or the rolling-hash helpers
/// inside the extraction loop for amortised O(1) computation per n-gram.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NGram(pub(crate) u32);

impl NGram {
    /// Build an `NGram` by hashing the given byte slice from scratch.
    pub fn from_bytes(src: &[u8]) -> Self {
        let mut hash = 0u32;
        for &byte in src {
            hash = hash.wrapping_mul(POLY_HASH_PRIME).wrapping_add(byte as u32);
        }
        Self((hash << 8) | src.len() as u32)
    }

    /// Build an `NGram` from a precomputed rolling hash and a length.
    #[inline]
    pub(crate) fn from_rolling_hash(hash: u32, len: usize) -> Self {
        Self((hash << 8) | len as u32)
    }

    /// The byte length of the n-gram (stored in the lower 8 bits).
    #[inline]
    pub fn len(&self) -> usize {
        (self.0 & 0xff) as usize
    }

    /// Whether this represents an empty gram (should never happen in practice).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The raw packed `u32` (hash ≪ 8 | len).
    #[inline]
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

impl fmt::Debug for NGram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NGram({:#x}, len={})", self.0 >> 8, self.len())
    }
}

/// Compute the rolling hash for `content[start..start+len]` from prefix hashes.
#[inline]
pub(crate) fn ngram_from_range(
    prefix_hashes: &[u32; MAX_SPARSE_GRAM_SIZE as usize],
    end_hash: u32,
    start: usize,
    len: usize,
) -> NGram {
    let hash = end_hash.wrapping_sub(
        prefix_hashes[start & (MAX_SPARSE_GRAM_SIZE as usize - 1)]
            .wrapping_mul(POLY_POWERS[len]),
    );
    NGram::from_rolling_hash(hash, len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_bytes_roundtrip() {
        let ngram = NGram::from_bytes(b"hello");
        assert_eq!(ngram.len(), 5);
    }

    #[test]
    fn test_equal_content_equal_ngram() {
        assert_eq!(NGram::from_bytes(b"abc"), NGram::from_bytes(b"abc"));
    }

    #[test]
    fn test_different_content_likely_different() {
        assert_ne!(NGram::from_bytes(b"abc"), NGram::from_bytes(b"abd"));
    }

    #[test]
    fn test_same_hash_different_length() {
        // Even if hashes collide, different lengths produce different NGrams.
        let a = NGram::from_bytes(b"ab");
        let b = NGram::from_bytes(b"abc");
        assert_ne!(a, b);
    }

    #[test]
    fn test_rolling_hash_matches_from_bytes() {
        let content = b"hello world";
        // Build prefix hashes the same way the extraction loop does.
        let mut prefix_hashes = [0u32; MAX_SPARSE_GRAM_SIZE as usize];
        if !content.is_empty() {
            prefix_hashes[1] = content[0] as u32;
        }
        for idx in 1..content.len() {
            let end_hash = prefix_hashes[idx & (MAX_SPARSE_GRAM_SIZE as usize - 1)]
                .wrapping_mul(POLY_HASH_PRIME)
                .wrapping_add(content[idx] as u32);
            // Check the bigram content[idx-1..idx+1]
            let rolling = ngram_from_range(&prefix_hashes, end_hash, idx - 1, 2);
            let direct = NGram::from_bytes(&content[idx - 1..idx + 1]);
            assert_eq!(rolling, direct, "mismatch at idx={idx}");
            prefix_hashes[(idx + 1) & (MAX_SPARSE_GRAM_SIZE as usize - 1)] = end_hash;
        }
    }
}
