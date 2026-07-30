//! Compact n-gram representation.
//!
//! An [`NGram`] packs a substring's byte length and a payload into the low **27 bits** of a
//! `u32` (the top 5 bits are always zero):
//!
//! ```text
//!  bit 31              27        24            0
//!   +-----------------+-----------+-----------+
//!   | 00000 (unused)  |  len - 2  |  payload  |
//!   +-----------------+-----------+-----------+
//!         5 bits         3 bits      24 bits
//! ```
//!
//! * **Length** (`len - 2`, bits 24..27): substring byte-lengths range from 2 (bigrams) to
//!   [`MAX_SPARSE_GRAM_SIZE`] (8), so biasing by 2 fits the 7 possible values into 3 bits.
//! * **Payload** (bits 0..24): for substrings of at most 3 bytes the bytes are packed
//!   losslessly (right-aligned in the low bits, most-significant byte first); longer substrings are
//!   hashed down to 24 bits with a multiplicative hash.
//!
//! Because the length lives in its own field, an `NGram` of one size never collides with an
//! `NGram` of another size. The packed value is finally run through a bijective [`mix27`]
//! permutation so the most-significant bits (which callers may use for bucketing or sorting) are
//! well distributed even though the packed value is highly structured.

use std::fmt::{self, Write as _};

use crate::MAX_SPARSE_GRAM_SIZE;

/// Odd multiplicative constant (the golden-ratio / Fibonacci hashing constant) used to hash grams
/// longer than 3 bytes down to the 24-bit payload.
const MULTIPLICATIVE_HASH: u64 = 0x9E37_79B9_7F4A_7C15;

/// A compact n-gram identifier. See the module-level documentation for the bit layout.
///
/// Note: we could store n-grams up to length 8 verbatim in a `u64`. However, that would explode
/// the number of distinct keys in a search dictionary. For that reason we compress n-grams into a
/// `u32`, which puts a more reasonable upper bound on the number of dictionary keys.
///
/// Note: by storing the length explicitly, we ensure that only n-grams of the same length can
/// collide. This is important because there are exponentially more long n-grams than short ones.
/// At the same time, longer n-grams occur less frequently, so colliding long n-grams won't
/// increase the false-positive rate too much.
///
/// # Construction
///
/// Use [`NGram::from_bytes`] for one-off hashing, or the rolling 8-byte window helper inside the
/// extraction loop for amortised O(1) computation per n-gram.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(transparent)]
pub struct NGram(pub(crate) u32);

impl NGram {
    /// Smallest indexed gram length (bigrams); subtracted from the stored length so the biased
    /// value fits in [`Self::LEN_BITS`] bits.
    const LEN_BIAS: u32 = 2;
    /// Number of bits used to encode the biased length.
    const LEN_BITS: u32 = 3;
    /// Mask selecting the biased-length field (once shifted down).
    const LEN_MASK: u32 = (1 << Self::LEN_BITS) - 1;
    /// Number of low bits used by the payload; the length sits just above it.
    const PAYLOAD_BITS: u32 = 24;
    /// Mask selecting the payload field.
    const PAYLOAD_MASK: u32 = (1 << Self::PAYLOAD_BITS) - 1;
    /// Total number of significant bits in the packed representation (the top 5 bits of the `u32`
    /// are always zero).
    pub(crate) const BITS: u32 = Self::PAYLOAD_BITS + Self::LEN_BITS;
    /// Mask selecting the significant [`Self::BITS`] bits.
    pub(crate) const MASK: u32 = (1 << Self::BITS) - 1;

    /// Build an `NGram` by hashing the given byte slice from scratch.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `src.len()` is not in `2..=`[`MAX_SPARSE_GRAM_SIZE`].
    pub fn from_bytes(src: &[u8]) -> Self {
        debug_assert!(
            (Self::LEN_BIAS as usize..=MAX_SPARSE_GRAM_SIZE).contains(&src.len()),
            "ngram length {} out of range [{}, {}]",
            src.len(),
            Self::LEN_BIAS,
            MAX_SPARSE_GRAM_SIZE,
        );
        // 24-bit payload: short grams are packed losslessly, longer ones hashed.
        let payload = if src.len() <= 3 {
            // Pack the 2-3 bytes into the low 24 bits, most-significant byte first.
            let mut p = 0u32;
            for &byte in src {
                p = (p << 8) | byte as u32;
            }
            p
        } else {
            // Grams here are 4..=MAX_SPARSE_GRAM_SIZE (8) bytes, so they fit in a single u64. A
            // multiplicative hash is much cheaper than a per-byte loop, and the top bits of the
            // product mix in every input byte. `from_le_bytes` keeps the result independent of
            // host endianness, and the gram length lives in its own field so the payload hash
            // needn't encode it.
            let mut buf = [0u8; 8];
            buf[..src.len()].copy_from_slice(src);
            let product = u64::from_le_bytes(buf).wrapping_mul(MULTIPLICATIVE_HASH);
            (product >> (u64::BITS - Self::PAYLOAD_BITS)) as u32
        };
        Self::pack(src.len(), payload)
    }

    /// Builds an `NGram` from a big-endian packing of its bytes: the `len` gram bytes occupy the
    /// most-significant bytes of `value` (the first gram byte in the top byte) and the low
    /// `8 - len` bytes are zero. This is the form the extraction loop's rolling 8-byte window
    /// produces, so it can construct grams without re-reading them from a slice. It returns exactly
    /// the same value as [`from_bytes`](Self::from_bytes) would for the same gram (see the
    /// `from_window_matches_from_bytes` test), so the two paths stay interchangeable.
    #[inline]
    pub(crate) fn from_window(value: u64, len: usize) -> Self {
        debug_assert!(
            (Self::LEN_BIAS as usize..=MAX_SPARSE_GRAM_SIZE).contains(&len),
            "ngram length {len} out of range [{}, {}]",
            Self::LEN_BIAS,
            MAX_SPARSE_GRAM_SIZE,
        );
        // 24-bit payload: short grams are packed losslessly, longer ones hashed.
        let payload = if len <= 3 {
            // The bytes sit in the top `len` bytes, most-significant byte first; shifting them down
            // to the low bits reproduces `from_bytes`'s lossless packing.
            (value >> (u64::BITS - len as u32 * 8)) as u32
        } else {
            // `swap_bytes` turns the big-endian window into the little-endian byte order that
            // `from_bytes` feeds to the multiplicative hash, so both paths agree bit-for-bit.
            let product = value.swap_bytes().wrapping_mul(MULTIPLICATIVE_HASH);
            (product >> (u64::BITS - Self::PAYLOAD_BITS)) as u32
        };
        Self::pack(len, payload)
    }

    /// Packs a length and payload into the structured value, then stores it *mixed* (via [`mix27`])
    /// so the hot sorting/bucketing paths read a well-distributed value directly from the field;
    /// [`len`](Self::len) and [`Debug`] unmix on demand.
    #[inline]
    fn pack(len: usize, payload: u32) -> Self {
        let packed = ((len as u32 - Self::LEN_BIAS) << Self::PAYLOAD_BITS) | payload;
        Self(mix27(packed))
    }

    /// The byte length of the n-gram.
    #[inline]
    pub fn len(&self) -> usize {
        // The length lives in the *packed* value; unmix the stored value first.
        let packed = unmix27(self.0);
        (((packed >> Self::PAYLOAD_BITS) & Self::LEN_MASK) + Self::LEN_BIAS) as usize
    }

    /// Whether this represents an empty gram. Valid n-grams are always at least 2 bytes long, so
    /// this only holds for a default-constructed placeholder.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The raw packed `u32`. This is an opaque, well-distributed identifier suitable as a hash-map
    /// or hash-set key.
    #[inline]
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

/// A bijective 27-bit mixing permutation. The packed gram value is highly structured — the length
/// sits in the top 3 bits and short grams carry raw ASCII bytes — which would make the
/// most-significant bits badly skewed. This xorshift-multiply finalizer, restricted to 27 bits,
/// spreads entropy across all bits while remaining a bijection on `[0, 2^27)` (each step is
/// invertible: xorshifts are triangular GF(2) maps, and multiplication by an odd constant is a
/// unit modulo `2^27`), so distinct grams stay distinct.
fn mix27(mut x: u32) -> u32 {
    debug_assert!(x <= NGram::MASK, "mix27 input must be a 27-bit value");
    x ^= x >> 15;
    x = x.wrapping_mul(0x2c1b_3c6d) & NGram::MASK;
    x ^= x >> 12;
    x = x.wrapping_mul(0x297a_2d39) & NGram::MASK;
    x ^= x >> 15;
    x
}

/// Inverse of [`mix27`]: recovers the packed (length + payload) value from the stored value. Each
/// step undoes the corresponding `mix27` step in reverse: the `>> 15` xorshifts are self-inverse
/// (since `2 * 15 >= 27`), the `>> 12` xorshift is undone by the doubling `>> 12` then `>> 24`, and
/// the multiplies by the modular inverses of their constants (mod `2^27`).
fn unmix27(mut x: u32) -> u32 {
    debug_assert!(x <= NGram::MASK, "unmix27 input must be a 27-bit value");
    x ^= x >> 15;
    x = x.wrapping_mul(0x4f0_b109) & NGram::MASK; // inverse of 0x297a_2d39 mod 2^27
    x ^= x >> 12;
    x ^= x >> 24;
    x = x.wrapping_mul(0x4ea_2d65) & NGram::MASK; // inverse of 0x2c1b_3c6d mod 2^27
    x ^= x >> 15;
    x
}

/// The encoded `u32` representation is not human readable. This formatter improves the situation
/// at least for short ascii grams.
impl fmt::Debug for NGram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let packed = unmix27(self.0);
        let len = (((packed >> Self::PAYLOAD_BITS) & Self::LEN_MASK) + Self::LEN_BIAS) as usize;
        let mut s = String::new();
        if len <= 3 {
            // The `len` payload bytes sit right-aligned in the low bits of the packed value,
            // most-significant byte first (see `from_bytes`).
            let payload = packed & Self::PAYLOAD_MASK;
            let bytes = [(payload >> 16) as u8, (payload >> 8) as u8, payload as u8];
            for &byte in &bytes[3 - len..3] {
                if byte.is_ascii_graphic() || byte == b' ' {
                    s.push(byte as char);
                } else {
                    write!(s, "\\x{byte:02x}")?;
                }
            }
        } else {
            write!(s, "{:#08x}", packed & Self::PAYLOAD_MASK)?;
        }
        write!(f, "NGram('{s}', len={len})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix27_is_invertible() {
        // `mix27` must be a bijection on the 27-bit space so distinct grams never collide; sample
        // the space plus boundaries and check the round-trip.
        for x in (0..=NGram::MASK).step_by(97) {
            assert_eq!(unmix27(mix27(x)), x, "round-trip failed for {x:#x}");
        }
        for x in [0, 1, 2, NGram::MASK - 1, NGram::MASK] {
            assert_eq!(unmix27(mix27(x)), x, "round-trip failed for {x:#x}");
        }
    }

    #[test]
    fn test_from_bytes_roundtrip() {
        for len in 2..=MAX_SPARSE_GRAM_SIZE {
            let bytes = vec![b'a'; len];
            assert_eq!(
                NGram::from_bytes(&bytes).len(),
                len,
                "len mismatch for {len}"
            );
        }
    }

    #[test]
    fn test_equal_content_equal_ngram() {
        assert_eq!(NGram::from_bytes(b"abc"), NGram::from_bytes(b"abc"));
        assert_eq!(NGram::from_bytes(b"abcdef"), NGram::from_bytes(b"abcdef"));
    }

    #[test]
    fn test_short_grams_are_lossless() {
        // Distinct grams of length <= 3 are packed losslessly, so they must never collide.
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for a in 0u8..64 {
            for b in 0u8..64 {
                assert!(seen.insert(NGram::from_bytes(&[a, b])), "bigram collision");
                for c in 0u8..8 {
                    assert!(
                        seen.insert(NGram::from_bytes(&[a, b, c])),
                        "trigram collision"
                    );
                }
            }
        }
    }

    #[test]
    fn test_same_content_different_length() {
        // Even if payloads were to collide, different lengths produce different NGrams.
        let a = NGram::from_bytes(b"ab");
        let b = NGram::from_bytes(b"abc");
        assert_ne!(a, b);
        assert_ne!(a.len(), b.len());
    }

    #[test]
    fn from_window_matches_from_bytes() {
        // The extraction loop builds grams from a big-endian rolling window via `from_window`; it
        // must produce the identical value to `from_bytes`. Check every indexable length with
        // distinct bytes.
        for len in (NGram::LEN_BIAS as usize)..=MAX_SPARSE_GRAM_SIZE {
            let bytes: Vec<u8> = (0..len as u8).map(|i| b'a' + i).collect();
            let mut buf = [0u8; 8];
            buf[..len].copy_from_slice(&bytes);
            // Gram bytes left-aligned in the most-significant bytes, low bytes zero.
            let window = u64::from_be_bytes(buf);
            assert_eq!(
                NGram::from_window(window, len),
                NGram::from_bytes(&bytes),
                "mismatch for {len}-byte gram",
            );
        }
    }

    #[test]
    fn test_default_is_not_empty() {
        // Valid n-grams are always at least 2 bytes, so nothing (not even the default placeholder)
        // is ever "empty".
        assert!(!NGram::default().is_empty());
        assert_eq!(NGram::default().len(), 2);
    }
}
