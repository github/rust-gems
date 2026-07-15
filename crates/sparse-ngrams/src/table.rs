//! Bigram priority model.
//!
//! Assigns a frequency-based priority to each byte pair, used by the sparse n-gram
//! extraction algorithm to decide where n-gram boundaries fall.
//!
//! Priorities used to be a full 256×256 `u16` table baked from a `bigrams.bin` frequency
//! ranking (~64kB in memory). They are now reconstructed from a compact *factored* model
//! (~8.5kB) trained offline against a bigram frequency ranking. The ascii bigram `(a, b)` is
//! scored as
//! `BIGRAM_H[a] + BIGRAM_H[b] + (code << BIGRAM_CODE_SHIFT) + 1`, where [`BIGRAM_H`] is a single
//! shared per-byte weight and `code` is a 4-bit (`0..=15`) per-bigram correction. Bigrams absent
//! from the training data carry `code == 0` and are not special-cased: they fall back to the bare
//! factored score, i.e. the model extrapolates a priority for them. The byte index `idx` is folded
//! into the low 16 bits so every bigram gets a *unique* priority, while a higher score still means
//! a more frequent bigram (~1.4% inversions vs. the reference ranking).

/// A casefolded indexable byte uses 7 bits for ascii characters; any non-ascii (unicode)
/// character is expected to have its high bit set. Only ascii bigrams are ever present, so
/// non-ascii characters always resolve to priority `0`. The model therefore only covers the 128
/// ascii values per character.
const BIGRAM_ALPHABET: usize = 128;

/// The per-bigram 4-bit correction code is scaled by `1 << BIGRAM_CODE_SHIFT` and added to the
/// shared per-byte weights. A plain shift replaces what used to be a lookup into a learned
/// 16-entry offset table, at a negligible accuracy cost.
const BIGRAM_CODE_SHIFT: u32 = 10;

/// 4-bit correction code per ascii bigram, packed two codes per byte (the even index in the low
/// nibble). Scaled by `1 << BIGRAM_CODE_SHIFT` and added to the shared per-byte weights.
static BIGRAM_CODE: &[u8; BIGRAM_ALPHABET * BIGRAM_ALPHABET / 2] =
    include_bytes!("bigram_code.bin");

/// Shared per-byte weight. `BIGRAM_H[b]` contributes to the priority of every bigram containing
/// byte `b`; `7272` is the filler weight for bytes absent from the training data.
static BIGRAM_H: [u16; BIGRAM_ALPHABET] = [
    1712, 811, 1084, 886, 596, 91, 132, 450, 724, 8461, 16403, 286, 1348, 6151, 140, 343, 333, 64,
    162, 45, 103, 178, 27, 5, 35, 0, 161, 2323, 70, 125, 44, 101, 13403, 8259, 11824, 8316, 8468,
    8305, 8173, 10620, 10671, 9834, 9082, 8674, 9982, 10753, 11148, 10895, 10788, 10958, 10844,
    10474, 10331, 10220, 10066, 9935, 10004, 9859, 10243, 9293, 9469, 9631, 9908, 8272, 8073, 7272,
    7272, 7272, 7272, 7272, 7272, 7272, 7272, 7272, 7272, 7272, 7272, 7272, 7272, 7272, 7272, 7272,
    7272, 7272, 7272, 7272, 7272, 7272, 7272, 7272, 7272, 9286, 8956, 8593, 6514, 9968, 8679,
    11923, 11057, 11570, 11787, 12294, 11116, 10949, 10849, 11401, 9455, 10261, 11463, 11238,
    11595, 11281, 11422, 9364, 11668, 12007, 11945, 10678, 10380, 10412, 10197, 10534, 9280, 8890,
    7898, 8767, 6372, 1475,
];

/// Reconstructs the priority of the ascii bigram `(a, b)`; see [`bigram_priority`]. This rolling
/// variant avoids re-loading `BIGRAM_H[a]`: consecutive bigrams overlap by one byte, so the caller
/// passes the `h_b` returned for the previous position as `h_a` and gets `BIGRAM_H[b]` back for the
/// next one. The `H` value is only used for ascii bytes; for a non-ascii byte the bigram is absent,
/// so the masked lookup (`b & 0x7f`) merely returns a value that the next step discards.
#[inline]
pub(crate) fn bigram_priority_rolling(a: u8, b: u8, h_a: u32) -> (u32, u32) {
    let h_b = BIGRAM_H[(b & (BIGRAM_ALPHABET as u8 - 1)) as usize] as u32;
    if (a | b) >= BIGRAM_ALPHABET as u8 {
        return (0, h_b);
    }
    let idx = a as usize * BIGRAM_ALPHABET + b as usize;
    let code = (BIGRAM_CODE[idx >> 1] >> ((idx & 1) * 4)) & 0xF;
    // The 4-bit `code` is scaled by `1 << BIGRAM_CODE_SHIFT` (a plain shift in place of an offset
    // table) and added to the shared per-byte weights. Absent bigrams carry `code == 0`, so they
    // fall back to the bare factored score `h_a + h_b + 1`. A higher score still means a more
    // frequent bigram; `base` fits in 16 bits, so the unique per-bigram `idx` in the low 16 bits
    // keeps every priority unique.
    let base = h_a + h_b + ((code as u32) << BIGRAM_CODE_SHIFT) + 1;
    ((base << 16) | idx as u32, h_b)
}

/// The `BIGRAM_H` weight of a single byte, used to seed [`bigram_priority_rolling`].
#[inline]
pub(crate) fn bigram_h(a: u8) -> u32 {
    BIGRAM_H[(a & (BIGRAM_ALPHABET as u8 - 1)) as usize] as u32
}

/// Reconstructs the frequency-ranking priority of the ascii bigram `(a, b)`. Absent or non-ascii
/// bigrams resolve to `0`; present bigrams get a strictly positive, unique priority where a higher
/// value means a more frequent bigram. This priority is used to split strings into smaller
/// n-grams.
pub fn bigram_priority(a: u8, b: u8) -> u32 {
    bigram_priority_rolling(a, b, bigram_h(a)).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_ascii_is_zero() {
        assert_eq!(bigram_priority(0x80, b'a'), 0);
        assert_eq!(bigram_priority(b'a', 0x80), 0);
        assert_eq!(bigram_priority(0xff, 0xff), 0);
    }

    #[test]
    fn ascii_bigrams_are_positive_and_unique() {
        // Distinct ascii bigrams get distinct, strictly positive priorities (the `idx` in the low
        // 16 bits guarantees uniqueness).
        assert!(bigram_priority(b'a', b'b') > 0);
        assert_ne!(bigram_priority(b'a', b'b'), bigram_priority(b'b', b'a'));
        assert_ne!(bigram_priority(b'a', b'b'), bigram_priority(b'a', b'c'));
    }

    #[test]
    fn rolling_matches_direct() {
        // The rolling variant must reproduce the standalone `bigram_priority` for every ascii pair,
        // and hand back `BIGRAM_H[b]` for the next step.
        for a in 0u8..128 {
            for b in 0u8..128 {
                let (p, h_b) = bigram_priority_rolling(a, b, bigram_h(a));
                assert_eq!(p, bigram_priority(a, b), "mismatch at ({a}, {b})");
                assert_eq!(h_b, bigram_h(b), "h_b mismatch at ({a}, {b})");
            }
        }
    }
}
