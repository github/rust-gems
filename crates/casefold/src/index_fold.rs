//! Compact one-byte-per-character *index* fold, built on the same paged-bitmap
//! run table as [`simple_fold`](crate::simple_fold).

use crate::table::*;
use crate::{popcount_up_to, scan_end_low};

/// Consumes `s` and returns its simple case-folded form as a compact byte
/// *index*: each character is folded with the same simple (1-to-1) fold as
/// [`simple_fold`](crate::simple_fold), then collapsed to **exactly one byte**
/// per input character.
///
/// Single-byte (ASCII) characters are emitted as their plain lowercased byte
/// (high bit clear). Every multibyte character is replaced by the single byte
/// `0x80 | (cp & 0x7F)`: the low 7 bits of its *folded* code point with the high
/// bit set. The high bit is set unconditionally, so a multibyte character that
/// folds to ASCII (e.g. U+212A KELVIN SIGN → `k`) still yields a high-bit byte
/// (`0x80 | b'k'`), not the bare ASCII byte.
///
/// The result has one byte per character and is therefore **not** valid UTF-8.
/// It is intended as a cheap, fixed-width key for case-insensitive indexing or
/// hashing where collisions between code points sharing the same low 7 bits are
/// acceptable.
///
/// Because every character collapses to exactly one byte, the output is never
/// longer than the input; pure-ASCII input is folded in place (the input's heap
/// buffer is returned untouched), and once a multibyte character is hit the
/// remainder is rewritten in place with a write cursor that never overtakes the
/// read cursor, so no second buffer is ever allocated.
///
/// Like [`simple_fold`](crate::simple_fold), characters are never fully decoded
/// and the fold needs no UTF-8 reconstruction: the page coordinates come from
/// the lead/continuation bytes, and on a fold hit the folded low 7 bits are
/// `(cp & 0x7F)` plus the run's 7-bit `INDEX_DELTA`, masked back to 7 bits.
///
/// # Example
///
/// ```
/// use casefold::index_fold;
/// assert_eq!(index_fold("Hi!".to_string()), b"hi!");
/// // U+212A KELVIN SIGN folds to ASCII 'k', but the high bit is still set:
/// assert_eq!(index_fold("\u{212A}".to_string()), &[0x80 | b'k']);
/// // 'Ü' (U+00DC) folds to 'ü' (U+00FC); 0x80 | (0xFC & 0x7F) == 0xFC:
/// assert_eq!(index_fold("Ü".to_string()), &[0xFC]);
/// ```
pub fn index_fold(s: String) -> Vec<u8> {
    let mut bytes = s.into_bytes();
    // Tier 1 — vectorizable straight-through pass (identical to `fold_into_bytes`):
    // lowercase every ASCII A..Z byte in place and OR all bytes together so a
    // single sign-bit test tells us whether any multibyte sequence is present.
    let mut high_bit_acc: u8 = 0;
    for b in &mut bytes {
        high_bit_acc |= *b;
        let is_upper = b.wrapping_sub(b'A') < 26;
        *b |= u8::from(is_upper) << 5;
    }
    if high_bit_acc & 0x80 == 0 {
        // Pure ASCII: already folded in place, one byte per character.
        return bytes;
    }
    // Tier 2 — collapse each character to one index byte, in place. The ASCII
    // prefix (already lowercased above, one byte per char) is left untouched;
    // from the first non-ASCII byte we rewrite with a `write` cursor that, since
    // every character yields exactly one byte from its >= 1 source bytes, never
    // overtakes `read`.
    let first_non_ascii = bytes
        .iter()
        .position(|&b| b & 0x80 != 0)
        .expect("a non-ASCII byte exists (the high-bit accumulator was set)");
    let mut write = first_non_ascii;
    let mut read = first_non_ascii;
    while read < bytes.len() {
        let lead = bytes[read];
        // ASCII (already lowercased by tier 1): copy through as a single byte.
        if lead & 0x80 == 0 {
            bytes[write] = lead;
            write += 1;
            read += 1;
            continue;
        }
        // Multibyte: recover the `PAGE_BITMAP` coordinates of `cp >> 6`
        // directly as `(word_idx, bit_idx)` — the high part `cp >> 12` indexes
        // the bitmap word, the next 6 bits `(cp >> 6) & 63` index the bit —
        // without ever materializing the combined page number.
        let (word_idx, bit_idx, c_len) = if lead < 0xE0 {
            (0usize, (lead & 0x1F) as u32, 2usize)
        } else if lead < 0xF0 {
            ((lead & 0x0F) as usize, (bytes[read + 1] & 0x3F) as u32, 3)
        } else {
            (
                (((lead & 0x07) as usize) << 6) | (bytes[read + 1] & 0x3F) as usize,
                (bytes[read + 2] & 0x3F) as u32,
                4,
            )
        };
        let low_v = bytes[read + c_len - 1] & 0x3F;
        // The source code point's low 7 bits, `cp & 0x7F`, as `((cp >> 6) & 1)
        // << 6 | (cp & 0x3F)`: `bit_idx`'s low bit is `(cp >> 6) & 1`. We don't
        // mask `bit_idx` to one bit — its higher bits land in output bit 7+,
        // which the unconditional `0x80 |` at write time overwrites anyway.
        let mut folded_index = ((bit_idx << 6) as u8) | low_v;
        if word_idx < PAGE_BITMAP.len() && (PAGE_BITMAP[word_idx] >> bit_idx) & 1 != 0 {
            let dense = popcount_up_to(word_idx, bit_idx) as usize;
            let lo = PAGE_OFFSET[dense] as usize;
            let n = PAGE_OFFSET[dense + 1] as usize - lo;
            let off = scan_end_low(lo, n, low_v);
            if off < n {
                let ss = RUN_START_STRIDE[lo + off];
                let start_low = ss & 0x3F;
                let stride_bit = ss >> 6;
                if low_v >= start_low && ((low_v - start_low) & stride_bit) == 0 {
                    // Folding character: by modular arithmetic the folded low 7
                    // bits are `(cp & 0x7F) + (delta & 0x7F)) mod 128`, so adding
                    // the run's 7-bit `INDEX_DELTA` yields them directly — no UTF-8
                    // reconstruction. The add may carry into bit 7, but that bit
                    // is overwritten by `0x80 |` below, so no `& 0x7F` is needed.
                    folded_index = folded_index.wrapping_add(INDEX_DELTA[lo + off]);
                }
            }
        }
        // `write <= read` here, and the source bytes this character needs were
        // all read above, so storing the single index byte never clobbers
        // bytes still to be read. The high bit always marks a multibyte origin.
        bytes[write] = 0x80 | folded_index;
        write += 1;
        read += c_len;
    }
    bytes.truncate(write);
    bytes
}

/// Folds a single `char` to its one-byte [`index_fold`] representation.
///
/// Equivalent to the per-character output of [`index_fold`]: an ASCII `char`
/// yields its lowercased byte (high bit clear); any other `char` yields
/// `0x80 | (cp & 0x7F)` of its *folded* code point (high bit set), including a
/// multibyte `char` that folds to ASCII (e.g. U+212A KELVIN SIGN → `0x80 | b'k'`).
///
/// # Example
///
/// ```
/// use casefold::index_fold_char;
/// assert_eq!(index_fold_char('A'), b'a');
/// assert_eq!(index_fold_char('Ü'), 0xFC); // ü → 0x80 | (0xFC & 0x7F)
/// assert_eq!(index_fold_char('中'), 0x80 | 0x2D);
/// ```
pub fn index_fold_char(c: char) -> u8 {
    let cp = c as u32;
    if cp < 0x80 {
        // ASCII: lowercase A..Z (a no-op otherwise), high bit stays clear.
        let b = cp as u8;
        let is_upper = b.wrapping_sub(b'A') < 26;
        return b | (u8::from(is_upper) << 5);
    }
    // Multibyte: the `PAGE_BITMAP` coordinates of `cp >> 6` are `word_idx =
    // cp >> 12` (the bitmap word) and `bit_idx = (cp >> 6) & 63` (the bit).
    let word_idx = (cp >> 12) as usize;
    let bit_idx = (cp >> 6) & 0x3F;
    let low_v = (cp & 0x3F) as u8;
    // `cp & 0x7F` is the source low 7 bits; a fold adds the run's 7-bit delta.
    let mut folded_index = (cp & 0x7F) as u8;
    if word_idx < PAGE_BITMAP.len() && (PAGE_BITMAP[word_idx] >> bit_idx) & 1 != 0 {
        let dense = popcount_up_to(word_idx, bit_idx) as usize;
        let lo = PAGE_OFFSET[dense] as usize;
        let n = PAGE_OFFSET[dense + 1] as usize - lo;
        let off = scan_end_low(lo, n, low_v);
        if off < n {
            let ss = RUN_START_STRIDE[lo + off];
            let start_low = ss & 0x3F;
            let stride_bit = ss >> 6;
            if low_v >= start_low && ((low_v - start_low) & stride_bit) == 0 {
                folded_index = folded_index.wrapping_add(INDEX_DELTA[lo + off]);
            }
        }
    }
    0x80 | folded_index
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::reference;
    use std::collections::HashMap;

    /// Per-character index fold via the reference map: fold each char, then
    /// collapse it to one byte the same way [`index_fold`] does. The high bit is
    /// set for every multibyte (source `cp >= 0x80`) character, even one that
    /// folds to ASCII.
    fn index_fold_oracle(r: &HashMap<u32, u32>, s: &str) -> Vec<u8> {
        let mut out = Vec::new();
        for c in s.chars() {
            let cp = c as u32;
            let folded = r.get(&cp).copied().unwrap_or(cp);
            if cp < 0x80 {
                out.push(folded as u8);
            } else {
                out.push(0x80 | (folded & 0x7F) as u8);
            }
        }
        out
    }

    #[test]
    fn index_fold_ascii() {
        assert_eq!(index_fold(String::new()), b"");
        assert_eq!(index_fold("Hello, WORLD!".into()), b"hello, world!");
        assert_eq!(index_fold("abc 123 XYZ".into()), b"abc 123 xyz");
    }

    #[test]
    fn index_fold_reuses_buffer_for_ascii_input() {
        // Pure-ASCII input is folded in place; the returned Vec must hold the
        // exact same allocation as the input String.
        let s = "MIXED case AsCiI 12345".to_string();
        let original_ptr = s.as_ptr();
        let out = index_fold(s);
        assert_eq!(out, b"mixed case ascii 12345");
        assert_eq!(out.as_ptr(), original_ptr);
    }

    #[test]
    fn index_fold_multibyte_to_single_byte() {
        // Ü (U+00DC) folds to ü (U+00FC); 0x80 | (0xFC & 0x7F) == 0xFC.
        assert_eq!(index_fold("Ü".into()), vec![0xFC]);
        // Length-preserving fold of three 2-byte chars to one byte each.
        assert_eq!(index_fold("ÄÖÜ".into()), vec![0x80 | 0x64, 0x80 | 0x76, 0xFC]);
        // Fold to ASCII keeps the high bit set: U+212A KELVIN SIGN -> 'k'.
        assert_eq!(
            index_fold("\u{212A}elvin".into()),
            vec![0x80 | b'k', b'e', b'l', b'v', b'i', b'n'],
        );
        // Growing fold U+023A -> U+2C65: 0x80 | (0x2C65 & 0x7F) == 0xE5.
        assert_eq!(index_fold("\u{023A}".into()), vec![0x80 | 0x65]);
        // Non-folding multibyte still collapses to its low 7 bits.
        assert_eq!(index_fold("中".into()), vec![0x80 | 0x2D]);
    }

    #[test]
    fn index_fold_matches_reference_map() {
        let r = reference();
        let input = "Quick BROWN Fox 🦊 ÜBER Größe ΣΟΦΙΑ \u{0130}\u{023A}漢";
        assert_eq!(index_fold(input.to_string()), index_fold_oracle(&r, input));
    }

    #[test]
    fn index_fold_matches_reference_map_exhaustive() {
        // Drive every assigned code point through the byte-oriented index path
        // and cross-check against the reference fold map.
        let r = reference();
        let mut input = String::from("X");
        for cp in 0x80..0x110000u32 {
            if (0xD800..0xE000).contains(&cp) {
                continue; // surrogates aren't valid chars
            }
            input.push(char::from_u32(cp).expect("cp is a valid non-surrogate char"));
        }
        let expected = index_fold_oracle(&r, &input);
        assert_eq!(index_fold(input), expected);
    }

    #[test]
    fn index_fold_char_examples() {
        assert_eq!(index_fold_char('A'), b'a');
        assert_eq!(index_fold_char('!'), b'!');
        assert_eq!(index_fold_char('Ü'), 0xFC);
        assert_eq!(index_fold_char('中'), 0x80 | 0x2D);
        // Fold to ASCII keeps the high bit set.
        assert_eq!(index_fold_char('\u{212A}'), 0x80 | b'k');
    }

    #[test]
    fn index_fold_char_matches_index_fold_exhaustive() {
        // Every code point's single-char `index_fold_char` must equal the lone
        // byte `index_fold` produces for that character.
        for cp in 0u32..0x110000 {
            if (0xD800..0xE000).contains(&cp) {
                continue; // surrogates aren't valid chars
            }
            let c = char::from_u32(cp).expect("cp is a valid non-surrogate char");
            let folded = index_fold(c.to_string());
            assert_eq!(folded.len(), 1, "cp {cp:#x} did not yield one byte");
            assert_eq!(index_fold_char(c), folded[0], "cp {cp:#x}");
        }
    }
}
