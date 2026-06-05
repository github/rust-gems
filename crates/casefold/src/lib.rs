//! Compact Unicode simple case-folding.
//!
//! Maps each character to its lower-case fold form as defined by the Unicode
//! [Case Folding](https://www.unicode.org/Public/UCD/latest/ucd/CaseFolding.txt)
//! data file, restricted to the **simple** (1-to-1) folds (statuses `C` and
//! `S`). Full multi-character folds (`F`) and Turkic locale folds (`T`) are
//! intentionally not supported.
//!
//! # Why this exists
//!
//! Unicode 16.0 defines roughly 1500 simple-fold mappings. A naive table
//! storing `(from, to)` pairs as `(u32, u32)` would consume ~12 KB, and the
//! representation chosen by the Go standard library (`unicode.SimpleFold`)
//! consumes ~7 KB. The 1500 mappings exhibit a great deal of structure:
//! adjacent code points typically share the same delta to their fold, and the
//! whole table can be compressed into about **1 KB** while remaining trivially
//! queryable with byte-wise loads.
//!
//! The encoding uses three ideas:
//!
//! 1. **Run-length grouping with stride.** Consecutive code points sharing a
//!    delta are coalesced into a single run. A 1-bit `stride` field allows
//!    runs that alternate (e.g. `0x0100, 0x0102, 0x0104 …`), which is the
//!    common pattern for paired upper/lower Latin Extended letters.
//! 2. **Paged bitmap index.** A `PAGE_BITMAP` of 64-cp pages (one bit per
//!    page) plus a cumulative-popcount sidetable lets us test in *one* bit
//!    load whether `cp`'s page contains any fold run at all. An unset bit is
//!    a definitive "no fold" — no cross-page search is ever required. Within
//!    a populated page, a short branch-predictable linear scan over the
//!    page's slice of `RUN_DATA` (located via `PAGE_OFFSET`) finds the
//!    candidate run.
//! 3. **Per-page run splitting and packed run records.** Any run whose
//!    `[start, end]` would straddle a 64-cp boundary is split at the boundary
//!    during the build, so every run lives in exactly one page (which is what
//!    makes "empty bit ⇒ no fold" sound). Each run is then a single packed
//!    `u32` in `RUN_DATA` holding `end_low` (6 b), `stride - 1` (1 b),
//!    `length` (7 b), and a signed `delta` (18 b). One indexed load both
//!    drives the slot scan and decodes the run on a hit — no parallel arrays,
//!    no escape table.
//!
//! # Example
//!
//! ```
//! use casefold::simple_fold;
//! assert_eq!(simple_fold('A'), 'a');
//! assert_eq!(simple_fold('Ä'), 'ä');
//! assert_eq!(simple_fold('a'), 'a');
//! assert_eq!(simple_fold('1'), '1');
//! ```

#![deny(missing_docs)]

mod table {
    include!(concat!(env!("OUT_DIR"), "/table.rs"));
}

use table::*;

const PAGE_BITS: u32 = 6;
const PAGE_MASK: u32 = (1u32 << PAGE_BITS) - 1;

/// Returns the simple (1-to-1) lower-case fold of `c`, or `c` itself if no
/// fold is defined for it.
///
/// This is suitable for case-insensitive comparison of individual characters.
/// For full case-insensitive string matching (which may require multi-character
/// folds such as `ß` → `ss`), use a dedicated locale-aware library.
#[inline]
pub fn simple_fold(c: char) -> char {
    let cp = c as u32;
    // ASCII fast path: branchless A..Z → a..z, identity otherwise.
    if cp < 0x80 {
        let is_upper = u32::from(cp.wrapping_sub(b'A' as u32) < 26);
        // Safe: `cp | 0x20` for any ASCII byte stays in the ASCII range, and
        // for cp in 'A'..='Z' it produces 'a'..='z'. For any other ASCII byte
        // `is_upper` is 0, so the OR is a no-op.
        return unsafe { char::from_u32_unchecked(cp | (is_upper << 5)) };
    }
    simple_fold_non_ascii(c)
}

/// Identical to [`simple_fold`], but assumes `c as u32 >= 0x80`. Use this in
/// hot loops where the caller has already separated ASCII from non-ASCII (the
/// generic `simple_fold` retests `cp < 0x80` on every call).
///
/// On an ASCII codepoint this still returns the correct (identity) result —
/// every ASCII char's lookup misses the bitmap and returns `c` — but the
/// ASCII fast path that `simple_fold` provides is skipped, so calling this
/// on ASCII is slower than `simple_fold`.
#[inline]
pub fn simple_fold_non_ascii(c: char) -> char {
    let cp = c as u32;
    if cp > LAST_COVERED {
        return c;
    }
    let (packed, end) = match successor(cp) {
        Some(x) => x,
        None => return c,
    };
    let length = (packed >> 7) & 0x7f;
    let stride = ((packed >> 6) & 1) + 1;
    let start = end - (length - 1) * stride;
    if cp < start {
        return c;
    }
    let off = cp - start;
    if stride == 2 && (off & 1) != 0 {
        return c;
    }
    // 18-bit signed delta: sign-extend by shifting an i32 right by 14.
    let delta = (packed as i32) >> 14;
    let folded = (cp as i64 + delta as i64) as u32;
    char::from_u32(folded).unwrap_or(c)
}

/// Consumes `s` and returns its simple-fold form as a `Vec<u8>`, reusing the
/// `String`'s heap buffer whenever every char folds to a sequence of the same
/// byte length.
///
/// Falls back to allocating a fresh buffer as soon as any non-ASCII byte
/// is encountered (folded characters may shrink, e.g. U+212A KELVIN SIGN is
/// 3 bytes but folds to `k` = 1 byte, or grow, e.g. U+023A `Ⱥ` is 2 bytes
/// but folds to U+2C65 `ⱥ` = 3 bytes, so in-place rewriting wouldn't work
/// in general — and benchmarking showed the always-reallocate path is
/// faster than a length-preserving in-place tier anyway). The returned
/// bytes are always valid UTF-8.
///
/// # Example
///
/// ```
/// use casefold::fold_into_bytes;
/// assert_eq!(fold_into_bytes("Hello, WORLD!".to_string()), b"hello, world!");
/// assert_eq!(fold_into_bytes("ÜBER".to_string()), "über".as_bytes());
/// // Length-changing fold (U+212A KELVIN SIGN → U+006B, 3 bytes → 1 byte):
/// assert_eq!(fold_into_bytes("\u{212A}elvin".to_string()), b"kelvin");
/// ```
pub fn fold_into_bytes(s: String) -> Vec<u8> {
    let mut bytes = s.into_bytes();
    // Tier 1 — full straight-through pass: lowercase every ASCII A..Z byte
    // in place (a no-op on any non-ASCII byte, since `b.wrapping_sub(b'A')`
    // is ≥ 26 for every byte outside 0x41..0x5A), and OR all bytes together
    // so a single sign-bit test afterwards tells us whether the input
    // contained any multibyte UTF-8 sequences. No early `break`, no
    // input-dependent control flow — LLVM auto-vectorizes the loop.
    let mut high_bit_acc: u8 = 0;
    for b in &mut bytes {
        high_bit_acc |= *b;
        let is_upper = b.wrapping_sub(b'A') < 26;
        *b |= u8::from(is_upper) << 5;
    }
    if high_bit_acc & 0x80 == 0 {
        return bytes;
    }
    // Non-ASCII bytes are present. Locate the first one (SIMD-fast via
    // `position`/memchr) and hand off to the UTF-8 path from there — the
    // ASCII prefix is already lowercased and `simple_fold` is idempotent
    // on lower-case ASCII, so skipping it is purely an optimization.
    let first_non_ascii = bytes.iter().position(|&b| b & 0x80 != 0).unwrap();
    fold_non_ascii_tail(&bytes, first_non_ascii)
}

/// Tier 2 — reallocating UTF-8 fold path. Copies the already-folded ASCII
/// prefix `bytes[..read]` into a fresh allocation, then walks the remaining
/// tail byte-by-byte: ASCII bytes (already lowercased by `fold_into_bytes`'
/// pass 1) are copied verbatim, multibyte characters are decoded, folded
/// via [`simple_fold_non_ascii`], and re-encoded into the output buffer.
fn fold_non_ascii_tail(bytes: &[u8], mut read: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() + 4);
    out.extend_from_slice(&bytes[..read]);
    let mut buf = [0u8; 4];
    while read < bytes.len() {
        // ASCII bytes were already lowercased by pass 1 — copy verbatim.
        if bytes[read] & 0x80 == 0 {
            out.push(bytes[read]);
            read += 1;
            continue;
        }
        // SAFETY: `bytes` originated from a `String` and `read` always sits
        // on a char boundary, so the tail is well-formed UTF-8.
        let c = {
            let tail = unsafe { std::str::from_utf8_unchecked(&bytes[read..]) };
            tail.chars().next().unwrap()
        };
        read += c.len_utf8();
        out.extend_from_slice(simple_fold_non_ascii(c).encode_utf8(&mut buf).as_bytes());
    }
    out
}

/// Number of distinct code points with a simple fold.
pub const fn num_fold_entries() -> u32 {
    NUM_FOLD_ENTRIES
}

/// Total compressed size of the embedded table, in bytes. Used by the
/// size-budget test.
#[cfg(test)]
const fn table_size_bytes() -> usize {
    PAGE_BITMAP.len() * 8 + POPCNT_SAMPLES.len() + PAGE_OFFSET.len() + RUN_DATA.len() * 4
}

// ---- Paged bitmap lookup ------------------------------------------------
//
// `PAGE_BITMAP` is a 1-bit-per-64-cp-page bitmap. A bit set at position
// `cp >> PAGE_BITS` means "the slot containing `cp` overlaps at least one
// (post-split) interval". Because every interval is fully contained in a
// single page (we split runs at page boundaries during the build), an empty
// page means definitively no fold — there is *no* need to look at any other
// page. `POPCNT_SAMPLES[i]` holds the cumulative popcount of
// `PAGE_BITMAP[0..i]`, so the dense index of any page is one load plus one
// masked popcount. `PAGE_OFFSET[j] .. PAGE_OFFSET[j+1]` is the slice of
// `RUN_DATA` belonging to the `j`-th populated page, sorted by end-low.
//
// Each `RUN_DATA[i]` is a packed u32:
//   bits  0..6   end & PAGE_MASK   (used by the within-page linear scan)
//   bit   6      stride - 1
//   bits  7..14  length
//   bits 14..32  delta (signed, sign-extended via `as i32 >> 14`)

/// Number of populated pages strictly before `page`.
#[inline]
fn popcount_up_to(page: u32) -> u32 {
    let word_idx = (page / 64) as usize;
    let bit_in_word = page % 64;
    let base = POPCNT_SAMPLES[word_idx] as u32;
    let partial = PAGE_BITMAP[word_idx] & ((1u64 << bit_in_word).wrapping_sub(1));
    base + partial.count_ones()
}

/// Smallest run with `end >= cp` *within `cp`'s own page*, or `None` if the
/// page has no intervals or none of its ends is `>= cp`. Returns the packed
/// `RUN_DATA` entry plus the run's full inclusive `end` code point. Because
/// intervals are split at page boundaries, this is also the only possible run
/// that could contain `cp` — no cross-page scan is ever needed.
#[inline]
fn successor(cp: u32) -> Option<(u32, u32)> {
    let page = cp >> PAGE_BITS;
    let low_v = cp & PAGE_MASK;

    let word_idx = (page / 64) as usize;
    let bit_in_word = page % 64;
    let word = PAGE_BITMAP[word_idx];
    if (word >> bit_in_word) & 1 == 0 {
        // Slot has no intervals at all → cp definitely doesn't fold.
        return None;
    }
    let dense_idx = popcount_up_to(page) as usize;
    let lo = PAGE_OFFSET[dense_idx] as usize;
    let hi = PAGE_OFFSET[dense_idx + 1] as usize;

    // Within-slot linear scan: intervals share a common high prefix, so the
    // right one is the first end-low (bits 0..6 of the packed run) that is
    // ≥ `low_v`. Slots hold at most 30 entries and ~3.8 on average; the scan
    // is branch-predictable and reads sequential u32s.
    let slice = &RUN_DATA[lo..hi];
    let mut off = 0;
    while off < slice.len() && (slice[off] & PAGE_MASK) < low_v {
        off += 1;
    }
    if off >= slice.len() {
        // All ends in this slot are < cp's low byte; since intervals don't
        // span slots, cp is past every interval and so doesn't fold.
        return None;
    }
    let packed = slice[off];
    let end = (page << PAGE_BITS) | (packed & PAGE_MASK);
    Some((packed, end))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs;

    fn reference() -> HashMap<u32, u32> {
        let text = fs::read_to_string("data/CaseFolding.txt").expect("CaseFolding.txt");
        let mut out = HashMap::new();
        for raw in text.lines() {
            let line = raw.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split(';').map(|s| s.trim());
            let cp = u32::from_str_radix(parts.next().unwrap(), 16).unwrap();
            let status = parts.next().unwrap();
            let mapping = parts.next().unwrap();
            if status != "C" && status != "S" {
                continue;
            }
            let target = u32::from_str_radix(mapping.split_whitespace().next().unwrap(), 16).unwrap();
            out.insert(cp, target);
        }
        out
    }

    #[test]
    fn matches_unicode_data_for_every_codepoint() {
        let r = reference();
        for cp in 0..0x110000u32 {
            // Surrogates are not valid `char`s, skip.
            if (0xD800..0xE000).contains(&cp) {
                continue;
            }
            let c = char::from_u32(cp).unwrap();
            let expected = r.get(&cp).copied().unwrap_or(cp);
            let got = simple_fold(c) as u32;
            assert_eq!(
                got, expected,
                "mismatch at U+{cp:04X}: got U+{got:04X}, want U+{expected:04X}"
            );
        }
    }

    #[test]
    fn ascii_fast_cases() {
        assert_eq!(simple_fold('A'), 'a');
        assert_eq!(simple_fold('Z'), 'z');
        assert_eq!(simple_fold('a'), 'a');
        assert_eq!(simple_fold('0'), '0');
        assert_eq!(simple_fold(' '), ' ');
    }

    #[test]
    fn extended_cases() {
        assert_eq!(simple_fold('Ä'), 'ä');
        assert_eq!(simple_fold('Ω'), 'ω');
        // Codepoints with no defined fold map to themselves.
        assert_eq!(simple_fold('漢'), '漢');
        assert_eq!(simple_fold('\u{1F600}'), '\u{1F600}'); // 😀
    }

    #[test]
    fn table_is_compact() {
        // Should be well under 1.4 KB.
        let sz = table_size_bytes();
        eprintln!("table size: {sz} bytes for {} entries", num_fold_entries());
        assert!(sz < 1300, "table size {sz} exceeds 1300 B budget");
    }

    #[test]
    fn fold_into_bytes_ascii() {
        assert_eq!(fold_into_bytes(String::new()), b"");
        assert_eq!(fold_into_bytes("Hello, WORLD!".into()), b"hello, world!");
        assert_eq!(fold_into_bytes("abc 123 XYZ".into()), b"abc 123 xyz");
    }

    #[test]
    fn fold_into_bytes_ascii_then_utf8_handoff() {
        // ASCII prefix gets lowercased by the tier-1 loop, then control
        // hands off to the tier-2 reallocating UTF-8 path at the first
        // multibyte lead.
        assert_eq!(
            fold_into_bytes("MIXED Größe TEXT".into()),
            "mixed größe text".as_bytes(),
        );
        // ASCII prefix, then a *shrinking* fold inside the tail.
        assert_eq!(
            fold_into_bytes("LORD \u{212A}elvin".into()),
            b"lord kelvin",
        );
        // ASCII prefix, then a *growing* fold.
        assert_eq!(
            fold_into_bytes("abc\u{023A}".into()),
            "abc\u{2C65}".as_bytes(),
        );
    }

    #[test]
    fn fold_into_bytes_length_preserving_bmp() {
        assert_eq!(fold_into_bytes("ÄÖÜ".into()), "äöü".as_bytes());
        assert_eq!(fold_into_bytes("ΑΒΓ".into()), "αβγ".as_bytes());
        assert_eq!(fold_into_bytes("漢字".into()), "漢字".as_bytes());
    }

    #[test]
    fn fold_into_bytes_reuses_buffer_for_ascii_input() {
        // Pure-ASCII inputs are lowercased in place — the returned Vec must
        // hold the exact same allocation as the input String.
        let s = "MIXED case AsCiI 12345".to_string();
        let original_ptr = s.as_ptr();
        let out = fold_into_bytes(s);
        assert_eq!(out, b"mixed case ascii 12345");
        assert_eq!(out.as_ptr(), original_ptr);
    }

    #[test]
    fn fold_into_bytes_handles_shrinking_fold() {
        // U+212A KELVIN SIGN (3 bytes) folds to U+006B 'k' (1 byte).
        assert_eq!(fold_into_bytes("\u{212A}elvin".into()), b"kelvin");
        // Shrink inside a longer string.
        let out = fold_into_bytes("LORD \u{212A}elvin RULES".into());
        assert_eq!(out, b"lord kelvin rules");
        // U+2126 OHM SIGN (3 bytes) folds to U+03C9 'ω' (2 bytes).
        assert_eq!(fold_into_bytes("\u{2126}".into()), "\u{03C9}".as_bytes());
    }

    #[test]
    fn fold_into_bytes_handles_growing_fold() {
        // The Unicode 16.0 simple-fold table has exactly two folds that
        // grow in UTF-8 length (verified by scanning CaseFolding.txt):
        // U+023A → U+2C65 and U+023E → U+2C66, both 2 B → 3 B.

        // U+023A 'Ⱥ' is 2 bytes, folds to U+2C65 'ⱥ' = 3 bytes.
        assert_eq!(
            fold_into_bytes("\u{023A}".into()),
            "\u{2C65}".as_bytes()
        );
        // U+023E 'Ⱦ' is 2 bytes, folds to U+2C66 'ⱦ' = 3 bytes.
        assert_eq!(
            fold_into_bytes("\u{023E}".into()),
            "\u{2C66}".as_bytes()
        );

        // Each one mid-string, with mixed length-preserving context on both
        // sides so that the bail-out path also copies a prefix that already
        // contains a length-preserving rewrite.
        let out = fold_into_bytes("ABC\u{023A}xyz".into());
        assert_eq!(out, "abc\u{2C65}xyz".as_bytes());
        let out = fold_into_bytes("ABC\u{023E}xyz".into());
        assert_eq!(out, "abc\u{2C66}xyz".as_bytes());

        // Both growing folds inside the same string: the second one occurs
        // after we have already switched to the allocating buffer.
        let out = fold_into_bytes("\u{023A}\u{023E}".into());
        assert_eq!(out, "\u{2C65}\u{2C66}".as_bytes());

        // Mixed: a length-preserving fold, then a shrinking fold, then both
        // growing folds — exercises every branch in one input.
        let out = fold_into_bytes("Ä\u{212A}\u{023A}\u{023E}".into());
        assert_eq!(out, "ä\u{006B}\u{2C65}\u{2C66}".as_bytes());
    }

    #[test]
    fn fold_into_bytes_matches_per_char_fold() {
        // Cross-check against per-char `simple_fold` on a varied input.
        let input = "Quick BROWN Fox 🦊 ÜBER Größe ΣΟΦΙΑ \u{0130}\u{023A}漢";
        let expected: String = input.chars().map(simple_fold).collect();
        assert_eq!(fold_into_bytes(input.to_string()), expected.as_bytes());
    }
}
