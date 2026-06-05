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
    // ASCII fast path: A..Z → a..z, every other ASCII byte is identity.
    if cp < 0x80 {
        return if cp.wrapping_sub(b'A' as u32) < 26 {
            // Safe: result is in 'a'..='z'.
            unsafe { char::from_u32_unchecked(cp | 0x20) }
        } else {
            c
        };
    }
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

/// Total compressed size of the embedded table, in bytes. Useful for tests
/// and documentation.
pub const fn table_size_bytes() -> usize {
    PAGE_BITMAP.len() * 8 + POPCNT_SAMPLES.len() + PAGE_OFFSET.len() + RUN_DATA.len() * 4
}

/// Number of distinct code points with a simple fold.
pub const fn num_fold_entries() -> u32 {
    NUM_FOLD_ENTRIES
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
    // ≥ `low_v`. Slots hold at most ~18 entries and ~3.8 on average; the scan
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
}
