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
//!    `u16` in `RUN_DATA` holding `end_low` (6 b), `stride - 1` (1 b) and
//!    `length - 1` (7 b); the fold itself is a little-endian byte delta stored
//!    in the parallel `BYTE_DELTA` table (see idea 5).
//! 4. **Page-precision byte-level reject.** The bulk [`fold_into_bytes`] path
//!    probes `PAGE_BITMAP` straight from the first one or two UTF-8 bytes —
//!    `cp >> 6` (the page index) is fully determined by `b0` (2-byte) or
//!    `b0,b1` (3-byte); only the final continuation byte holds the within-page
//!    offset `cp & 0x3F`. A clear page bit copies the whole character verbatim
//!    without assembling `cp` or consulting the run table, so fold-free scripts
//!    (CJK, Hangul, Kana, Arabic, Hebrew, Indic) *and* the empty 64-cp pages
//!    inside otherwise-foldable blocks are skipped at memory speed. No extra
//!    table: it reuses the same `PAGE_BITMAP` the per-`char` lookup uses.
//! 5. **Little-endian byte-delta fold.** On a little-endian machine a folded
//!    character is the source character's UTF-8 bytes, read as a `u32`, plus a
//!    per-run constant. `BYTE_DELTA[i]` stores that constant (32 b, since the
//!    low code-point bits land in the high word byte), so folding is a masked
//!    load + `wrapping_add` + a single 4-byte store — no decode, no encode, and
//!    it handles length-changing folds (`K`→`k`, `Ⱥ`→`ⱥ`) by writing fewer or
//!    more bytes than were read.
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
    let (idx, end) = match successor(cp) {
        Some(x) => x,
        None => return c,
    };
    let packed = RUN_DATA[idx] as u32;
    let len_m1 = (packed >> 7) & 0x7f;
    let stride_bit = (packed >> 6) & 1;
    // `start = end - (length - 1) * stride`, computed with a shift since
    // `stride` is `1 << stride_bit`.
    let start = end - (len_m1 << stride_bit);
    if cp < start || ((cp - start) & stride_bit) != 0 {
        return c;
    }
    // Fold with the run's little-endian byte delta: encode `cp`, add the delta,
    // decode the result. The same `BYTE_DELTA` table serves the bulk byte path.
    let mut buf = [0u8; 4];
    c.encode_utf8(&mut buf);
    let folded = u32::from_le_bytes(buf).wrapping_add(BYTE_DELTA[idx]);
    char::from_u32(decode_utf8_le(folded)).unwrap_or(c)
}

/// Decodes the little-endian UTF-8 byte word `word` (low bytes hold the
/// character, length implied by the lead byte) back into a code point.
#[inline]
fn decode_utf8_le(word: u32) -> u32 {
    let b = word.to_le_bytes();
    match utf8_len(b[0]) {
        1 => b[0] as u32,
        2 => (((b[0] & 0x1F) as u32) << 6) | (b[1] & 0x3F) as u32,
        3 => {
            (((b[0] & 0x0F) as u32) << 12)
                | (((b[1] & 0x3F) as u32) << 6)
                | (b[2] & 0x3F) as u32
        }
        _ => {
            (((b[0] & 0x07) as u32) << 18)
                | (((b[1] & 0x3F) as u32) << 12)
                | (((b[2] & 0x3F) as u32) << 6)
                | (b[3] & 0x3F) as u32
        }
    }
}

/// Consumes `s` and returns its simple-fold form as a `Vec<u8>`. The input's
/// heap buffer is handed back untouched whenever folding changes no bytes —
/// that covers pure-ASCII / already-lowercase input (folded in place) *and*
/// any input whose multibyte characters never fold (CJK, Hangul, Kana,
/// Arabic, Hebrew, Indic, symbols, …). A fresh buffer is allocated only once
/// an actual case fold is encountered; from there, unmodified spans are
/// bulk-copied and folded characters are re-encoded in between.
///
/// Folds may shrink (e.g. U+212A KELVIN SIGN is 3 bytes but folds to `k` =
/// 1 byte) or grow (e.g. U+023A `Ⱥ` is 2 bytes but folds to U+2C65 `ⱥ` =
/// 3 bytes), so in-place rewriting isn't possible in general — but inputs that
/// don't fold at all skip the second buffer entirely. The returned bytes are
/// always valid UTF-8.
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
    fold_non_ascii_tail(bytes, first_non_ascii)
}

/// Tier 2 — copy-on-fold UTF-8 path. Scans the non-ASCII tail of the
/// already-(ASCII-)lowercased `bytes` for the first character that actually
/// folds to *different* bytes. Until one is found nothing is copied, so an
/// input whose multibyte content never folds is returned in its original
/// allocation untouched. Once a folding character is hit, a fresh buffer is
/// allocated and the rest is built by bulk-copying each contiguous unmodified
/// span and re-encoding the folded characters in between. The returned bytes
/// are always valid UTF-8.
///
/// Characters are never fully decoded: the page index (`cp >> 6`) comes from
/// the first one or two bytes for the `PAGE_BITMAP` reject, and on a page hit
/// the remaining `cp & 0x3F` is read directly from the final byte to drive the
/// within-page run search and delta — so the fold logic of
/// [`simple_fold_non_ascii`] is inlined here without repeating its page test.
fn fold_non_ascii_tail(bytes: Vec<u8>, start: usize) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let src = bytes.as_ptr();
    // Raw write cursor into `out`'s buffer (valid once `building` is set). We
    // bypass the Vec push/reserve API: the buffer is reserved once for the
    // worst case, so every copy/store below is unchecked.
    let mut dst: *mut u8 = core::ptr::null_mut();
    // `flushed` marks the start of the contiguous run of `bytes` that is
    // already correct but not yet copied out.
    let mut flushed = 0usize;
    let mut building = false;
    let mut read = start;
    while read < bytes.len() {
        // ASCII (already lowercased by pass 1) — unchanged, keep scanning.
        let lead = bytes[read];
        if lead & 0x80 == 0 {
            read += 1;
            continue;
        }
        // Page-precision reject probe (see the module docs).
        let (page, c_len) = if lead < 0xE0 {
            ((lead & 0x1F) as u32, 2usize)
        } else if lead < 0xF0 {
            ((((lead & 0x0F) as u32) << 6) | (bytes[read + 1] & 0x3F) as u32, 3)
        } else {
            (
                (((lead & 0x07) as u32) << 12)
                    | (((bytes[read + 1] & 0x3F) as u32) << 6)
                    | (bytes[read + 2] & 0x3F) as u32,
                4,
            )
        };
        let word_idx = (page >> 6) as usize;
        if word_idx >= PAGE_BITMAP.len() || (PAGE_BITMAP[word_idx] >> (page & 63)) & 1 == 0 {
            read += c_len;
            continue;
        }
        let low_v = (bytes[read + c_len - 1] & 0x3F) as u32;
        let cp = (page << PAGE_BITS) | low_v;
        let dense = popcount_up_to(page) as usize;
        let lo = PAGE_OFFSET[dense] as usize;
        let slice = &RUN_DATA[lo..PAGE_OFFSET[dense + 1] as usize];
        let mut off = 0;
        while off < slice.len() && (slice[off] as u32 & PAGE_MASK) < low_v {
            off += 1;
        }
        let idx = if off < slice.len() {
            let packed = slice[off] as u32;
            let len_m1 = (packed >> 7) & 0x7F;
            let stride_bit = (packed >> 6) & 1;
            let end = (page << PAGE_BITS) | (packed & PAGE_MASK);
            let run_start = end - (len_m1 << stride_bit);
            if cp < run_start || ((cp - run_start) & stride_bit) != 0 {
                read += c_len;
                continue;
            }
            lo + off
        } else {
            read += c_len;
            continue;
        };
        // Load the character's bytes as a little-endian u32, mask off the lanes
        // past it, add the run's constant byte delta. Over-reading 4 bytes is
        // safe except within ≤3 bytes of the buffer end; the variable-length
        // fallback there is far slower (a `memcpy` call per fold), so the fast
        // path is worth the branch.
        let raw = if read + 4 <= bytes.len() {
            u32::from_le_bytes(bytes[read..read + 4].try_into().unwrap())
        } else {
            let mut w = [0u8; 4];
            w[..c_len].copy_from_slice(&bytes[read..read + c_len]);
            u32::from_le_bytes(w)
        };
        let word = raw & (u32::MAX >> ((4 - c_len) * 8));
        let folded = word.wrapping_add(BYTE_DELTA[idx]);
        let dest_len = utf8_len((folded & 0xFF) as u8);
        if !building {
            // Reserve once for the worst case so the writes below never need a
            // per-store capacity check. Output is at most 1.5× the input: the
            // only folds that grow are U+023A/U+023E (2→3 bytes), so every 2
            // input bytes yield ≤3 output bytes; `+ 4` covers the 4-byte
            // over-store of the final character.
            out = Vec::with_capacity(bytes.len() + bytes.len() / 2 + 4);
            dst = out.as_mut_ptr();
            building = true;
        }
        // SAFETY: the buffer is reserved for the worst-case 1.5× output plus 4
        // bytes of over-store headroom, so `dst` (the running output length)
        // plus the 4-byte store stays in bounds for every iteration. `src` and
        // `dst` are distinct allocations.
        unsafe {
            let run = read - flushed;
            if run != 0 {
                core::ptr::copy_nonoverlapping(src.add(flushed), dst, run);
                dst = dst.add(run);
            }
            // Store a full 4-byte word, advance only by the real folded length.
            dst.cast::<u32>().write_unaligned(folded.to_le());
            dst = dst.add(dest_len);
        }
        read += c_len;
        flushed = read;
    }
    if !building {
        // Nothing folded — return the original buffer with no extra copy.
        return bytes;
    }
    // SAFETY: the trailing unmodified run fits in the reserved buffer; `dst`
    // minus the base pointer is the total number of bytes written.
    unsafe {
        let tail = bytes.len() - flushed;
        core::ptr::copy_nonoverlapping(src.add(flushed), dst, tail);
        dst = dst.add(tail);
        out.set_len(dst as usize - out.as_ptr() as usize);
    }
    out
}

/// Number of bytes in the UTF-8 sequence whose lead byte is `lead`.
///
/// The length only depends on the top 4 bits of the lead byte, so the 16
/// possible lengths are packed one-nibble-each into a `u64` lookup constant
/// (`0x0..7` ASCII → 1; `0x8..0xB` continuation → 1, never a valid lead;
/// `0xC/0xD` → 2; `0xE` → 3; `0xF` → 4). The length is then a single shift and
/// mask rather than a comparison chain.
#[inline]
pub fn utf8_len(lead: u8) -> usize {
    const UTF8_LEN_BY_LEAD: u64 = 0x4322_1111_1111_1111;
    ((UTF8_LEN_BY_LEAD >> (4 * (lead >> 4))) & 0xF) as usize
}

/// Number of distinct code points with a simple fold.
pub const fn num_fold_entries() -> u32 {
    NUM_FOLD_ENTRIES
}

/// Total compressed size of the embedded table, in bytes. Used by the
/// size-budget test.
#[cfg(test)]
const fn table_size_bytes() -> usize {
    PAGE_BITMAP.len() * 8
        + POPCNT_SAMPLES.len()
        + PAGE_OFFSET.len()
        + RUN_DATA.len() * 2
        + BYTE_DELTA.len() * 4
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
// Each `RUN_DATA[i]` is a packed u16 (the fold delta lives in `BYTE_DELTA[i]`):
//   bits  0..6   end & PAGE_MASK   (used by the within-page linear scan)
//   bit   6      stride - 1
//   bits  7..14  length - 1

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
/// page has no intervals or none of its ends is `>= cp`. Returns the absolute
/// `RUN_DATA`/`BYTE_DELTA` index of the run plus its full inclusive `end` code
/// point. Because intervals are split at page boundaries, this is also the only
/// possible run that could contain `cp` — no cross-page scan is ever needed.
#[inline]
fn successor(cp: u32) -> Option<(usize, u32)> {
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
    // is branch-predictable and reads sequential u16s.
    let slice = &RUN_DATA[lo..hi];
    let mut off = 0;
    while off < slice.len() && (slice[off] as u32 & PAGE_MASK) < low_v {
        off += 1;
    }
    if off >= slice.len() {
        // All ends in this slot are < cp's low byte; since intervals don't
        // span slots, cp is past every interval and so doesn't fold.
        return None;
    }
    let end = (page << PAGE_BITS) | (slice[off] as u32 & PAGE_MASK);
    Some((lo + off, end))
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
        // The parallel BYTE_DELTA table (one u32 per run) roughly doubles the
        // run storage in exchange for a decode/encode-free fold path.
        let sz = table_size_bytes();
        eprintln!("table size: {sz} bytes for {} entries", num_fold_entries());
        assert!(sz < 2400, "table size {sz} exceeds 2400 B budget");
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
    fn fold_into_bytes_reuses_buffer_for_nonfolding_nonascii() {
        // Non-ASCII content that never folds (CJK + Hebrew) plus ASCII upper
        // case: the ASCII is lowercased in place and, because no multibyte
        // character folds, the original allocation is handed back with no
        // second buffer — same pointer as the input String.
        let s = "HELLO 日本語 שלום WORLD".to_string();
        let original_ptr = s.as_ptr();
        let out = fold_into_bytes(s);
        assert_eq!(out, "hello 日本語 שלום world".as_bytes());
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

    #[test]
    fn fold_into_bytes_matches_per_char_fold_exhaustive() {
        // Drive every assigned code point through the byte-oriented fold path
        // and cross-check against per-char `simple_fold`. This guarantees the
        // UTF-8 lead-byte reject filter never skips a code point that actually
        // folds (a false reject would corrupt output here). A leading 'X'
        // forces the tier-2 UTF-8 tail to run from the very first char.
        let mut input = String::from("X");
        for cp in 0x80..0x110000u32 {
            if (0xD800..0xE000).contains(&cp) {
                continue; // surrogates aren't valid chars
            }
            input.push(char::from_u32(cp).unwrap());
        }
        let expected: String = input.chars().map(simple_fold).collect();
        assert_eq!(fold_into_bytes(input), expected.into_bytes());
    }
}
