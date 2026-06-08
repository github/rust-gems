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
//!    page's slice of `RUN_END_LOW` (located via `PAGE_OFFSET`) finds the
//!    candidate run.
//! 3. **Per-page run splitting and split byte records.** Any run whose
//!    `[start, end]` would straddle a 64-cp boundary is split at the boundary
//!    during the build, so every run lives in exactly one page (which is what
//!    makes "empty bit ⇒ no fold" sound). Because both ends then fit in 6
//!    bits, a run needs only two bytes: `RUN_END_LOW[i]` (the clean scan key,
//!    compared byte-to-byte against `cp & 0x3F` with no masking) and
//!    `RUN_START_STRIDE[i]` (`start_low | (stride−1) << 6`, read only on a
//!    hit). The membership test compares `cp & 0x3F` directly — no code-point
//!    reconstruction. The fold itself is a little-endian byte delta stored in
//!    the parallel `BYTE_DELTA` table (see idea 5).
//! 4. **Page-precision byte-level reject.** The bulk [`simple_fold`] path
//!    probes `PAGE_BITMAP` straight from the first one or two UTF-8 bytes —
//!    `cp >> 6` (the page index) is fully determined by `b0` (2-byte) or
//!    `b0,b1` (3-byte); only the final continuation byte holds the within-page
//!    offset `cp & 0x3F`. A clear page bit copies the whole character verbatim
//!    without assembling `cp` or consulting the run table, so fold-free scripts
//!    (CJK, Hangul, Kana, Arabic, Hebrew, Indic) *and* the empty 64-cp pages
//!    inside otherwise-foldable blocks are skipped at memory speed. No extra
//!    table: it reuses the same `PAGE_BITMAP` the run lookup uses.
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
//! assert_eq!(simple_fold("Hello, WORLD!".to_string()), "hello, world!");
//! assert_eq!(simple_fold("ÜBER".to_string()), "über");
//! ```

#![deny(missing_docs)]

mod table {
    include!(concat!(env!("OUT_DIR"), "/table.rs"));
}

use table::*;

/// Consumes `s` and returns its simple case-folded form as a `String`. The
/// input's heap buffer is reused untouched whenever folding changes no bytes —
/// that covers pure-ASCII / already-lowercase input (folded in place) *and*
/// any input whose multibyte characters never fold (CJK, Hangul, Kana,
/// Arabic, Hebrew, Indic, symbols, …). A fresh buffer is allocated only once
/// an actual case fold is encountered; from there, unmodified spans are
/// bulk-copied and folded characters are re-encoded in between.
///
/// Folds may shrink (e.g. U+212A KELVIN SIGN is 3 bytes but folds to `k` =
/// 1 byte) or grow (e.g. U+023A `Ⱥ` is 2 bytes but folds to U+2C65 `ⱥ` =
/// 3 bytes), so in-place rewriting isn't possible in general — but inputs that
/// don't fold at all skip the second buffer entirely.
///
/// Only **simple** (1-to-1) folds are applied; multi-character folds such as
/// `ß` → `ss` and Turkic locale folds are left unchanged.
///
/// # Example
///
/// ```
/// use casefold::simple_fold;
/// assert_eq!(simple_fold("Hello, WORLD!".to_string()), "hello, world!");
/// assert_eq!(simple_fold("ÜBER".to_string()), "über");
/// // Length-changing fold (U+212A KELVIN SIGN → U+006B, 3 bytes → 1 byte):
/// assert_eq!(simple_fold("\u{212A}elvin".to_string()), "kelvin");
/// ```
pub fn simple_fold(s: String) -> String {
    // SAFETY: `fold_into_bytes` only lowercases ASCII bytes in place and
    // re-encodes whole characters through the fold table, so its output is
    // always valid UTF-8 (see the exhaustive round-trip test).
    unsafe { String::from_utf8_unchecked(fold_into_bytes(s)) }
}

/// Byte-level core of [`simple_fold`]. Returns the fold as a `Vec<u8>` that is
/// always valid UTF-8; see [`simple_fold`] for the allocation behavior.
fn fold_into_bytes(s: String) -> Vec<u8> {
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
    // ASCII prefix is already lowercased and folding is idempotent on
    // lower-case ASCII, so skipping it is purely an optimization.
    let first_non_ascii = bytes
        .iter()
        .position(|&b| b & 0x80 != 0)
        .expect("a non-ASCII byte exists (the high-bit accumulator was set)");
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
/// within-page run search and byte-delta fold — no code-point reconstruction.
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
            (
                (((lead & 0x0F) as u32) << 6) | (bytes[read + 1] & 0x3F) as u32,
                3,
            )
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
        let low_v = bytes[read + c_len - 1] & 0x3F;
        let dense = popcount_up_to(page) as usize;
        let lo = PAGE_OFFSET[dense] as usize;
        let n = PAGE_OFFSET[dense + 1] as usize - lo;
        let off = scan_end_low(lo, n, low_v);
        let idx = if off < n {
            // The scan guarantees `low_v <= end_low`; the run covers `low_v`
            // iff `low_v >= start_low` (and, for stride 2, the offset is even).
            // No code-point reconstruction — `low_v` is compared directly.
            let ss = RUN_START_STRIDE[lo + off];
            let start_low = ss & 0x3F;
            let stride_bit = ss >> 6;
            if low_v < start_low || ((low_v - start_low) & stride_bit) != 0 {
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
            u32::from_le_bytes(bytes[read..read + 4].try_into().expect("4-byte slice"))
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

/// Total compressed size of the embedded table, in bytes. Used by the
/// size-budget test.
#[cfg(test)]
const fn table_size_bytes() -> usize {
    PAGE_BITMAP.len() * 8
        + POPCNT_SAMPLES.len()
        + PAGE_OFFSET.len()
        + RUN_END_LOW.len()
        + RUN_START_STRIDE.len()
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
// `PAGE_OFFSET[j] .. PAGE_OFFSET[j+1]` is the slice of `RUN_END_LOW` /
// `RUN_START_STRIDE` / `BYTE_DELTA` belonging to the `j`-th populated page,
// sorted by end-low.
//
// The run record is split across two byte arrays so the hot scan reads clean
// keys with no masking:
//   RUN_END_LOW[i]      = end   & PAGE_MASK   (the within-page scan key)
//   RUN_START_STRIDE[i] = (start & PAGE_MASK) | ((stride - 1) << 6)
//                                              (membership, vs `cp & 0x3F`)

/// Number of populated pages strictly before `page`.
#[inline]
fn popcount_up_to(page: u32) -> u32 {
    let word_idx = (page / 64) as usize;
    let bit_in_word = page % 64;
    let base = POPCNT_SAMPLES[word_idx] as u32;
    let partial = PAGE_BITMAP[word_idx] & ((1u64 << bit_in_word).wrapping_sub(1));
    base + partial.count_ones()
}

/// Offset of the first run with `end_low >= low_v` in a page of `n` runs at
/// `RUN_END_LOW[lo..]`, or `n` if none. Scans 8 `end_low` bytes at a time via
/// SWAR: one branchless chunk covers the average ~3.8-run page; the outer loop
/// advances only for the rare page with >8 runs. Padding / next-page bytes that
/// the over-read pulls in are discarded by the `j < n` bound, so no per-lane
/// validity mask is needed.
#[inline]
fn scan_end_low(lo: usize, n: usize, low_v: u8) -> usize {
    const HIGH: u64 = 0x8080_8080_8080_8080;
    const ONES: u64 = 0x0101_0101_0101_0101;
    let bcast = (low_v as u64).wrapping_mul(ONES);
    let mut base = 0;
    while base < n {
        // `RUN_END_LOW` is padded by 8 bytes (build.rs) so this read is always
        // in bounds; the slice length is statically 8.
        let chunk = u64::from_le_bytes(
            RUN_END_LOW[lo + base..lo + base + 8]
                .try_into()
                .expect("8-byte slice"),
        );
        // `(b | 0x80) - low_v` keeps its high bit iff `b >= low_v` (no
        // cross-lane borrow). The first set lane is the first run `>= low_v`.
        let ge = (chunk | HIGH).wrapping_sub(bcast) & HIGH;
        if ge != 0 {
            let j = base + (ge.trailing_zeros() / 8) as usize;
            return if j < n { j } else { n };
        }
        base += 8;
    }
    n
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
            let line = raw
                .split('#')
                .next()
                .expect("split yields at least one item")
                .trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split(';').map(|s| s.trim());
            let cp = u32::from_str_radix(parts.next().expect("code point field"), 16)
                .expect("code point is hex");
            let status = parts.next().expect("status field");
            let mapping = parts.next().expect("mapping field");
            if status != "C" && status != "S" {
                continue;
            }
            let target = u32::from_str_radix(
                mapping
                    .split_whitespace()
                    .next()
                    .expect("mapping has a target"),
                16,
            )
            .expect("mapping is hex");
            out.insert(cp, target);
        }
        out
    }

    /// Per-character fold via the reference map, used as the oracle for the
    /// byte-oriented `fold_into_bytes` cross-checks below.
    fn fold_oracle(r: &HashMap<u32, u32>, s: &str) -> Vec<u8> {
        let mut out = String::new();
        for c in s.chars() {
            let cp = c as u32;
            let folded = r.get(&cp).copied().unwrap_or(cp);
            out.push(char::from_u32(folded).expect("reference fold is a valid char"));
        }
        out.into_bytes()
    }

    #[test]
    fn table_is_compact() {
        // The parallel BYTE_DELTA table (one u32 per run) roughly doubles the
        // run storage in exchange for a decode/encode-free fold path.
        let sz = table_size_bytes();
        eprintln!("table size: {sz} bytes for {NUM_FOLD_ENTRIES} entries");
        assert!(sz < 2400, "table size {sz} exceeds 2400 B budget");
    }

    #[test]
    fn fold_into_bytes_ascii() {
        assert_eq!(fold_into_bytes(String::new()), b"");
        assert_eq!(fold_into_bytes("Hello, WORLD!".into()), b"hello, world!");
        assert_eq!(fold_into_bytes("abc 123 XYZ".into()), b"abc 123 xyz");
    }

    #[test]
    fn simple_fold_returns_string() {
        // Public `String` wrapper: ASCII, length-preserving, shrinking and
        // growing folds all yield valid UTF-8.
        assert_eq!(simple_fold("Hello, WORLD!".to_string()), "hello, world!");
        assert_eq!(simple_fold("ÜBER Größe".to_string()), "über größe");
        assert_eq!(simple_fold("\u{212A}elvin".to_string()), "kelvin");
        assert_eq!(simple_fold("abc\u{023A}".to_string()), "abc\u{2C65}");
        // Non-folding multibyte content is returned unchanged.
        assert_eq!(simple_fold("漢字 שלום".to_string()), "漢字 שלום");
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
        assert_eq!(fold_into_bytes("LORD \u{212A}elvin".into()), b"lord kelvin",);
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
        assert_eq!(fold_into_bytes("\u{023A}".into()), "\u{2C65}".as_bytes());
        // U+023E 'Ⱦ' is 2 bytes, folds to U+2C66 'ⱦ' = 3 bytes.
        assert_eq!(fold_into_bytes("\u{023E}".into()), "\u{2C66}".as_bytes());

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
    fn fold_into_bytes_matches_reference_map() {
        // Cross-check against the reference fold map on a varied input.
        let r = reference();
        let input = "Quick BROWN Fox 🦊 ÜBER Größe ΣΟΦΙΑ \u{0130}\u{023A}漢";
        assert_eq!(fold_into_bytes(input.to_string()), fold_oracle(&r, input));
    }

    #[test]
    fn fold_into_bytes_matches_reference_map_exhaustive() {
        // Drive every assigned code point through the byte-oriented fold path
        // and cross-check against the reference fold map. This guarantees the
        // UTF-8 lead-byte reject filter never skips a code point that actually
        // folds (a false reject would corrupt output here). A leading 'X'
        // forces the tier-2 UTF-8 tail to run from the very first char.
        let r = reference();
        let mut input = String::from("X");
        for cp in 0x80..0x110000u32 {
            if (0xD800..0xE000).contains(&cp) {
                continue; // surrogates aren't valid chars
            }
            input.push(char::from_u32(cp).expect("cp is a valid non-surrogate char"));
        }
        let expected = fold_oracle(&r, &input);
        assert_eq!(fold_into_bytes(input), expected);
    }
}
