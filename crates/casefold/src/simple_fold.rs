//! Unicode simple case-folding to a `String`, built on the shared paged-bitmap
//! run table.

use crate::table::*;
use crate::{popcount_up_to, scan_end_low, utf8_len};

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
    // Raw write cursor into `out`'s buffer. Null until the first real fold
    // allocates `out` (its pointer is then non-null), so `dst.is_null()` doubles
    // as the "haven't started building the output yet" flag. We bypass the Vec
    // push/reserve API: the buffer is reserved once for the worst case, so every
    // copy/store below is unchecked.
    let mut dst: *mut u8 = core::ptr::null_mut();
    // `flushed` marks the start of the contiguous run of `bytes` that is
    // already correct but not yet copied out.
    let mut flushed = 0usize;
    let mut read = start;
    while read < bytes.len() {
        // ASCII (already lowercased by pass 1) — unchanged, keep scanning.
        let lead = bytes[read];
        if lead & 0x80 == 0 {
            read += 1;
            continue;
        }
        // Page-precision reject probe (see the module docs). Recover the
        // `PAGE_BITMAP` coordinates of `cp >> 6` directly as `(word_idx,
        // bit_idx)` — `cp >> 12` indexes the bitmap word and `(cp >> 6) & 63`
        // the bit — without materializing the combined page number.
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
        if word_idx >= PAGE_BITMAP.len() || (PAGE_BITMAP[word_idx] >> bit_idx) & 1 == 0 {
            read += c_len;
            continue;
        }
        let low_v = bytes[read + c_len - 1] & 0x3F;
        let dense = popcount_up_to(word_idx, bit_idx) as usize;
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
        if dst.is_null() {
            // Reserve once for the worst case so the writes below never need a
            // per-store capacity check. Output is at most 1.5× the input: the
            // only folds that grow are U+023A/U+023E (2→3 bytes), so every 2
            // input bytes yield ≤3 output bytes; `+ 4` covers the 4-byte
            // over-store of the final character. The non-zero capacity makes
            // `out.as_mut_ptr()` non-null, so `dst` is non-null from here on.
            out = Vec::with_capacity(bytes.len() + bytes.len() / 2 + 4);
            dst = out.as_mut_ptr();
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
    if dst.is_null() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::reference;
    use std::collections::HashMap;

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
