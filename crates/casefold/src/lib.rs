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

mod index_fold;
mod simple_fold;
pub use index_fold::index_fold;
pub use simple_fold::simple_fold;

use table::*;

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
        + INDEX_DELTA.len()
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

/// Number of populated pages strictly before the page located at
/// `PAGE_BITMAP[word_idx]` bit `bit_idx`.
#[inline]
fn popcount_up_to(word_idx: usize, bit_idx: u32) -> u32 {
    let base = POPCNT_SAMPLES[word_idx] as u32;
    let partial = PAGE_BITMAP[word_idx] & ((1u64 << bit_idx).wrapping_sub(1));
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
pub(crate) mod test_support {
    use std::collections::HashMap;
    use std::fs;

    /// Parse `data/CaseFolding.txt` into a simple-fold map (statuses `C` and
    /// `S`), shared by the `simple_fold` and `index_fold` cross-check tests.
    pub(crate) fn reference() -> HashMap<u32, u32> {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_compact() {
        // The parallel BYTE_DELTA table (one u32 per run) roughly doubles the
        // run storage in exchange for a decode/encode-free fold path.
        let sz = table_size_bytes();
        eprintln!("table size: {sz} bytes for {NUM_FOLD_ENTRIES} entries");
        assert!(sz < 2400, "table size {sz} exceeds 2400 B budget");
    }
}
