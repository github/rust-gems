//! Shared helpers for casefold benchmarks: a reference HashMap implementation
//! and a few representative workloads.

use std::collections::HashMap;
use std::fs;

use foldhash::fast::FixedState;

/// `HashMap<u32, u32>` using `foldhash`'s fast fixed-seed hasher — the same
/// hasher hashbrown 0.15 uses by default. Avoids the ~4× penalty of std's
/// `RandomState` for tiny keys.
pub type FoldHashMap = HashMap<u32, u32, FixedState>;

/// Parses `CaseFolding.txt` and returns a `FoldHashMap` containing every
/// simple (1-to-1) fold. Used as the baseline against which the compact table
/// is compared.
pub fn reference_map() -> FoldHashMap {
    // The benchmark crate sits at `crates/casefold/benchmarks`; the data file
    // lives one directory up.
    let text = fs::read_to_string("../data/CaseFolding.txt")
        .expect("read CaseFolding.txt (run from crate dir)");
    let mut out = FoldHashMap::with_hasher(FixedState::default());
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split(';').map(|s| s.trim());
        let cp_str = parts.next().expect("code point field");
        let cp = u32::from_str_radix(cp_str, 16).expect("code point is hex");
        let status = parts.next().expect("status field");
        let mapping = parts.next().expect("mapping field");
        if status != "C" && status != "S" {
            continue;
        }
        let target_str = mapping.split_whitespace().next().unwrap_or("");
        let target = u32::from_str_radix(target_str, 16).expect("mapping is hex");
        out.insert(cp, target);
    }
    out
}

/// Look up via the HashMap baseline.
#[inline]
pub fn hashmap_fold(map: &FoldHashMap, c: char) -> char {
    let cp = c as u32;
    let folded = map.get(&cp).copied().unwrap_or(cp);
    char::from_u32(folded).unwrap_or(c)
}

/// Encodes a character's UTF-8 bytes into a little-endian `u32`, with any
/// unused high bytes left zero. This is the writable low-byte form the fold
/// *output* uses: the lead byte sits in the low byte, so storing the whole
/// `u32` little-endian and advancing by the encoded length writes the
/// character back out correctly.
pub fn encode_utf8_word(c: char) -> u32 {
    let mut buf = [0u8; 4];
    let encoded = c.encode_utf8(&mut buf).len();
    let mut word = 0u32;
    for (j, &b) in buf[..encoded].iter().enumerate() {
        word |= (b as u32) << (8 * j);
    }
    word
}

/// Left-aligned UTF-8 key: the character's bytes packed by [`encode_utf8_word`]
/// and then shifted up by `4 - utf8_len` bytes so only this character's bytes
/// occupy the high end of the `u32` and the low bytes are zero. This is exactly
/// what [`hashmap_fold_utf8`] expects: loading 4 raw bytes little-endian and
/// shifting left by `4 - len` bytes discards any trailing bytes belonging to
/// the *next* character, so no masking is needed at the lookup site.
pub fn encode_utf8_key(c: char) -> u32 {
    let word = encode_utf8_word(c);
    let len = casefold::utf8_len((word & 0xFF) as u8);
    word << (8 * (4 - len) as u32)
}

/// Like [`reference_map`], but keyed by each character's left-aligned UTF-8
/// encoding (see [`encode_utf8_key`]) and valued by the folded character's
/// writable low-byte encoding (see [`encode_utf8_word`]). This is the table
/// [`hashmap_fold_utf8`] queries.
pub fn reference_map_utf8() -> FoldHashMap {
    let mut out = FoldHashMap::with_hasher(FixedState::default());
    for (&cp, &target) in reference_map().iter() {
        let (Some(from), Some(to)) = (char::from_u32(cp), char::from_u32(target)) else {
            continue;
        };
        out.insert(encode_utf8_key(from), encode_utf8_word(to));
    }
    out
}

/// Look up via the HashMap baseline, operating directly on a UTF-8 character
/// supplied as a *left-aligned* `u32` `key` (the form produced by
/// [`encode_utf8_key`]): the character's bytes sit in the high end of the word
/// and the low bytes are zero. Callers obtain this by loading 4 raw bytes
/// little-endian and shifting left by `4 - utf8_len` bytes, which drops any
/// trailing bytes from the following character — so no masking is needed here.
///
/// Returns the folded character in writable low-byte order (lead byte first),
/// ready to be stored as a single 4-byte word. On a miss the original
/// character is returned in the same low-byte order by undoing the caller's
/// left shift.
#[inline]
pub fn hashmap_fold_utf8(map: &FoldHashMap, key: u32) -> u32 {
    match map.get(&key) {
        Some(&folded) => folded,
        // Miss: shift the left-aligned key back down to writable low-byte order
        // (the inverse of `word << (4 - len)`). The lead byte is the lowest
        // non-zero byte, so its position is the shift amount; clamp to the
        // 1-byte (ASCII / NUL) maximum of 24 bits so an all-zero key is safe.
        None => key >> (key.trailing_zeros() & !7).min(24),
    }
}
