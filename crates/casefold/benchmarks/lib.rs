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
        let target =
            u32::from_str_radix(mapping.split_whitespace().next().unwrap(), 16).unwrap();
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
