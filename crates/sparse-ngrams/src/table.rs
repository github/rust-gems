//! Bigram priority table.
//!
//! Assigns a frequency-based priority to each byte pair, used by the sparse n-gram
//! extraction algorithm to decide where n-gram boundaries fall.

use std::sync::OnceLock;

use crate::NUM_FREQUENT_BIGRAMS;

/// The bigrams in this string are sorted by how frequently they occur in code (descending).
/// Bigrams are separated by null bytes.
/// Currently contains only the top 5845 bigrams (ascii, case-insensitive).
static BIGRAMS_STR: &str = include_str!("bigrams.bin");

/// Flat 256×256 lookup table indexed by `a as usize * 256 + b`.
/// Entries default to 0 for bigrams not in the frequency table.
static BIGRAM_TABLE: OnceLock<Box<[u16; 256 * 256]>> = OnceLock::new();

/// Returns the bigram priority table. The first call initializes it (thread-safe).
pub(crate) fn get_bigram_table() -> &'static [u16; 256 * 256] {
    BIGRAM_TABLE.get_or_init(|| {
        let mut table = Box::new([0u16; 256 * 256]);
        for (idx, s) in BIGRAMS_STR
            .split('\0')
            .take(NUM_FREQUENT_BIGRAMS)
            .enumerate()
        {
            let mut chars = s.chars();
            let Some((a, b)) = chars.next().zip(chars.next()) else {
                continue;
            };
            let a = a as u8;
            let b = b as u8;
            assert_eq!(table[a as usize * 256 + b as usize], 0);
            // Higher-frequency bigrams get HIGHER values so they are more often
            // encompassed by longer grams.
            table[a as usize * 256 + b as usize] = (NUM_FREQUENT_BIGRAMS - idx) as u16;
        }
        table
    })
}
