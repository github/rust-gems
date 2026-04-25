//! Bigram priority table.
//!
//! Assigns a frequency-based priority to each byte pair, used by the sparse n-gram
//! extraction algorithm to decide where n-gram boundaries fall.

use std::sync::OnceLock;

use crate::murmur::hash_bigram;
use crate::NUM_FREQUENT_BIGRAMS;

/// The bigrams in this string are sorted by how frequently they occur in code (descending).
/// Bigrams are separated by null bytes. Only the first [`NUM_FREQUENT_BIGRAMS`] entries
/// receive nonzero priority; all other byte pairs default to `(0, 0)`.
static BIGRAMS_STR: &str = include_str!("bigrams.bin");

/// Flat 256×256 lookup table indexed by `a as usize * 256 + b`.
/// Entries default to `(0, 0)` for bigrams not in the frequency table.
static BIGRAM_TABLE: OnceLock<Box<[(u32, u32); 256 * 256]>> = OnceLock::new();

/// Returns the bigram priority table. The first call initializes it (thread-safe).
pub(crate) fn get_bigram_table() -> &'static [(u32, u32); 256 * 256] {
    BIGRAM_TABLE.get_or_init(|| {
        let mut table = Box::new([(0u32, 0u32); 256 * 256]);
        for (idx, s) in BIGRAMS_STR
            .split('\0')
            .take(NUM_FREQUENT_BIGRAMS)
            .enumerate()
        {
            let mut chars = s.chars();
            let Some((a, b)) = chars.next().zip(chars.next()) else {
                continue;
            };
            // All top-200 bigrams are ASCII; apply lowercase folding.
            let a = (a as u8).to_ascii_lowercase();
            let b = (b as u8).to_ascii_lowercase();
            // Higher-frequency bigrams get HIGHER values so they are more often
            // encompassed by longer grams.
            table[a as usize * 256 + b as usize] =
                ((NUM_FREQUENT_BIGRAMS - idx) as u32, hash_bigram((a, b)));
        }
        table
    })
}
