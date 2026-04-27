//! Sparse n-gram extraction from byte slices.
//!
//! Sparse grams are a way of selecting variable-length n-grams (longer than 2 bytes) without
//! extracting all possible n-grams. The algorithm is deterministic: the same extraction logic
//! works for every substring, so that substring searches are supported.
//!
//! # How it works
//!
//! Each consecutive byte pair (bigram) is assigned a priority based on how frequently it occurs
//! in a large code corpus. A monotone deque tracks potential n-gram boundaries: an n-gram
//! boundary occurs wherever a bigram has lower priority than all bigrams between it and the
//! previous boundary.
//!
//! For a document of N bytes, this produces roughly 2N n-grams: all bigrams plus algorithmically
//! selected longer n-grams (up to [`MAX_SPARSE_GRAM_SIZE`] bytes).
//!
//! # Example
//!
//! ```
//! use sparse_ngrams::{NGram, collect_sparse_grams, MAX_SPARSE_GRAM_SIZE};
//!
//! let input = b"hello world";
//! let grams = collect_sparse_grams(input);
//! assert!(grams.len() > input.len() - 1);
//! for gram in &grams {
//!     assert!(gram.len() >= 2);
//!     assert!(gram.len() <= MAX_SPARSE_GRAM_SIZE as usize);
//! }
//! ```

mod deque;
mod extract;
mod ngram;
mod table;

pub use ngram::NGram;

/// Number of high-frequency bigrams used to build the priority table.
pub const NUM_FREQUENT_BIGRAMS: usize = 200;

/// Maximum length (in bytes) of a sparse n-gram.
pub const MAX_SPARSE_GRAM_SIZE: u32 = 8;

pub use extract::{collect_sparse_grams, collect_sparse_grams_deque, collect_sparse_grams_masked, collect_sparse_grams_scan, collect_sparse_grams_wide, max_sparse_grams};
#[cfg(target_arch = "x86_64")]
pub use extract::{collect_sparse_grams_masked_avx, collect_sparse_grams_wide_avx};

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(mut v: Vec<NGram>) -> Vec<NGram> {
        v.sort();
        v
    }

    fn collect_to_vec(content: &[u8], f: fn(&[u8], &mut [NGram]) -> usize) -> Vec<NGram> {
        let mut buf = vec![NGram::from_rolling_hash(0, 0); max_sparse_grams(content.len())];
        let count = f(content, &mut buf);
        buf.truncate(count);
        buf
    }

    #[test]
    fn test_empty_input() {
        assert!(collect_sparse_grams(b"").is_empty());
    }

    #[test]
    fn test_single_byte() {
        assert!(collect_sparse_grams(b"a").is_empty());
    }

    #[test]
    fn test_two_bytes() {
        let grams = collect_sparse_grams(b"ab");
        assert_eq!(grams.len(), 1);
        assert_eq!(grams[0], NGram::from_bytes(b"ab"));
    }

    #[test]
    fn test_three_bytes() {
        let grams = collect_sparse_grams(b"abc");
        assert!(grams.len() >= 2);
        assert_eq!(grams[0], NGram::from_bytes(b"ab"));
        assert_eq!(grams[1], NGram::from_bytes(b"bc"));
    }

    #[test]
    fn test_gram_lengths_bounded() {
        let input = b"self.reset_states(the_quick_brown_fox_jumps";
        let grams = collect_sparse_grams(input);
        for gram in &grams {
            assert!(gram.len() >= 2, "gram too short: {gram:?}");
            assert!(gram.len() <= MAX_SPARSE_GRAM_SIZE as usize, "gram too long: {gram:?}");
        }
    }

    #[test]
    fn test_produces_longer_grams() {
        let grams = collect_sparse_grams(b"self.reset_states(");
        assert!(grams.iter().any(|g| g.len() > 2));
    }

    #[test]
    fn test_max_gram_size_boundary() {
        let grams = collect_sparse_grams(b"abcdefgh");
        for gram in &grams {
            assert!(gram.len() <= MAX_SPARSE_GRAM_SIZE as usize);
        }
    }

    #[test]
    fn test_repeated_bytes() {
        let grams = collect_sparse_grams(b"aaaaaaaaaa");
        assert!(grams.iter().filter(|g| g.len() == 2).count() >= 9);
    }

    #[test]
    fn test_gram_count_scales_linearly() {
        let input: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let grams = collect_sparse_grams(&input);
        assert!(grams.len() >= input.len() - 1);
        assert!(grams.len() <= input.len() * 3);
    }

    // -- Equivalence: scan vs deque --

    #[test]
    fn test_scan_equivalence_small() {
        for input in [b"" as &[u8], b"x", b"ab", b"abc", b"abcdefgh", b"abcdefghi"] {
            assert_eq!(
                collect_to_vec(input, collect_sparse_grams_deque),
                collect_to_vec(input, collect_sparse_grams_scan),
                "mismatch on {:?}",
                std::str::from_utf8(input).unwrap_or("?")
            );
        }
    }

    #[test]
    fn test_scan_equivalence_hello_world() {
        let input = b"hello world";
        assert_eq!(
            collect_to_vec(input, collect_sparse_grams_deque),
            collect_to_vec(input, collect_sparse_grams_scan),
        );
    }

    #[test]
    fn test_scan_equivalence_large() {
        let input: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        assert_eq!(
            collect_to_vec(&input, collect_sparse_grams_deque),
            collect_to_vec(&input, collect_sparse_grams_scan),
        );
    }

    #[test]
    fn test_scan_equivalence_source_code() {
        let input = include_bytes!("lib.rs");
        assert_eq!(
            collect_to_vec(input, collect_sparse_grams_deque),
            collect_to_vec(input, collect_sparse_grams_scan),
        );
    }

    // -- Equivalence: wide vs deque (sorted, order differs) --

    #[test]
    fn test_wide_equivalence_small() {
        for input in [b"" as &[u8], b"x", b"ab", b"abc", b"abcdefgh", b"abcdefghi"] {
            assert_eq!(
                sorted(collect_to_vec(input, collect_sparse_grams_deque)),
                sorted(collect_to_vec(input, collect_sparse_grams_wide)),
                "mismatch on {:?}",
                std::str::from_utf8(input).unwrap_or("?")
            );
        }
    }

    #[test]
    fn test_wide_equivalence_hello_world() {
        assert_eq!(
            sorted(collect_to_vec(b"hello world", collect_sparse_grams_deque)),
            sorted(collect_to_vec(b"hello world", collect_sparse_grams_wide)),
        );
    }

    #[test]
    fn test_wide_equivalence_long_input() {
        let input = b"this is a very long substring where we don't want to construct too long substrings";
        assert_eq!(
            sorted(collect_to_vec(input, collect_sparse_grams_deque)),
            sorted(collect_to_vec(input, collect_sparse_grams_wide)),
        );
    }

    #[test]
    fn test_wide_equivalence_large_diverse() {
        let input: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        assert_eq!(
            sorted(collect_to_vec(&input, collect_sparse_grams_deque)),
            sorted(collect_to_vec(&input, collect_sparse_grams_wide)),
        );
    }

    #[test]
    fn test_wide_equivalence_source_code() {
        let input = include_bytes!("lib.rs");
        assert_eq!(
            sorted(collect_to_vec(input, collect_sparse_grams_deque)),
            sorted(collect_to_vec(input, collect_sparse_grams_wide)),
        );
    }

    // -- Equivalence: masked_avx vs deque (sorted, order differs) --

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_masked_avx_equivalence_small() {
        for input in [b"" as &[u8], b"x", b"ab", b"abc", b"abcdefgh", b"abcdefghi"] {
            assert_eq!(
                sorted(collect_to_vec(input, collect_sparse_grams_deque)),
                sorted(collect_to_vec(input, collect_sparse_grams_masked_avx)),
                "mismatch on {:?}",
                std::str::from_utf8(input).unwrap_or("?")
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_masked_avx_equivalence_large() {
        let input: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        assert_eq!(
            sorted(collect_to_vec(&input, collect_sparse_grams_deque)),
            sorted(collect_to_vec(&input, collect_sparse_grams_masked_avx)),
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_masked_avx_equivalence_source_code() {
        let input = include_bytes!("lib.rs");
        assert_eq!(
            sorted(collect_to_vec(input, collect_sparse_grams_deque)),
            sorted(collect_to_vec(input, collect_sparse_grams_masked_avx)),
        );
    }

    // -- Equivalence: wide_avx vs deque (sorted, order differs) --

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_wide_avx_equivalence_small() {
        for input in [b"" as &[u8], b"x", b"ab", b"abc", b"abcdefgh", b"abcdefghi"] {
            assert_eq!(
                sorted(collect_to_vec(input, collect_sparse_grams_deque)),
                sorted(collect_to_vec(input, collect_sparse_grams_wide_avx)),
                "mismatch on {:?}",
                std::str::from_utf8(input).unwrap_or("?")
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_wide_avx_equivalence_large() {
        let input: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        assert_eq!(
            sorted(collect_to_vec(&input, collect_sparse_grams_deque)),
            sorted(collect_to_vec(&input, collect_sparse_grams_wide_avx)),
        );
    }

    // -- Equivalence: masked vs deque (sorted, order differs) --

    #[test]
    fn test_masked_equivalence_small() {
        for input in [b"" as &[u8], b"x", b"ab", b"abc", b"abcdefgh", b"abcdefghi"] {
            assert_eq!(
                sorted(collect_to_vec(input, collect_sparse_grams_deque)),
                sorted(collect_to_vec(input, collect_sparse_grams_masked)),
                "mismatch on {:?}",
                std::str::from_utf8(input).unwrap_or("?")
            );
        }
    }

    #[test]
    fn test_masked_equivalence_hello_world() {
        assert_eq!(
            sorted(collect_to_vec(b"hello world", collect_sparse_grams_deque)),
            sorted(collect_to_vec(b"hello world", collect_sparse_grams_masked)),
        );
    }

    #[test]
    fn test_masked_equivalence_large() {
        let input: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        assert_eq!(
            sorted(collect_to_vec(&input, collect_sparse_grams_deque)),
            sorted(collect_to_vec(&input, collect_sparse_grams_masked)),
        );
    }

    #[test]
    fn test_masked_equivalence_source_code() {
        let input = include_bytes!("lib.rs");
        assert_eq!(
            sorted(collect_to_vec(input, collect_sparse_grams_deque)),
            sorted(collect_to_vec(input, collect_sparse_grams_masked)),
        );
    }
}

