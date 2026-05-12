//! Core sparse n-gram extraction algorithm.

use crate::deque::{FixedDeque, PosStateBytes};
use crate::ngram::{NGram, POLY_HASH_PRIME, POLY_POWERS};
use crate::table::get_bigram_table;
use crate::MAX_SPARSE_GRAM_SIZE;

/// Returns the maximum number of sparse n-grams that can be produced from
/// `content_len` bytes of input. Use this to pre-allocate the output slice.
#[inline]
pub const fn max_sparse_grams(content_len: usize) -> usize {
    if content_len < 2 {
        0
    } else {
        (content_len - 1) * 3
    }
}

/// Collect all sparse n-grams from the input byte slice into a new [`Vec`].
pub fn collect_sparse_grams(content: &[u8]) -> Vec<NGram> {
    let mut buf = vec![NGram::from_rolling_hash(0, 0); max_sparse_grams(content.len())];
    let count = collect_sparse_grams_deque(content, &mut buf);
    buf.truncate(count);
    buf
}

/// Deque-based extraction. Writes n-grams into `out` (must have at least
/// [`max_sparse_grams`]`(content.len())` slots). Returns the count written.
///
/// # Panics
///
/// Panics if `out` is too small.
pub fn collect_sparse_grams_deque(content: &[u8], out: &mut [NGram]) -> usize {
    let n = content.len();
    if n < 2 {
        return 0;
    }
    assert!(out.len() >= max_sparse_grams(n));
    let table = get_bigram_table();
    let mut queue = FixedDeque::<MAX_SPARSE_GRAM_SIZE>::new();
    let mut prefix_hashes = [0u32; MAX_SPARSE_GRAM_SIZE];
    prefix_hashes[1] = content[0] as u32;
    let mut w = 0usize;

    for idx in 1..n as u32 {
        let mask = MAX_SPARSE_GRAM_SIZE - 1;
        let end_hash = prefix_hashes[idx as usize & mask]
            .wrapping_mul(POLY_HASH_PRIME)
            .wrapping_add(content[idx as usize] as u32);

        // Bigram
        let bigram_hash = end_hash
            .wrapping_sub(prefix_hashes[(idx as usize - 1) & mask].wrapping_mul(POLY_POWERS[2]));
        out[w] = NGram::from_rolling_hash(bigram_hash, 2);
        w += 1;

        let v1 = table[content[idx as usize - 1] as usize * 256 + content[idx as usize] as usize];

        if let Some(begin) = queue.front() {
            if idx - begin.index + 1 >= MAX_SPARSE_GRAM_SIZE as u32 {
                queue.pop_front();
            }
        }
        while let Some(begin) = queue.back() {
            let start = begin.index as usize - 1;
            let len = (idx - begin.index + 2) as usize;
            let hash =
                end_hash.wrapping_sub(prefix_hashes[start & mask].wrapping_mul(POLY_POWERS[len]));
            out[w] = NGram::from_rolling_hash(hash, len);
            w += 1;
            if begin.value == v1 {
                queue.pop_back();
                break;
            } else if begin.value <= v1 {
                break;
            }
            queue.pop_back();
        }
        queue.push_back(PosStateBytes {
            index: idx,
            value: v1,
        });
        prefix_hashes[(idx as usize + 1) & mask] = end_hash;
    }
    w
}

/// Queue-free scan-based extraction. Writes n-grams into `out` (must have at least
/// [`max_sparse_grams`]`(content.len())` slots). Returns the count written.
///
/// Produces identical output (same order) as [`collect_sparse_grams_deque`].
///
/// # Panics
///
/// Panics if `out` is too small.
pub fn collect_sparse_grams_scan(content: &[u8], out: &mut [NGram]) -> usize {
    let n = content.len();
    if n < 2 {
        return 0;
    }
    assert!(out.len() >= max_sparse_grams(n));

    let table = get_bigram_table();
    const MASK: usize = MAX_SPARSE_GRAM_SIZE - 1;
    let mut w = 0usize;
    let mut prefix_hashes = [0u32; MAX_SPARSE_GRAM_SIZE];
    prefix_hashes[1] = content[0] as u32;
    let mut priorities = [u16::MAX; MAX_SPARSE_GRAM_SIZE];
    for idx in 1..n as u32 {
        let end_hash = prefix_hashes[idx as usize & MASK]
            .wrapping_mul(POLY_HASH_PRIME)
            .wrapping_add(content[idx as usize] as u32);
        // Bigram
        let bigram_hash = end_hash
            .wrapping_sub(prefix_hashes[(idx as usize - 1) & MASK].wrapping_mul(POLY_POWERS[2]));
        out[w] = NGram::from_rolling_hash(bigram_hash, 2);
        w += 1;
        let v1 = table[content[idx as usize - 1] as usize * 256 + content[idx as usize] as usize];
        priorities[idx as usize & MASK] = v1;
        let mut running_min = u16::MAX;
        for d in 1..=(MAX_SPARSE_GRAM_SIZE as u32 - 2) {
            if d >= idx {
                break;
            }
            let p = idx.wrapping_sub(d) as usize & MASK;
            let v_p = priorities[p];
            if v_p < running_min {
                running_min = v_p;
                let start = p.wrapping_sub(1) & MASK;
                let len = d as usize + 2;
                let hash =
                    end_hash.wrapping_sub(prefix_hashes[start].wrapping_mul(POLY_POWERS[len]));
                out[w] = NGram::from_rolling_hash(hash, len);
                w += 1;
                if v_p <= v1 {
                    break;
                }
            }
        }
        prefix_hashes[(idx as usize + 1) & MASK] = end_hash;
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::get_bigram_table;
    use std::collections::HashSet;

    fn collect_to_vec(content: &[u8], f: fn(&[u8], &mut [NGram]) -> usize) -> Vec<NGram> {
        let mut buf = vec![NGram::from_rolling_hash(0, 0); max_sparse_grams(content.len())];
        let count = f(content, &mut buf);
        buf.truncate(count);
        buf
    }

    /// Brute-force reference implementation.
    ///
    /// Enumerates all substrings of length 2..=MAX_SPARSE_GRAM_SIZE and emits those
    /// where every interior bigram priority is strictly greater than `max(left, right)`
    /// boundary bigram priority. All bigrams (len=2) are always emitted.
    fn brute_force_sparse_grams(content: &[u8]) -> HashSet<NGram> {
        let table = get_bigram_table();
        let n = content.len();
        let mut result = HashSet::new();
        if n < 2 {
            return result;
        }
        // All bigrams.
        for i in 0..n - 1 {
            result.insert(NGram::from_bytes(&content[i..i + 2]));
        }
        // Longer grams: length 3..=MAX_SPARSE_GRAM_SIZE.
        for len in 3..=MAX_SPARSE_GRAM_SIZE {
            'outer: for start in 0..=n.saturating_sub(len) {
                if start + len > n {
                    break;
                }
                let left = table[content[start] as usize * 256 + content[start + 1] as usize];
                let right = table
                    [content[start + len - 2] as usize * 256 + content[start + len - 1] as usize];
                let boundary = left.max(right);
                // Inner bigrams: bytes [start+1,start+2], ..., [start+len-3,start+len-2]
                for k in 1..len - 2 {
                    let p =
                        table[content[start + k] as usize * 256 + content[start + k + 1] as usize];
                    if p <= boundary {
                        continue 'outer;
                    }
                }
                result.insert(NGram::from_bytes(&content[start..start + len]));
            }
        }
        result
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
            assert!(
                gram.len() <= MAX_SPARSE_GRAM_SIZE,
                "gram too long: {gram:?}"
            );
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
            assert!(gram.len() <= MAX_SPARSE_GRAM_SIZE);
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

    // -- Brute-force equivalence --

    fn assert_matches_brute_force(input: &[u8]) {
        let grams = collect_sparse_grams(input);
        let actual: HashSet<NGram> = grams.into_iter().collect();
        let expected = brute_force_sparse_grams(input);
        let only_actual: Vec<_> = actual.difference(&expected).collect();
        let only_expected: Vec<_> = expected.difference(&actual).collect();
        if !only_actual.is_empty() || !only_expected.is_empty() {
            panic!(
                "mismatch on input len={}\n  only in algorithm: {:?}\n  only in brute force: {:?}",
                input.len(),
                only_actual,
                only_expected
            );
        }
    }

    #[test]
    fn test_brute_force_small() {
        for input in [
            b"" as &[u8],
            b"x",
            b"ab",
            b"abc",
            b"abcd",
            b"abcdefgh",
            b"abcdefghi",
        ] {
            assert_matches_brute_force(input);
        }
    }

    #[test]
    fn test_brute_force_hello_world() {
        assert_matches_brute_force(b"hello world");
    }

    #[test]
    fn test_brute_force_repeated() {
        assert_matches_brute_force(b"aaaaaaaaaa");
    }

    #[test]
    fn test_brute_force_code_snippet() {
        assert_matches_brute_force(b"self.reset_states(the_quick_brown_fox_jumps");
    }

    #[test]
    fn test_brute_force_diverse() {
        let input: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        assert_matches_brute_force(&input);
    }

    #[test]
    fn test_brute_force_source_code() {
        let input = include_bytes!("lib.rs");
        assert_matches_brute_force(input);
    }
}
