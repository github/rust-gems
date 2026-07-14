//! Core sparse n-gram extraction algorithm.

use crate::deque::{FixedDeque, PosStateBytes};
use crate::ngram::NGram;
use crate::table::{bigram_h, bigram_priority_rolling};
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

/// Builds the `len`-byte gram ending at the current position from the rolling `window` (newest byte
/// in the low byte). The gram is the `len` low bytes of `window`; shifting them into the
/// most-significant bytes gives the big-endian layout [`NGram::from_window`] consumes.
#[inline]
fn window_to_gram(window: u64, len: usize) -> NGram {
    debug_assert!(len <= MAX_SPARSE_GRAM_SIZE);
    NGram::from_window(window << ((MAX_SPARSE_GRAM_SIZE - len) * 8), len)
}

/// Collect all sparse n-grams from the input byte slice into a new [`Vec`].
pub fn collect_sparse_grams(content: &[u8]) -> Vec<NGram> {
    let mut buf = vec![NGram::default(); max_sparse_grams(content.len())];
    let count = collect_sparse_grams_deque(content, &mut buf);
    buf.truncate(count);
    buf
}

/// Deque-based extraction. Writes n-grams into `out` (must have at least
/// [`max_sparse_grams`]`(content.len())` slots). Returns the count written.
///
/// A rolling `u64` holds the last 8 bytes of `content` (newest byte in the low byte). A gram is at
/// most [`MAX_SPARSE_GRAM_SIZE`] bytes and always ends at the current index, so it is exactly the
/// `len` low bytes of the window; `window_to_gram` shifts those up to build it straight from
/// registers, without re-slicing `content`.
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
    let mut queue = FixedDeque::<MAX_SPARSE_GRAM_SIZE>::new();
    // The rolling window starts holding the first two bytes (the first bigram).
    let mut window = ((content[0] as u64) << 8) | content[1] as u64;
    // `BIGRAM_H` of the most recent byte, carried between positions so consecutive bigrams (which
    // overlap by one byte) only load one new H value each. Seeded with the first byte's H.
    let mut h = bigram_h(content[0]);
    let mut w = 0usize;

    for idx in 1..n as u32 {
        if idx >= 2 {
            window = (window << 8) | content[idx as usize] as u64;
        }
        // The bigram for this position is the low two bytes of the rolling window. `h` is the H
        // value of its first byte, carried over from the previous position; `h_b` feeds the next.
        let (value, h_b) = bigram_priority_rolling((window >> 8) as u8, window as u8, h);
        h = h_b;

        // Keep produced grams within `MAX_SPARSE_GRAM_SIZE` bytes by dropping the oldest boundary.
        // To account for the full bigram of the begin state, the delta is incremented by one.
        if let Some(begin) = queue.front() {
            if idx - begin.index + 1 >= MAX_SPARSE_GRAM_SIZE as u32 {
                queue.pop_front();
            }
        }

        // The bigram (length 2) is always emitted.
        out[w] = window_to_gram(window, 2);
        w += 1;

        // Longer grams: one per boundary candidate from the back. We emit each, then stop once the
        // candidate's priority is below the current bigram's (it becomes the new left boundary).
        while let Some(begin) = queue.back() {
            let len = (idx - begin.index + 2) as usize;
            out[w] = window_to_gram(window, len);
            w += 1;
            if begin.value < value {
                break;
            }
            queue.pop_back();
        }
        queue.push_back(PosStateBytes { index: idx, value });
    }
    w
}

/// Queue-free scan-based extraction. Writes n-grams into `out` (must have at least
/// [`max_sparse_grams`]`(content.len())` slots). Returns the count written.
///
/// Produces identical output (same order) as [`collect_sparse_grams_deque`]: the boundary
/// candidates are exactly the positions where a backward scan hits a new suffix minimum, so a
/// fixed-size ring of recent priorities replaces the monotone deque.
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

    const MASK: usize = MAX_SPARSE_GRAM_SIZE - 1;
    let mut w = 0usize;
    let mut window = ((content[0] as u64) << 8) | content[1] as u64;
    let mut h = bigram_h(content[0]);
    // Ring buffer of the most recent bigram priorities, indexed by `idx & MASK`.
    let mut priorities = [0u32; MAX_SPARSE_GRAM_SIZE];
    for idx in 1..n as u32 {
        if idx >= 2 {
            window = (window << 8) | content[idx as usize] as u64;
        }
        let (v1, h_b) = bigram_priority_rolling((window >> 8) as u8, window as u8, h);
        h = h_b;
        priorities[idx as usize & MASK] = v1;

        // The bigram (length 2) is always emitted.
        out[w] = window_to_gram(window, 2);
        w += 1;

        // Scan backwards, tracking the minimum interior priority seen so far. Each new strict
        // minimum is a boundary candidate; emit its gram while the right boundary `v1` does not
        // exceed the interior minimum, then stop.
        let mut running_min = u32::MAX;
        for d in 1..=(MAX_SPARSE_GRAM_SIZE as u32 - 2) {
            if d >= idx {
                break;
            }
            let v_p = priorities[(idx - d) as usize & MASK];
            if v_p < running_min {
                if running_min < v1 {
                    break;
                }
                let len = d as usize + 2;
                out[w] = window_to_gram(window, len);
                w += 1;
                running_min = v_p;
            }
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::bigram_priority;
    use std::collections::HashSet;

    fn collect_to_vec(content: &[u8], f: fn(&[u8], &mut [NGram]) -> usize) -> Vec<NGram> {
        let mut buf = vec![NGram::default(); max_sparse_grams(content.len())];
        let count = f(content, &mut buf);
        buf.truncate(count);
        buf
    }

    /// Brute-force reference implementation.
    ///
    /// Enumerates all substrings of length 2..=MAX_SPARSE_GRAM_SIZE and emits those where the left
    /// boundary bigram priority is strictly less than every interior priority and the right
    /// boundary priority is at most every interior priority. All bigrams (len=2) are always
    /// emitted. The left/right asymmetry mirrors the extraction algorithm, which processes bigrams
    /// left to right and treats the current (right) position as the incoming boundary.
    fn brute_force_sparse_grams(content: &[u8]) -> HashSet<NGram> {
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
            for start in 0..=n.saturating_sub(len) {
                if start + len > n {
                    break;
                }
                let left = bigram_priority(content[start], content[start + 1]);
                let right =
                    bigram_priority(content[start + len - 2], content[start + len - 1]);
                // Interior bigrams: (start+1,start+2), ..., (start+len-3,start+len-2).
                let mut min_interior = u32::MAX;
                for k in 1..len - 2 {
                    let p = bigram_priority(content[start + k], content[start + k + 1]);
                    min_interior = min_interior.min(p);
                }
                if left < min_interior && right <= min_interior {
                    result.insert(NGram::from_bytes(&content[start..start + len]));
                }
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
            assert!(gram.len() <= MAX_SPARSE_GRAM_SIZE, "gram too long: {gram:?}");
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
        let input = include_bytes!("extract.rs");
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
    fn test_brute_force_tie_break() {
        // Repeated bigrams create exact ties between an interior priority and a boundary; this
        // exercises the `L < interior && R <= interior` asymmetry where deque, scan and brute
        // force must agree.
        assert_matches_brute_force(b"ababababab");
        assert_matches_brute_force(b"the the the the");
        assert_matches_brute_force(b"a.b.a.b.a.b.");
    }

    #[test]
    fn test_brute_force_diverse() {
        let input: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        assert_matches_brute_force(&input);
    }

    #[test]
    fn test_brute_force_source_code() {
        let input = include_bytes!("extract.rs");
        assert_matches_brute_force(input);
    }
}
