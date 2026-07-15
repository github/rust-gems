//! Core sparse n-gram extraction algorithm.

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

/// Monotone-deque extraction, with the deque held in fixed ring buffers. Writes n-grams into `out`
/// (must have at least [`max_sparse_grams`]`(content.len())` slots). Returns the count written.
///
/// The boundary candidates form a monotone run, kept in fixed `[_; MAX_SPARSE_GRAM_SIZE]` ring
/// buffers addressed by a single running depth `tail` — there is no head. Rather than dropping the
/// oldest candidate up front, the walk-back simply stops at the first candidate too far back to
/// form a gram of at most `MAX_SPARSE_GRAM_SIZE` bytes; out-of-window candidates are never read and
/// get overwritten by later pushes (the live window spans fewer than `MAX_SPARSE_GRAM_SIZE`
/// candidates, so the ring never loses one it still needs).
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
    const MASK: usize = MAX_SPARSE_GRAM_SIZE - 1;
    // Sentinel index for empty ring slots. It is chosen so the "too-large gram" test in the
    // walk-back (`idx - begin + 1 >= MAX_SPARSE_GRAM_SIZE`) fires for it at every `idx >= 1` — the
    // subtraction wraps to `idx + MAX_SPARSE_GRAM_SIZE + 1` — so the walk terminates at the bottom
    // of the stack purely from the stored values, with no separate emptiness check.
    const EMPTY: u32 = 0u32.wrapping_sub(MAX_SPARSE_GRAM_SIZE as u32);
    // Monotone deque of boundary candidates (position index + priority), strictly increasing in
    // both, held in fixed ring buffers. Only `tail` (the running stack depth) is tracked; a
    // candidate at depth `d` lives in slot `d & MASK`. The live window never spans
    // `MAX_SPARSE_GRAM_SIZE` candidates, so overwriting slot `tail & MASK` only clobbers an
    // out-of-window entry that the walk-back below never reaches.
    let mut idx_buf = [EMPTY; MAX_SPARSE_GRAM_SIZE];
    let mut val_buf = [0u32; MAX_SPARSE_GRAM_SIZE];
    let mut tail = 0usize;

    // The rolling window starts with the first byte; each loop iteration shifts in `content[idx]`.
    let mut window = content[0] as u64;
    // `BIGRAM_H` of the most recent byte, carried between positions so consecutive bigrams (which
    // overlap by one byte) only load one new H value each. Seeded with the first byte's H.
    let mut h = bigram_h(content[0]);
    let mut w = 0usize;

    for idx in 1..n as u32 {
        window = (window << 8) | content[idx as usize] as u64;
        // The bigram for this position is the low two bytes of the rolling window. `h` is the H
        // value of its first byte, carried over from the previous position; `h_b` feeds the next.
        let (value, h_b) = bigram_priority_rolling((window >> 8) as u8, window as u8, h);
        h = h_b;

        // The bigram (length 2) is always emitted.
        out[w] = window_to_gram(window, 2);
        w += 1;

        // Walk back over the candidates from the tail, emitting one gram each. Stop at the first
        // candidate too far back to form a gram of at most `MAX_SPARSE_GRAM_SIZE` bytes (this
        // replaces the front-drop and, via the `EMPTY` sentinel, also terminates at the stack
        // bottom), or one whose priority is < the current bigram's (it stays as the new left
        // boundary). Equal-or-larger priorities are dropped, and the current position overwrites
        // the first dropped slot.
        let mut t = tail;
        loop {
            let slot = t.wrapping_sub(1) & MASK;
            let begin = idx_buf[slot];
            if idx.wrapping_sub(begin) + 1 >= MAX_SPARSE_GRAM_SIZE as u32 {
                break;
            }
            out[w] = window_to_gram(window, (idx.wrapping_sub(begin) + 2) as usize);
            w += 1;
            let bval = val_buf[slot];
            if bval < value {
                break;
            }
            t -= 1;
            if bval == value {
                break;
            }
        }

        // Push the current position by overwriting the first dropped (or next free) slot.
        idx_buf[t & MASK] = idx;
        val_buf[t & MASK] = value;
        tail = t + 1;
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
    let mut window = content[0] as u64;
    let mut h = bigram_h(content[0]);
    // Ring buffer of the most recent bigram priorities, indexed by `idx & MASK`.
    let mut priorities = [0u32; MAX_SPARSE_GRAM_SIZE];
    for idx in 1..n as u32 {
        window = (window << 8) | content[idx as usize] as u64;
        let (v1, h_b) = bigram_priority_rolling((window >> 8) as u8, window as u8, h);
        h = h_b;
        priorities[idx as usize & MASK] = v1;

        // The bigram (length 2) is always emitted.
        out[w] = window_to_gram(window, 2);
        w += 1;

        // Scan backwards, tracking the minimum interior priority seen so far. Each new strict
        // minimum is a boundary candidate; emit its gram while the right boundary `v1` is strictly
        // below the interior minimum, then stop.
        let mut running_min = u32::MAX;
        for d in 1..=(MAX_SPARSE_GRAM_SIZE as u32 - 2) {
            if d >= idx {
                break;
            }
            let v_p = priorities[(idx - d) as usize & MASK];
            if v_p < running_min {
                if running_min <= v1 {
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
    /// Enumerates all substrings of length 2..=MAX_SPARSE_GRAM_SIZE and emits those where both the
    /// left and right boundary bigram priorities are strictly less than every interior priority.
    /// All bigrams (len=2) are always emitted.
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
                if left < min_interior && right < min_interior {
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
        // exercises the both-strict `max(L, R) < interior` rule where deque, scan and brute force
        // must agree (a gram whose right boundary only ties the smallest interior priority is
        // dropped as redundant).
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
    fn test_brute_force_long_ascending() {
        // Long ascending ASCII runs create monotone priority stretches that grow the deque's
        // `tail` counter far beyond the ring size, exercising the `EMPTY` sentinel that lets the
        // walk-back terminate at the stack bottom without a `t > 0` check. Also runs a scan/deque
        // cross-check on the same input.
        let input: Vec<u8> = (0..3000u32).map(|i| 33 + (i % 90) as u8).collect();
        assert_matches_brute_force(&input);
        assert_eq!(
            collect_to_vec(&input, collect_sparse_grams_deque),
            collect_to_vec(&input, collect_sparse_grams_scan),
        );
    }

    #[test]
    fn test_brute_force_source_code() {
        let input = include_bytes!("extract.rs");
        assert_matches_brute_force(input);
    }
}
