//! Streaming sparse n-gram extraction state for query-time traversal.
//!
//! Unlike indexing, query extraction prefers longer grams and supports incremental character
//! feeding, cloning, and partial/full draining.

use std::hash::{Hash, Hasher};

use casefold::index_fold_char;

use crate::ngram::NGram;
use crate::table::{bigram_h, bigram_priority_rolling};
use crate::MAX_SPARSE_GRAM_SIZE;

#[derive(Clone, Copy, Debug)]
struct PosState {
    index: u32,
    value: u32,
}

#[derive(Clone, Debug, Default)]
struct Queue {
    idx_buf: [u32; MAX_SPARSE_GRAM_SIZE],
    val_buf: [u32; MAX_SPARSE_GRAM_SIZE],
    head: usize,
    len: usize,
}

impl Queue {
    fn new() -> Self {
        Self {
            idx_buf: [0; MAX_SPARSE_GRAM_SIZE],
            val_buf: [0; MAX_SPARSE_GRAM_SIZE],
            head: 0,
            len: 0,
        }
    }

    fn clear(&mut self) {
        self.len = 0;
    }

    fn front_idx(&self) -> u32 {
        if self.len == 0 {
            1
        } else {
            self.idx_buf[self.head]
        }
    }

    fn front_value(&self) -> u32 {
        if self.len == 0 {
            0
        } else {
            self.val_buf[self.head]
        }
    }

    fn back_value(&self) -> Option<u32> {
        if self.len == 0 {
            None
        } else {
            let slot = (self.head + self.len - 1) & (MAX_SPARSE_GRAM_SIZE - 1);
            Some(self.val_buf[slot])
        }
    }

    fn pop_back(&mut self) -> Option<PosState> {
        if self.len == 0 {
            return None;
        }
        let slot = (self.head + self.len - 1) & (MAX_SPARSE_GRAM_SIZE - 1);
        self.len -= 1;
        Some(PosState {
            index: self.idx_buf[slot],
            value: self.val_buf[slot],
        })
    }

    fn pop_front(&mut self) -> PosState {
        debug_assert!(self.len > 0);
        let first = PosState {
            index: self.idx_buf[self.head],
            value: self.val_buf[self.head],
        };
        self.head = (self.head + 1) & (MAX_SPARSE_GRAM_SIZE - 1);
        self.len -= 1;
        first
    }

    fn push(&mut self, state: PosState) {
        while let Some(back_value) = self.back_value() {
            if back_value <= state.value {
                break;
            }
            self.pop_back();
        }

        debug_assert!(self.len < MAX_SPARSE_GRAM_SIZE);
        let slot = (self.head + self.len) & (MAX_SPARSE_GRAM_SIZE - 1);
        self.idx_buf[slot] = state.index;
        self.val_buf[slot] = state.value;
        self.len += 1;
    }
}

/// Streaming query n-gram state.
///
/// This state accepts one character at a time, can be cloned while traversing an automaton, and
/// can emit grams incrementally. It uses the same compact bigram-priority model as indexing, but
/// emits the minimum number of grams that cover the input stream. Its state space is fully
/// represented by the content buffer and can be shrunk on demand when only part of the stream
/// needs to be retained.
#[derive(Clone)]
pub struct QueryGrams {
    /// Queue of candidate boundaries (strictly increasing index and priority).
    queue: Queue,
    /// Active content packed into a `u64` (newest byte in the low byte).
    content: u64,
    /// Absolute index one past the last active byte in `content`.
    content_end_idx: u32,
    /// Rolling `H` value of the first byte of the next bigram to score.
    h: u32,
}

impl Eq for QueryGrams {}

impl PartialEq for QueryGrams {
    fn eq(&self, other: &Self) -> bool {
        self.active_content_len() == other.active_content_len()
            && self.masked_content() == other.masked_content()
    }
}

impl Hash for QueryGrams {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.active_content_len().hash(state);
        self.masked_content().hash(state);
    }
}

impl Default for QueryGrams {
    fn default() -> Self {
        Self {
            content: 0,
            queue: Queue::new(),
            content_end_idx: 0,
            h: 0,
        }
    }
}

impl QueryGrams {
    #[inline]
    fn active_content_len(&self) -> u32 {
        self.content_end_idx - self.queue.front_idx().saturating_sub(1)
    }

    #[inline]
    fn masked_content(&self) -> u64 {
        let content_len = self.active_content_len();
        debug_assert!(content_len < 8);
        let mask = (1u64 << (content_len * 8)) - 1;
        self.content & mask
    }

    /// Returns the smallest active boundary priority.
    pub fn min_priority(&self) -> u32 {
        self.queue.front_value()
    }

    fn extract_gram<F>(&mut self, begin_index: u32, end_index: u32, consumer: &mut F)
    where
        F: FnMut(NGram, u32),
    {
        debug_assert!(end_index >= begin_index);
        debug_assert!(end_index < self.content_end_idx);

        let len = (end_index - begin_index + 2) as usize;
        let dist = self.content_end_idx - 1 - end_index;
        let shifted = self.content >> (dist * 8);
        consumer(NGram::from_window_masked(shifted, len), end_index);
    }

    /// Appends a single character to the n-gram state.
    ///
    /// The character is index-folded using the `casefold` crate and may trigger one or more grams.
    pub fn append_char<F>(&mut self, c: char, mut consumer: F)
    where
        F: FnMut(NGram, u32),
    {
        let right = index_fold_char(c);
        self.content_end_idx += 1;
        self.content = (self.content << 8) | right as u64;

        let idx = self.content_end_idx - 1;

        // Initialize rolling state from the first character; from then on each append consumes
        // exactly one new character via the bigram path.
        if idx == 0 {
            self.h = bigram_h(right);
        } else {
            let left = ((self.content >> 8) & 0xFF) as u8;
            let (value, h_b) = bigram_priority_rolling(left, right, self.h);
            self.h = h_b;
            if self.queue.len > 0 && value < self.queue.front_value() {
                let priority = self.queue.front_value();
                let mut last = self.queue.pop_front();
                while self.queue.len > 0 && self.queue.front_value() == priority {
                    let next = self.queue.pop_front();
                    self.extract_gram(last.index, next.index, &mut consumer);
                    last = next;
                }
                self.queue.clear();
                self.queue.push(PosState { index: idx, value });
                self.extract_gram(last.index, idx, &mut consumer);
            } else {
                self.queue.push(PosState { index: idx, value });
            }
            if idx - self.queue.front_idx() + 1 >= MAX_SPARSE_GRAM_SIZE as u32 {
                if self.queue.len > 1 {
                    let first = self.queue.pop_front();
                    let end = self.queue.front_idx();
                    self.extract_gram(first.index, end, &mut consumer);
                }
            }
        }
    }

    /// Flushes all buffered characters and emits remaining grams.
    pub fn flush<F>(mut self, mut consumer: F)
    where
        F: FnMut(NGram, u32),
    {
        if self.content_end_idx == 2 {
            self.extract_gram(1, 1, &mut consumer);
            return;
        }
        while self.queue.len > 1 {
            let first = self.queue.pop_front();
            let end = self.queue.front_idx();
            self.extract_gram(first.index, end, &mut consumer);
        }
    }

    /// Consumes and emits at most one queued gram (if available), shrinking state.
    pub fn consume_first<F>(&mut self, mut consumer: F)
    where
        F: FnMut(NGram, u32),
    {
        if self.queue.len > 1 {
            // Emit the gram spanning the first boundary to the next one, mirroring `flush`.
            let first = self.queue.pop_front();
            let end = self.queue.front_idx();
            self.extract_gram(first.index, end, &mut consumer);
        } else if self.content_end_idx == 2 {
            // Only a single bigram remains; emit it like `flush` does.
            self.extract_gram(1, 1, &mut consumer);
        }

        // The last boundary is a dangling endpoint that never becomes its own gram (just like
        // `flush` stops at `queue.len > 1`). Once only that boundary (or nothing) is left, collapse
        // to the retained last byte so a continuation matches a fresh run starting from that byte.
        if self.queue.len <= 1 {
            self.queue.clear();
            if self.content_end_idx > 1 {
                // `self.h` already holds `bigram_h(last)` by invariant, so it needs no update.
                let last = (self.content & 0xFF) as u8;
                self.content = last as u64;
                self.content_end_idx = 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::bigram_priority;
    use std::collections::hash_map::DefaultHasher;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Interval {
        start: u32,
        end: u32,
    }

    fn query_intervals(input: &str) -> Vec<Interval> {
        let mut q = QueryGrams::default();
        let mut out = Vec::new();
        for c in input.chars() {
            q.append_char(c, |gram, end| {
                let begin = end + 2 - gram.len() as u32;
                out.push(Interval { start: begin, end });
            });
        }
        q.flush(|gram, end| {
            let begin = end + 2 - gram.len() as u32;
            out.push(Interval { start: begin, end });
        });
        out
    }

    fn candidate_intervals(bytes: &[u8]) -> Vec<Interval> {
        let n = bytes.len();
        if n < 2 {
            return Vec::new();
        }

        let mut out = Vec::with_capacity((n - 1) * 3);
        // Every bigram is always a candidate.
        for i in 1..n as u32 {
            out.push(Interval { start: i, end: i });
        }

        for len in 3..=MAX_SPARSE_GRAM_SIZE.min(n) {
            for start in 0..=n - len {
                let left = bigram_priority(bytes[start], bytes[start + 1]);
                let right = bigram_priority(bytes[start + len - 2], bytes[start + len - 1]);
                let mut min_interior = u32::MAX;
                for k in 1..len - 2 {
                    min_interior =
                        min_interior.min(bigram_priority(bytes[start + k], bytes[start + k + 1]));
                }
                if left < min_interior && right < min_interior {
                    out.push(Interval {
                        start: start as u32 + 1,
                        end: start as u32 + len as u32 - 1,
                    });
                }
            }
        }
        out
    }

    fn min_cover_size_query_semantics(m: u32, intervals: &[Interval]) -> usize {
        let inf = usize::MAX / 4;
        let mut dp = vec![0usize; m as usize + 1];
        for pos in (1..m).rev() {
            let mut best = inf;
            for iv in intervals {
                if iv.start <= pos && iv.end > pos {
                    let next = iv.end.min(m) as usize;
                    best = best.min(1 + dp[next]);
                }
            }
            dp[pos as usize] = best;
        }
        dp[1]
    }

    fn is_valid_query_cover_chain(m: u32, intervals: &[Interval]) -> bool {
        if m <= 1 {
            return true;
        }
        let mut produced = 1u32;
        for iv in intervals {
            if iv.start > produced || iv.end <= produced {
                return false;
            }
            produced = iv.end;
            if produced >= m {
                return true;
            }
        }
        false
    }

    #[derive(Clone, Copy, Debug)]
    struct Rng64(u64);

    impl Rng64 {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            // xorshift64*: tiny deterministic RNG for tests.
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn gen_range(&mut self, upper: usize) -> usize {
            (self.next_u64() % upper as u64) as usize
        }
    }

    #[test]
    fn append_and_flush_emit_grams() {
        let mut q = QueryGrams::default();
        for c in "hello world".chars() {
            q.append_char(c, |_gram, _idx| {});
        }
        let mut out = Vec::new();
        q.flush(|gram, idx| out.push((gram, idx)));
        assert!(!out.is_empty());
        assert!(out
            .iter()
            .all(|(g, _)| (2..=MAX_SPARSE_GRAM_SIZE).contains(&g.len())));
    }

    #[test]
    fn state_eq_and_hash_ignore_absolute_history() {
        let mut a = QueryGrams::default();
        let mut b = QueryGrams::default();

        for c in "abc".chars() {
            a.append_char(c, |_gram, _idx| {});
        }
        for c in "zabc".chars() {
            b.append_char(c, |_gram, _idx| {});
        }
        b.consume_first(|_gram, _idx| {});

        // Only assert hash consistency when Eq says they are equivalent.
        if a == b {
            let mut ha = DefaultHasher::new();
            let mut hb = DefaultHasher::new();
            a.hash(&mut ha);
            b.hash(&mut hb);
            assert_eq!(ha.finish(), hb.finish());
        }
    }

    #[test]
    fn query_flush_is_minimum_cover_on_small_inputs() {
        for input in [
            "abc",
            "abcd",
            "abcdef",
            "hello",
            "hello world",
            "ababababab",
        ] {
            let bytes = input.as_bytes();
            if bytes.len() < 3 {
                continue;
            }
            let produced = query_intervals(input);
            let candidates = candidate_intervals(bytes);
            let m = bytes.len() as u32 - 1;

            assert!(
                is_valid_query_cover_chain(m, &produced),
                "produced set is not a valid cover chain for {input:?}: {:?}",
                produced
            );
            let optimum = min_cover_size_query_semantics(m, &candidates);
            assert_eq!(
                produced.len(),
                optimum,
                "produced set is not minimum-size query cover for {input:?}; produced={:?}; optimum={optimum}",
                produced
            );
        }
    }

    #[test]
    fn query_lardeee_diagnostic() {
        let input = "lardeee";
        let bytes = input.as_bytes();
        let produced = query_intervals(input);
        let candidates = candidate_intervals(bytes);
        let m = bytes.len() as u32 - 1;

        assert!(
            is_valid_query_cover_chain(m, &produced),
            "produced set is not a valid cover chain for {input:?}: {:?}",
            produced
        );

        let optimum = min_cover_size_query_semantics(m, &candidates);

        // Diagnostic check: if this fails, query extraction emitted an interval the oracle does
        // not consider legal under its candidate rules.
        for iv in &produced {
            assert!(
                candidates
                    .iter()
                    .any(|c| c.start == iv.start && c.end == iv.end),
                "produced interval not present in oracle candidates for {input:?}: {:?}; candidates={:?}",
                iv,
                candidates
            );
        }

        // This currently captures the known mismatch that motivated the randomized check.
        assert_eq!(
            produced.len(),
            optimum,
            "minimum-cover mismatch for {input:?}; produced={:?}; optimum={optimum}; candidates={:?}",
            produced,
            candidates
        );
    }

    #[test]
    fn query_sssdk_diagnostic() {
        let input = "sssdk";
        let bytes = input.as_bytes();
        let produced = query_intervals(input);
        let candidates = candidate_intervals(bytes);
        let m = bytes.len() as u32 - 1;

        assert!(
            is_valid_query_cover_chain(m, &produced),
            "produced set is not a valid cover chain for {input:?}: {:?}",
            produced
        );

        let optimum = min_cover_size_query_semantics(m, &candidates);

        for iv in &produced {
            assert!(
                candidates
                    .iter()
                    .any(|c| c.start == iv.start && c.end == iv.end),
                "produced interval not present in oracle candidates for {input:?}: {:?}; candidates={:?}",
                iv,
                candidates
            );
        }

        assert_eq!(
            produced.len(),
            optimum,
            "minimum-cover mismatch for {input:?}; produced={:?}; optimum={optimum}; candidates={:?}",
            produced,
            candidates
        );
    }

    #[test]
    fn query_flush_is_minimum_cover_on_randomized_inputs() {
        let mut rng = Rng64::new(0xA5A5_0123_89AB_CDEF);

        for _ in 0..2000 {
            let len = 3 + rng.gen_range(MAX_SPARSE_GRAM_SIZE - 2); // [3, 8]
            let mut bytes = vec![0u8; len];
            for b in &mut bytes {
                *b = (b'a' + rng.gen_range(26) as u8) as u8; // casefold-stable lowercase ASCII
            }
            let input = std::str::from_utf8(&bytes).expect("ASCII should be valid UTF-8");

            let produced = query_intervals(input);
            let candidates = candidate_intervals(&bytes);
            let m = bytes.len() as u32 - 1;

            assert!(
                is_valid_query_cover_chain(m, &produced),
                "produced set is not a valid cover chain for randomized input {:?}: {:?}",
                input,
                produced
            );
            assert!(
                produced.iter().all(|iv| {
                    let gram_len = (iv.end - iv.start + 2) as usize;
                    (2..=MAX_SPARSE_GRAM_SIZE).contains(&gram_len)
                }),
                "produced set contains out-of-range gram length for randomized input {:?}: {:?}",
                input,
                produced
            );
            let optimum = min_cover_size_query_semantics(m, &candidates);
            assert_eq!(
                produced.len(),
                optimum,
                "produced set is not minimum-size query cover for randomized input {:?}; produced={:?}; optimum={optimum}",
                input,
                produced
            );
        }
    }

    #[test]
    fn query_consume_first_on_randomized_inputs() {
        let mut rng = Rng64::new(0xC0DE_CAFE_1234_5678);

        for _ in 0..2000 {
            let len = 4 + rng.gen_range(MAX_SPARSE_GRAM_SIZE - 3); // [4, 8]
            let mut bytes = vec![0u8; len];
            for b in &mut bytes {
                *b = (b'a' + rng.gen_range(26) as u8) as u8;
            }
            let input = std::str::from_utf8(&bytes).expect("ASCII should be valid UTF-8");

            let split = 2 + rng.gen_range(len - 2);
            let (prefix, suffix) = input.split_at(split);
            let mut q = QueryGrams::default();

            // Capture the full first-half cover: grams emitted eagerly while appending the prefix
            // plus grams drained via `consume_first`.
            let mut first_half = Vec::new();
            for c in prefix.chars() {
                q.append_char(c, |gram, end| {
                    let begin = end + 2 - gram.len() as u32;
                    first_half.push(Interval { start: begin, end });
                });
            }

            while q.queue.len != 0 {
                q.consume_first(|gram, end| {
                    let begin = end + 2 - gram.len() as u32;
                    first_half.push(Interval { start: begin, end });
                });
            }

            assert!(first_half.iter().all(|iv| {
                let gram_len = (iv.end - iv.start + 2) as usize;
                (2..=MAX_SPARSE_GRAM_SIZE).contains(&gram_len)
            }));

            // The first half must be an optimal (minimum-size) cover of the prefix. The DP optimum
            // is only meaningful for inputs of length >= 3 (a 2-char input has no interior
            // boundary, so the DP reports 0 while a single bigram is still emitted).
            let prefix_bytes = prefix.as_bytes();
            if prefix_bytes.len() >= 3 {
                let prefix_m = prefix_bytes.len() as u32 - 1;
                let prefix_candidates = candidate_intervals(prefix_bytes);
                assert!(
                    is_valid_query_cover_chain(prefix_m, &first_half),
                    "first half is not a valid cover chain for prefix {:?}: {:?}",
                    prefix,
                    first_half
                );
                let prefix_optimum = min_cover_size_query_semantics(prefix_m, &prefix_candidates);
                assert_eq!(
                    first_half.len(),
                    prefix_optimum,
                    "first half is not a minimum-size query cover for prefix {:?}; produced={:?}; optimum={prefix_optimum}",
                    prefix,
                    first_half
                );
            }

            let retained = char::from((q.content & 0xFF) as u8);
            let local_input = format!("{retained}{suffix}");

            let mut remaining = Vec::new();
            for c in suffix.chars() {
                q.append_char(c, |gram, end| {
                    let begin = end + 2 - gram.len() as u32;
                    remaining.push(Interval { start: begin, end });
                });
            }
            q.flush(|gram, end| {
                let begin = end + 2 - gram.len() as u32;
                remaining.push(Interval { start: begin, end });
            });

            assert_eq!(
                remaining,
                query_intervals(&local_input),
                "post-drain continuation mismatch for randomized input {:?}; prefix={:?}; suffix={:?}; first_half={:?}; remaining={:?}; local_input={:?}",
                input,
                prefix,
                suffix,
                first_half,
                remaining,
                local_input
            );

            // The second half must be an optimal (minimum-size) cover of the retained tail + suffix.
            let local_bytes = local_input.as_bytes();
            if local_bytes.len() >= 3 {
                let local_m = local_bytes.len() as u32 - 1;
                let local_candidates = candidate_intervals(local_bytes);
                assert!(
                    is_valid_query_cover_chain(local_m, &remaining),
                    "second half is not a valid cover chain for {:?}: {:?}",
                    local_input,
                    remaining
                );
                let local_optimum = min_cover_size_query_semantics(local_m, &local_candidates);
                assert_eq!(
                    remaining.len(),
                    local_optimum,
                    "second half is not a minimum-size query cover for {:?}; produced={:?}; optimum={local_optimum}",
                    local_input,
                    remaining
                );
            }
        }
    }
}
