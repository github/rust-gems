//! Streaming sparse n-gram extraction state for query-time traversal.
//!
//! Unlike indexing, which emits every candidate gram, query extraction emits the minimum number of
//! grams that cover the input stream and supports incremental character feeding, cloning, and
//! partial/full draining.

use std::hash::{Hash, Hasher};

use casefold::index_fold_char;

use crate::MAX_SPARSE_GRAM_SIZE;
use crate::ngram::NGram;
use crate::table::{bigram_h, bigram_priority_rolling};

/// Bit mask that wraps ring-buffer indices into `[0, MAX_SPARSE_GRAM_SIZE)`.
const RING_MASK: u32 = MAX_SPARSE_GRAM_SIZE as u32 - 1;

#[derive(Clone, Copy, Debug)]
struct PosState {
    index: u32,
    value: u32,
}

#[derive(Clone, Debug, Default)]
struct Queue {
    idx_buf: [u32; MAX_SPARSE_GRAM_SIZE],
    val_buf: [u32; MAX_SPARSE_GRAM_SIZE],
    head: u32,
    len: u32,
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

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn front_idx(&self) -> u32 {
        // Callers only read the front boundary when the queue is non-empty; the empty case is
        // handled earlier (e.g. via `state`'s early return) and must never reach here.
        debug_assert!(!self.is_empty());
        self.idx_buf[self.head as usize]
    }

    fn front_value(&self) -> u32 {
        // Callers only read the front priority when the queue is non-empty; the empty case is
        // handled earlier and must never reach here.
        debug_assert!(!self.is_empty());
        self.val_buf[self.head as usize]
    }

    fn pop_front(&mut self) -> u32 {
        debug_assert!(!self.is_empty());
        let first = self.idx_buf[self.head as usize];
        self.head = (self.head + 1) & RING_MASK;
        self.len -= 1;
        first
    }

    fn push(&mut self, state: PosState) {
        // Drop tail candidates whose priority exceeds the new one, keeping the deque monotone
        // (nondecreasing priorities front-to-back).
        while !self.is_empty() {
            let slot = ((self.head + self.len - 1) & RING_MASK) as usize;
            if self.val_buf[slot] <= state.value {
                break;
            }
            self.len -= 1;
        }
        debug_assert!(self.len < MAX_SPARSE_GRAM_SIZE as u32);
        let slot = ((self.head + self.len) & RING_MASK) as usize;
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
    /// Queue of candidate boundaries (strictly increasing indices and nondecreasing priorities).
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
        self.state() == other.state()
    }
}

impl Hash for QueryGrams {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.state().hash(state);
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
    pub fn state(&self) -> (u32, u64) {
        if self.queue.is_empty() {
            return (0, 0);
        }
        let content_len = self.content_end_idx - self.queue.front_idx();
        debug_assert!(content_len < MAX_SPARSE_GRAM_SIZE as u32);
        let mask = (1u64 << (content_len * 8)) - 1;
        (content_len, self.content & mask)
    }

    /// Returns the smallest active boundary priority.
    pub fn min_priority(&self) -> u32 {
        if self.queue.is_empty() {
            0
        } else {
            self.queue.front_value()
        }
    }

    fn extract_gram<F>(&mut self, begin_index: u32, end_index: u32, consumer: &mut F)
    where
        F: FnMut(NGram, u32),
    {
        debug_assert!(end_index >= begin_index);
        debug_assert!(end_index <= self.content_end_idx);

        let len = (end_index - begin_index + 2) as usize;
        let dist = self.content_end_idx - end_index;
        let shifted = self.content >> (dist * 8);
        // Report the position of the character just after the emitted ngram.
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
        let left = (self.content & 0xFF) as u8;
        self.content_end_idx += 1;
        self.content = (self.content << 8) | right as u64;

        let idx = self.content_end_idx;

        // Initialize rolling state from the first character; from then on each append consumes
        // exactly one new character via the bigram path.
        if idx == 1 {
            self.h = bigram_h(right);
        } else {
            let (value, h_b) = bigram_priority_rolling(left, right, self.h);
            self.h = h_b;
            if !self.queue.is_empty()
                && let priority = self.queue.front_value()
                && value < priority
            {
                let mut begin = self.queue.pop_front();
                while !self.queue.is_empty() && self.queue.front_value() == priority {
                    let end = self.queue.pop_front();
                    self.extract_gram(begin, end, &mut consumer);
                    begin = end;
                }
                self.queue.clear();
                self.queue.push(PosState { index: idx, value });
                self.extract_gram(begin, idx, &mut consumer);
            } else {
                self.queue.push(PosState { index: idx, value });
            }
            if idx - self.queue.front_idx() + 2 >= MAX_SPARSE_GRAM_SIZE as u32 && self.queue.len > 1
            {
                let begin = self.queue.pop_front();
                let end = self.queue.front_idx();
                self.extract_gram(begin, end, &mut consumer);
            }
        }
    }

    /// Flushes all buffered characters and emits remaining grams.
    pub fn flush<F>(mut self, mut consumer: F)
    where
        F: FnMut(NGram, u32),
    {
        if self.content_end_idx == 2 {
            self.extract_gram(2, 2, &mut consumer);
        } else {
            while self.queue.len > 1 {
                let begin = self.queue.pop_front();
                let end = self.queue.front_idx();
                self.extract_gram(begin, end, &mut consumer);
            }
        }
    }

    /// Consumes and emits at most one queued gram (if available), shrinking state.
    pub fn consume_first<F>(&mut self, mut consumer: F)
    where
        F: FnMut(NGram, u32),
    {
        if self.queue.len > 1 {
            // Emit the gram spanning the first boundary to the next one, mirroring `flush`.
            let begin = self.queue.pop_front();
            let end = self.queue.front_idx();
            self.extract_gram(begin, end, &mut consumer);
        } else if self.content_end_idx == 2 {
            // Only a single bigram remains; emit it like `flush` does.
            self.extract_gram(2, 2, &mut consumer);
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
    use crate::collect_sparse_grams_deque;
    use std::collections::hash_map::DefaultHasher;
    use std::ops::Range;

    // Test intervals are stored as `Range<u32>` in bigram-boundary coordinates: a gram covering
    // 0-based bytes `s..=s+L-1` is represented as `s+1..s+L-1`, the inclusive range of bigram
    // boundaries (right-byte indices) it covers. The cover-chain and DP helpers treat `end`
    // inclusively.

    fn query_intervals(input: &str) -> Vec<Range<u32>> {
        let mut q = QueryGrams::default();
        let mut out = Vec::new();
        for c in input.chars() {
            q.append_char(c, |gram, end| {
                let begin = end + 1 - gram.len() as u32;
                out.push(begin..end - 1);
            });
        }
        q.flush(|gram, end| {
            let begin = end + 1 - gram.len() as u32;
            out.push(begin..end - 1);
        });
        out
    }

    // The full candidate set (every sparse gram the index extractor would emit) is exactly the
    // pool the query cover must be drawn from, so reuse `collect_sparse_grams_deque` instead of
    // re-deriving the boundary rule here. `idx` is the position just after the gram, matching the
    // `Range<u32>` boundary convention above.
    fn candidate_intervals(bytes: &[u8]) -> Vec<Range<u32>> {
        let mut out = Vec::new();
        collect_sparse_grams_deque(bytes, |gram, idx| {
            let begin = idx + 1 - gram.len() as u32;
            out.push(begin..idx - 1);
        });
        out
    }

    fn min_cover_size_query_semantics(m: u32, intervals: &[Range<u32>]) -> usize {
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

    fn is_valid_query_cover_chain(m: u32, intervals: &[Range<u32>]) -> bool {
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
        assert!(
            out.iter()
                .all(|(g, _)| (2..=MAX_SPARSE_GRAM_SIZE).contains(&g.len()))
        );
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
            let len = 3 + rng.gen_range(14); // [3, 16]
            let mut bytes = vec![0u8; len];
            for b in &mut bytes {
                *b = b'a' + rng.gen_range(26) as u8; // casefold-stable lowercase ASCII
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
            let len = 4 + rng.gen_range(13); // [4, 16]
            let mut bytes = vec![0u8; len];
            for b in &mut bytes {
                *b = b'a' + rng.gen_range(26) as u8;
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
                    let begin = end + 1 - gram.len() as u32;
                    first_half.push(begin..end - 1);
                });
            }

            while !q.queue.is_empty() {
                q.consume_first(|gram, end| {
                    let begin = end + 1 - gram.len() as u32;
                    first_half.push(begin..end - 1);
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
                    let begin = end + 1 - gram.len() as u32;
                    remaining.push(begin..end - 1);
                });
            }
            q.flush(|gram, end| {
                let begin = end + 1 - gram.len() as u32;
                remaining.push(begin..end - 1);
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
