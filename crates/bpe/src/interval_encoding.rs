use std::ops::Range;

use crate::backtrack_encoder::BacktrackEncoder;
use crate::byte_pair_encoding::BytePairEncoding;

/// This data structure allows fast, i.e. O(1), counting of tokens for arbitrary substrings of the original input text.
/// It achieves this by precomputing for every position the last token which ends at this position.
/// These last tokens represent a token tree with its root being the empty input text where each path starting at the root represents
/// the encoded tokens of the corresponding text prefix.
/// The struct stores a topological ordering in `tree_id` over this tree which then enables O(1) testing whether one node
/// is the predecessor of another node.
/// With the `tree_depth` field the number of path length (which is equivalent to the number of encoded tokens) can be determined
/// in O(1) as well.
///
/// Note: the fields `tree_end` and `tree_depth` could also be represented by succinct data structures, reducing their size drastically.
/// Since we still need the `tree_id` and `last_token` fields, this would in total reduce memory footprint by a bit less than 50%.
pub struct IntervalEncoding<'a> {
    bpe: &'a BytePairEncoding,
    text: &'a [u8],
    last_token: Vec<u32>,
    /// index 0 is reserved as root of the tree and corresponds to the empty prefix.
    tree_id: Vec<u32>,
    tree_end: Vec<u32>,
    tree_depth: Vec<u32>,
}

impl<'a> IntervalEncoding<'a> {
    pub fn new(bpe: &'a BytePairEncoding, text: &'a [u8]) -> Self {
        let last_token = bpe.encode_all_prefixes(text);
        let mut tree_size = vec![1; text.len() + 1];
        for (id, token) in last_token.iter().copied().enumerate().rev() {
            let id = id + 1;
            tree_size[id - bpe.token_len(token)] += tree_size[id];
        }
        let mut tree_end = vec![1];
        let mut tree_id = vec![0];
        let mut tree_depth = vec![0];
        for (id, token) in last_token.iter().copied().enumerate() {
            let id = id + 1;
            let parent = id - bpe.token_len(token);
            tree_id.push(tree_end[parent]);
            tree_end.push(tree_end[parent] + 1);
            tree_depth.push(tree_depth[parent] + 1);
            tree_end[parent] += tree_size[id];
        }
        Self {
            bpe,
            text,
            last_token,
            tree_id,
            tree_end,
            tree_depth,
        }
    }

    /// Computes in typically O(1) time the number of tokens required to encode the specified range.
    /// Thereby it reencodes the prefix with the `BacktrackEncoder` until the encoding sequence becomes
    /// compatible with the precomputed tables. Once that's the case, the remainder of the range becomes
    /// a simple O(1) lookup.
    pub fn count(&self, range: Range<usize>) -> usize {
        let leaf = self.tree_id[range.end];
        let mut encoder = BacktrackEncoder::with_capacity(self.bpe, &self.text[range.clone()], 8);
        // TODO: Consider adding a short-cut when the range starts at a good position.
        while let Some(next_token) = encoder.step() {
            if let Some(prev_token) = encoder.last_token() {
                let end_pos = encoder.pos() + range.start;
                // is the position compatible with the chosen leaf node
                // and does next and prev token match?
                if (self.tree_id[end_pos]..self.tree_end[end_pos]).contains(&leaf)
                    && self.last_token[end_pos - 1] == prev_token
                    && self.last_token[end_pos - 1 + self.bpe.token_len(next_token)] == next_token
                {
                    return encoder.count()
                        + (self.tree_depth[range.end] - self.tree_depth[end_pos]) as usize;
                }
            }
        }
        encoder.count()
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use crate::byte_pair_encoding::{create_test_bytes, BytePairEncoding};

    use super::IntervalEncoding;

    #[test]
    fn test_interval_count() {
        let bpe = BytePairEncoding::cl100k();
        let text = create_test_bytes(bpe, 10000);
        let intervals = IntervalEncoding::new(bpe, &text);
        for _ in 0..1000 {
            let start = thread_rng().gen_range(0..text.len());
            let end = thread_rng().gen_range(0..text.len());
            let range = start.min(end)..start.max(end);
            assert_eq!(
                intervals.count(range.clone()),
                bpe.encode_via_backtracking(&text[range]).len()
            );
        }
    }
}
