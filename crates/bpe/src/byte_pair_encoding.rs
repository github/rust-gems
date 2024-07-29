use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::time::Instant;

use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder};
use fnv::{FnvHashMap, FnvHasher};
use itertools::Itertools;
use once_cell::sync::Lazy;
use serde::de::Visitor;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tiktoken_rs::CoreBPE;

use crate::backtrack_encoder::BacktrackEncoder;
use crate::bitfield::BitField;

static BPE_CL100K: Lazy<BytePairEncoding> = Lazy::new(|| {
    let bytes = include_bytes!("data/bpe_cl100k.dict");
    rmp_serde::from_slice(bytes).expect("")
});

/// Representation of the byte pair dictionary.
/// This struct provides various conversions.
/// We put all of them into a single struct so that they can be reused by different implementations.
#[derive(Serialize, Deserialize)]
pub struct BytePairEncoding {
    /// All the decoded tokens concatenated into
    all_tokens: Vec<u8>,
    /// Start index of each token in all_tokens.
    /// The end is simply the next entry in this vector.
    token_starts: Vec<u32>,
    /// Mapping from hash of token to token id.
    bytes_hash_to_token: FnvHashMap<u32, u32>,
    /// The two tokens from which the token got merged.
    /// If the token is an original one, than the two tokens point back to itself.
    split_table: Vec<(u32, u32)>,
    /// Mapping from a pair of tokens to a merged token if such a merged token exists.
    pair_lookup: FnvHashMap<(u32, u32), u32>,
    /// An aho corasick automaton to find the next longest token in a byte sequence.
    #[serde(
        serialize_with = "serialize_daac",
        deserialize_with = "deserialize_daac"
    )]
    longest_searcher: DoubleArrayAhoCorasick<u32>,
    /// An aho corasick automaton to find ALL tokens in a byte sequence.
    #[serde(
        serialize_with = "serialize_daac",
        deserialize_with = "deserialize_daac"
    )]
    pub(crate) overlapping_searcher: DoubleArrayAhoCorasick<u32>,
    /// Mapping from a token to the next longest prefix token.
    /// This is in principle information represented by the AhoCorasick automaton.
    /// But we don't have efficient access to it and therefore store it here again.
    /// If there is none, then the value is set to u32::MAX.
    next_prefix_match: Vec<u32>,
}

fn serialize_daac<S: Serializer>(
    daac: &DoubleArrayAhoCorasick<u32>,
    s: S,
) -> Result<S::Ok, S::Error> {
    s.serialize_bytes(&daac.serialize())
}

struct DaacVisitor;
impl<'de> Visitor<'de> for DaacVisitor {
    type Value = DoubleArrayAhoCorasick<u32>;

    fn expecting(&self, _formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        Err(std::fmt::Error)
    }

    fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<Self::Value, E> {
        Ok(unsafe { DoubleArrayAhoCorasick::deserialize_unchecked(v).0 })
    }
}

fn deserialize_daac<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<DoubleArrayAhoCorasick<u32>, D::Error> {
    d.deserialize_bytes(DaacVisitor)
}

fn token_iter<'a>(all_tokens: &'a [u8], token_starts: &'a [u32]) -> impl Iterator<Item = &'a [u8]> {
    token_starts
        .iter()
        .tuple_windows()
        .map(|(start, end)| &all_tokens[*start as usize..*end as usize])
}

fn next_match(longest_searcher: &DoubleArrayAhoCorasick<u32>, text: &[u8]) -> Option<u32> {
    longest_searcher
        .leftmost_find_iter(text)
        .map(|m| m.value())
        .next()
}

fn is_valid_token_pair(
    pair_lookup: &FnvHashMap<(u32, u32), u32>,
    split_table: &[(u32, u32)],
    mut token1: u32,
    mut token2: u32,
) -> bool {
    // Keep track of the maximum token which can still be chosen across the split point.
    let mut limit = u32::MAX;
    loop {
        // Check whether BPE would choose a different token pair across the split point.
        if let Some(combined) = pair_lookup.get(&(token1, token2)) {
            if *combined < limit {
                return false;
            }
        }
        // Reverse the merge operation from BPE.
        if token1 > token2 {
            limit = token1;
            token1 = unsafe { split_table.get_unchecked(token1 as usize).1 };
            if token1 == limit {
                limit = token2 + 1;
                token2 = unsafe { split_table.get_unchecked(token2 as usize).0 };
                if token2 + 1 == limit {
                    return true;
                }
            }
        } else {
            limit = token2 + 1;
            token2 = unsafe { split_table.get_unchecked(token2 as usize).0 };
            if token2 + 1 == limit {
                limit = token1;
                token1 = unsafe { split_table.get_unchecked(token1 as usize).1 };
                if token1 == limit {
                    return true;
                }
            }
        }
    }
}

fn token_range(token_starts: &[u32], token_id: u32) -> Range<usize> {
    unsafe {
        *token_starts.get_unchecked(token_id as usize) as usize
            ..*token_starts.get_unchecked(token_id as usize + 1) as usize
    }
}

fn token_bytes<'a>(all_tokens: &'a [u8], token_starts: &[u32], token_id: u32) -> &'a [u8] {
    &all_tokens[token_range(token_starts, token_id)]
}

fn hash_bytes(bytes: &[u8]) -> u32 {
    let mut hasher = FnvHasher::default();
    bytes.hash(&mut hasher);
    // Note: we save 1/3 of space for the hashmap by only using the most significant bits of the hash.
    // To make them unique for the given tokens, we have to add unfortunately another multiplication.
    ((hasher.finish().wrapping_mul(37493864257)) >> 32) as u32
}

fn find_token_by_bytes(
    all_tokens: &[u8],
    token_starts: &[u32],
    bytes_hash_to_token: &FnvHashMap<u32, u32>,
    bytes: &[u8],
) -> Option<u32> {
    let hash = hash_bytes(bytes);
    let token = *bytes_hash_to_token.get(&hash)?;
    if token_bytes(all_tokens, token_starts, token) == bytes {
        Some(token)
    } else {
        None
    }
}

impl BytePairEncoding {
    pub fn cl100k() -> &'static Self {
        &BPE_CL100K
    }

    pub fn from_tiktoken(tiktoken_bpe: &CoreBPE, num_tokens: usize) -> Self {
        let start = Instant::now();
        println!("loaded tiktoken: {:?}", start.elapsed());
        let mut all_tokens = Vec::new();
        let mut token_starts = vec![0];
        let mut bytes_hash_to_token = FnvHashMap::default();
        for i in 0..num_tokens {
            let token = tiktoken_bpe._decode_native(&[i]);
            bytes_hash_to_token.insert(hash_bytes(&token), i as u32);
            all_tokens.extend(token);
            token_starts.push(all_tokens.len() as u32);
        }
        assert_eq!(bytes_hash_to_token.len() + 1, token_starts.len());
        println!("copied tokens: {:?}", start.elapsed());

        let longest_searcher = DoubleArrayAhoCorasickBuilder::new()
            .match_kind(daachorse::MatchKind::LeftmostLongest)
            .build(token_iter(&all_tokens, &token_starts))
            .expect("failed to build AhoCorasick");
        println!("constructed longest searcher: {:?}", start.elapsed());

        let overlapping_searcher =
            DoubleArrayAhoCorasick::<u32>::new(token_iter(&all_tokens, &token_starts)).expect("");
        println!("constructed overlapping searcher: {:?}", start.elapsed());

        let next_prefix_match: Vec<_> = token_iter(&all_tokens, &token_starts)
            .map(|token| {
                next_match(&longest_searcher, &token[0..token.len() - 1]).unwrap_or(u32::MAX)
            })
            .collect();
        println!("constructed next_prefix_match: {:?}", start.elapsed());

        let mut split_table = vec![];
        let mut pair_lookup = FnvHashMap::default();
        // Reverse engineer the merge/split table.
        for (id, token) in token_iter(&all_tokens, &token_starts).enumerate() {
            let mut token1 = next_prefix_match[id];
            while token1 != u32::MAX {
                let rest = &token[token_range(&token_starts, token1).len()..];
                if let Some(token2) =
                    find_token_by_bytes(&all_tokens, &token_starts, &bytes_hash_to_token, rest)
                {
                    if token1 < id as u32
                        && token2 < id as u32
                        && is_valid_token_pair(&pair_lookup, &split_table, token1, token2)
                    {
                        pair_lookup.insert((token1, token2), id as u32);
                        split_table.push((token1, token2));
                        break;
                    }
                }
                token1 = next_prefix_match[token1 as usize];
            }
            if token1 == u32::MAX {
                split_table.push((id as u32, id as u32));
            }
        }
        println!("constructed split table: {:?}", start.elapsed());

        Self {
            all_tokens,
            token_starts,
            bytes_hash_to_token,
            overlapping_searcher,
            longest_searcher,
            next_prefix_match,
            pair_lookup,
            split_table,
        }
    }

    /// Return the number of tokens in this BPE dictionary.
    pub fn num_tokens(&self) -> usize {
        self.token_starts.len() - 1
    }

    /// Converts a token id into its corresponding token bytes.
    /// Panics if the token_id is not within the valid 0..num_tokens() range!
    pub fn token_bytes(&self, token_id: u32) -> &[u8] {
        token_bytes(&self.all_tokens, &self.token_starts, token_id)
    }

    pub(crate) fn is_valid_token_pair(&self, token1: u32, token2: u32) -> bool {
        is_valid_token_pair(&self.pair_lookup, &self.split_table, token1, token2)
    }

    /// Returns the length of the decoded byte slice of a token.
    pub fn token_len(&self, token_id: u32) -> usize {
        token_range(&self.token_starts, token_id).len()
    }

    /// Returns the first longest match in the provided text.
    pub(crate) fn next_match(&self, text: &[u8]) -> Option<u32> {
        next_match(&self.longest_searcher, text)
    }

    /// Returns the next token which shares the longest prefix with the specified token.
    pub(crate) fn next_prefix(&self, token_id: u32) -> Option<u32> {
        let prefix = self.next_prefix_match[token_id as usize];
        if prefix == u32::MAX {
            None
        } else {
            Some(prefix)
        }
    }

    fn find_token_by_bytes(&self, bytes: &[u8]) -> Option<u32> {
        find_token_by_bytes(
            &self.all_tokens,
            &self.token_starts,
            &self.bytes_hash_to_token,
            bytes,
        )
    }

    /// Decode a sequence of tokens back to its original byte sequence.
    /// Note: we don't return here a str, since not every token sequence corresponds to a valid
    /// utf8 sequence.
    pub fn decode_tokens(&self, tokens: &[u32]) -> Vec<u8> {
        let mut text = vec![];
        for token in tokens {
            text.extend(self.token_bytes(*token));
        }
        text
    }

    /// Computes for every prefix of the input text a corresponding last token.
    pub(crate) fn encode_all_prefixes(&self, text: &[u8]) -> Vec<u32> {
        let mut last_token = Vec::with_capacity(text.len());
        let mut stepper = self.overlapping_searcher.overlapping_stepper();
        for c in text {
            stepper.consume(*c);
            while let Some(m) = stepper.next() {
                let new_token = m.value();
                let new_range = m.start()..m.end();
                assert_eq!(new_range.end, last_token.len() + 1);
                if new_range.start == 0 {
                    last_token.push(new_token);
                    break;
                } else {
                    let prev_token = unsafe { *last_token.get_unchecked(new_range.start - 1) };
                    if self.is_valid_token_pair(prev_token, new_token) {
                        last_token.push(new_token);
                        break;
                    }
                }
            }
        }
        last_token
    }

    pub fn count(&self, text: &[u8]) -> usize {
        let mut enc = BacktrackEncoder::new(self, text);
        while enc.step().is_some() {}
        enc.count()
    }

    pub fn encode_via_table(&self, text: &[u8]) -> Vec<u32> {
        let last_token = self.encode_all_prefixes(text);
        let mut encoded = Vec::with_capacity(text.len() / 3);
        let mut pos = text.len();
        while pos > 0 {
            let token = last_token[pos - 1];
            encoded.push(token);
            pos -= self.token_len(token);
        }
        encoded.reverse();
        encoded
    }

    pub fn encode_via_backtracking(&self, text: &[u8]) -> Vec<u32> {
        let mut enc = BacktrackEncoder::new(self, text);
        while enc.step().is_some() {}
        enc.into_tokens()
    }

    fn encode_into_bitfield(&self, bytes: &[u8]) -> (BitField, usize) {
        // Reserve for every byte a bit in the bitfield.
        let mut bitfield = BitField::new(bytes.len() + 1);
        let mut heap = BinaryHeap::with_capacity(bytes.len() * 2);
        heap.extend((0..bytes.len().saturating_sub(1)).filter_map(|i| {
            self.find_token_by_bytes(&bytes[i..i + 2])
                .map(|e| Reverse((e, i as u32)))
        }));
        let mut count = bytes.len();
        while let Some(Reverse((token, start))) = heap.pop() {
            let start = start as usize;
            if !bitfield.is_set(start) {
                continue;
            }
            let mid = bitfield.successor(start + 1);
            if mid >= bytes.len() {
                continue;
            }
            let end = bitfield.successor(mid + 1);
            if self.token_len(token) != end - start {
                continue;
            }
            bitfield.clear(mid);
            count -= 1;
            if end < bytes.len() {
                let new_end = bitfield.successor(end + 1);
                if let Some(e) = self.find_token_by_bytes(&bytes[start..new_end]) {
                    heap.push(Reverse((e, start as u32)));
                }
            }
            if start > 0 {
                let new_start = bitfield.predecessor(start - 1);
                if let Some(e) = self.find_token_by_bytes(&bytes[new_start..end]) {
                    heap.push(Reverse((e, new_start as u32)));
                }
            }
        }
        (bitfield, count)
    }

    fn bitfield_into_tokens(&self, bytes: &[u8], bitfield: BitField, count: usize) -> Vec<u32> {
        let mut encoded = Vec::with_capacity(count);
        let mut start = 0;
        while start < bytes.len() {
            let end = bitfield.successor(start + 1);
            let token = self.find_token_by_bytes(&bytes[start..end]).expect("");
            encoded.push(token);
            start = end;
        }
        encoded
    }

    pub fn encode_via_bitfield(&self, text: &[u8]) -> Vec<u32> {
        let (bitfield, count) = self.encode_into_bitfield(text);
        self.bitfield_into_tokens(text, bitfield, count)
    }

    /// It is not recommended to use this function, since it doesn't output the correct BPE encoded sequence.
    pub fn encode_greedy(&self, text: &[u8]) -> Vec<u32> {
        self.longest_searcher
            .leftmost_find_iter(text)
            .map(|m| m.value())
            .collect()
    }

    /// This function computes the shortest possible encoding sequence which will usually differ from the
    /// tokenization produced by the original BPE algorithm.
    pub fn encode_minimal(&self, text: &[u8]) -> Vec<u32> {
        let mut last_token: Vec<(u32, u32)> = Vec::with_capacity(text.len());
        let mut stepper = self.overlapping_searcher.overlapping_stepper();
        for c in text {
            stepper.consume(*c);
            let mut best = (0, u32::MAX);
            while let Some(m) = stepper.next() {
                if m.start() == 0 {
                    best = (m.value(), 1);
                    break;
                } else if last_token[m.start() - 1].1 + 1 < best.1 {
                    best = (m.value(), last_token[m.start() - 1].1 + 1)
                }
            }
            last_token.push(best);
        }
        let mut encoded = Vec::with_capacity(last_token.last().map(|l| l.1 as usize).unwrap_or(0));
        let mut pos = text.len();
        while pos > 0 {
            let token = last_token[pos - 1].0;
            encoded.push(token);
            pos -= self.token_len(token);
        }
        encoded.reverse();
        encoded
    }
}

pub fn create_test_bytes(bpe: &BytePairEncoding, tokens: usize) -> Vec<u8> {
    use rand::{thread_rng, Rng};
    let mut text = vec![];
    for _ in 0..tokens {
        let i = thread_rng().gen_range(0..bpe.num_tokens());
        let s = bpe.token_bytes(i as u32);
        text.extend_from_slice(s);
    }
    text
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::path::PathBuf;
    use std::time::Instant;

    use itertools::Itertools;
    use serde::Serialize;
    use tiktoken_rs::{cl100k_base, cl100k_base_singleton};

    use crate::byte_pair_encoding::{create_test_bytes, BytePairEncoding};

    #[test]
    fn test_correctness() {
        // This is quite a challenging test case...
        let test_string = std::str::from_utf8(&[
            125, 34, 10, 10, 46, 109, 107, 100, 105, 114, 115, 32, 102, 100, 115, 32, 97, 100, 105,
            112, 105, 115, 105, 99, 105, 110, 103, 105, 116, 121, 69, 110, 103, 105, 110, 101, 32,
            69, 67, 105, 114, 105, 101, 32, 111, 112, 116, 105, 109, 97, 108, 95, 68, 65, 32, 111,
            102, 102, 101, 110, 100,
        ])
        .unwrap();
        let time = Instant::now();
        let bpe = BytePairEncoding::cl100k();
        println!("{:?}", time.elapsed());
        let encoded1 = cl100k_base_singleton()
            .lock()
            .encode_ordinary(test_string)
            .into_iter()
            .map(|t| t as u32)
            .collect_vec();
        let encoded2 = bpe.encode_via_backtracking(test_string.as_bytes());
        assert_eq!(encoded1, encoded2);
        let encoded3 = bpe.encode_via_table(test_string.as_bytes());
        assert_eq!(encoded1, encoded3);
        let encoded4 = bpe.encode_via_bitfield(test_string.as_bytes());
        assert_eq!(encoded1, encoded4);
    }

    #[test]
    fn test_bpe_equivalence() {
        let bpe = BytePairEncoding::cl100k();
        for tokens in [10, 1000, 10000] {
            for _ in 0..5 {
                let test_input = create_test_bytes(bpe, tokens);
                let encoded1 = bpe.encode_via_backtracking(&test_input);
                let encoded2 = bpe.encode_via_bitfield(&test_input);
                assert_eq!(encoded1, encoded2, "{} {}", encoded1.len(), encoded2.len());
            }
        }
    }

    // TODO: Move the generation of the dictionary into some build procedure?
    #[test]
    fn test_serialize() {
        let path = PathBuf::from(file!());
        let dir = path.parent().unwrap();
        let data_file = dir.join("data/bpe_cl100k.dict");
        let current_dir = std::env::current_dir().unwrap();
        let abs_path = current_dir.parent().unwrap().parent().unwrap();
        let file = File::create(abs_path.join(data_file)).unwrap();
        let mut serializer = rmp_serde::Serializer::new(file);
        let cl100_dict = cl100k_base().expect("tiktoken initialization must not fail!");
        BytePairEncoding::from_tiktoken(&cl100_dict, 100256)
            .serialize(&mut serializer)
            .unwrap();
    }
}