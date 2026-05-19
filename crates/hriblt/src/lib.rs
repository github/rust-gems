//! This module implements a set reconciliation algorithm using XOR-based hashes.
//!
//! The core algorithm is based on the idea of set similarity sketching where pure hashes are identified
//! by testing whether one of the 5 hash functions would map the candidate value back to the index
//! of the value. Taking advantage of this necessary condition reduces the number of necessary checksum
//! bits by a bit.
//!
//! To make this algorithm rateless, we simply observe that any given IBLT becomes another IBLT if
//! we XOR the hashes of the upper half of the hashes with the lower half.
//! This process can also be realized by mapping one code symbol at a time into the lower half, such
//! that we can generate ANY number of coded symbols. Since this procedure is revertable when the next coded symbol
//! is provided by the client, one ends up with a rateless set reconciliation algorithm.
//!
//! The decoding algorithm essentially reconstructs in each iteration an IBLT with one more coded symbol
//! and then tries to decode the two coded symbols which have been modified by the expansion operation.
//!
//! This algorithm has similar properties to the "Practical Rateless Set Reconciliation" algorithm.
//! Main differences are:
//! * We only use 5 hash functions instead of log(n).
//! * As a result we only require 5 independent hash functions instead of log(n) many.
//! * The amount of data being transferred is comparable to the one in the paper.
//! * The chance that two documents collide on all hash functions is higher, but still very low,
//!   since we utilize 5 hash functions instead of just 3.
//! * There is no complicated math involved. In the paper it is important that the exact same computation
//!   is performed by both sides or the scheme will fall apart. (I.e. any kind of math optimizations must be disabled
//!   and a stable math library must be used!)
//! * Encoding/decoding is faster due to the fixed number of hash functions and the simpler operations.
//! * Since we have a fixed number of hash functions, we can utilize the coded symbol index as
//!   an additional condition. In fact, we need to compute just a single hash function (on average).
mod coded_symbol;
mod decoded_value;
mod decoding_session;
mod encoding_session;
mod error;

use std::{
    fmt::Debug,
    hash::{DefaultHasher, Hash, Hasher},
    ops::BitXorAssign,
};

pub use coded_symbol::CodedSymbol;
pub use decoded_value::DecodedValue;
pub use decoding_session::DecodingSession;
pub use encoding_session::EncodingSession;
pub use error::SetReconciliationError;

/// Computes independent hash functions.
pub trait HashFunctions<T>: Eq + Copy + Debug {
    /// Hashes the given value with the n-th hash function.
    /// This trait should provide 5 independent hash functions!
    fn hash(&self, value: &T, n: u32) -> u32;
    /// Computes a checksum. Note this checksum must become invalid
    /// under xor operations!
    fn check_sum(&self, value: &T) -> u64;
}

/// Hasher builder implementing equality operation of seed value.
#[derive(Default, Debug, Clone, Copy, Eq, PartialEq)]
pub struct DefaultHashFunctions;

impl<T: Hash> HashFunctions<T> for DefaultHashFunctions {
    fn hash(&self, value: &T, n: u32) -> u32 {
        let mut hasher = DefaultHasher::new();
        n.hash(&mut hasher);
        value.hash(&mut hasher);
        hasher.finish() as u32
    }

    fn check_sum(&self, value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }
}

/// Trait for value types that can be used in the set reconciliation algorithm.
pub trait Encodable: Copy + Eq + PartialEq + Hash {
    /// Returns a zero value for this type which is usually the default value.
    fn zero() -> Self;
    /// Xor the current value with another value of the same type.
    /// This must not strictly be an XOR operation. We only require that applying the operation twice
    /// returns the original value.
    fn xor(&mut self, other: Self);
}

impl<T: Copy + BitXorAssign + Default + Eq + PartialEq + Hash> Encodable for T {
    fn zero() -> Self {
        Self::default()
    }

    fn xor(&mut self, other: Self) {
        *self ^= other;
    }
}

fn indices<T>(
    builder: &impl HashFunctions<T>,
    value: &T,
    stream_len: usize,
) -> impl Iterator<Item = usize> {
    (0..5).map(move |seed| index_for_seed(builder, value, stream_len, seed))
}

/// This function computes with 5 distinct hash functions 5 indices to which a value maps.
/// Essentially, we are "fighting" here two contradicting requirements:
/// - More hash functions with larger partitions reduce the probability that two values maps
///   to exactly the same indices for all hash functions! In this situation, we can decode the stream.
/// - Fewer (ideally 3) hash functions with larger partitions lead to a higher chance to
///   find a pure value in the stream, i.e. the stream can be decoded with fewer coded symbols.
///
/// After testing various schemes, I settled for this one which uses 4 equally sized partitions,
/// one for each of the first 4 hash functions. This ensures that the first 4 hash functions
/// map to distinct indices.
/// The last hash function is used to reduce the probability of hash collisions further without
/// reducing the chance to find pure values in the stream.
///
/// Note: we want an odd number of hash functions, so that collapsing the stream to a single coded symbol
/// (or small number of coded symbols) won't erase the value information.
///
/// The second return value indicates whether the entry should be stored negated.
fn index_for_seed<T>(
    builder: &impl HashFunctions<T>,
    value: &T,
    stream_len: usize,
    seed: u32,
) -> usize {
    let mut hash = builder.hash(value, seed);
    let mask = 31;
    let mut lsb = hash & mask;
    hash -= lsb;
    if seed == 4 {
        lsb = 0;
    } else {
        lsb &= !3;
        lsb += seed;
    }
    hash += lsb;
    hash_to_index(hash, stream_len)
}

/// Determines the parent index in the rateless/hierarchical representation.
fn parent(i: usize) -> usize {
    i - ((i + 1).next_power_of_two() / 2)
}

/// Function to map a hash value into a correct index for a given number of coded symbols.
/// This function is compatible with the above `parent` function in the sense that
/// repeatedly applying `parent` until the index is less than `n` will yield the same result!
fn hash_to_index(hash: u32, n: usize) -> usize {
    let power_of_two = (n as u32).next_power_of_two();
    let hash = hash % power_of_two;
    let res = if hash >= n as u32 {
        hash - power_of_two / 2
    } else {
        hash
    };
    res as usize
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fmt::Debug, time::Instant};

    use crate::{
        DefaultHashFunctions, EncodingSession, decoded_value::DecodedValue,
        decoding_session::DecodingSession, hash_to_index, parent,
    };

    /// This test ensures that the parent and hash_to_index functions are consistent to each other!
    #[test]
    fn test_parent() {
        for i in 0..1000 {
            let mut ii = i;
            while ii > 0 {
                let jj = parent(ii);
                for k in jj + 1..=ii {
                    assert_eq!(hash_to_index(i as u32, k), jj, "{i} {ii} {k}");
                }
                assert_eq!(hash_to_index(i as u32, jj + 1), jj, "{i} {ii} {jj}");
                ii = jj;
            }
        }
    }

    /// Test that moving the split point works correctly.
    /// This test simply goes through all possible size and split point combinations and compares
    /// the results when the split point is changed before or after inserting elements.
    /// In both cases, the result should be the same!
    #[test]
    fn test_move_split() {
        let state = DefaultHashFunctions;
        for n in 1..64 {
            for s in 0..=n {
                let mut encoding1 = EncodingSession::new(state, 0..n);
                let mut encoding2 = EncodingSession::new(state, 0..n);
                encoding2.move_split_point(s);
                for k in 1..100 {
                    encoding1.insert(k);
                    encoding2.insert(k);
                }
                encoding1.move_split_point(s);
                assert_eq!(
                    encoding1.coded_symbols, encoding2.coded_symbols,
                    "n: {n}, s: {s}"
                );
            }
        }
    }

    /// Test encoding and decoding of a large stream.
    #[test]
    fn test_single_stream() {
        let mut stats = Stats::default();
        let mut bits = Stats::default();
        let mut encoding_time = Stats::default();
        let mut decoding_time = Stats::default();
        let mut deocding_time_fast = Stats::default();
        for i in 0..10 {
            let items = 100000;
            let entries: Vec<_> = (1u64..=items).collect();
            let state = DefaultHashFunctions;
            let start = Instant::now();
            let mut stream1 = EncodingSession::new(state, 0..items as usize * 15 / 10);
            stream1.extend(entries.iter().cloned());
            encoding_time.add(start.elapsed().as_secs_f32());

            let start = Instant::now();
            let mut decoding_session = DecodingSession::new(state);
            decoding_session.append(stream1.clone());
            assert!(
                decoding_session.is_done(),
                "{} {i}",
                decoding_session.non_zero,
            );
            stats.add(decoding_session.consumed_coded_symbols() as f32);
            bits.add(decoding_session.required_bits as f32);
            decoding_time.add(start.elapsed().as_secs_f32());
            let mut decoded: Vec<_> = decoding_session
                .into_decoded_iter()
                .map(|decoded_value| {
                    let DecodedValue::Addition(e) = decoded_value else {
                        panic!("Value was deleted, but expected added");
                    };
                    e
                })
                .collect();
            decoded.sort();
            assert_eq!(decoded.len(), items as usize);
            assert_eq!(decoded, entries);

            let start = Instant::now();
            let decoding_session = DecodingSession::from_encoding(stream1);
            assert!(decoding_session.is_done());
            deocding_time_fast.add(start.elapsed().as_secs_f32());
            let decoded2: Vec<_> = decoding_session.into_decoded_iter().collect();
            assert_eq!(decoded2.len(), items as usize);
        }
        println!("stream size: {stats:?}");
        println!("required bits: {bits:?}");
        println!("encoding time: {encoding_time:?}");
        println!("decoding time: {decoding_time:?}");
        println!("decoding time fast: {deocding_time_fast:?}");
    }

    /// Test that splitting an encoding session and reassembling it into a decoding session works.
    /// The comparison must be done in the hierarchical representation which is enforced by
    /// calling move_split_point.
    /// Note: we abort the loop, once the stream was successfully decoded, since otherwise some
    /// assertion would trigger :)
    #[test]
    fn test_splitting_of_decoding() {
        let state = DefaultHashFunctions;
        let mut stream1 = EncodingSession::new(state, 0..200);
        let items = 100;
        stream1.extend(1..=items);

        // Test that decoding would actually work...
        let mut decoding_session = DecodingSession::new(state);
        decoding_session.append(stream1.clone());
        assert!(decoding_session.is_done());

        let mut expected = stream1.clone();
        expected.move_split_point(0);
        let expected: Vec<_> = expected.into_coded_symbols().collect();
        let mut decoding_session = DecodingSession::new(state);
        for i in 0.. {
            let mut got = stream1.split_off(10);
            decoding_session.append(got.clone());
            assert_eq!(got.range.start, i * 10);
            got.move_split_point(i * 10);
            let got: Vec<_> = got.into_coded_symbols().collect();
            assert_eq!(got, expected[i * 10..(i + 1) * 10]);
            if decoding_session.is_done() {
                break;
            }
        }
        assert_eq!(decoding_session.into_decoded_iter().count(), items);
    }

    #[derive(Default)]
    struct Stats {
        sum: f32,
        sum2: f32,
        cnt: f32,
        max: f32,
    }

    impl Stats {
        fn add(&mut self, v: f32) {
            self.sum += v;
            self.sum2 += v * v;
            self.cnt += 1.0;
            if v > self.max {
                self.max = v;
            }
        }

        fn finish(&self) -> (f32, f32, f32) {
            let mean = self.sum / self.cnt;
            let var = self.sum2 / self.cnt - mean * mean;
            (mean, var.max(0.0).sqrt() / mean, self.max)
        }
    }

    impl Debug for Stats {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let (mean, stddev, max) = self.finish();
            write!(f, "{mean},{stddev},{max}")
        }
    }

    #[test]
    fn test_merging() {
        let state = DefaultHashFunctions;
        let mut stream1 = EncodingSession::new(state, 0..200);
        stream1.extend(0..20);
        let mut stream2 = EncodingSession::new(state, 0..200);
        stream2.extend(10..30);
        let merged = stream1.merge(stream2, true);
        let decoding_session = DecodingSession::from_encoding(merged);
        assert!(decoding_session.is_done());
        let mut decoded: Vec<_> = decoding_session.into_decoded_iter().collect();
        decoded.sort();
        assert_eq!(
            &decoded[0..10],
            (0..10).map(DecodedValue::Addition).collect::<Vec<_>>()
        );
        assert_eq!(
            &decoded[10..20],
            (20..30).map(DecodedValue::Deletion).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_statistics() {
        let state = DefaultHashFunctions;
        let mut stats = HashMap::new();

        for start in 0..100 {
            let mut stream = EncodingSession::new(state, 0..1500);
            stream.move_split_point(0);
            for value in 1u64..1000 {
                stream.insert(value);
                if ((value as f32).log2() * 8.0).floor()
                    != ((value as f32 + 1.0).log2() * 8.0).floor()
                {
                    let mut decoding_session = DecodingSession::new(state);
                    decoding_session.append(stream.clone());
                    assert!(decoding_session.is_done(), "start: {start}, value: {value}");
                    {
                        // It is in theory possible that the fast decoding requires a longer stream than the
                        // incremental decoding. This can happen when collisions cancel out in the incremental decoding
                        // case. This test shows that this is extremely unlikely to happen.
                        let decoding_session = DecodingSession::from_encoding(
                            stream
                                .clone()
                                .split_off(decoding_session.consumed_coded_symbols()),
                        );
                        assert!(decoding_session.is_done(), "start: {start}, value: {value}");
                    }
                    let s = stats
                        .entry(value)
                        .or_insert((Stats::default(), Stats::default()));
                    s.0.add(decoding_session.consumed_coded_symbols() as f32);
                    s.1.add(decoding_session.required_bits as f32);
                }
            }
        }
        let mut stats: Vec<_> = stats.into_iter().collect();
        stats.sort_by_key(|(value, _)| *value);
        for (value, (stat, bits)) in stats {
            println!("{value},{stat:?},{bits:?}");
        }
    }
}

#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDocTests;
