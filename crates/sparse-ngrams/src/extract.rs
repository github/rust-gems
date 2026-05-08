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
    let mut queue = FixedDeque::<{ MAX_SPARSE_GRAM_SIZE as usize }>::new();
    let mut prefix_hashes = [0u32; MAX_SPARSE_GRAM_SIZE as usize];
    prefix_hashes[1] = content[0] as u32;
    let mut w = 0usize;

    for idx in 1..n as u32 {
        let mask = MAX_SPARSE_GRAM_SIZE as usize - 1;
        let end_hash = prefix_hashes[idx as usize & mask]
            .wrapping_mul(POLY_HASH_PRIME)
            .wrapping_add(content[idx as usize] as u32);

        // Bigram
        let bigram_hash = end_hash
            .wrapping_sub(prefix_hashes[(idx as usize - 1) & mask].wrapping_mul(POLY_POWERS[2]));
        out[w] = NGram::from_rolling_hash(bigram_hash, 2);
        w += 1;

        let v1 =
            table[content[idx as usize - 1] as usize * 256 + content[idx as usize] as usize];

        if let Some(begin) = queue.front() {
            if idx - begin.index + 1 >= MAX_SPARSE_GRAM_SIZE {
                queue.pop_front();
            }
        }
        while let Some(begin) = queue.back() {
            let start = begin.index as usize - 1;
            let len = (idx - begin.index + 2) as usize;
            let hash = end_hash.wrapping_sub(prefix_hashes[start & mask].wrapping_mul(POLY_POWERS[len]));
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
    const MASK: usize = MAX_SPARSE_GRAM_SIZE as usize - 1;
    let mut w = 0usize;
    let mut prefix_hashes = [0u32; MAX_SPARSE_GRAM_SIZE as usize];
    prefix_hashes[1] = content[0] as u32;
    let mut priorities = [u16::MAX; MAX_SPARSE_GRAM_SIZE as usize];
    for idx in 1..n as u32 {
        let end_hash = prefix_hashes[idx as usize & MASK]
            .wrapping_mul(POLY_HASH_PRIME)
            .wrapping_add(content[idx as usize] as u32);
        // Bigram
        let bigram_hash = end_hash
            .wrapping_sub(prefix_hashes[(idx as usize - 1) & MASK].wrapping_mul(POLY_POWERS[2]));
        out[w] = NGram::from_rolling_hash(bigram_hash, 2);
        w += 1;
        let v1 =
            table[content[idx as usize - 1] as usize * 256 + content[idx as usize] as usize];
        priorities[idx as usize & MASK] = v1;
        let mut running_min = u16::MAX;
        for d in 1..=(MAX_SPARSE_GRAM_SIZE - 2) {
            if d >= idx {
                break;
            }
            let p = idx.wrapping_sub(d) as usize & MASK;
            let v_p = priorities[p];
            if v_p < running_min {
                running_min = v_p;
                let start = p.wrapping_sub(1) & MASK;
                let len = d as usize + 2;
                let hash = end_hash.wrapping_sub(prefix_hashes[start].wrapping_mul(POLY_POWERS[len]));
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
