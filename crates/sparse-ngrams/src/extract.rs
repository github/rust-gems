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
    let mut w = 0usize;

    let mut prefix_hashes = [0u32; MAX_SPARSE_GRAM_SIZE as usize];
    prefix_hashes[1] = content[0] as u32;

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

        let (v1, _v2) =
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
            if begin.value < v1 {
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
    let mut priorities = [0u32; MAX_SPARSE_GRAM_SIZE as usize];
    for idx in 1..n as u32 {
        let end_hash = prefix_hashes[idx as usize & MASK]
            .wrapping_mul(POLY_HASH_PRIME)
            .wrapping_add(content[idx as usize] as u32);
        // Bigram
        let bigram_hash = end_hash
            .wrapping_sub(prefix_hashes[(idx as usize - 1) & MASK].wrapping_mul(POLY_POWERS[2]));
        out[w] = NGram::from_rolling_hash(bigram_hash, 2);
        w += 1;
        let (v1, _v2) =
            table[content[idx as usize - 1] as usize * 256 + content[idx as usize] as usize];
        priorities[idx as usize & MASK] = v1;
        let mut running_min = u32::MAX;
        for d in 1..=(MAX_SPARSE_GRAM_SIZE - 2) {
            if d >= idx {
                break;
            }
            let p = (idx - d) as usize;
            let v_p = priorities[p & MASK];
            if v_p < running_min {
                running_min = v_p;
                let start = p - 1;
                let len = d as usize + 2;
                let hash = end_hash.wrapping_sub(prefix_hashes[start & MASK].wrapping_mul(POLY_POWERS[len]));
                out[w] = NGram::from_rolling_hash(hash, len);
                w += 1;
                if v_p < v1 {
                    break;
                }
            }
        }
        prefix_hashes[(idx as usize + 1) & MASK] = end_hash;
    }
    w
}

/// Masked variant: maintains the deque as a bitmask across iterations.
///
/// Since deque elements have strictly increasing priorities (front→back), comparing
/// each lookback priority with the current priority `v1` is enough to determine
/// which elements to emit and pop — no prefix-min scan needed.
///
/// Bit `k` in the `active` mask means the position at lookback offset `k+1` is
/// currently in the deque. At each step: compare all 6 lookback priorities with `v1`,
/// emit from active positions, update the mask with a shift+OR+AND.
///
/// Produces the same *set* of n-grams as [`collect_sparse_grams_scan`], but in a different order.
///
/// # Panics
///
/// Panics if `out` is too small.
pub fn collect_sparse_grams_masked(content: &[u8], out: &mut [NGram]) -> usize {
    const MAX_D: usize = MAX_SPARSE_GRAM_SIZE as usize - 2; // 6
    const CMASK: usize = MAX_SPARSE_GRAM_SIZE as usize - 1; // 7

    let n = content.len();
    if n < 2 {
        return 0;
    }
    assert!(out.len() >= max_sparse_grams(n));

    let table = get_bigram_table();
    let mut w = 0usize;

    let mut prefix_hashes = [0u32; MAX_SPARSE_GRAM_SIZE as usize];
    prefix_hashes[1] = content[0] as u32;

    let mut prio_circ = [0u32; MAX_SPARSE_GRAM_SIZE as usize];
    let mut active: u32 = 0; // bitmask: bit k ↔ array slot k is in the deque
    let mut dist = [2usize, 9, 8, 7, 6, 5, 4, 3]; // dist[k] = ((idx - k) & CMASK) + 2
    // poly_circ[k] = POLY_HASH_PRIME^(dist[k] + 2) — tracks the poly power for each slot.
    // After each iteration dist increments, so we multiply all by PRIME and reset
    // the wrapping slot. Initial values correspond to dist = [0, 7, 6, 5, 4, 3, 2, 1],
    // but we need power dist+2. For the slot with dist=7 that means power 9, which
    // we'll get naturally after one multiply (it starts at power 8, becomes 9 after
    // the first multiply, then gets reset to 2 when it wraps).
    // So initialize poly_circ[k] = POLY_POWERS[dist_init[k] + 1] instead, since the
    // first thing in the loop is the multiply.
    let mut poly_circ = [
        POLY_POWERS[1], POLY_POWERS[8], POLY_POWERS[7], POLY_POWERS[6],
        POLY_POWERS[5], POLY_POWERS[4], POLY_POWERS[3], POLY_POWERS[2],
    ];

    for idx in 1..n as u32 {
        // Advance all distances by 1 and multiply poly powers.
        for k in 0..MAX_SPARSE_GRAM_SIZE as usize {
            dist[k] += 1;
            poly_circ[k] = poly_circ[k].wrapping_mul(POLY_HASH_PRIME);
        }
        // The current-position slot just exceeded MAX_SPARSE_GRAM_SIZE+1; reset it.
        dist[idx as usize & CMASK] = 2;
        poly_circ[idx as usize & CMASK] = POLY_POWERS[2];
        let end_hash = prefix_hashes[idx as usize & CMASK]
            .wrapping_mul(POLY_HASH_PRIME)
            .wrapping_add(content[idx as usize] as u32);
        let (v1, _) =
            table[content[idx as usize - 1] as usize * 256 + content[idx as usize] as usize];
        prio_circ[idx as usize & CMASK] = v1;
        // Compare all lookback priorities with v1 (no idx dependency in the loop).
        let mut ge_bits: u32 = 0;
        for k in 0..MAX_SPARSE_GRAM_SIZE as usize {
            ge_bits |= ((prio_circ[k] >= v1) as u32) << k;
        }
        // Emit from active deque positions:
        //  - all with v >= v1 (they get popped)
        //  - plus the nearest one with v < v1 (it stays)
        let cur_bit = 1u32 << (idx as usize & CMASK);
        let emit_ge = active & ge_bits;
        let active_lt = active & !ge_bits;
        // Find nearest (smallest distance) active position with v < v1.
        // Rotate so bit 7 = distance 1, bit 6 = distance 2, ..., bit 1 = distance 7.
        // Bit 0 = distance 0 (cur_bit's slot, never in active_lt) → used as sentinel.
        // When empty, leading_zeros finds the sentinel; first_lt becomes cur_bit (harmless).
        let doubled = active_lt | (active_lt << 8);
        let shift = idx as usize & CMASK;
        let rotated = ((doubled >> shift) & 0xFF) | 1;
        let b = 31 - rotated.leading_zeros() as usize;
        let first_lt = 1u32 << ((shift + b) & CMASK);
        let emit = emit_ge | first_lt | cur_bit;
        // Write emitted ngrams.
        let mut m = emit;
        while m != 0 {
            let k = m.trailing_zeros() as usize;
            let len = dist[k];
            let hash = end_hash
                .wrapping_sub(prefix_hashes[k.wrapping_sub(1) & CMASK].wrapping_mul(poly_circ[k]));
            out[w] = NGram((hash << 8) | len as u32);
            w += 1;
            m &= m - 1;
        }
        // Update active: keep survivors with v < v1, add current position,
        // and evict the slot that just left the lookback window.
        active = (active & !ge_bits) | (1 << (idx as usize & CMASK));
        active &= !(1u32 << (idx.wrapping_sub(MAX_D as u32) as usize & CMASK));
        prefix_hashes[(idx as usize + 1) & CMASK] = end_hash;
    }
    w
}

/// AVX2 variant of [`collect_sparse_grams_masked`].
///
/// Uses 256-bit SIMD to update the 8-element circular arrays (`poly_circ`,
/// `prio_circ`, `dist`, `prefix_hashes`) and to compute the ge_bits comparison
/// and hash outputs in parallel. The bitmask logic (active/emit/first_lt) remains
/// scalar since it operates on 8-bit masks.
///
/// Produces the same *set* of n-grams as [`collect_sparse_grams_masked`].
///
/// # Panics
///
/// Panics if `out` is too small.
#[cfg(target_arch = "x86_64")]
pub fn collect_sparse_grams_masked_avx(content: &[u8], out: &mut [NGram]) -> usize {
    // SAFETY: AVX2 is enabled globally via .cargo/config.toml target-feature flags.
    unsafe { collect_sparse_grams_masked_avx_inner(content, out) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512vl")]
unsafe fn collect_sparse_grams_masked_avx_inner(content: &[u8], out: &mut [NGram]) -> usize {
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    const MAX_D: usize = MAX_SPARSE_GRAM_SIZE as usize - 2; // 6
    const CMASK: usize = MAX_SPARSE_GRAM_SIZE as usize - 1; // 7

    let n = content.len();
    if n < 2 {
        return 0;
    }
    assert!(out.len() >= max_sparse_grams(n));

    let table = get_bigram_table();
    let mut w = 0usize;

    // Aligned arrays for SIMD load/store.
    #[repr(align(32))]
    struct A32([u32; 8]);

    // Shifted layout: lane k stores prefix_hashes[(k - 1) & 7].
    // This lets the hash step load the needed rotated prefix vector directly.
    let mut prefix_hashes = A32([0u32; 8]);
    prefix_hashes.0[2] = content[0] as u32;

    let mut prio_arr = A32([0u32; 8]);
    let mut active: u32 = 0;

    // Doubled LUTs let us load the 8-lane circular view with one unaligned load.
    let dist_lut = [2u32, 9, 8, 7, 6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3];
    let poly_pow9 = POLY_POWERS[8].wrapping_mul(POLY_HASH_PRIME);
    let poly_lut = [
        POLY_POWERS[2], poly_pow9, POLY_POWERS[8], POLY_POWERS[7],
        POLY_POWERS[6], POLY_POWERS[5], POLY_POWERS[4], POLY_POWERS[3],
        POLY_POWERS[2], poly_pow9, POLY_POWERS[8], POLY_POWERS[7],
        POLY_POWERS[6], POLY_POWERS[5], POLY_POWERS[4], POLY_POWERS[3],
    ];

    for idx in 1..n as u32 {
        let slot = idx as usize & CMASK;

        // Load slot-aligned circular views for dist/poly with a single unaligned read.
        let lut_off = (MAX_SPARSE_GRAM_SIZE as usize - slot) & CMASK;
        let v_dist = _mm256_loadu_si256(dist_lut.as_ptr().add(lut_off) as *const __m256i);
        let v_poly = _mm256_loadu_si256(poly_lut.as_ptr().add(lut_off) as *const __m256i);

        // --- Scalar: compute end_hash, v1, update prio ---
        let end_hash = prefix_hashes.0[(slot + 1) & CMASK]
            .wrapping_mul(POLY_HASH_PRIME)
            .wrapping_add(content[idx as usize] as u32);

        let (v1, _) =
            table[content[idx as usize - 1] as usize * 256 + content[idx as usize] as usize];
        prio_arr.0[slot] = v1;

        // --- AVX-512: compare all 8 priorities with v1 and get the lane mask directly ---
        let v_prio = _mm256_load_si256(prio_arr.0.as_ptr() as *const __m256i);
        let v_v1 = _mm256_set1_epi32(v1 as i32);
        let ge_bits = _mm256_cmp_epu32_mask(v_prio, v_v1, _MM_CMPINT_NLT) as u32;

        // --- Scalar bitmask logic (same as masked variant) ---
        let cur_bit = 1u32 << slot;
        let emit_ge = active & ge_bits;
        let active_lt = active & !ge_bits;

        let doubled = active_lt | (active_lt << 8);
        let shift = idx as usize & CMASK;
        let rotated = ((doubled >> shift) & 0xFF) | 1;
        let b = 31 - rotated.leading_zeros() as usize;
        let first_lt = 1u32 << ((shift + b) & CMASK);
        let emit = emit_ge | first_lt | cur_bit;

        // --- AVX2: compute all 8 hashes in parallel ---
        // hash[k] = end_hash - prefix_hashes[(k-1) & 7] * poly_circ[k]
        let v_ph_rot = _mm256_load_si256(prefix_hashes.0.as_ptr() as *const __m256i);
        let v_prod = _mm256_mullo_epi32(v_ph_rot, v_poly);
        let v_end = _mm256_set1_epi32(end_hash as i32);
        let v_hash = _mm256_sub_epi32(v_end, v_prod);

        // Compute (hash << 8) | dist for all lanes.
        let v_hash_shifted = _mm256_slli_epi32(v_hash, 8);
        let v_ngram = _mm256_or_si256(v_hash_shifted, v_dist);

        // --- AVX-512 compressed store (VPCOMPRESSD): selected lanes written contiguously ---
        let count = emit.count_ones() as usize;
        // SAFETY: NGram is #[repr(transparent)] over u32, and out[w..w+count] is in bounds.
        _mm256_mask_compressstoreu_epi32(
            out.as_mut_ptr().add(w) as *mut i32,
            emit as __mmask8,
            v_ngram,
        );
        w += count;

        // --- Update active ---
        active = (active & !ge_bits) | cur_bit;
        active &= !(1u32 << (idx.wrapping_sub(MAX_D as u32) as usize & CMASK));

        let next = (idx as usize + 1) & CMASK;
        prefix_hashes.0[(next + 1) & CMASK] = end_hash;
    }
    w
}

/// SIMD-friendly wide variant that processes 16 end positions in parallel.
///
/// Uses two fixed-size `[u32; LANES * 2]` sliding buffers (priorities + prefix hashes)
/// on the stack — no heap allocation. Each chunk copies the back half to the front,
/// then fills the back half from `content`, so all reads are simple flat-array indexing.
///
/// Writes n-grams into `out` (must have at least [`max_sparse_grams`]`(content.len())`
/// slots). Returns the count written.
///
/// Produces the same *set* of n-grams as the other variants, but in a different order.
///
/// # Panics
///
/// Panics if `out` is too small.
pub fn collect_sparse_grams_wide(content: &[u8], out: &mut [NGram]) -> usize {
    const LANES: usize = 16;
    const BUF: usize = LANES * 2;
    const MASK: usize = MAX_SPARSE_GRAM_SIZE as usize - 1;

    let n = content.len();
    if n < 2 {
        return 0;
    }

    let table = get_bigram_table();
    let num_bigrams = n - 1;
    assert!(out.len() >= max_sparse_grams(n));

    let mut w = 0usize;

    // --- Scalar prefix: first LANES positions (fills the initial buffer) ----
    let scalar_prefix = LANES.min(num_bigrams);

    let mut circ_ph = [0u32; MAX_SPARSE_GRAM_SIZE as usize];
    circ_ph[1] = content[0] as u32;
    let mut circ_prio = [0u32; MAX_SPARSE_GRAM_SIZE as usize];

    for idx in 1..=scalar_prefix as u32 {
        let end_hash = circ_ph[idx as usize & MASK]
            .wrapping_mul(POLY_HASH_PRIME)
            .wrapping_add(content[idx as usize] as u32);

        let bigram_hash = end_hash
            .wrapping_sub(circ_ph[(idx as usize - 1) & MASK].wrapping_mul(POLY_POWERS[2]));
        out[w] = NGram::from_rolling_hash(bigram_hash, 2);
        w += 1;

        let (v1, _) =
            table[content[idx as usize - 1] as usize * 256 + content[idx as usize] as usize];
        circ_prio[idx as usize & MASK] = v1;

        let mut running_min = u32::MAX;
        for d in 1..=(MAX_SPARSE_GRAM_SIZE - 2) {
            if d >= idx {
                break;
            }
            let v_p = circ_prio[(idx - d) as usize & MASK];
            if v_p < running_min {
                let start = (idx - d) as usize - 1;
                let len = d as usize + 2;
                let hash = end_hash.wrapping_sub(circ_ph[start & MASK].wrapping_mul(POLY_POWERS[len]));
                out[w] = NGram::from_rolling_hash(hash, len);
                w += 1;
                if v_p < v1 {
                    break;
                }
            }
            running_min = running_min.min(v_p);
        }

        circ_ph[(idx as usize + 1) & MASK] = end_hash;
    }

    if num_bigrams <= LANES {
        return w;
    }

    // --- Seed the sliding buffers -------------------------------------------
    // Each buffer has BUF=32 slots.  The back half [LANES..BUF) corresponds to
    // the LANES positions we just processed.  Before the first wide chunk we'll
    // copy back→front, then fill back with fresh data.
    //
    // ph_buf[LANES + j] = ph[j + 1]  for j in 0..LANES
    // prio_buf[LANES + j] = priority at 1-based position j+1

    let mut ph_buf = [0u32; BUF + 1];
    let mut prio_buf = [0u32; BUF];

    ph_buf[LANES] = content[0] as u32;
    for j in 1..=LANES {
        ph_buf[LANES + j] = ph_buf[LANES + j - 1]
            .wrapping_mul(POLY_HASH_PRIME)
            .wrapping_add(content[j] as u32);
    }
    for j in 0..LANES {
        prio_buf[LANES + j] =
            table[content[j] as usize * 256 + content[j + 1] as usize].0;
    }

    // --- Wide chunk loop (includes partial last chunk) -----------------------
    let remaining = num_bigrams - LANES;
    let total_chunks = (remaining + LANES - 1) / LANES; // ceiling division
    let mut content_pos = LANES; // 0-based: next byte to process

    for _chunk in 0..total_chunks {
        let chunk_lanes = LANES.min(num_bigrams - (content_pos)); // active lanes this chunk

        // Slide: back → front.
        prio_buf.copy_within(LANES..BUF, 0);
        ph_buf.copy_within(LANES..BUF + 1, 0);

        // Fill back half — only chunk_lanes+1 prefix hashes and chunk_lanes priorities.
        ph_buf[LANES] = ph_buf[LANES - 1]
            .wrapping_mul(POLY_HASH_PRIME)
            .wrapping_add(content[content_pos] as u32);
        for j in 1..=chunk_lanes {
            ph_buf[LANES + j] = ph_buf[LANES + j - 1]
                .wrapping_mul(POLY_HASH_PRIME)
                .wrapping_add(content[content_pos + j] as u32);
        }
        for j in 0..chunk_lanes {
            let ci = content_pos + j;
            prio_buf[LANES + j] =
                table[content[ci] as usize * 256 + content[ci + 1] as usize].0;
        }

        let lane_mask: u32 = (1u32 << chunk_lanes) - 1; // bits 0..chunk_lanes

        // Pad unfilled back-half slots so the 0..LANES loops read valid memory.
        // Inactive lanes will be masked out by lane_mask.
        for j in chunk_lanes..LANES {
            prio_buf[LANES + j] = 0;
            ph_buf[LANES + j + 1] = ph_buf[LANES + chunk_lanes]; // any valid value
        }

        // These loops MUST iterate 0..LANES (compile-time constant) for auto-vectorization.
        let mut v_current = [0u32; LANES];
        let mut end_hashes = [0u32; LANES];
        for i in 0..LANES {
            v_current[i] = prio_buf[LANES + i];
            end_hashes[i] = ph_buf[LANES + i + 1];
        }

        // Emit bigrams for active lanes.
        for i in 0..chunk_lanes {
            let hash = end_hashes[i]
                .wrapping_sub(ph_buf[LANES + i - 1].wrapping_mul(POLY_POWERS[2]));
            out[w + i] = NGram::from_rolling_hash(hash, 2);
        }
        w += chunk_lanes;

        // Backward suffix-minimum scan — inner loops always 0..LANES for vectorization.
        let mut running_min = [u32::MAX; LANES];
        let mut active: u32 = lane_mask;

        for d in 1..=(MAX_SPARSE_GRAM_SIZE as usize - 2) {
            if active == 0 {
                break;
            }

            let len = d + 2;
            let poly_power = POLY_POWERS[len];

            let mut v_shifted = [0u32; LANES];
            let mut candidate_hashes = [0u32; LANES];
            let mut is_suffix_min = [false; LANES];
            let mut should_break = [false; LANES];

            for i in 0..LANES {
                v_shifted[i] = prio_buf[LANES + i - d];
                candidate_hashes[i] = end_hashes[i]
                    .wrapping_sub(ph_buf[LANES + i - d - 1].wrapping_mul(poly_power));
                is_suffix_min[i] = v_shifted[i] < running_min[i];
                should_break[i] = is_suffix_min[i] && v_shifted[i] < v_current[i];
                running_min[i] = running_min[i].min(v_shifted[i]);
            }

            let mut emit_mask: u32 = 0;
            let mut break_mask: u32 = 0;
            for i in 0..LANES {
                emit_mask |= (is_suffix_min[i] as u32) << i;
                break_mask |= (should_break[i] as u32) << i;
            }
            emit_mask &= active;

            let mut m = emit_mask;
            while m != 0 {
                let i = m.trailing_zeros() as usize;
                out[w] = NGram::from_rolling_hash(candidate_hashes[i], len);
                w += 1;
                m &= m - 1;
            }

            active &= !break_mask;
        }

        content_pos += chunk_lanes;
    }

    w
}

/// AVX-512 version of [`collect_sparse_grams_wide`].
///
/// Keeps the same high-level algorithm and output set, but vectorizes the
/// 16-lane suffix-min/hash inner step explicitly with AVX-512 intrinsics.
#[cfg(target_arch = "x86_64")]
pub fn collect_sparse_grams_wide_avx(content: &[u8], out: &mut [NGram]) -> usize {
    // SAFETY: AVX-512 is enabled globally via .cargo/config.toml target-feature flags.
    unsafe { collect_sparse_grams_wide_avx_inner(content, out) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512vl")]
unsafe fn collect_sparse_grams_wide_avx_inner(content: &[u8], out: &mut [NGram]) -> usize {
    use core::arch::x86_64::*;

    const LANES: usize = 16;
    const BUF: usize = LANES * 2;
    const MASK: usize = MAX_SPARSE_GRAM_SIZE as usize - 1;

    let n = content.len();
    if n < 2 {
        return 0;
    }

    let table = get_bigram_table();
    let num_bigrams = n - 1;
    assert!(out.len() >= max_sparse_grams(n));

    let mut w = 0usize;

    // Scalar prefix (same as wide) to seed buffers.
    let scalar_prefix = LANES.min(num_bigrams);

    let mut circ_ph = [0u32; MAX_SPARSE_GRAM_SIZE as usize];
    circ_ph[1] = content[0] as u32;
    let mut circ_prio = [0u32; MAX_SPARSE_GRAM_SIZE as usize];

    for idx in 1..=scalar_prefix as u32 {
        let end_hash = circ_ph[idx as usize & MASK]
            .wrapping_mul(POLY_HASH_PRIME)
            .wrapping_add(content[idx as usize] as u32);

        let bigram_hash = end_hash
            .wrapping_sub(circ_ph[(idx as usize - 1) & MASK].wrapping_mul(POLY_POWERS[2]));
        out[w] = NGram::from_rolling_hash(bigram_hash, 2);
        w += 1;

        let (v1, _) =
            table[content[idx as usize - 1] as usize * 256 + content[idx as usize] as usize];
        circ_prio[idx as usize & MASK] = v1;

        let mut running_min = u32::MAX;
        for d in 1..=(MAX_SPARSE_GRAM_SIZE - 2) {
            if d >= idx {
                break;
            }
            let v_p = circ_prio[(idx - d) as usize & MASK];
            if v_p < running_min {
                let start = (idx - d) as usize - 1;
                let len = d as usize + 2;
                let hash =
                    end_hash.wrapping_sub(circ_ph[start & MASK].wrapping_mul(POLY_POWERS[len]));
                out[w] = NGram::from_rolling_hash(hash, len);
                w += 1;
                if v_p < v1 {
                    break;
                }
            }
            running_min = running_min.min(v_p);
        }

        circ_ph[(idx as usize + 1) & MASK] = end_hash;
    }

    if num_bigrams <= LANES {
        return w;
    }

    let mut ph_buf = [0u32; BUF + 1];
    let mut prio_buf = [0u32; BUF];

    ph_buf[LANES] = content[0] as u32;
    for j in 1..=LANES {
        ph_buf[LANES + j] = ph_buf[LANES + j - 1]
            .wrapping_mul(POLY_HASH_PRIME)
            .wrapping_add(content[j] as u32);
    }
    for j in 0..LANES {
        prio_buf[LANES + j] = table[content[j] as usize * 256 + content[j + 1] as usize].0;
    }

    let remaining = num_bigrams - LANES;
    let total_chunks = (remaining + LANES - 1) / LANES;
    let mut content_pos = LANES;

    for _chunk in 0..total_chunks {
        let chunk_lanes = LANES.min(num_bigrams - content_pos);

        prio_buf.copy_within(LANES..BUF, 0);
        ph_buf.copy_within(LANES..BUF + 1, 0);

        ph_buf[LANES] = ph_buf[LANES - 1]
            .wrapping_mul(POLY_HASH_PRIME)
            .wrapping_add(content[content_pos] as u32);
        for j in 1..=chunk_lanes {
            ph_buf[LANES + j] = ph_buf[LANES + j - 1]
                .wrapping_mul(POLY_HASH_PRIME)
                .wrapping_add(content[content_pos + j] as u32);
        }
        for j in 0..chunk_lanes {
            let ci = content_pos + j;
            prio_buf[LANES + j] = table[content[ci] as usize * 256 + content[ci + 1] as usize].0;
        }

        let lane_mask: u32 = (1u32 << chunk_lanes) - 1;

        for j in chunk_lanes..LANES {
            prio_buf[LANES + j] = 0;
            ph_buf[LANES + j + 1] = ph_buf[LANES + chunk_lanes];
        }

        // Emit bigrams.
        for i in 0..chunk_lanes {
            let hash = ph_buf[LANES + i + 1]
                .wrapping_sub(ph_buf[LANES + i - 1].wrapping_mul(POLY_POWERS[2]));
            out[w + i] = NGram::from_rolling_hash(hash, 2);
        }
        w += chunk_lanes;

        let v_current = _mm512_loadu_si512(prio_buf[LANES..].as_ptr() as *const __m512i);
        let v_end_hashes = _mm512_loadu_si512(ph_buf[LANES + 1..].as_ptr() as *const __m512i);
        let mut v_running_min = _mm512_set1_epi32(-1);
        let mut active: u32 = lane_mask;

        let poly3 = _mm512_set1_epi32(POLY_POWERS[3] as i32);
        let poly4 = _mm512_set1_epi32(POLY_POWERS[4] as i32);
        let poly5 = _mm512_set1_epi32(POLY_POWERS[5] as i32);
        let poly6 = _mm512_set1_epi32(POLY_POWERS[6] as i32);
        let poly7 = _mm512_set1_epi32(POLY_POWERS[7] as i32);
        let poly8 = _mm512_set1_epi32(POLY_POWERS[8] as i32);

        let len3 = _mm512_set1_epi32(3);
        let len4 = _mm512_set1_epi32(4);
        let len5 = _mm512_set1_epi32(5);
        let len6 = _mm512_set1_epi32(6);
        let len7 = _mm512_set1_epi32(7);
        let len8 = _mm512_set1_epi32(8);

        macro_rules! step {
            ($d:expr, $poly:expr, $vlen:expr) => {{
                let v_shifted =
                    _mm512_loadu_si512(prio_buf[LANES - $d..].as_ptr() as *const __m512i);
                let v_is_suffix_min =
                    _mm512_cmp_epu32_mask(v_shifted, v_running_min, _MM_CMPINT_LT);
                let v_should_break =
                    v_is_suffix_min & _mm512_cmp_epu32_mask(v_shifted, v_current, _MM_CMPINT_LT);

                let v_prev = _mm512_loadu_si512(
                    ph_buf[LANES - $d - 1..].as_ptr() as *const __m512i,
                );
                let v_prod = _mm512_mullo_epi32(v_prev, $poly);
                let v_hashes = _mm512_sub_epi32(v_end_hashes, v_prod);

                let emit_mask = active & v_is_suffix_min as u32;
                let break_mask = active & v_should_break as u32;

                let v_hash_shifted = _mm512_slli_epi32(v_hashes, 8);
                let v_ngram = _mm512_add_epi32(v_hash_shifted, $vlen);
                _mm512_mask_compressstoreu_epi32(
                    out.as_mut_ptr().add(w) as *mut i32,
                    emit_mask as __mmask16,
                    v_ngram,
                );
                w += emit_mask.count_ones() as usize;

                active &= !break_mask;
                v_running_min = _mm512_min_epu32(v_running_min, v_shifted);
            }};
        }

        macro_rules! step_last {
            ($d:expr, $poly:expr, $vlen:expr) => {{
                let v_shifted =
                    _mm512_loadu_si512(prio_buf[LANES - $d..].as_ptr() as *const __m512i);
                let v_is_suffix_min =
                    _mm512_cmp_epu32_mask(v_shifted, v_running_min, _MM_CMPINT_LT);

                let v_prev = _mm512_loadu_si512(
                    ph_buf[LANES - $d - 1..].as_ptr() as *const __m512i,
                );
                let v_prod = _mm512_mullo_epi32(v_prev, $poly);
                let v_hashes = _mm512_sub_epi32(v_end_hashes, v_prod);

                let emit_mask = active & v_is_suffix_min as u32;

                let v_hash_shifted = _mm512_slli_epi32(v_hashes, 8);
                let v_ngram = _mm512_add_epi32(v_hash_shifted, $vlen);
                _mm512_mask_compressstoreu_epi32(
                    out.as_mut_ptr().add(w) as *mut i32,
                    emit_mask as __mmask16,
                    v_ngram,
                );
                w += emit_mask.count_ones() as usize;
            }};
        }

        if active != 0 {
            step!(1, poly3, len3);
            if active != 0 {
                step!(2, poly4, len4);
                if active != 0 {
                    step!(3, poly5, len5);
                    if active != 0 {
                        step!(4, poly6, len6);
                        if active != 0 {
                            step!(5, poly7, len7);
                            if active != 0 {
                                step_last!(6, poly8, len8);
                            }
                        }
                    }
                }
            }
        }

        content_pos += chunk_lanes;
    }

    w
}
