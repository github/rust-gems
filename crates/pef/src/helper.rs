#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::*;

/// Transform the Elias-Fano high-bits mask into a bitmask which can be used for
/// simple AND intersection before decoding low-bits.
/// At the end of the AND operation, one knows which buckets are still relevant
/// for further decoding.
#[inline(always)]
fn transform_mask(mask: u64) -> u64 {
    unsafe {
        // We extract bits from (mask >> 1) based on the selector (!mask)
        _pext_u64(mask >> 1, !mask)
    }
}

/// Constructs the Elias-Fano high-bits mask for 16 values.
/// 
/// `high_parts`: A ZMM register containing 16 `u32` high-part values (bucket indices).
/// Returns: A `u64` bitmask where the bit at `i + high_parts[i]` is set.
/// 
/// Note: This assumes `i + high_parts[i] < 64`. If the data is very sparse,
/// you may need to handle offsets or multiple words.
#[inline(always)]
pub unsafe fn construct_high_bits_avx512(high_parts: __m512i) -> u64 {
    // 1. Add the implicit index (0..15) to the high parts
    // The i-th set bit belongs at position: i + high_parts[i]
    let increment = _mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    let bit_indices = _mm512_add_epi32(high_parts, increment);

    // 2. We need to shift '1' by these indices.
    // Since _mm512_sllv_epi32 only works within 32-bit lanes (masking shift to 5 bits),
    // and our indices can be > 31, we must promote to 64-bit integers to use _mm512_sllv_epi64.
    
    // Split into low and high halves (8 integers each) and promote to u64
    let idx_lo = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(bit_indices));
    let idx_hi = _mm512_cvtepu32_epi64(_mm512_extracti32x8_epi32(bit_indices, 1));

    // 3. Create bitmasks: 1 << index
    let one = _mm512_set1_epi64(1);
    let bits_lo = _mm512_sllv_epi64(one, idx_lo);
    let bits_hi = _mm512_sllv_epi64(one, idx_hi);

    // 4. Combine all 16 bitmasks into one register
    let combined = _mm512_or_si512(bits_lo, bits_hi);

    // 5. Horizontal OR to reduce the 8 u64 lanes into a single u64 scalar
    _mm512_reduce_or_epi64(combined)
}

/// Given the Elias-Fano high-bits sequence and a mask of interesting buckets,
/// returns a bitmask of the values that belong to the interesting buckets.
///
/// `high_bits`: The Elias-Fano high-bits sequence (1 = value, 0 = bucket boundary).
/// `bucket_mask`: A bitmask where bit `k` is set if bucket `k` is interesting.
///
/// Returns: A bitmask where bit `i` is set if the value at position `i` in `high_bits`
/// belongs to an interesting bucket.
#[inline(always)]
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
pub fn filter_values_by_bucket(high_bits: u64, bucket_mask: u64) -> u64 {
    unsafe {
        #[cfg(not(target_feature = "avx512f"))]
        use std::arch::x86_64::_pdep_u64;

        // 1. Identify the start of each bucket's value run.
        // The start of a run is marked by a 0->1 transition in high_bits, or index 0.
        // By shifting high_bits left by 1, we align the '0' separators 
        // with the start of the *next* bucket.
        // Inverting gives us 1s at these start positions.
        let starts = !(high_bits << 1);
        
        // 2. Distribute the bucket_mask to these start positions.
        // If bucket k is set in bucket_mask, the k-th set bit in 'starts'
        // (which is the start of bucket k) gets a 1.
        let mask_at_starts = _pdep_u64(bucket_mask, starts);
        
        // 3. Propagate the mask through the runs of ones.
        // Adding the start bit to the run of ones causes a carry ripple
        // that turns the run of 1s into 0s (and sets the next 0 to 1).
        // Example: 111 + 1 = 1000.
        let sum = high_bits.wrapping_add(mask_at_starts);
        
        // 4. Recover the original runs.
        // The runs that were added to are now 0s. Inverting makes them 1s.
        // The runs that were NOT added to are still 1s. Inverting makes them 0s.
        // ANDing with the original high_bits selects only the runs that were inverted (selected).
        (!sum) & high_bits
    }
}