use crate::elias_fano::EliasFano;
use std::arch::x86_64::*;

pub trait BatchDecoder {
    /// Decodes up to 16 values from the Elias-Fano representation.
    ///
    /// Updates internal state.
    /// Writes decoded values to `output` and returns the number of values written.
    ///
    /// # Safety
    /// This function requires AVX-512 features.
    fn decode_batch(&mut self, output: &mut [u32]) -> usize;
}

pub fn new_decoder<'a>(ef: &'a EliasFano) -> Box<dyn BatchDecoder + 'a> {
    macro_rules! dispatch {
        ($($b:literal),*) => {
            match ef.bits_per_value {
                $( $b => Box::new(AvxBatchDecoder::<$b>::new(ef)), )*
                _ => panic!("Unsupported bits per value: {}", ef.bits_per_value),
            }
        }
    }
    dispatch!(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32
    )
}

pub struct AvxBatchDecoder<'a, const BITS: u32> {
    high_bits: &'a [u64],
    low_bits: &'a [u64],
    high_idx: u32,
    value_idx: u32,
}

impl<'a, const BITS: u32> AvxBatchDecoder<'a, BITS> {
    pub fn new(ef: &'a EliasFano) -> Self {
        assert_eq!(ef.bits_per_value, BITS);
        Self {
            high_bits: &ef.high_bits,
            low_bits: &ef.low_bits,
            high_idx: 0,
            value_idx: 0,
        }
    }

    #[inline(always)]
    unsafe fn get_low_bits(&self) -> __m512i {
        if BITS == 4 {
            let byte_idx = (self.value_idx / 2) as usize;
            let nibble_offset = (self.value_idx & 1) as u32; // 0 or 1 (which nibble in the first byte)

            let ptr = self.low_bits.as_ptr() as *const u8;
            
            // We need 16 nibbles = 64 bits of data
            // If nibble_offset == 0, we need exactly 8 bytes
            // If nibble_offset == 1, we need 8.5 bytes (9 bytes total)
            // Load 16 bytes to be safe, then combine
            let w0 = (ptr.add(byte_idx) as *const u64).read_unaligned();
            let w1 = (ptr.add(byte_idx + 1) as *const u64).read_unaligned();
            
            // If nibble_offset == 1, shift right by 4 bits to align
            // This requires combining bits from w0 and w1
            let aligned = w0 >> (nibble_offset * 4) | w1 << (8 - nibble_offset * 4);
            
            // Now 'aligned' contains 16 nibbles in the lower 64 bits
            // Put the 8 bytes into a vector
            let v_bytes = _mm_cvtsi64_si128(aligned as i64);
            
            // We need to separate low and high nibbles of each byte
            // Byte layout: [n1 n0] [n3 n2] [n5 n4] [n7 n6] [n9 n8] [n11 n10] [n13 n12] [n15 n14]
            // We want: n0, n1, n2, n3, ..., n15
            
            // Simpler: mask low nibbles and high nibbles separately, then interleave
            let mask_0f = _mm_set1_epi8(0x0F);
            let lo_nibbles = _mm_and_si128(v_bytes, mask_0f);  // n0, n2, n4, n6, n8, n10, n12, n14
            let hi_nibbles = _mm_and_si128(_mm_srli_epi16(v_bytes, 4), mask_0f); // n1, n3, n5, n7, n9, n11, n13, n15
            
            // Interleave: we want n0, n1, n2, n3, ...
            // unpacklo_epi8 interleaves bytes from lo and hi
            let interleaved = _mm_unpacklo_epi8(lo_nibbles, hi_nibbles);
            // Result: n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15
            
            // Now expand 16 bytes to 16 32-bit integers
            return _mm512_cvtepu8_epi32(interleaved);
        }

        let mut low_vals = [0i32; 16];
        let mut bit_idx = self.value_idx * BITS;
        for i in 0..16 {
            // Inline get_bits logic for simplicity and performance
            let start = bit_idx / 64;
            let end = (bit_idx + BITS - 1) / 64;
            let offset = bit_idx % 64;
            let mut val = self.low_bits.get_unchecked(start as usize) >> offset;
            if start != end {
                val |= self.low_bits.get_unchecked((start + 1) as usize) << (64 - offset);
            }
            low_vals[i] = (val as u32 & !(!0 << BITS)) as i32;
            bit_idx += BITS;
        }
        _mm512_loadu_si512(low_vals.as_ptr() as *const _)
    }

    /// Decodes a batch of 16 values using AVX-512 instructions.
    ///
    /// # Safety
    /// This function requires AVX-512F, AVX-512VBMI2, and AVX-512BW features.
    #[target_feature(enable = "avx512f", enable = "avx512vbmi2", enable = "avx512bw")]
    pub unsafe fn decode(high_bits: u64, low_bits: __m512i) -> __m512i {
        // Identity indices 0..63 for the byte positions
        let identity = _mm512_set_epi8(
            63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42,
            41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,
            19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        );

        // Compress the indices of the set bits in high_bits.
        // This gathers the byte values from 'identity' where the corresponding bit in 'high_bits' is 1.
        let compressed_indices = _mm512_maskz_compress_epi8(high_bits, identity);

        // Expand the first 16 byte positions (indices of the first 16 set bits) into epi32 values.
        let first_16_indices = _mm512_castsi512_si128(compressed_indices);
        let expanded_indices = _mm512_cvtepu8_epi32(first_16_indices);

        // Subtract their position in the register (0..15).
        // This corresponds to calculating the high part value: position - rank.
        let range_0_15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let high_parts = _mm512_sub_epi32(expanded_indices, range_0_15);

        // Shift each value by bits_per_value.
        let shifted_high = _mm512_slli_epi32(high_parts, BITS);

        // Add the expanded low bits.
        _mm512_add_epi32(shifted_high, low_bits)
    }

    /// Decodes up to 16 values from the Elias-Fano representation.
    ///
    /// Updates internal state.
    /// Writes decoded values to `output` and returns the number of values written.
    #[target_feature(
        enable = "avx512f",
        enable = "avx512vbmi2",
        enable = "avx512bw",
        enable = "popcnt"
    )]
    pub unsafe fn decode_batch_impl(&mut self, output: &mut [u32]) -> usize {
        let mut current_word;
        let mut byte_idx;

        loop {
            byte_idx = self.high_idx / 8;
            let bit_offset = self.high_idx % 8;
            if byte_idx as usize >= self.high_bits.len() * 8 {
                return 0;
            }
            let ptr = self.high_bits.as_ptr() as *const u8;
            // Handle potential out-of-bounds read at the very end of the slice
            if (byte_idx as usize) + 8 > self.high_bits.len() * 8 {
                let mut word = 0u64;
                let len = self.high_bits.len() * 8 - byte_idx as usize;
                std::ptr::copy_nonoverlapping(
                    ptr.add(byte_idx as usize),
                    &mut word as *mut u64 as *mut u8,
                    len,
                );
                current_word = word;
            } else {
                current_word = ptr.add(byte_idx as usize).cast::<u64>().read_unaligned();
            }
            // Clear bits that have already been processed
            current_word &= !0u64 << bit_offset;
            if current_word != 0 {
                break;
            }
            // Skip the zeros in this window
            self.high_idx = (self.high_idx + 64) & !7;
        }

        let k = _popcnt64(current_word as i64) as usize;
        let count = k.min(16);

        // 1. Prepare High Bits
        let identity = _mm512_set_epi8(
            63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42,
            41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,
            19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        );
        let compressed_indices = _mm512_maskz_compress_epi8(current_word, identity);
        let first_16_indices = _mm512_castsi512_si128(compressed_indices);
        let expanded_indices = _mm512_cvtepu8_epi32(first_16_indices);

        // Calculate high parts: (position - rank_in_batch) + correction
        // correction = base_position - global_rank_start
        // base_position = byte_idx * 8
        let base_position = (byte_idx * 8) as i32;
        let correction = base_position - self.value_idx as i32;

        let range_0_15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        // (pos - rank_in_batch)
        let high_parts_base = _mm512_sub_epi32(expanded_indices, range_0_15);
        // Add correction
        let correction_vec = _mm512_set1_epi32(correction);
        let high_parts = _mm512_add_epi32(high_parts_base, correction_vec);

        // 2. Prepare Low Bits
        let low_bits_vec = self.get_low_bits();

        // 3. Combine
        let shifted_high = _mm512_slli_epi32(high_parts, BITS);
        let result_vec = _mm512_add_epi32(shifted_high, low_bits_vec);

        // 4. Store Result
        _mm512_storeu_si512(output.as_mut_ptr() as *mut _, result_vec);

        // 5. Update State
        // Find position of the last processed bit to update high_idx
        // We want the value at index (count - 1) from compressed_indices.
        // compressed_indices contains byte indices in the lower 128 bits (since count <= 16).
        let lower_128 = _mm512_castsi512_si128(compressed_indices);
        let last_pos = if count == 16 {
            _mm_extract_epi8(lower_128, 15) as u32
        } else {
            let lo = _mm_cvtsi128_si64(lower_128) as u64;
            let hi = _mm_extract_epi64(lower_128, 1) as u64;

            let index = count - 1;
            let val = if index < 8 { lo } else { hi };
            let shift = (index & 7) * 8;
            (val >> shift) as u32 & 0xFF
        };

        self.high_idx = byte_idx * 8 + last_pos + 1;
        self.value_idx += count as u32;
        count
    }
}

impl<'a, const BITS: u32> BatchDecoder for AvxBatchDecoder<'a, BITS> {
    fn decode_batch(&mut self, output: &mut [u32]) -> usize {
        unsafe { self.decode_batch_impl(output) }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::generate_markov_chain_data;

    use super::*;

    #[test]
    fn test_decode_first_16() {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512vbmi2")
            || !is_x86_feature_detected!("avx512bw")
        {
            println!("Skipping test: AVX-512 not supported on this hardware");
            return;
        }
        unsafe {
            // Construct high_bits with 16 set bits at specific positions.
            // Positions: 0, 2, 3, 5, 8, 10, 12, 15, 17, 20, 22, 25, 28, 30, 32, 35
            let positions = [0, 2, 3, 5, 8, 10, 12, 15, 17, 20, 22, 25, 28, 30, 32, 35];
            let mut high_bits: u64 = 0;
            for &p in &positions {
                high_bits |= 1u64 << p;
            }

            // Construct low_bits (arbitrary values)
            let low_vals: [i32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
            let low_bits = _mm512_loadu_si512(low_vals.as_ptr() as *const _);

            const BITS: u32 = 3;

            // Calculate expected result
            let mut expected = [0i32; 16];
            for i in 0..16 {
                let p = positions[i];
                let rank = i as i32;
                let high_part = (p as i32) - rank;
                expected[i] = (high_part << BITS) + low_vals[i];
            }

            // Decode
            let result_vec = AvxBatchDecoder::<'_, BITS>::decode(high_bits, low_bits);

            // Store result
            let mut result = [0i32; 16];
            _mm512_storeu_si512(result.as_mut_ptr() as *mut _, result_vec);

            assert_eq!(
                result, expected,
                "Decoded values do not match expected values"
            );
        }
    }

    #[test]
    fn test_decode_batch_1024() {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512vbmi2")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("popcnt")
        {
            println!("Skipping test: AVX-512 not supported on this hardware");
            return;
        }

        let data = generate_markov_chain_data(32 * 32, 12345);
        let max = *data.last().unwrap() + 1;
        let ef = EliasFano::new(data.iter().copied(), max, data.len() as u32);

        unsafe {
            let mut decoder = new_decoder(&ef);
            let mut decoded = Vec::new();
            let mut buffer = [0u32; 16];

            loop {
                let count = decoder.decode_batch(&mut buffer);
                if count == 0 {
                    break;
                }
                decoded.extend_from_slice(&buffer[..count]);
            }

            assert_eq!(decoded, data);
        }
    }
}
