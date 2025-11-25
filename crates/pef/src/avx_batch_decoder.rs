use crate::elias_fano::EliasFano;
use std::arch::x86_64::*;

pub struct AvxBatchDecoder<'a> {
    high_bits: &'a [u64],
    low_bits: &'a [u64],
    bits_per_value: u32,
    current_word: u64,
    high_idx: usize,
    value_idx: usize,
}

impl<'a> AvxBatchDecoder<'a> {
    pub fn new(ef: &'a EliasFano) -> Self {
        Self {
            high_bits: &ef.high_bits,
            low_bits: &ef.low_bits,
            bits_per_value: ef.bits_per_value,
            current_word: 0,
            high_idx: 0,
            value_idx: 0,
        }
    }

    /// Decodes a batch of 16 values using AVX-512 instructions.
    ///
    /// # Safety
    /// This function requires AVX-512F, AVX-512VBMI2, and AVX-512BW features.
    #[target_feature(enable = "avx512f", enable = "avx512vbmi2", enable = "avx512bw")]
    pub unsafe fn decode(high_bits: u64, low_bits: __m512i, bits_per_value: u32) -> __m512i {
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
        let shift_count = _mm_cvtsi32_si128(bits_per_value as i32);
        let shifted_high = _mm512_sll_epi32(high_parts, shift_count);

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
    pub unsafe fn decode_batch(&mut self, output: &mut [u32]) -> usize {
        // Ensure we have a non-zero current word
        while self.current_word == 0 {
            if self.high_idx >= self.high_bits.len() {
                return 0;
            }
            self.current_word = *self.high_bits.get_unchecked(self.high_idx);
            self.high_idx += 1;
        }

        let k = _popcnt64(self.current_word as i64) as usize;
        let count = k.min(16);

        // 1. Prepare High Bits
        let identity = _mm512_set_epi8(
            63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42,
            41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,
            19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        );
        let compressed_indices = _mm512_maskz_compress_epi8(self.current_word, identity);
        let first_16_indices = _mm512_castsi512_si128(compressed_indices);
        let expanded_indices = _mm512_cvtepu8_epi32(first_16_indices);

        // Calculate high parts: (position - rank_in_batch) + correction
        // correction = base_position - global_rank_start
        // base_position = (high_idx - 1) * 64
        let base_position = (self.high_idx - 1) as i32 * 64;
        let correction = base_position - self.value_idx as i32;

        let range_0_15 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        // (pos - rank_in_batch)
        let high_parts_base = _mm512_sub_epi32(expanded_indices, range_0_15);
        // Add correction
        let correction_vec = _mm512_set1_epi32(correction);
        let high_parts = _mm512_add_epi32(high_parts_base, correction_vec);

        // 2. Prepare Low Bits
        let mut low_vals = [0i32; 16];
        let mut bit_idx = self.value_idx as u32 * self.bits_per_value;
        for i in 0..count {
            // Inline get_bits logic for simplicity and performance
            let start = bit_idx / 64;
            let end = (bit_idx + self.bits_per_value - 1) / 64;
            let offset = bit_idx % 64;
            let mut val = self.low_bits.get_unchecked(start as usize) >> offset;
            if start != end {
                val |= self.low_bits.get_unchecked((start + 1) as usize) << (64 - offset);
            }
            low_vals[i] = (val as u32 & !(!0 << self.bits_per_value)) as i32;
            bit_idx += self.bits_per_value;
        }
        let low_bits_vec = _mm512_loadu_si512(low_vals.as_ptr() as *const _);

        // 3. Combine
        let shift_count = _mm_cvtsi32_si128(self.bits_per_value as i32);
        let shifted_high = _mm512_sll_epi32(high_parts, shift_count);
        let result_vec = _mm512_add_epi32(shifted_high, low_bits_vec);

        // 4. Store Result
        let mut temp_output = [0u32; 16];
        _mm512_storeu_si512(temp_output.as_mut_ptr() as *mut _, result_vec);
        output[..count].copy_from_slice(&temp_output[..count]);

        // 5. Update State
        // Find position of the last processed bit to clear bits in current_word
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

        let mask = (!0u64 << 1) << last_pos;
        self.current_word &= mask;

        self.value_idx += count;
        count
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

            let bits_per_value = 3;

            // Calculate expected result
            let mut expected = [0i32; 16];
            for i in 0..16 {
                let p = positions[i];
                let rank = i as i32;
                let high_part = (p as i32) - rank;
                expected[i] = (high_part << bits_per_value) + low_vals[i];
            }

            // Decode
            let result_vec = AvxBatchDecoder::<'_>::decode(high_bits, low_bits, bits_per_value);

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
            let mut decoder = AvxBatchDecoder::new(&ef);
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
