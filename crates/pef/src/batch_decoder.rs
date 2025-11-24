use crate::elias_fano::{BitReader, EliasFano};

pub struct EFBatchDecoder<'a> {
    bits_per_value: u32,
    high_bits: BitReader<'a>,
    low_bits: BitReader<'a>,
    value_idx: u32,
    word_idx: u32,
}

impl<'a> EFBatchDecoder<'a> {
    pub fn new(ef: &'a EliasFano) -> Self {
        Self {
            bits_per_value: ef.bits_per_value,
            high_bits: BitReader::new(&ef.high_bits),
            low_bits: BitReader::new(&ef.low_bits),
            value_idx: 0,
            word_idx: 0,
        }
    }

    pub fn decode_batch(&mut self, buffer: &mut [u32]) -> Option<usize> {
        assert!(buffer.len() >= 64);

        let mut count = 0;
        let word_idx = self.word_idx;
        let mut value_idx = self.value_idx;
        
        // Cache frequently accessed fields
        let bits_per_value = self.bits_per_value;
        let high_bits_len = self.high_bits.len();

        if word_idx * 64 >= high_bits_len {
            return None;
        }

        let mut current_word = self.high_bits.get_word(word_idx);
        let mut bit_pos = word_idx * 64;
        while current_word != 0 {
            let zeros = current_word.trailing_zeros();
            current_word >>= zeros + 1;
            
            let bucket_id = bit_pos + zeros - value_idx;
            let low = self.low_bits.get_bits(value_idx, bits_per_value);
            
            unsafe { *buffer.get_unchecked_mut(count) = low | (bucket_id << bits_per_value) };
            count += 1;
            
            bit_pos += zeros + 1;
            value_idx += 1;
        }
        self.word_idx = word_idx + 1;
        self.value_idx = value_idx;
        Some(count)
    }
}
