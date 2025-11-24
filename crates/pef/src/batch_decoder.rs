use crate::elias_fano::{BitReader, EliasFano};

pub struct EFBatchDecoder<'a> {
    bits_per_value: u32,
    high_bits: BitReader<'a>,
    low_bits: BitReader<'a>,
    value_idx: u32,
    bit_pos: u32,
    current_word: u64,
    buffer: [u32; 32],
}

impl<'a> EFBatchDecoder<'a> {
    pub fn new(ef: &'a EliasFano) -> Self {
        let high_bits = BitReader::new(&ef.high_bits);
        Self {
            bits_per_value: ef.bits_per_value,
            current_word: high_bits.get_word(0),
            high_bits,
            low_bits: BitReader::new(&ef.low_bits),
            value_idx: 0,
            bit_pos: 0,
            buffer: [0; 32],
        }
    }

    #[inline]
    pub fn decode_batch(&mut self) -> &[u32] {
        let mut current_word = self.current_word;
        let mut bit_pos = self.bit_pos;
        let mut value_idx = self.value_idx;
        
        // Cache frequently accessed fields
        let bits_per_value = self.bits_per_value;
        let high_bits_len = self.high_bits.len();

        for count in 0..32 {
            while current_word == 0 {
                let next_word_idx = bit_pos / 64 + 1;
                let next_bit_pos = next_word_idx * 64;
                
                if next_bit_pos >= high_bits_len {
                    self.current_word = 0;
                    self.bit_pos = next_bit_pos;
                    self.value_idx = value_idx;
                    return &self.buffer[..count];
                }
                current_word = self.high_bits.get_word(next_word_idx);
                bit_pos = next_bit_pos;
            }

            let zeros = current_word.trailing_zeros();
            current_word >>= zeros + 1;
            
            let bucket_id = bit_pos + zeros - value_idx;
            let low = self.low_bits.get_bits(value_idx * bits_per_value, bits_per_value);
            
            unsafe { *self.buffer.get_unchecked_mut(count) = low | (bucket_id << bits_per_value) };
            
            bit_pos += zeros + 1;
            value_idx += 1;

            if bit_pos % 64 == 0 {
                if bit_pos >= high_bits_len {
                    current_word = 0;
                } else {
                    current_word = self.high_bits.get_word(bit_pos / 64);
                }
            }
        }

        self.current_word = current_word;
        self.bit_pos = bit_pos;
        self.value_idx = value_idx;
        
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elias_fano::EliasFano;
    use rand::prelude::*;

    #[test]
    fn test_batch_decoder() {
        let mut rng = StdRng::seed_from_u64(12345);
        let mut data = Vec::new();
        let mut current = 0;
        for _ in 0..1000 {
            current += (rng.random::<u32>() % 10) + 1;
            data.push(current);
        }
        let max = *data.last().unwrap() + 1;
        let ef = EliasFano::new(data.iter().copied(), max, data.len() as u32);
        
        let mut decoder = EFBatchDecoder::new(&ef);
        let mut decoded = Vec::new();
        
        loop {
            let batch = decoder.decode_batch();
            if batch.is_empty() {
                break;
            }
            decoded.extend_from_slice(batch);
        }
        
        assert_eq!(decoded, data);
    }
}
