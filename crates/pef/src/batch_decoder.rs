use crate::elias_fano::{BitReader, EliasFano};

const BATCH_SIZE: usize = 8;

pub struct EFBatchDecoder<'a> {
    bits_per_value: u32,
    high_bits: BitReader<'a>,
    low_bits: BitReader<'a>,
    value_idx: u32,
    bit_pos: u32,
    buffer: [u32; BATCH_SIZE],
}

impl<'a> EFBatchDecoder<'a> {
    pub fn new(ef: &'a EliasFano) -> Self {
        let high_bits = BitReader::new(&ef.high_bits);
        Self {
            bits_per_value: ef.bits_per_value,
            high_bits,
            low_bits: BitReader::new(&ef.low_bits),
            value_idx: 0,
            bit_pos: 0,
            buffer: [0; BATCH_SIZE],
        }
    }

    #[inline]
    pub fn decode_batch(&mut self) -> &[u32] {
        // Cache frequently accessed fields
        let high_bits_len = self.high_bits.len();
        let mut bit_pos = self.bit_pos;
        if bit_pos >= high_bits_len {
            return &[];
        }
        let mut value_idx = self.value_idx;
        let mut count = 0;
        loop {
            let current_word =  self.high_bits.get_word_unaligned(bit_pos / 8);
            let total_ones = (current_word >> (bit_pos % 8)).count_ones();
            let ones = total_ones.min((BATCH_SIZE - count) as u32);
            bit_pos = self.process_word_dispatch(ones, current_word, bit_pos, value_idx, count);
            value_idx += ones;
            count += ones as usize;
            if count == BATCH_SIZE {
                break;
            }
            if bit_pos >= high_bits_len {
                break;
            }
        }
        self.bit_pos = bit_pos;
        self.value_idx = value_idx;
        &self.buffer[0..count]
    }

    #[inline(always)]
    fn process_word_dispatch(
        &mut self,
        ones: u32,
        current_word: u64,
        bit_pos: u32,
        value_idx: u32,
        count: usize,
    ) -> u32 {
        match ones {
            0 => (bit_pos + 64) & !7,
            1 => self.process_word::<1>(current_word, bit_pos, value_idx, count),
            2 => self.process_word::<2>(current_word, bit_pos, value_idx, count),
            3 => self.process_word::<3>(current_word, bit_pos, value_idx, count),
            4 => self.process_word::<4>(current_word, bit_pos, value_idx, count),
            5 => self.process_word::<5>(current_word, bit_pos, value_idx, count),
            6 => self.process_word::<6>(current_word, bit_pos, value_idx, count),
            7 => self.process_word::<7>(current_word, bit_pos, value_idx, count),
            8 => self.process_word::<8>(current_word, bit_pos, value_idx, count),
            /*9 => self.process_word::<9>(current_word, bit_pos, value_idx, count),
            10 => self.process_word::<10>(current_word, bit_pos, value_idx, count),
            11 => self.process_word::<11>(current_word, bit_pos, value_idx, count),
            12 => self.process_word::<12>(current_word, bit_pos, value_idx, count),
            13 => self.process_word::<13>(current_word, bit_pos, value_idx, count),
            14 => self.process_word::<14>(current_word, bit_pos, value_idx, count),
            15 => self.process_word::<15>(current_word, bit_pos, value_idx, count),
            16 => self.process_word::<16>(current_word, bit_pos, value_idx, count),
            17 => self.process_word::<17>(current_word, bit_pos, value_idx, count),
            18 => self.process_word::<18>(current_word, bit_pos, value_idx, count),
            19 => self.process_word::<19>(current_word, bit_pos, value_idx, count),
            20 => self.process_word::<20>(current_word, bit_pos, value_idx, count),
            21 => self.process_word::<21>(current_word, bit_pos, value_idx, count),
            22 => self.process_word::<22>(current_word, bit_pos, value_idx, count),
            23 => self.process_word::<23>(current_word, bit_pos, value_idx, count),
            24 => self.process_word::<24>(current_word, bit_pos, value_idx, count),
            25 => self.process_word::<25>(current_word, bit_pos, value_idx, count),
            26 => self.process_word::<26>(current_word, bit_pos, value_idx, count),
            27 => self.process_word::<27>(current_word, bit_pos, value_idx, count),
            28 => self.process_word::<28>(current_word, bit_pos, value_idx, count),
            29 => self.process_word::<29>(current_word, bit_pos, value_idx, count),
            30 => self.process_word::<30>(current_word, bit_pos, value_idx, count),
            31 => self.process_word::<31>(current_word, bit_pos, value_idx, count),
            32 => self.process_word::<32>(current_word, bit_pos, value_idx, count),*/
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    fn process_word<const N: u32>(
        &mut self,
        mut current_word: u64,
        bit_pos: u32,
        value_idx: u32,
        count: usize,
    ) -> u32 {
        let mut bucket_id = bit_pos - value_idx;
        current_word >>= bit_pos % 8;
        for i in 0..N {
            let zeros = current_word.trailing_zeros();
            current_word >>= zeros + 1;
            bucket_id += zeros;
            let low = self
                .low_bits
                .get_bits((value_idx + i) * self.bits_per_value, self.bits_per_value);
            unsafe {
                *self.buffer.get_unchecked_mut(count + i as usize) = low | (bucket_id << self.bits_per_value);
            }
        }
        bucket_id + value_idx + N
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

