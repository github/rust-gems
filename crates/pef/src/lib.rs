struct BitStream {
    data: Vec<u32>,
    bit_pos: u32,
    curr: u64,
}

impl BitStream {
    fn new() -> Self {
        Self {
            data: vec![],
            bit_pos: 0,
            curr: 0,
        }
    }

    fn push(&mut self, bits: u32, num_bits: u32) {
        self.curr |= (bits as u64) << (self.bit_pos % 32);
        if (self.bit_pos % 32) + num_bits >= 32 {
            self.data.push(self.curr as u32);
            self.curr >>= 32;
        }
        self.bit_pos += num_bits;
    }

    fn finish(mut self) -> (Vec<u64>, u32) {
        if self.bit_pos % 32 != 0 {
            self.data.push(self.curr as u32);
        }
        if self.data.len() % 2 != 0 {
            self.data.push(0);
        }
        let (head, _) = self.data.as_slice().as_chunks::<2>();
        let data = head
            .iter()
            .map(|[a, b]| (*b as u64) << 32 | *a as u64)
            .collect::<Vec<_>>();
        (data, self.bit_pos)
    }
}

struct EliasFano {
    bits_per_value: u32,
    high_bits: Vec<u64>,
    low_bits: Vec<u64>,
}

impl EliasFano {
    fn new(iter: impl Iterator<Item = u32>, max: u32, len: usize) -> Self {
        let bits_per_value = optimal_bits_per_value(max, len as u32);
        let mut high_bits = BitStream::new();
        let mut low_bits = BitStream::new();
        let mut bucket_id = 0;
        for value in iter {
            assert!(value < max);
            let high = value >> bits_per_value;
            let low = value & !(!0 << bits_per_value);
            while high > bucket_id {
                high_bits.push(0, 1);
                bucket_id += 1;
            }
            println!("{value} {bucket_id} {high} {low} {bits_per_value}");
            high_bits.push(1, 1);
            low_bits.push(low, bits_per_value);
        }
        high_bits.push(0, 1);
        EliasFano {
            bits_per_value,
            high_bits: high_bits.finish().0,
            low_bits: low_bits.finish().0,
        }
    }

    fn iter(&self) -> EliasFanoDecoder<'_> {
        EliasFanoDecoder {
            bits_per_value: self.bits_per_value,
            high_bits: BitReader::new(&self.high_bits),
            low_bits: BitReader::new(&self.low_bits),
            bucket_id: 0,
            bit_pos: 0,
        }
    }
}

struct EliasFanoDecoder<'a> {
    bits_per_value: u32,
    high_bits: BitReader<'a>,
    low_bits: BitReader<'a>,
    bucket_id: u32,
    bit_pos: u32,
}

struct BitReader<'a> {
    data: &'a [u64],
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u64]) -> Self {
        Self { data }
    }

    fn get_bit(&self, index: u32) -> bool {
        self.data[index as usize / 64] & (1 << (index % 64)) != 0
    }

    fn get_bits(&self, index: u32, num_bits: u32) -> u32 {
        let start = index / 64;
        let end = (index + num_bits) / 64;
        let mut temp = self.data[start as usize] >> (index % 64);
        if start != end {
            temp |= self.data[(start + 1) as usize] << (64 - (index % 64));
        }
        (temp as u32) & !(!0 << num_bits)
    }

    fn len(&self) -> u32 {
        (self.data.len() * 64) as u32
    }
}

impl<'a> Iterator for EliasFanoDecoder<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.bit_pos >= self.high_bits.len() {
                return None;
            }
            if self.high_bits.get_bit(self.bit_pos) {
                break;
            }
            self.bucket_id += 1;
            self.bit_pos += 1;
        }
        let value_idx = self.bit_pos - self.bucket_id;
        let value = self.low_bits.get_bits(value_idx, self.bits_per_value);
        self.bit_pos += 1;
        Some(value | (self.bucket_id << self.bits_per_value))
    }
}

fn optimal_bits_per_value(max: u32, len: u32) -> u32 {
    let mut best_cost = (u32::MAX, 0);
    for bits in 0..32 {
        let cost = bits * len as u32 + ((max >> bits) + 1) + len;
        if cost < best_cost.0 {
            best_cost = (cost, bits);
        }
    }
    best_cost.1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elias_fano() {
        let data = vec![1, 2, 5, 13, 15];
        let max = 16;
        let len = data.len();
        let elias_fano = EliasFano::new(data.iter().copied(), max, len);
        println!("{:b}", elias_fano.high_bits[0]);
        println!("{:b}", elias_fano.low_bits[0]);
        let mut decoder = elias_fano.iter();
        for value in data {
            assert_eq!(decoder.next(), Some(value));
        }
    }
}
