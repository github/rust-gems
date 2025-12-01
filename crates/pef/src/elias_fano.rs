struct BitStream {
    data: Vec<u64>,
}

impl BitStream {
    fn new(bits: u32) -> Self {
        Self {
            data: vec![0; ((bits + 63) / 64) as usize],
        }
    }

    fn set_bit(&mut self, index: u32) {
        self.data[index as usize / 64] |= 1 << (index % 64);
    }

    fn set_bits(&mut self, index: u32, value: u32, bits: u32) {
        let start = index / 64;
        let end = (index + bits - 1) / 64;
        self.data[start as usize] |= (value as u64) << (index % 64);
        if start != end {
            self.data[(start + 1) as usize] |= (value >> (64 - (index % 64))) as u64;
        }
    }

    fn finish(self) -> Vec<u64> {
        self.data
    }
}

pub struct EliasFano {
    pub(crate) bits_per_value: u32,
    pub(crate) high_bits: Vec<u64>,
    pub(crate) low_bits: Vec<u64>,
}

impl EliasFano {
    pub fn new(iter: impl Iterator<Item = u32>, max: u32, len: u32) -> Self {
        let (_bits, bits_per_value) = optimal_bits_per_value(max, len as u32);
        // TODO: we want to get the buffers as input...
        // In particular in the partitioned case, we somehow want all the
        // high bits in one stream and all the low bits in another stream.
        // The question is how to store the starting positions of each partition
        // most efficiently...
        let mut high_bits = BitStream::new(((max >> bits_per_value) + 1) + len);
        // Note: add some buffer, so that we don't have to worry about out-of-bounds reads!
        let mut low_bits = BitStream::new(bits_per_value * (len + 15));
        for (i, value) in iter.enumerate() {
            assert!(value < max);
            let high = value >> bits_per_value;
            let low = value & !(!0 << bits_per_value);
            high_bits.set_bit(high + i as u32);
            low_bits.set_bits((i as u32) * bits_per_value, low, bits_per_value);
        }
        EliasFano {
            bits_per_value,
            high_bits: high_bits.finish(),
            low_bits: low_bits.finish(),
        }
    }

    pub fn iter(&self) -> EliasFanoDecoder<'_> {
        let high_bits = BitReader::new(&self.high_bits);
        EliasFanoDecoder {
            current_word: high_bits.get_word(0),
            bits_per_value: self.bits_per_value,
            high_bits,
            low_bits: BitReader::new(&self.low_bits),
            value_idx: 0,
            bit_pos: 0,
        }
    }
}

pub struct EliasFanoDecoder<'a> {
    pub(crate) bits_per_value: u32,
    pub(crate) high_bits: BitReader<'a>,
    pub(crate) low_bits: BitReader<'a>,
    pub(crate) value_idx: u32,
    pub(crate) bit_pos: u32,
    pub(crate) current_word: u64,
}

pub(crate) struct BitReader<'a> {
    data: &'a [u64],
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u64]) -> Self {
        Self { data }
    }

    fn get_bit(&self, index: u32) -> bool {
        self.data[index as usize / 64] & (1 << (index % 64)) != 0
    }

    pub(crate) fn get_word(&self, index: u32) -> u64 {
        unsafe { *self.data.get_unchecked(index as usize) }
    }

    pub(crate) fn get_word_unaligned(&self, byte_pos: u32) -> u64 {
        unsafe {
            let ptr = self.data.as_ptr() as *const u8;
            std::ptr::read_unaligned(ptr.add(byte_pos as usize) as *const u64)
        }
    }

    pub(crate) fn get_bits(&self, index: u32, num_bits: u32) -> u32 {
        let start = index / 64;
        let end = (index + num_bits - 1) / 64;
        let mut temp = self.data[start as usize] >> (index % 64);
        if start != end {
            temp |= self.data[(start + 1) as usize] << (64 - (index % 64));
        }
        (temp as u32) & !(!0 << num_bits)
    }

    pub(crate) fn len(&self) -> u32 {
        (self.data.len() * 64) as u32
    }
}

impl<'a> Iterator for EliasFanoDecoder<'a> {
    type Item = u32;

    // Decoding 32 values in a row could be done more efficient by
    // a) expanding the low bits into a full words.
    // b) using masked SIMD operations to process multiple high bits in parallel
    //    but masking those with 0 bits.
    // c) adding the two results together.
    // Note: this is VERY similar to how bit packing with exceptions works.
    // The main difference is that we can use the high bits only to skip forward!
    // Advantages of EF are that:
    // 1. we don't need to store deltas explicitly, i.e.
    //    no need for partial sum computations as final step.
    // 2. we can skip ahead and decode individual values only.
    // 3. NEW: we can intersect quickly purely based on high bits.
    // 4. Due to 2, EF leads to a compact two-layer representation, while
    //    preserving skip list functionality.
    //    - start value of each block (<- EF)
    //      (encode number of low bits in start value with trailing ones!)
    //    - start bit/byte of high bits (<- EF)
    //   [- start bit/byte of low bits (<- EF)]
    //      (these could also be stored in reverse order ending at next partition)
    //   [- counting ones in high bit representation]
    //      (succinct; only needed for interval encoding)
    // 5. NEW: EF can be used for interval encodings, where it is
    //    important to know the index of each value.
    // 6. The encoding size is fully deterministic and can be computed with
    //    simple bit operations.
    // So, in total, the EF encoding is much more flexible than bin-packing.
    fn next(&mut self) -> Option<Self::Item> {
        let mut current_word = self.current_word;
        let mut bit_pos = self.bit_pos;
        loop {
            let zeros = current_word.trailing_zeros();
            if zeros < 64 {
                current_word >>= zeros;
                current_word >>= 1;
                self.current_word = current_word;
                let bucket_id = bit_pos + zeros - self.value_idx;
                let value = self
                    .low_bits
                    .get_bits(self.value_idx * self.bits_per_value, self.bits_per_value);
                self.bit_pos = bit_pos + zeros + 1;
                self.value_idx += 1;
                if self.bit_pos & 63 == 0 {
                    self.bit_pos -= 1;
                }
                return Some(value | (bucket_id << self.bits_per_value));
            } else if bit_pos + 64 >= self.high_bits.len() {
                return None;
            } else {
                bit_pos = (bit_pos + 64) & !63;
                current_word = self.high_bits.get_word(bit_pos / 64);
            }
        }
    }
}

impl<'a> EliasFanoDecoder<'a> {
    /// Skips forward to the given target value.
    /// Returns `Some(value)` if the target exists in the sequence,
    /// or `Some(next_value)` where `next_value > target` is the next value in the sequence,
    /// or `None` if there are no more values >= target.
    pub fn skip_to(&mut self, target: u32) -> Option<u32> {
        while let Some(value) = self.next() {
            if value >= target {
                return Some(value);
            }
        }
        None
    }

    /// Returns an iterator that yields values present in both this sequence
    /// and the provided iterator. Both must yield values in sorted order.
    pub fn intersect<I: Iterator<Item = u32>>(self, other: I) -> Intersection<'a, I> {
        Intersection {
            ef_decoder: self,
            other,
        }
    }
}

pub struct Intersection<'a, I> {
    ef_decoder: EliasFanoDecoder<'a>,
    other: I,
}

impl<'a, I: Iterator<Item = u32>> Iterator for Intersection<'a, I> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        let mut ef_val = None;
        while let Some(other_val) = self.other.next() {
            if Some(other_val) < ef_val {
                continue;
            } else if Some(other_val) == ef_val {
                // Found a match
                return ef_val;
            }
            ef_val = self.ef_decoder.skip_to(other_val);
            if ef_val == None {
                return None;
            } else if ef_val == Some(other_val) {
                // Found a match
                return ef_val;
            }
        }
        None
    }
}

// TODO: Implement an iterator which only retrieves the high bit buckets.
// Such an iterator can be used to approximate an intersection!
// This way, one can potentially erase parts before running a second pass
// where the low values are being decoded...
// When does this work?
// Usually, one would start off the intersection with the shortest posting list.
// This posting list has the coarsest high mask granularity, i.e. for a random
// sequence, 40% of the buckets will have a value.
// If the second posting list is also random and at the same level, then
// we get 40% * 40% = 16% of the buckets remaining.
// If the second posting list is at a higher level, i.e. more dense, then
// the fill level would be roughly 63% and the intersection would be 40% * 63% = 25%.
// For another level difference, the fill level increases to 86% ==> 34% remaining.
// I.e. for more than 2 levels difference there is no benefit of this approach!
// The fill rate can be reduced by spending more bits per value.
// For example moving one level down, increases the number of bits from:
// n * log(u/n) + 2n to n * log(u/n) + 4n.

// TODO: Implement an interval based iterator where the even indices represent
// the start of an interval and the odd indices represent the end.
// In this case, the intersection of two iterators   cannot be represented as a
// simple bitmask unfortunately. Instead, one has to serialize the new
// set.
// If however one of the two sets is represented as points, then the masking
// works again... One has to decide however, whether too many points would be
// filtered out and therefore the bitmask would become larger than the encoded
// representation.

// TODO: Implement an intersecting iterator which masks off values from the first
// iterator as we go.

// TODO: Implement test presence of a value. This requires the rank/select stuff...
//
fn optimal_bits_per_value(max: u32, len: u32) -> (u32, u32) {
    let mut best_cost = (u32::MAX, 0);
    for bits in 0..32 {
        let cost = bits * len as u32 + ((max >> bits) + 1) + len;
        if cost < best_cost.0 {
            best_cost = (cost, bits);
        }
    }
    best_cost
}

#[cfg(test)]
pub mod test_utils {
    use rand::prelude::*;

    pub fn generate_markov_chain_data(len: usize, seed: u64) -> Vec<u32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Vec::with_capacity(len);
        let mut current = 0;
        let mut state = 0;
        for _ in 0..len {
            let gap = if state == 0 {
                if rng.random_bool(0.1) {
                    state = 1;
                } // Transition to sparse
                if rng.random_bool(0.9) {
                    1
                } else {
                    (rng.random::<u32>() % 5) + 1
                }
            } else {
                if rng.random_bool(0.1) {
                    state = 0;
                } // Transition to dense
                (rng.random::<u32>() % 100) + 1
            };

            current += gap;
            data.push(current);
        }
        data
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;

    use crate::elias_fano::test_utils::generate_markov_chain_data;

    use super::*;

    #[test]
    fn test_elias_fano() {
        let data = vec![1, 2, 5, 13, 15];
        let max = 16;
        let len = data.len();
        let elias_fano = EliasFano::new(data.iter().copied(), max, len as u32);
        println!("{:b}", elias_fano.high_bits[0]);
        println!("{:b}", elias_fano.low_bits[0]);
        let mut decoder = elias_fano.iter();
        for value in data {
            assert_eq!(decoder.next(), Some(value));
        }
    }

    #[test]
    fn test_elias_fano_decoder() {
        let data = generate_markov_chain_data(32 * 32, 12345);
        let max = *data.last().unwrap() + 1;
        let ef = EliasFano::new(data.iter().copied(), max, data.len() as u32);

        let decoder = ef.iter();
        let decoded = decoder.collect::<Vec<_>>();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_intersect_basic() {
        let data1 = vec![1, 5, 10, 15, 20, 100, 200, 500];
        let data2 = vec![2, 5, 10, 12, 20, 50, 200, 300, 500];

        let max = 501;
        let ef = EliasFano::new(data1.iter().copied(), max, data1.len() as u32);

        let intersection: Vec<u32> = ef.iter().intersect(data2.into_iter()).collect();
        assert_eq!(intersection, vec![5, 10, 20, 200, 500]);
    }

    #[test]
    fn test_intersect_empty_result() {
        let data1 = vec![1, 3, 5, 7];
        let data2 = vec![2, 4, 6, 8];

        let max = 10;
        let ef = EliasFano::new(data1.iter().copied(), max, data1.len() as u32);

        let intersection: Vec<u32> = ef.iter().intersect(data2.into_iter()).collect();
        assert!(intersection.is_empty());
    }

    #[test]
    fn test_intersect_all_match() {
        let data = vec![1, 5, 10, 15, 20];

        let max = 21;
        let ef = EliasFano::new(data.iter().copied(), max, data.len() as u32);

        let intersection: Vec<u32> = ef.iter().intersect(data.clone().into_iter()).collect();
        assert_eq!(intersection, data);
    }

    #[test]
    fn test_intersect_two_ef() {
        let data1 = generate_markov_chain_data(1000, 11111);
        let data2 = generate_markov_chain_data(500, 22222);

        let max1 = *data1.last().unwrap() + 1;
        let max2 = *data2.last().unwrap() + 1;

        let ef1 = EliasFano::new(data1.iter().copied(), max1, data1.len() as u32);
        let ef2 = EliasFano::new(data2.iter().copied(), max2, data2.len() as u32);

        // Compute expected intersection
        let set1: std::collections::HashSet<u32> = data1.iter().copied().collect();
        let expected: Vec<u32> = data2.iter().copied().filter(|v| set1.contains(v)).collect();

        let intersection: Vec<u32> = ef1.iter().intersect(ef2.iter()).collect();
        assert_eq!(intersection, expected);
    }

    #[test]
    fn test_intersect_first_empty() {
        let data1: Vec<u32> = vec![];
        let data2 = vec![1, 2, 3];

        let max = 10;
        let ef = EliasFano::new(data1.iter().copied(), max, data1.len() as u32);

        let intersection: Vec<u32> = ef.iter().intersect(data2.into_iter()).collect();
        assert!(intersection.is_empty());
    }

    #[test]
    fn test_intersect_second_empty() {
        let data1 = vec![1, 2, 3];
        let data2: Vec<u32> = vec![];

        let max = 10;
        let ef = EliasFano::new(data1.iter().copied(), max, data1.len() as u32);

        let intersection: Vec<u32> = ef.iter().intersect(data2.into_iter()).collect();
        assert!(intersection.is_empty());
    }

    #[test]
    fn test_intersect_large_gap() {
        let data1 = vec![1, 1000000];
        let data2 = vec![1, 500000, 1000000];

        let max = 1000001;
        let ef = EliasFano::new(data1.iter().copied(), max, data1.len() as u32);

        let intersection: Vec<u32> = ef.iter().intersect(data2.into_iter()).collect();
        assert_eq!(intersection, vec![1, 1000000]);
    }

    #[test]
    fn test_optimal_bits() {
        for i in 1..100 {
            for max in i..300 {
                let max = max - i;
                let (cost, bits) = optimal_bits_per_value(max, i);
                let b_f32 = (max as f32 / i as f32).log2();
                let b = (max / i).max(1).ilog2();
                let fast_cost = b * i + max / (1 << b) + i + 1;
                if fast_cost != cost {
                    println!(
                        "{i} cost: {}, bits: {} {b_f32} {b} {}",
                        cost, bits, fast_cost
                    );
                }
            }
        }
    }
}
