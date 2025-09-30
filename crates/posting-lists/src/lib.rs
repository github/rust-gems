pub struct TebIterator<'a> {
    data: &'a [u64],
    index: u32,
    bit_pos: u32,

    ranges: [u32; 8],
    range_idx: u32,
}

impl<'a> TebIterator<'a> {
    pub fn new(data: &'a [u64]) -> Self {
        Self {
            data,
            index: 0x80000000,
            bit_pos: Default::default(),
            ranges: Default::default(),
            range_idx: Default::default(),
        }
    }

    pub fn decode_batch(&mut self) {
        let mut index = self.index;
        let mut bit_pos = self.bit_pos;
        let mut range_idx = self.range_idx;
        let mut data = self.data[(bit_pos / 64) as usize];
        while data != 0 {
            let level = index.trailing_zeros();
            let d = data as u32;
            let b = d & 1;
            //let down = ((d >> 1)).trailing_zeros().min(level);
            let down = (d >> 1).trailing_zeros();
            let end = index + (1 << (level - down));
            unsafe { *self.ranges.get_unchecked_mut(range_idx as usize) = index & !0x80000000};
            range_idx += (range_idx & 1) ^ b;
            // let bits = down + 1 + (down != level) as u32;
            let bits = down + 2;
            bit_pos += bits;
            index = end;
            data >>= bits;
        }
        self.index = index;
        self.bit_pos = bit_pos;
        self.range_idx = range_idx;
    }
}

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
        self.bit_pos += num_bits;
        if self.bit_pos % 64 >= 32 {
            self.data.push(self.curr as u32);
            self.curr >>= 32;
            self.bit_pos -= 32;
        }
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

struct Encoder {
    index: u32,
    out: BitStream,
}

impl Encoder {
    fn new() -> Self {
        Encoder {
            index: 0x80000000,
            out: BitStream::new(),
        }
    }

    fn encode_till(&mut self, end: u32, flag: bool) {
        let end = end | 0x80000000;
        while self.index < end {
            // let level = (self.index | 0x8000000).trailing_zeros();
            let level = (self.index).trailing_zeros();
            let rest = self.index ^ end;
            let shift = if level < 32 && rest >= (1 << level) {
                self.index += 1 << level;
                0
            } else {
                let down = 31 - rest.leading_zeros();
                self.index += 1 << down;
                level - down
            };
            if false && level == shift {
                self.out.push(flag as u32, shift + 1);
            } else {
                self.out.push((2 << shift) | (flag as u32), shift + 2);
            }
        }
    }

    fn encode(mut self, ranges: &[u32]) -> (Vec<u64>, u32) {
        for [start, end] in ranges.as_chunks().0 {
            self.encode_till(*start, false);
            self.encode_till(*end, true);
        }
        self.index &= !0x80000000;
        self.encode_till(1 << 31, false);
        self.out.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding() {
        let encoder = Encoder::new();
        let seq = [2, 3, 5, 7, 10, 15];
        //let seq = [1002, 1003, 1005, 1007];
        let postings = seq.as_chunks().0.iter().map(|[a, b]| *b - *a).sum::<u32>();
        let (data, _) = encoder.encode(&seq);
        println!("{:064b} {:064b}", data[1], data[0]);

        let mut iter = TebIterator::new(&data);
        iter.decode_batch();
        assert_eq!(&iter.ranges[..iter.range_idx as usize], &seq);

        let start = std::time::Instant::now();
        let iterations = 1000000;
        for _ in 0..iterations {
            std::hint::black_box({
                let mut iter = TebIterator::new(&data);
                iter.decode_batch();
            });
        }
        let duration = start.elapsed();
        println!(
            "Decoding took: {:?} {:?} {:?}",
            iterations as f32 / duration.as_secs_f32(),
            postings as f32 * iterations as f32 / duration.as_secs_f32(),
            duration / 13 / iterations
        );
    }

    #[test]
    fn test_decoding() {
        let data = [
            0b11111110100000010101110011000000,
            0b00110011001100110011001100110011,
        ];
        let mut iter = TebIterator::new(&data);
        //iter.decode_batch();
        //assert_eq!(iter.ranges[..iter.range_idx as usize], [0, 4, 8, 12]);
    }
}
