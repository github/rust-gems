use std::ops::Range;

pub const MAX_DEPTH: u32 = 31;

pub struct BitReader<'a> {
    bits: &'a [u32],
    current: u32,
    index: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(bits: &'a [u32]) -> Self {
        Self { bits, current: 0, index: 0 }
    }
}

impl<'a> Iterator for BitReader<'a> {
    type Item = bool;

    #[inline]
    fn next(&mut self) -> Option<bool> {
        let bit = unsafe { *self.bits.get_unchecked(self.index as usize / 32) }
            & (1 << self.index % 32)
            != 0;
        /*if self.index % 32 == 0 {
            self.current = unsafe { *self.bits.get_unchecked(self.index as usize / 32) };
        }
        let bit = self.current & (1 << self.index % 32) != 0;*/
        self.index += 1;
        Some(bit)
    }
}

pub struct RunReader<'a> {
    words: &'a [u32],
    remaining: u64,
    index: u32,
    limit: u32,
}

impl<'a> RunReader<'a> {
    pub fn new(words: &'a [u32], limit: u32) -> Self {
        let remaining = words[0] as u64 + ((words[1] as u64) << 32);
        Self {
            words,
            index: 0,
            remaining,
            limit,
        }
    }
}

impl<'a> Iterator for RunReader<'a> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        /*if self.index > self.limit {
            return None;
        }*/
        /*let consumed = self.index % 32;
        let word = unsafe {(self.words.as_ptr().add(self.index as usize / 32) as *const u64).read_unaligned() >> consumed};
        let run = word.trailing_ones();
        self.index += run + 1;
        return Some(run);*/

        let consumed = self.index % 32;
        let remaining = self.remaining >> consumed;
        let run = remaining.trailing_ones();
        self.index += run + 1;
        if consumed + run >= 31 {
            // self.remaining = unsafe {(self.words.as_ptr().add(self.index as usize / 32) as *const u64).read_unaligned()};
            self.remaining = self.remaining >> 32;
            self.remaining |=
                unsafe { *self.words.get_unchecked((self.index / 32 + 1) as usize) as u64 } << 32;
        }
        Some(run)
    }
}

pub struct Decoder<T> {
    bits: T,
    index: u32,
}

impl<T> Decoder<T> {
    pub fn new(bits: T) -> Self {
        Self { bits, index: 0 }
    }
}

impl<T: Iterator<Item = bool>> Iterator for Decoder<T> {
    type Item = Range<u32>;

    fn next(&mut self) -> Option<Range<u32>> {
        let mut index = self.index;
        // Implementation without popcount!
        let mut depth = if index == 0 {
            1 << (MAX_DEPTH - 1)
        } else {
            index & !(index - 1)
        };
        while depth < (1 << MAX_DEPTH) {
            let bit = self.bits.next().unwrap();
            let bit = if depth == 1 {
                bit
            } else if bit {
                depth >>= 1;
                continue;
            } else {
                self.bits.next().unwrap()
            };
            index += depth;
            if bit {
                self.index = index;
                return Some(index - depth..index);
            }
            depth = index & !(index - 1);
        }
        None
    }
}

pub struct RunDecoder<T> {
    runs: T,
    index: u32,
    run: u32,
}

impl<T: Iterator<Item = u32>> RunDecoder<T> {
    pub fn new(mut runs: T) -> Self {
        let run = runs.next().unwrap();
        Self {
            runs,
            index: 0,
            run,
        }
    }
}

impl<T: Iterator<Item = u32>> Iterator for RunDecoder<T> {
    type Item = Range<u32>;

    fn next(&mut self) -> Option<Range<u32>> {
        let mut index = self.index;
        let mut depth = if index == 0 {
            1 << (MAX_DEPTH - 1)
        } else {
            index & !(index - 1)
        };
        let mut run = self.run;
        while depth < (1 << MAX_DEPTH) {
            if (depth >> run) > 1 {
                depth >>= run;
                run = self.runs.next().unwrap();
            } else {
                run -= depth.trailing_zeros();
                depth = 1;
            }
            let bit = run != 0;
            if run == 0 {
                run = self.runs.next().unwrap();
            } else {
                run -= 1;
            }
            index += depth;
            if bit {
                self.index = index;
                self.run = run;
                return Some(index - depth..index);
            }
            depth = index & !(index - 1);
        }
        None
    }
}

pub fn encode(mut runs: impl Iterator<Item = Range<u32>>) -> Vec<bool> {
    let mut res = vec![];

    let mut index = 0u32;
    let mut depth = MAX_DEPTH - 1;
    let mut run = runs.next().unwrap();
    while depth < MAX_DEPTH {
        if index >= run.end {
            run = runs.next().unwrap_or(1 << MAX_DEPTH..u32::MAX);
        }
        if index + (1 << depth) <= run.start {
            if depth > 0 {
                res.push(false);
            }
            res.push(false);
            index += 1 << depth;
            depth = index.trailing_zeros();
        } else if index < run.start {
            depth -= 1;
            res.push(true);
        } else if index + (1 << depth) <= run.end {
            if depth > 0 {
                res.push(false);
            }
            res.push(true);
            index += 1 << depth;
            depth = index.trailing_zeros();
        } else if index < run.end {
            depth -= 1;
            res.push(true);
        } else {
            panic!("something went wrong!");
        }
    }
    res
}

pub fn flatten(runs: impl Iterator<Item = Range<u32>>) -> Vec<u32> {
    runs.flat_map(|r| r.start..r.end).collect()
}

pub fn encode_runs(bits: impl Iterator<Item = bool>) -> Vec<u32> {
    let mut res = vec![];
    let mut run = 0;
    for bit in bits {
        if bit {
            run += 1;
        } else {
            res.push(run);
            run = 0;
        }
    }
    res.push(run);
    res
}

pub fn encode_ranges(bits: impl Iterator<Item = bool>) -> Vec<Range<u32>> {
    let mut res = vec![];
    let mut end = 0;
    let mut run = 0;
    for bit in bits {
        end += 1;
        if !bit {
            if run > 0 {
                res.push(end - run..end);
            }
            run = 0;
        } else {
            run += 1;
        }
    }
    if run > 0 {
        res.push(end - run..end);
    }
    res
}

pub fn encode_bits_into_bytes(bits: impl Iterator<Item = bool>) -> Vec<u8> {
    let mut res = vec![];
    let mut byte = 0u8;
    let mut index = 0;
    for bit in bits {
        if bit {
            byte |= 1 << index % 8;
        }
        index += 1;
        if index % 8 == 0 {
            res.push(byte);
            byte = 0;
        }
    }
    if index % 8 != 0 {
        res.push(byte);
    }
    res
}

pub fn encode_bits_into_words(bits: impl Iterator<Item = bool>) -> Vec<u32> {
    let mut res = vec![];
    let mut word = 0u32;
    let mut index = 0;
    for bit in bits {
        if bit {
            word |= 1 << index % 32;
        }
        index += 1;
        if index % 32 == 0 {
            res.push(word);
            word = 0;
        }
    }
    if index % 32 != 0 {
        res.push(word);
    }
    res.push(0); // Add a dummy 0 to avoid overflows during reading.
    res.push(0); // Add a dummy 0 to avoid overflows during reading.
    res
}

#[cfg(test)]
mod tests {
    use rand::{distr::Uniform, Rng};

    use crate::{
        encode, encode_bits_into_words, encode_ranges, encode_runs,
        flatten, BitReader, Decoder, RunDecoder, RunReader, MAX_DEPTH,
    };

    #[test]
    fn it_works() {
        let runs = encode_ranges(
            rand::rng()
                .sample_iter(Uniform::new(0u32, 2u32).unwrap())
                .map(|v| v != 0)
                .take(10000),
        );
        println!("ones: {}", runs.iter().map(|r| r.len()).sum::<usize>());
        // let runs = vec![3..8, 13..19, 63..129];
        let encoded = encode(runs.iter().cloned());
        let encoded_runs = encode_runs(encoded.iter().copied());
        println!("max run: {:?}", encoded_runs.iter().copied().max());
        assert!(encoded_runs.iter().all(|&v| v <= MAX_DEPTH));
        //println!("{encoded:?}");
        //println!("{encoded_runs:?}");
        let decoded: Vec<_> = Decoder::new(encoded.iter().copied()).collect();
        //println!("{decoded:?}");
        assert_eq!(flatten(decoded.into_iter()), flatten(runs.iter().cloned()));

        let decoded_runs: Vec<_> = RunDecoder::new(encoded_runs.iter().copied()).collect();
        //println!("{decoded_runs:?}");
        assert_eq!(
            flatten(decoded_runs.into_iter()),
            flatten(runs.iter().cloned())
        );

        // let encoded_bytes = encode_bits_into_bytes(encoded.iter().copied());
        let encoded_words = encode_bits_into_words(encoded.iter().copied());
        let decoded: Vec<_> = Decoder::new(BitReader::new(&encoded_words)).collect();
        assert_eq!(flatten(decoded.into_iter()), flatten(runs.iter().cloned()));
        assert_eq!(
            RunReader::new(&encoded_words, encoded.len() as u32).take(encoded_runs.len()).collect::<Vec<_>>(),
            encoded_runs
        );
    }
}
