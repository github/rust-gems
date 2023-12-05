use std::ops::Range;

use crate::footer;
use crate::read_uint::read_u128_le;

type Overflow = u32;
type Offset = u32;
const OVERFLOW_SIZE: usize = std::mem::size_of::<Overflow>();
const OFFSET_SIZE: usize = std::mem::size_of::<Offset>();
const BLOCK_SIZE: usize = 16;
const BLOCK_OVERFLOW_MASK: u128 = 0x80808080808080808080808080808080;

// TODO: Reduce the number of fields in the struct, since they are heavily redundant.
// Instead, introduce some functions to derive fields on demand. But, these optimizations
// should be driven by benchmarks.
#[derive(Clone, Default)]
pub struct PSVec<'a> {
    values: &'a [u8],
    overflow_counts: &'a [Overflow],
    offsets: &'a [Offset],
}

pub struct PSVecBuilder<W: std::io::Write> {
    writer: W,
    bytes: Vec<u8>,
    offsets: Vec<Offset>,
    overflow_counts: Vec<Overflow>,
    partial_sum: u32,
}

pub struct PSVecIterator<'a> {
    values: &'a [u8],
    offsets: &'a [Offset],
    offset: Offset,
    first: bool,
}

impl<W: std::io::Write> PSVecBuilder<W> {
    pub fn from_deltas(writer: W, deltas: &[u32]) -> std::io::Result<usize> {
        let mut builder = Self::new(writer);
        for value in deltas {
            builder.push(*value)?;
        }
        builder.finish()
    }

    pub fn new(writer: W) -> Self {
        Self {
            writer,
            bytes: Vec::new(),
            offsets: Vec::new(),
            overflow_counts: Vec::new(),
            partial_sum: 0,
        }
    }

    pub fn push(&mut self, delta: u32) -> std::io::Result<()> {
        self.partial_sum += delta;

        let current_offset = self.offsets.last().unwrap_or(&0);
        self.bytes.push(if self.partial_sum - current_offset < 128 {
            (self.partial_sum - current_offset) as u8
        } else {
            // The value is too big, write to offset layer
            self.offsets.push(self.partial_sum);
            128
        });

        if self.bytes.len() % BLOCK_SIZE == 0 {
            self.overflow_counts.push(self.offsets.len() as Overflow);
        }
        Ok(())
    }

    /// If we store sorted values, then we can store up to 5 small values without a footer.
    /// Also, decoding can be optimized in this case.
    /// While this won't help for normal PSStruct encodings, this can be used to store e.g. one bit
    /// positions of masks in posting lists more efficiently!
    /// Note: When decoding the function `from_bytes_vec_only` MUST be called!
    pub fn finish_vec_only(mut self) -> std::io::Result<usize> {
        // As soon as one number overflows, the size of the serialized representation becomes immediately 6 bytes:
        // 1 byte for the packed content, 4 bytes for the offset, and 1 byte for the footer!
        // That means that we can safely assume that less than 6 bytes represent only numbers without overflow.
        // In this case, we also don't have to store the footer, since the number of bytes corresponds to the
        // number of values.
        if self.bytes.len() <= 5 && self.offsets.is_empty() {
            self.writer.write_all(&self.bytes)?;
            Ok(self.bytes.len())
        } else {
            self.finish()
        }
    }

    pub fn finish(mut self) -> std::io::Result<usize> {
        if self.bytes.is_empty() {
            return Ok(0);
        }

        // Total size in bytes
        let content_size = self.bytes.len()
            + self.overflow_counts.len() * OVERFLOW_SIZE
            + self.offsets.len() * OFFSET_SIZE;

        for offset in self.offsets.iter().rev() {
            self.writer.write_all(&offset.to_le_bytes())?;
        }
        for overflow in self.overflow_counts {
            self.writer.write_all(&overflow.to_le_bytes())?;
        }
        let len = self.bytes.len();
        self.writer.write_all(&self.bytes)?;
        let footer_size = footer::write(&mut self.writer, len)?;
        Ok(content_size + footer_size)
    }
}

impl<'a> PSVec<'a> {
    pub fn empty() -> Self {
        Self {
            values: &[],
            overflow_counts: &[],
            offsets: &[],
        }
    }

    // Returns the number of items in the vector.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn from_bytes_vec_only(buffer: &'a [u8]) -> Self {
        if buffer.len() <= 5 {
            Self {
                values: buffer,
                overflow_counts: &[],
                offsets: &[],
            }
        } else {
            Self::from_bytes(buffer)
        }
    }

    pub fn from_bytes(buffer: &'a [u8]) -> Self {
        if buffer.is_empty() {
            return Self::empty();
        }

        let (num_values, footer_size) = footer::read(buffer).expect("Why is there no footer?");
        let body = &buffer[..buffer.len() - footer_size];

        let overflow_counts_begin = num_values + (num_values / BLOCK_SIZE) * OVERFLOW_SIZE;
        let overflow_counts_bytes =
            &body[body.len() - overflow_counts_begin..body.len() - num_values];

        // Question: should we worry about unaligned transmute here? Also, this is not portable
        // (pretty sure this requires that the system is little endian)
        let overflow_counts = unsafe {
            std::slice::from_raw_parts(
                overflow_counts_bytes.as_ptr() as *const Overflow,
                overflow_counts_bytes.len() / OVERFLOW_SIZE,
            )
        };

        let offset_bytes_len = body.len() - overflow_counts_begin;
        // We need a slice which ends at offset_bytes_len and whose length is a multiple of 4.
        // Maybe we can just subtract indices from the end pointer instead...
        let offsets_bytes = &body[offset_bytes_len % OFFSET_SIZE..offset_bytes_len];
        let offsets = unsafe {
            std::slice::from_raw_parts(
                offsets_bytes.as_ptr() as *const Offset,
                offsets_bytes.len() / OFFSET_SIZE,
            )
        };

        Self {
            values: &body[body.len() - num_values..],
            overflow_counts,
            offsets,
        }
    }

    // `i` must be in the range from 0..=len().
    // The function returns `sum_{0<=j<i} v[j]`, i.e. the partial sum
    // of all values up to index `i` (exclusive).
    pub fn get(&self, i: usize) -> Option<u32> {
        if i == 0 {
            return Some(0);
        }

        let packed_value = *self.values.get(i - 1)?;
        let overflow_count = self.overflow_count(i);
        Some(self.offset(overflow_count) + ((packed_value as u32) & 127))
    }

    #[inline]
    fn offset(&self, overflow_count: usize) -> u32 {
        match overflow_count {
            0 => 0,
            x => self.offsets[self.offsets.len() - x],
        }
    }

    /// Number of elements in the range `0..i` that overflow.
    pub fn overflow_count(&self, i: usize) -> usize {
        // Count the number of overflows in this block (excluding current value)
        let block_start = i & !(BLOCK_SIZE - 1);
        let block = read_u128_le(&self.values[block_start..]);
        let mask = !(!0u128 << (8 * (i % BLOCK_SIZE))) & BLOCK_OVERFLOW_MASK;
        let block_overflows = (block & mask).count_ones() as usize;

        let prior_overflows = self.overflow_count_by_block(i / BLOCK_SIZE);
        prior_overflows + block_overflows
    }

    /// Number of elements in the range `0..nblocks * BLOCK_SIZE` that overflow.
    #[inline]
    fn overflow_count_by_block(&self, nblocks: usize) -> usize {
        match nblocks {
            0 => 0,
            x => self.overflow_counts[x - 1] as usize,
        }
    }

    fn get_block_value(&self, i: usize) -> u32 {
        if i == 0 {
            return 0;
        }
        let packed_value = self.values[i * BLOCK_SIZE];
        // Count how many overflows we have seen up to this block.
        self.offset(self.overflow_count_by_block(i) + (packed_value >> 7) as usize)
            + ((packed_value as u32) & 127)
    }

    // Returns the index `i` for which the following property holds:
    // `get(i) <= value < get(i+1)`
    // If none such value exists, len() is returned which spans the interval
    // `get(len()) <= value < oo`
    pub fn lower_bound(&self, value: u32) -> usize {
        let mut lower = 0usize;
        let mut upper = self.overflow_counts.len();
        while lower + 1 < upper {
            let mid = (lower + 1 + upper) / 2;
            if self.get_block_value(mid) <= value {
                lower = mid;
            } else {
                upper = mid;
            }
        }
        lower *= BLOCK_SIZE;
        upper = self.len().min(lower + BLOCK_SIZE);
        // Search brute force within a block.
        // TODO: Use an iterator to make scanning faster.
        for idx in lower..upper {
            if self.get(idx + 1).unwrap_or_else(|| {
                panic!(
                    "there wasn't a value at index {:?}, but there should be",
                    idx + 1
                )
            }) > value
            {
                return idx;
            }
        }
        upper
    }

    pub fn begin(&self) -> PSVecIterator<'a> {
        PSVecIterator::<'a> {
            values: self.values,
            offsets: self.offsets,
            offset: 0,
            first: true,
        }
    }

    /// Returns the range `at(i)..at(i+1)` with about the cost of a single lookup.
    /// Note: can return `0..0` when the range is empty or doesn't exist!
    pub fn range(&self, i: usize) -> Range<usize> {
        if i >= self.values.len() {
            0..0
        } else if i == 0 {
            self.get(1).map(|e| (0..e as usize)).expect("")
        } else {
            let mut overflow_count = self.overflow_count(i);
            let start = self.offset(overflow_count) + (self.values[i - 1] & 127) as u32;
            overflow_count += (self.values[i] >> 7) as usize;
            let end = self.offset(overflow_count) + (self.values[i] & 127) as u32;
            start as usize..end as usize
        }
    }

    pub fn at(&self, i: usize) -> PSVecIterator<'a> {
        if i == 0 {
            return self.begin();
        }

        if i > self.values.len() {
            return PSVecIterator::empty();
        }

        let i = i - 1;
        let overflow_count = self.overflow_count(i);
        if overflow_count == 0 {
            PSVecIterator::<'a> {
                values: &self.values[i..],
                offsets: self.offsets,
                offset: 0,
                first: false,
            }
        } else {
            PSVecIterator::<'a> {
                values: &self.values[i..],
                offsets: &self.offsets[..self.offsets.len() - overflow_count],
                offset: self.offsets[self.offsets.len() - overflow_count],
                first: false,
            }
        }
    }
}

impl<'a> PSVecIterator<'a> {
    pub fn empty() -> Self {
        PSVecIterator::<'a> {
            values: &[],
            offsets: &[],
            offset: 0,
            first: false,
        }
    }
}

impl<'a> Iterator for PSVecIterator<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.first {
            self.first = false;
            return Some(0);
        }
        let (byte, rest) = self.values.split_first()?;
        self.values = rest;
        if (byte & 0x80) != 0 {
            self.offset = *self.offsets.last().expect("error deserializing PSVec data");
            self.offsets = &self.offsets[..self.offsets.len() - 1];
        }
        Some(self.offset + (byte & !0x80) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = PSVecBuilder::new(writer);
            builder.push(1).unwrap();
            builder.push(1).unwrap();
            builder.push(1).unwrap();
            builder.finish().unwrap();
        }

        assert_eq!(buffer, &[1, 2, 3, 3]);
    }

    #[test]
    fn test_builder_more_numbers() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = PSVecBuilder::new(writer);
            builder.push(1).unwrap();
            builder.push(0x0345).unwrap();
            builder.push(1).unwrap(); // Does not require an overflow
            builder.push(0x5678).unwrap();
            assert_eq!(builder.finish().unwrap(), 13);
        }
    }

    #[test]
    fn test_builder_many_numbers() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = PSVecBuilder::new(writer);
            builder.push(0).unwrap();
            for _ in 1..4096 {
                builder.push(1).unwrap();
            }

            // Expected size = 4096 byte values + 512 bit counts (16 bits each) + 32 offsets (32
            // bits each) + 2 bytes varint footer = 4738
            assert_eq!(builder.finish().unwrap(), 5246);
        }

        // Varint decode the last 8 bits
        assert_eq!(footer::read(buffer.as_slice()).unwrap().0, 4096);
    }

    #[test]
    fn test_decode() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = PSVecBuilder::new(writer);
            builder.push(10).unwrap();
            builder.push(5).unwrap();
            builder.push(135).unwrap();
            builder.push(800).unwrap();
            builder.finish().unwrap();
        }

        let reader = PSVec::from_bytes(&buffer);
        assert_eq!(reader.get(0), Some(0));
        assert_eq!(reader.get(1), Some(10));
        assert_eq!(reader.get(2), Some(15));
        assert_eq!(reader.get(3), Some(150));
        assert_eq!(reader.get(4), Some(950));
        assert_eq!(reader.get(5), None);
    }

    #[test]
    fn test_decode_empty() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let builder = PSVecBuilder::new(writer);
            builder.finish().unwrap();
        }

        // No elements in vec --> zero space
        assert!(buffer.is_empty());

        let reader = PSVec::from_bytes(&buffer);
        assert_eq!(reader.get(0), Some(0));
        assert_eq!(reader.get(1), None);
    }

    #[test]
    fn test_decode_big() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = PSVecBuilder::new(writer);
            for _ in 0..16000 {
                builder.push(2).unwrap();
            }
            builder.finish().unwrap();
        }

        let reader = PSVec::from_bytes(&buffer);
        for i in 0..16000 {
            assert_eq!(reader.get(i).unwrap(), 2 * i as u32);
        }
    }

    #[test]
    fn test_decode_big_gaps() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = PSVecBuilder::new(writer);
            for _ in 0..512 {
                builder.push(1000).unwrap();
            }
            builder.finish().unwrap();
        }

        let reader = PSVec::from_bytes(&buffer);
        for i in 0..512 {
            assert_eq!(reader.get(i).unwrap(), (i * 1000) as u32);
        }
    }

    #[test]
    fn test_lower_bound() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = PSVecBuilder::new(writer);
            for i in 0..1600 {
                builder.push(i).unwrap();
            }
            builder.finish().unwrap();
        }

        let reader = PSVec::from_bytes(&buffer);
        for i in 0..1600 {
            let j = reader.lower_bound(i);
            assert!(
                reader.get(j).unwrap() <= i,
                "{:?} <= {} < {:?}",
                reader.get(j),
                i,
                reader.get(j + 1)
            );
            assert!(
                i < reader.get(j + 1).unwrap(),
                "{:?} <= {} < {:?}",
                reader.get(j),
                i,
                reader.get(j + 1)
            );
        }
    }

    #[test]
    fn test_iterator() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = PSVecBuilder::new(writer);
            for i in 0..1600 {
                builder.push(i).unwrap();
            }
            builder.finish().unwrap();
        }

        let reader = PSVec::from_bytes(&buffer);
        let mut it = reader.begin();
        for i in 0..=1600 {
            assert_eq!(reader.get(i), it.next());
            let mut jt = reader.at(i);
            assert_eq!(reader.get(i), jt.next());
            assert_eq!(reader.get(i + 1), jt.next());
        }
        assert_eq!(None, it.next());

        // Constructing an iterator past the end of the list is fine, it
        // just has no entries
        assert_eq!(reader.at(3200).next(), None);
    }

    /// Check that `values` can round-trip through PSVec serialization.
    ///
    /// Precondition: `values` is sorted and `values[0] == 0`.
    #[track_caller]
    fn check_values(values: Vec<u32>) {
        let buffer = {
            let mut buffer = Vec::new();
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = PSVecBuilder::new(writer);
            for i in 1..values.len() {
                builder.push(values[i] - values[i - 1]).unwrap();
            }
            builder.finish().unwrap();
            buffer
        };

        let reader = PSVec::from_bytes(&buffer);

        // Decoding the buffer using an iterator produces exactly the original data.
        let iter_values = reader.at(0).collect::<Vec<u32>>();
        assert_eq!(iter_values, values);

        // Decoding the buffer using `reader.get()` produces exactly the original data.
        let get_values = (0..values.len())
            .map(|i| reader.get(i).expect("i is in range"))
            .collect::<Vec<u32>>();
        assert_eq!(get_values, values);
    }

    #[test]
    fn test_overflows_always() {
        let values = (0..300).map(|n| n * 1000).collect();
        check_values(values);
    }

    #[test]
    fn test_overflows_sometimes() {
        // By choosing multiples of 47, every 3rd value overflows.
        // This way, we can test all possible combinations at block borders.
        let values = (0..100).map(|n| n * 47).collect();
        check_values(values);
    }

    use proptest::collection::vec;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_interpolated_search_small(mut data in vec(0..256u32, 0..64)) {
            data.sort_unstable();
            data.insert(0, 0);
            check_values(data);
        }

        #[test]
        fn test_interpolated_search_large(mut data in vec(any::<u32>(), 0..1024)) {
            data.sort_unstable();
            data.insert(0, 0);
            check_values(data);
        }
    }
}
