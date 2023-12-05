use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;

use crate::vec::PSVecIterator;
use crate::{PSVec, PSVecBuilder, Serializable};

pub struct RepeatedField<'a, T: Serializable<'a>> {
    content: &'a [u8],
    offsets: PSVec<'a>,
    _marker: PhantomData<T>,
}

pub struct RepeatedFieldBuilder<'a, T: Serializable<'a>, W: std::io::Write> {
    writer: W,
    sizes: Vec<u32>,
    _marker: PhantomData<&'a T>,
}

pub struct RepeatedFieldBuilderWrapper<'a, T: Serializable<'a>, W: std::io::Write> {
    sizes: &'a mut Vec<u32>,
    builder: RepeatedFieldBuilder<'a, T, W>,
}

impl<'a, T: Serializable<'a>, W: std::io::Write> RepeatedFieldBuilderWrapper<'a, T, W> {
    pub fn new(sizes: &'a mut Vec<u32>, builder: RepeatedFieldBuilder<'a, T, W>) -> Self {
        Self { sizes, builder }
    }

    pub fn push(&mut self, data: &T) -> std::io::Result<u32> {
        self.builder.push(data)
    }

    /// TODO: Remove this function. There is only one place where it is used and that's because rust
    /// complains about lifetimes.
    pub fn push_bytes(&mut self, data: &[u8]) -> std::io::Result<u32> {
        self.builder.push_bytes(data)
    }

    /// This is a work-around to push Cow types instead of the original type T.
    /// A better solution is to introduce a Serializable<DecodeAs=T::DecodeAs> parameter
    /// which indicates that the two are compatible types.
    /// TODO: Implement this!
    pub fn push_any<S: Serializable<'a>>(&mut self, data: &S) -> std::io::Result<u32> {
        self.builder.push_any(data)
    }
}

impl<'a, T: Serializable<'a>, W: std::io::Write> Drop for RepeatedFieldBuilderWrapper<'a, T, W> {
    fn drop(&mut self) {
        let size = self.builder._finish().expect("error writing to builder");
        self.sizes.push(size as u32);
    }
}

impl<'a, W: std::io::Write, T: Serializable<'a>> RepeatedFieldBuilder<'a, T, W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            sizes: Vec::new(),
            _marker: PhantomData,
        }
    }

    pub fn push(&mut self, data: &T) -> std::io::Result<u32> {
        let size = data.write(&mut self.writer)?;
        self.sizes.push(size as u32);
        Ok(size as u32)
    }

    pub fn push_bytes(&mut self, data: &[u8]) -> std::io::Result<u32> {
        self.writer.write_all(data)?;
        let size = data.len() as u32;
        self.sizes.push(size);
        Ok(size)
    }

    pub fn push_any<S: Serializable<'a>>(&mut self, data: &S) -> std::io::Result<u32> {
        let size = data.write(&mut self.writer)?;
        self.sizes.push(size as u32);
        Ok(size as u32)
    }

    fn _finish(&mut self) -> std::io::Result<usize> {
        // If there are no fields, just return empty
        if self.sizes.is_empty() {
            return Ok(0);
        }

        let mut offset_builder = PSVecBuilder::new(&mut self.writer);
        let mut total_size = 0;
        for size in &self.sizes {
            offset_builder.push(*size)?;
            total_size += *size as usize;
        }
        let vec_size = offset_builder.finish()?;
        total_size += vec_size;
        Ok(total_size)
    }

    pub fn finish(mut self) -> std::io::Result<usize> {
        self._finish()
    }
}

impl<'a, T: Serializable<'a>> RepeatedField<'a, T> {
    pub fn empty() -> Self {
        Self {
            content: &[],
            offsets: PSVec::empty(),
            _marker: PhantomData,
        }
    }

    pub fn from_bytes(buffer: &'a [u8]) -> Self {
        if buffer.is_empty() {
            Self::empty()
        } else {
            Self {
                content: buffer,
                offsets: PSVec::from_bytes(buffer),
                _marker: PhantomData,
            }
        }
    }

    pub fn get(&self, idx: usize) -> T {
        let mut iter = self.offsets.at(idx);
        let content_start = iter.next().unwrap_or(0);
        let content_end = iter.next().unwrap_or(content_start);
        T::from_bytes(&self.content[content_start as usize..content_end as usize])
    }

    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    pub fn iter(&self) -> RepeatedFieldIterator<'a, T> {
        let mut offset_iter = self.offsets.begin();
        let start = offset_iter.next().expect("first offset is always 0");

        RepeatedFieldIterator {
            content: self.content,
            offset_iter,
            start,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: Serializable<'a>> Serializable<'a> for RepeatedField<'a, T> {
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        writer.write_all(self.content)?;
        Ok(self.content.len())
    }

    fn from_bytes(buf: &'a [u8]) -> Self {
        Self::from_bytes(buf)
    }
}

pub struct RepeatedFieldIterator<'a, T> {
    content: &'a [u8],
    offset_iter: PSVecIterator<'a>,
    start: u32,
    _marker: PhantomData<T>,
}

impl<'a, T: Serializable<'a>> Iterator for RepeatedFieldIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let end = match self.offset_iter.next() {
            Some(end) => end,
            None => return None,
        };

        let out = T::from_bytes(&self.content[self.start as usize..end as usize]);
        self.start = end;
        Some(out)
    }
}

impl<'a, T: Serializable<'a>> IntoIterator for &RepeatedField<'a, T> {
    type IntoIter = RepeatedFieldIterator<'a, T>;
    type Item = T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: Serializable<'a>> IntoIterator for RepeatedField<'a, T> {
    type IntoIter = RepeatedFieldIterator<'a, T>;
    type Item = T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: Serializable<'a> + Debug> Debug for RepeatedField<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self).finish()
    }
}
