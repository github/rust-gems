use crate::{PSVec, PSVecBuilder, RepeatedField, Serializable};

#[derive(Clone, Default)]
pub struct PSStruct<'a> {
    pub content: &'a [u8],
    offsets: PSVec<'a>,
}

pub struct PSStructBuilder<W: std::io::Write> {
    pub writer: W,
    pub sizes: Vec<u32>,
}

impl<W: std::io::Write> PSStructBuilder<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            sizes: Vec::new(),
        }
    }

    pub fn push<'a, S: Serializable<'a>>(&mut self, data: S) -> std::io::Result<()> {
        let size = data.write(&mut self.writer)?;
        self.sizes.push(size as u32);
        Ok(())
    }

    pub fn skip(&mut self) -> std::io::Result<()> {
        self.sizes.push(0);
        Ok(())
    }

    // This function can be used when writing a nested field.
    // In this case, one would simply write the nested field to the writer and
    // then push the written size with this function.
    // TODO: Figure out a more elegant solution to this problem (e.g. by returning a wrapped writer)
    // which pushes the written size automatically when being dropped.
    pub fn push_size(&mut self, size: usize) {
        self.sizes.push(size as u32);
    }

    pub fn finish(mut self) -> std::io::Result<usize> {
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
}

impl<'a> PSStruct<'a> {
    pub fn empty() -> Self {
        Self {
            content: &[],
            offsets: PSVec::empty(),
        }
    }

    pub fn from_bytes(buffer: &'a [u8]) -> Self {
        if buffer.is_empty() {
            PSStruct::empty()
        } else {
            Self {
                content: buffer,
                offsets: PSVec::from_bytes(buffer),
            }
        }
    }

    pub fn read<S: Serializable<'a>>(&self, field_number: u32) -> S {
        S::from_bytes(&self.content[self.offsets.range(field_number as usize)])
    }

    pub fn read_repeated<S: Serializable<'a>>(&self, field_number: u32) -> RepeatedField<'a, S> {
        RepeatedField::from_bytes(&self.content[self.offsets.range(field_number as usize)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_layout() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = PSStructBuilder::new(writer);
            builder.push("hello").unwrap();
            builder.finish().unwrap();
        }

        fn ch(c: char) -> u8 {
            c as u8
        }

        assert_eq!(
            &buffer,
            &[
                ch('h'),
                ch('e'),
                ch('l'),
                ch('l'),
                ch('o'),
                5, // Single element in PSStruct
                1, // number of elements in PSStruct
            ],
        )
    }

    #[test]
    fn test_struct_layout_two_values() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = PSStructBuilder::new(writer);
            builder.push("hello").unwrap();
            builder.push("world").unwrap();
            builder.finish().unwrap();
        }

        fn ch(c: char) -> u8 {
            c as u8
        }

        assert_eq!(
            &buffer,
            &[
                ch('h'),
                ch('e'),
                ch('l'),
                ch('l'),
                ch('o'),
                ch('w'),
                ch('o'),
                ch('r'),
                ch('l'),
                ch('d'),
                5,  // End byte of 1st payload
                10, // End byte of 2nd payload
                2,  // PSVec elements
            ],
        )
    }

    #[test]
    fn test_struct_layout_skipped_value() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = PSStructBuilder::new(writer);
            builder.skip().unwrap();
            builder.push("hello").unwrap();
            builder.push("world").unwrap();
            builder.finish().unwrap();
        }

        fn ch(c: char) -> u8 {
            c as u8
        }

        assert_eq!(
            &buffer,
            &[
                ch('h'),
                ch('e'),
                ch('l'),
                ch('l'),
                ch('o'),
                ch('w'),
                ch('o'),
                ch('r'),
                ch('l'),
                ch('d'),
                0,  // Skip marker
                5,  // End byte of 1st payload
                10, // End byte of 2nd payload
                3,  // PSVec elements
            ],
        )
    }

    #[test]
    fn test_read_write_struct() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = PSStructBuilder::new(writer);
            builder.push("hello").unwrap();
            builder.push("world").unwrap();
            builder.finish().unwrap();
        }

        let reader = PSStruct::from_bytes(&buffer);
        assert_eq!(reader.read::<&str>(0), "hello");
        assert_eq!(reader.read::<&str>(1), "world");
        // Test reading a non existing field ==> this should return an empty default instance.
        assert_eq!(reader.read::<&str>(2), "");
    }

    #[test]
    fn test_read_write_struct_many_fields() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = PSStructBuilder::new(writer);
            for i in 0..5000 {
                builder.push(&*format!("item{i}")).unwrap();
            }
            builder.finish().unwrap();
        }

        let reader = PSStruct::from_bytes(&buffer);
        for i in 0..5000 {
            let value = reader.read::<&str>(i);
            assert_eq!(value, &*format!("item{i}"));
        }
    }

    #[test]
    fn test_struct_empty() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let builder = PSStructBuilder::new(writer);
            builder.finish().unwrap();
        }

        assert!(buffer.is_empty());

        let reader = PSStruct::from_bytes(&buffer);

        // Reading things from here be invalid...
        assert_eq!(reader.read::<&str>(0), "");
    }
}
