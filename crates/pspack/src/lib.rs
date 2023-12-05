#![deny(unsafe_op_in_unsafe_fn)]

#[macro_use]
mod macros;

mod footer;
mod psstruct;
mod read_uint;
mod repeatedfield;
mod slices;
mod vec;

use base64::prelude::BASE64_STANDARD;
use base64::Engine;
// It's necessary to export this so that macros can refer to it
pub use paste::paste;
pub use psstruct::{PSStruct, PSStructBuilder};
use read_uint::{read_u32_le, read_u64_le};
pub use repeatedfield::{
    RepeatedField, RepeatedFieldBuilder, RepeatedFieldBuilderWrapper, RepeatedFieldIterator,
};
pub use vec::{PSVec, PSVecBuilder};

// The Serializable trait defines how to serialize individual fields to bytes. Only types
// implementing Serializable are allowed in PSStructs.
pub trait Serializable<'a>: Sized {
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize>;

    fn from_bytes(buf: &'a [u8]) -> Self;

    /// Convenience function to convert a serializable instance directly into a base64 encoded
    /// representation.
    /// Note: Decoding is not supported, since the temporary decoded string would cause
    /// lifetime issues!
    fn as_base64(&self) -> String {
        let mut buffer = vec![];
        self.write(&mut buffer)
            .expect("Serialization shouldn't fail!");
        BASE64_STANDARD.encode(&buffer)
    }
}

impl<'a> Serializable<'a> for &'a str {
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        writer.write_all(self.as_bytes())?;
        Ok(self.len())
    }

    fn from_bytes(buf: &'a [u8]) -> Self {
        unsafe { std::str::from_utf8_unchecked(buf) }
    }
}

impl<'a> Serializable<'a> for u32 {
    // Since we know the size at read time, we can drop leading zero bytes during writing.
    // This saves a lot of space when writing small numbers.
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        let mut bytes: &[u8] = &self.to_le_bytes();
        while let Some((0, rest)) = bytes.split_last() {
            bytes = rest
        }
        writer.write_all(bytes)?;
        Ok(bytes.len())
    }

    fn from_bytes(buf: &'a [u8]) -> Self {
        read_u32_le(buf)
    }
}

impl<'a> Serializable<'a> for u64 {
    // Since we know the size at read time, we can drop leading zero bytes during writing.
    // This saves a lot of space when writing small numbers.
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        let mut bytes: &[u8] = &self.to_le_bytes();
        while let Some((0, rest)) = bytes.split_last() {
            bytes = rest
        }
        writer.write_all(bytes)?;
        Ok(bytes.len())
    }

    fn from_bytes(buf: &'a [u8]) -> Self {
        read_u64_le(buf)
    }
}

impl<'a> Serializable<'a> for i64 {
    // Since we know the size at read time, we can drop leading zero bytes during writing.
    // This saves a lot of space when writing small numbers.
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        (*self as u64).write(writer)
    }

    fn from_bytes(buf: &'a [u8]) -> Self {
        u64::from_bytes(buf) as i64
    }
}

impl<'a> Serializable<'a> for bool {
    // Since we know the size at read time, we can drop leading zero bytes during writing.
    // This saves a lot of space when writing small numbers.
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        if *self {
            writer.write_all(&[1])?;
            Ok(1)
        } else {
            Ok(0)
        }
    }

    fn from_bytes(buf: &'a [u8]) -> Self {
        !buf.is_empty()
    }
}

impl<'a> Serializable<'a> for i32 {
    // TODO: implement some kind of variable encoding for i32 like u32
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        writer.write_all(&self.to_le_bytes())?;
        Ok(std::mem::size_of::<i32>())
    }

    fn from_bytes(buf: &'a [u8]) -> Self {
        assert!(buf.is_empty() || buf.len() == 4);
        let mut bytes = [0; 4];
        bytes[..buf.len()].copy_from_slice(buf);
        i32::from_le_bytes(bytes)
    }
}

impl<'a> Serializable<'a> for u8 {
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        writer.write_all(&[*self])?;
        Ok(std::mem::size_of::<u8>())
    }

    fn from_bytes(buf: &'a [u8]) -> Self {
        if buf.is_empty() {
            0
        } else {
            buf[0]
        }
    }
}

impl<'a, T> Serializable<'a> for Option<T>
where
    T: Serializable<'a>,
{
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        match self {
            None => Ok(0),
            Some(t) => {
                writer.write_all(&[1])?;
                t.write(writer).map(|s| s + 1)
            }
        }
    }

    fn from_bytes(buf: &'a [u8]) -> Self {
        if buf.is_empty() {
            None
        } else {
            Some(T::from_bytes(&buf[1..]))
        }
    }
}

impl<'a> Serializable<'a> for f32 {
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        writer.write_all(&self.to_le_bytes())?;
        Ok(std::mem::size_of::<i32>())
    }

    fn from_bytes(buf: &'a [u8]) -> Self {
        if buf.is_empty() {
            0.0
        } else {
            assert_eq!(4, buf.len());
            let mut bytes = [0; 4];
            bytes[..buf.len()].copy_from_slice(buf);
            f32::from_le_bytes(bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_u32() {
        let mut buffer = Vec::new();
        let mut writer = std::io::Cursor::new(&mut buffer);
        assert_eq!(0, 0u32.write(&mut writer).unwrap());
        assert_eq!(1, 1u32.write(&mut writer).unwrap());
        assert_eq!(1, 255u32.write(&mut writer).unwrap());
        assert_eq!(2, 256u32.write(&mut writer).unwrap());
        assert_eq!(2, 65535u32.write(&mut writer).unwrap());
        assert_eq!(3, 65536u32.write(&mut writer).unwrap());
        assert_eq!(3, 100000u32.write(&mut writer).unwrap());
        assert_eq!(4, 100000000u32.write(&mut writer).unwrap());

        assert_eq!(0, u32::from_bytes(&[]));
        assert_eq!(1, u32::from_bytes(&[1]));
        assert_eq!(256, u32::from_bytes(&[0, 1]));
        assert_eq!(65536, u32::from_bytes(&[0, 0, 1]));
        assert_eq!(16777216, u32::from_bytes(&[0, 0, 0, 1]));
    }

    fn serialize_u32(data: u32) -> Vec<u8> {
        let mut buffer = Vec::new();
        {
            let mut writer = std::io::Cursor::new(&mut buffer);
            data.write(&mut writer).unwrap();
        }
        buffer
    }

    #[test]
    fn test_u32_serialization_more() {
        let buffer = serialize_u32(0);
        assert_eq!(buffer.len(), 0);
        assert_eq!(u32::from_bytes(&buffer), 0);

        let buffer = serialize_u32(128);
        assert_eq!(buffer.len(), 1);
        assert_eq!(u32::from_bytes(&buffer), 128);

        let buffer = serialize_u32(350);
        assert_eq!(buffer.len(), 2);
        assert_eq!(u32::from_bytes(&buffer), 350);

        let buffer = serialize_u32(15790320);
        assert_eq!(buffer.len(), 3);
        assert_eq!(u32::from_bytes(&buffer), 15790320);

        let buffer = serialize_u32(u32::MAX);
        assert_eq!(buffer.len(), 4);
        assert_eq!(u32::from_bytes(&buffer), u32::MAX);
    }

    psstruct!(
        #[derive(Clone, Default)]
        pub struct StructWithArray {
            array: &[u32],
            opt: Option<i64>,
        }
    );

    #[test]
    fn test_struct_with_array() {
        let mut buffer = Vec::new();
        {
            let mut builder = StructWithArrayBuilder::new(&mut buffer);
            builder.array(&[1, 2, 3]).expect("writing must pass!");
            builder.opt(Some(-234)).expect("writing must pass!");
            builder.finish().expect("finish must pass!");
        }
        let a = StructWithArray::from_bytes(&buffer);
        assert_eq!(&[1, 2, 3], a.array());
        assert_eq!(Some(-234), a.opt());
    }

    #[test]
    fn test_struct_with_empty_array() {
        let mut buffer = Vec::new();
        {
            let mut builder = StructWithArrayBuilder::new(&mut buffer);
            builder.array(&[]).expect("writing must pass!");
            builder.opt(None).expect("writing must pass!");
            builder.finish().expect("finish must pass!");
        }
        let a = StructWithArray::from_bytes(&buffer);
        assert_eq!(a.array(), &[]);
        assert_eq!(a.opt(), None);
    }
}
