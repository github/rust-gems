use crate::Serializable;

fn any_as_u8_slice<T: Sized>(p: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(p.as_ptr() as *const u8, std::mem::size_of_val::<[T]>(p)) }
}

// NOTE: This may result in an unaligned read. Technically this is undefined behavior (and not
// possible on some hardware). But it seems to work OK. A slightly more general approach might be
// to use std::mem::transmute_copy, but this complicates ownership
unsafe fn any_from_u8_slice<T: Sized>(slice: &[u8]) -> &[T] {
    assert_eq!(slice.len() % std::mem::size_of::<T>(), 0);
    unsafe {
        std::slice::from_raw_parts(
            std::mem::transmute(slice.as_ptr()),
            slice.len() / std::mem::size_of::<T>(),
        )
    }
}

impl<'a, T: Sized> Serializable<'a> for &'a [T] {
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        let buf = any_as_u8_slice(self);
        writer.write_all(buf)?;
        Ok(buf.len())
    }

    fn from_bytes(buf: &'a [u8]) -> Self {
        unsafe { any_from_u8_slice(buf) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repeated_fixed() {
        let mut buffer = Vec::new();
        {
            let numbers = [5, 1];

            let mut writer = std::io::Cursor::new(&mut buffer);
            let s: &[i32] = <&[i32]>::from(&numbers);
            s.write(&mut writer).unwrap();
        }
        let decoded: &[i32] = <&[i32] as Serializable>::from_bytes(&buffer);
        assert_eq!(decoded, &[5, 1]);
    }

    #[derive(Debug, Eq, PartialEq)]
    pub struct Version {
        timestamp: u64,
        is_deleted: bool,
    }

    psstruct!(
        struct Data {
            header: bool,
            versions: &[Version],
        }
    );

    #[test]
    fn test_repeated_fixed2() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = DataBuilder::new(writer);
            let versions = [
                Version {
                    timestamp: 123,
                    is_deleted: false,
                },
                Version {
                    timestamp: 567,
                    is_deleted: true,
                },
            ];
            builder.header(true).unwrap();
            builder.versions(&versions).unwrap();
            builder.finish().unwrap();
        }
        let data = Data::from_bytes(&buffer);
        assert!(data.header());
        assert_eq!(data.versions().len(), 2);
        assert_eq!(data.versions()[0].timestamp, 123);
        assert!(!data.versions()[0].is_deleted);
        assert_eq!(data.versions()[1].timestamp, 567);
        assert!(data.versions()[1].is_deleted);
    }

    #[test]
    fn test_repeated_fixed_multi_alignment() {
        let versions = [
            Version {
                timestamp: 123,
                is_deleted: false,
            },
            Version {
                timestamp: 567,
                is_deleted: true,
            },
        ];

        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = DataBuilder::new(writer);
            builder.header(true).unwrap();
            builder.versions(&versions).unwrap();
            builder.finish().unwrap();
        }

        for alignment in 0..8 {
            let data = Data::from_bytes(&buffer[alignment..]);
            assert!(data.header());
            assert_eq!(data.versions(), &versions[..]);

            // Try another alignment by shifting everything right 1 byte
            buffer.insert(0, 0);
        }
    }
}
