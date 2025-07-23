//! Convert a [`GeoDiffCount`] to and from byte arrays.
//!
//! Since most of our target platforms are little endian there are more optimised approaches
//! for little endian platforms, just splatting the bytes into the writer. This is contrary
//! to the usual "network endian" approach where big endian is the default, but most of our
//! consumers are little endian so it makes sense for this to be the optimal approach.
//!
//! We still need to support big endian platforms though, but they get a less efficient path.
use std::{borrow::Cow, ops::Deref as _};

use crate::{config::GeoConfig, Diff};

use super::{bitvec::BitVec, GeoDiffCount};

impl<'a, C: GeoConfig<Diff>> GeoDiffCount<'a, C> {
    /// Create a new [`GeoDiffCount`] from a slice of bytes
    #[cfg(target_endian = "little")]
    pub fn from_bytes(c: C, buf: &'a [u8]) -> Self {
        if buf.is_empty() {
            return Self::new(c);
        }

        // The number of most significant bits stores in the MSB sparse repr
        let msb_len = (buf.len() / size_of::<C::BucketType>()).min(c.max_msb_len());

        let msb =
            unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const C::BucketType, msb_len) };

        // The number of bytes representing the MSB - this is how many bytes we need to
        // skip over to reach the LSB
        let msb_bytes_len = msb_len * size_of::<C::BucketType>();

        Self {
            config: c,
            msb: Cow::Borrowed(msb),
            lsb: BitVec::from_bytes(&buf[msb_bytes_len..]),
        }
    }

    /// Create a new [`GeoDiffCount`] from a slice of bytes
    #[cfg(target_endian = "big")]
    pub fn from_bytes(c: C, buf: &'a [u8]) -> Self {
        unimplemented!("not supported on big endian platforms")
    }

    #[cfg(target_endian = "little")]
    pub fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        if self.msb.is_empty() {
            return Ok(0);
        }

        let msb_buckets = self.msb.deref();
        let msb_bytes = unsafe {
            std::slice::from_raw_parts(
                msb_buckets.as_ptr() as *const u8,
                msb_buckets.len() * size_of::<C::BucketType>(),
            )
        };
        writer.write_all(msb_bytes)?;

        let mut bytes_written = msb_bytes.len();

        bytes_written += self.lsb.write(writer)?;

        Ok(bytes_written)
    }

    #[cfg(target_endian = "big")]
    pub fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        unimplemented!("not supported on big endian platforms")
    }
}
