use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::str::FromStr;

use dataview::Pod;
use hex::FromHexError;
use serde::{Deserializer, Serializer};
use thiserror::Error;

use github_lock_free::SelfContained;
use github_stable_hash::StableHash;

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Default, Hash, Pod)]
#[repr(transparent)]
pub struct Sha([u8; 20]);

impl SelfContained for Sha {}

impl Sha {
    pub const fn new(sha: [u8; 20]) -> Sha {
        Sha(sha)
    }

    pub fn to_hex_string(self) -> String {
        hex::encode(self.0)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn to_vec(self) -> Vec<u8> {
        self.0.to_vec()
    }

    pub fn least_significant_bits(&self) -> u64 {
        // TODO: Once this feature is in stable this can be rewritten as
        //   u64::from_be_bytes(self.0.rsplit_array_ref::<8>().1.clone())
        u64::from_be_bytes(
            self.0[12..20]
                .try_into()
                .expect("unable to convert slice to array"),
        )
    }

    pub fn trailing_zeros(&self) -> usize {
        self.least_significant_bits().trailing_zeros() as usize
    }

    /// Compute the Git SHA for a blob with the given `content`. Like `git hash-object`.
    pub fn git_blob_oid(content: &[u8]) -> Self {
        use sha1::Digest;
        let mut hasher = sha1::Sha1::new();
        hasher.update(format!("blob {}\0", content.len()).as_bytes());
        hasher.update(content);

        Self(hasher.finalize().into())
    }

    /// Converts the least significant bits of the SHA into a random floating number between 0 and 1.
    /// Note: both limits are inclusive!
    ///
    /// The algorithm works by essentially treating the SHA as a 20 byte integer number in little endian
    /// representation (we use little endian, since the first bytes are used for sharding by SHA!).
    /// Furthermore, we assume that these integers are uniformly sampled and as a consequence the resulting
    /// f64 values should inherit the same distribution.
    /// Note that f64 values represent a whole range and not just a single value!
    ///
    /// To make the conversion simple, we cut off the most significant zero bytes from the right. Those
    /// zero bytes are simply added to the conversion at the end by dividing the result by 256 for every
    /// truncated byte. This division can be computed exactly in floating space.
    /// The following 8 non-zero bytes are simply converted via intrinsics into a f64 number.
    /// Note: f64 can only represent a subset of these bits, so its not worth using more bytes for the conversion.
    ///
    /// At the end we scale down the number by 0.5^64 in order to get a number between 0 and 1.
    /// Note: the f64 conversion has to round the 8byte integer. That's why both 0 and 1 are possible.
    pub fn as_uniform_f64(&self) -> f64 {
        let slice = &self.0;
        let mut end = slice.len();
        let mut factor = 0.5f64.powi(64);
        while end > 8 && slice[end - 1] == 0 {
            end -= 1;
            factor /= 256.0;
        }
        let v = u64::from_le_bytes(
            slice[end - 8..end]
                .try_into()
                .expect("we definitely have 8 bytes here!"),
        );
        (v as f64) * factor
    }

    pub fn xor(mut self, other: &Self) -> Self {
        for i in 0..20 {
            self.0[i] ^= other.0[i];
        }
        self
    }
}

impl StableHash for Sha {
    fn stable_hash(&self) -> u64 {
        // Skip the hashing: a SHA is already a hash code.
        u64::from_le_bytes(
            self.0[20 - 8..]
                .try_into()
                .expect("can't fail; we built a slice of length 8"),
        )
    }
}

impl serde::Serialize for Sha {
    fn serialize<S>(&self, serializer: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(&self.0[..])
    }
}

struct SerdeShaVisitor;

impl<'de> serde::de::Visitor<'de> for SerdeShaVisitor {
    type Value = Sha;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a SHA-like 20 byte array")
    }

    fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Sha::try_from(v).map_err(|_| serde::de::Error::invalid_length(v.len(), &self))
    }
}

impl<'de> serde::Deserialize<'de> for Sha {
    fn deserialize<D>(deserializer: D) -> Result<Self, <D as Deserializer<'de>>::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_bytes(SerdeShaVisitor)
    }
}

impl From<[u8; 20]> for Sha {
    fn from(sha: [u8; 20]) -> Self {
        Sha(sha)
    }
}

impl From<Sha> for [u8; 20] {
    fn from(sha: Sha) -> [u8; 20] {
        sha.0
    }
}

impl<'a> github_pspack::Serializable<'a> for Sha {
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        writer.write_all(&self.0)?;
        Ok(20)
    }

    fn from_bytes(buf: &'a [u8]) -> Self {
        if buf.is_empty() {
            Sha::default()
        } else {
            Sha::new(buf.try_into().expect("expected a sha with 20 bytes!"))
        }
    }
}

impl TryFrom<&[u8]> for Sha {
    type Error = ShaError;

    fn try_from(value: &[u8]) -> std::result::Result<Self, Self::Error> {
        let sha: [u8; 20] = value
            .try_into()
            .map_err(|_| ShaError::InvalidShaLength(value.len()))?;
        Ok(Self(sha))
    }
}

impl FromStr for Sha {
    type Err = ShaError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let bytes = hex::decode(value)
            .map_err(|hex_err| ShaError::InvalidShaHex(value.to_string(), hex_err))?;
        bytes.as_slice().try_into()
    }
}

impl fmt::Debug for Sha {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Sha {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex_string())
    }
}

impl AsRef<[u8]> for Sha {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Supports `rand::random::<Sha>()`.
impl rand::distributions::Distribution<Sha> for rand::distributions::Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Sha {
        rng.gen::<[u8; 20]>().into()
    }
}

#[derive(Debug, Error)]
pub enum ShaError {
    #[error("invalid sha (expected 20 bytes, got {0} bytes)")]
    InvalidShaLength(usize),
    #[error("invalid sha {0:?}")]
    InvalidShaHex(String, #[source] FromHexError),
}

// TODO: convert this type to a struct and move the implementation for constructors like `sha256_digest` into
// trait implementations
pub type Sha256Digest = sha2::digest::Output<sha2::Sha256>;

pub fn decode_hex_digest(hex_digest: &str) -> Result<Sha256Digest, hex::FromHexError> {
    use sha2::digest::generic_array::GenericArray;
    let v = hex::decode(hex_digest)?;
    Ok(GenericArray::clone_from_slice(v.as_slice()))
}

pub fn encode_hex_digest(digest: &Sha256Digest) -> String {
    format!("{digest:x}")
}

pub fn sha256_digest(content: &[u8]) -> Sha256Digest {
    use sha2::digest::Digest;
    use sha2::Sha256;
    let mut hasher = Sha256::new();
    hasher.update(content);
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use crate::Sha;

    #[test_log::test]
    fn test_as_uniform_f64() {
        assert_eq!(Sha::new([255; 20]).as_uniform_f64(), 1.0f64);
        assert_eq!(Sha::new([0; 20]).as_uniform_f64(), 0.0f64);
        assert_eq!(
            Sha::new([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128])
                .as_uniform_f64(),
            0.5
        );
        assert_eq!(
            Sha::new([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).as_uniform_f64(),
            1.0 / 256.0
        );
        assert_eq!(
            Sha::new([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0])
                .as_uniform_f64(),
            1.0 / 512.0
        );
        assert_eq!(
            Sha::new([0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                .as_uniform_f64(),
            0.5f64.powf(97.0)
        );
    }
}
