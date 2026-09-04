//! This crate provides a hasher that makes it possible to hash a stream of data that is provided
//! out of order. It does so by taking advantage of the fact that the addition of points on an
//! elliptic curve is commutative. The stream of data is split into blocks, each block is mapped
//! to a Ristretto point, and all those points are added together. As a result, it is possible to
//! process the blocks in any order and still obtain the same hash value.
//!
//! There are two different hashers provided by this crate. They have different properties, but
//! always return the same hash value for the same inputs:
//! - `ParallelHasher` - Use this to process a stream of data in parallel. It requires that the
//!   data is provided to the hasher in multiples of the block size.
//! - `SequentialHasher` - Use this to process a stream of data serially. The data can be provided
//!   to the hasher in any sizes. It will internally buffer the computation to ensure that each full
//!   block is processed appropriately.
//!
//! When a data stream is complete, you must call `finalize()` on the hasher to obtain a
//! `CommutativeHashDigest`, which provides access to the hash value as either bytes or a hex
//! string for serialization. It also provides ways to deserialize a hex string and compare
//! different values.

use curve25519_dalek::RistrettoPoint;
use dataview::Pod;
pub use digest;
use parking_lot::Mutex;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use thiserror::Error;

/// `CommutativeHashBuilder` is a simple wrapper around a `RistrettoPoint` that makes it easy to
/// generate the final digest value.
#[derive(Debug, Default)]
struct CommutativeHashBuilder<Hash>
where
    Hash: digest::Digest<OutputSize = digest::consts::U64> + Default,
{
    point: RistrettoPoint,
    _marker: PhantomData<Hash>,
}

impl<Hash> CommutativeHashBuilder<Hash>
where
    Hash: digest::Digest<OutputSize = digest::consts::U64> + Default,
{
    fn add_hash(&mut self, hash: Hash) {
        self.point += RistrettoPoint::from_hash(hash);
    }

    fn finalize(&self) -> CommutativeHashDigest {
        CommutativeHashDigest(self.point.compress().to_bytes())
    }
}

impl<Hash> std::ops::AddAssign for CommutativeHashBuilder<Hash>
where
    Hash: digest::Digest<OutputSize = digest::consts::U64> + Default,
{
    fn add_assign(&mut self, rhs: Self) {
        self.point += rhs.point;
    }
}

/// `CommutativeHashDigestError` represents errors that can arise when generating a `CommutativeHashDigest`.
#[derive(Debug, Error)]
pub enum CommutativeHashDigestError {
    /// An invalid character was found. Valid ones are: `0...9`, `a...f` or `A...F`.
    #[error("Invalid character {:?} at position {}", c, index)]
    InvalidHexCharacter { c: char, index: usize },
    /// A hex string's length needs to be even, as two digits correspond to one byte.
    #[error("Odd number of characters")]
    OddLength,
    /// If the hex string is decoded into a fixed sized container, such as an array, the
    /// hex string's length * 2 has to match the container's length.
    #[error("Invalid string length")]
    InvalidStringLength,
    /// Digest is not 32 bytes long.
    #[error("Digest is not 32 bytes long")]
    Not32Bytes,
}

/// Wrapper struct that represents a hash value from a commutative hasher. It provides
/// serialization and other helpers.
#[derive(Copy, Clone, Ord, PartialOrd, PartialEq, Eq, Hash, Debug, Default, Pod)]
#[repr(C)]
pub struct CommutativeHashDigest([u8; 32]);

impl CommutativeHashDigest {
    /// Return the hash as bytes.
    pub fn to_bytes(&self) -> [u8; 32] {
        self.0
    }

    /// Return the hash as a hex-encoded string.
    pub fn hex_digest(&self) -> String {
        hex::encode(self.0)
    }

    /// Convert a hex-encoded string into a `CommutativeHashDigest`, if possible.
    pub fn decode_hex_digest(
        hex_digest: &str,
    ) -> core::result::Result<Self, CommutativeHashDigestError> {
        let value = hex::decode(hex_digest).map_err(|e| match e {
            hex::FromHexError::InvalidHexCharacter { c, index } => {
                CommutativeHashDigestError::InvalidHexCharacter { c, index }
            }
            hex::FromHexError::OddLength => CommutativeHashDigestError::OddLength,
            hex::FromHexError::InvalidStringLength => {
                CommutativeHashDigestError::InvalidStringLength
            }
        })?;
        value
            .try_into()
            .map(Self)
            .map_err(|_| CommutativeHashDigestError::Not32Bytes)
    }
}

#[cfg(feature = "serde")]
impl Serialize for CommutativeHashDigest {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Convert the CommutativeHashDigest to a string and serialize it
        let s = self.hex_digest();
        serializer.serialize_str(&s)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for CommutativeHashDigest {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize the string and convert it to a CommutativeHashDigest
        let s = String::deserialize(deserializer)?;
        CommutativeHashDigest::decode_hex_digest(&s).map_err(serde::de::Error::custom)
    }
}

/// `ParallelHasherError` represents possible errors that could be returned by a `ParallelHasher`.
#[derive(Debug, Error)]
pub enum ParallelHasherError {
    /// Data is not aligned on a block boundary.
    #[error("Data is not aligned on a block boundary")]
    MisalignedData,
    /// Provided data is larger than possible.
    #[error("Provided data is larger than possible")]
    OutOfBounds,
}

/// Result
pub type Result<T, E = ParallelHasherError> = std::result::Result<T, E>;

/// `ParallelHasher` is a hasher that can be used to efficiently generate a hash value from
/// a large amount of data. It does not require the data to be provided in order, instead
/// requiring that the location of each piece of provided data within the complete data stream
/// is provided.
///
/// In order to do this, it hashes the data in blocks, converts each hashed block into a point
/// on an elliptic curve, and adds those points together. Addition of points on an elliptic
/// curve is commutative, which means that adding data blocks out of order is acceptable. The
/// block size must be specified by the caller, and the hasher will split each piece of data
/// given to it into blocks. To ensure that reordered data does not cause a hash collision,
/// the byte location and length of each block is appended to the block of data when hashing it.
/// The data must be provided in multiples of the block size (except for the one representing
/// the final part of the data stream), and each block must be provided exactly once to get a
/// proper hash (adding a block multiple times will lead to a different hash).
#[derive(Debug)]
pub struct ParallelHasher<Hash>
where
    Hash: digest::Digest<OutputSize = digest::consts::U64> + Default,
{
    inner: Mutex<CommutativeHashBuilder<Hash>>,
    block_size: usize,
}

impl<Hash> ParallelHasher<Hash>
where
    Hash: digest::Digest<OutputSize = digest::consts::U64> + Default,
{
    /// Create a new `ParallelHasher`. The data provided to `update` will be split into pieces
    /// of size `block_size`, and that data must always be sized to be a multiple of `block_size`,
    /// except for the part representing the final part of data stream.
    pub fn new(block_size: NonZeroUsize) -> Self {
        ParallelHasher {
            inner: Mutex::new(CommutativeHashBuilder::default()),
            block_size: block_size.get(),
        }
    }

    /// Update the computed hash with some additional `data`. The size of `data` must be a multiple
    /// of the block size unless this is called for the end of the data stream.
    pub fn update(&self, start_byte: usize, data: &[u8]) -> Result<()> {
        if !start_byte.is_multiple_of(self.block_size) {
            return Err(ParallelHasherError::MisalignedData);
        }
        if start_byte.checked_add(data.len()).is_none() {
            return Err(ParallelHasherError::OutOfBounds);
        };

        // The calculation of the hash and point take a while, so they are done separately before
        // locking the mutex to minimize contention.
        let mut start = start_byte;
        let chunk_hash = data.chunks(self.block_size).fold(
            CommutativeHashBuilder::<Hash>::default(),
            |mut acc, chunk| {
                let len = chunk.len();
                let mut hash = Hash::default();
                hash.update(chunk);
                // The start and length are added to ensure that blocks are not reordered.
                hash.update(to_padded_bytes(start));
                hash.update(to_padded_bytes(len));
                start += len;
                acc.add_hash(hash);
                acc
            },
        );

        let mut inner = self.inner.lock();
        *inner += chunk_hash;
        Ok(())
    }

    /// Finish the computation and return the final hash value.
    pub fn finalize(self) -> CommutativeHashDigest {
        self.inner.lock().finalize()
    }
}

/// `SequentialHasher` is a hasher that can be used to generate the same hash values that
/// `ParallelHasher` generates, but without the limitation that the data must be provided in
/// chunks whose sizes are multiples of the block size.
///
/// In order to do this, the hasher must receive the data in order. Instead of requiring the data
/// to be in multiples of the block size, the hasher buffers data internally across `update` calls
/// to fill up a block before processing it.
#[derive(Debug)]
pub struct SequentialHasher<Hash>
where
    Hash: digest::Digest<OutputSize = digest::consts::U64> + Default,
{
    inner: CommutativeHashBuilder<Hash>,
    scratch: Hash,
    scratch_start: usize,
    scratch_size: usize,
    block_size: usize,
}

impl<Hash> SequentialHasher<Hash>
where
    Hash: digest::Digest<OutputSize = digest::consts::U64> + Default,
{
    /// Create a new `SequentialHasher`.
    pub fn new(block_size: NonZeroUsize) -> Self {
        SequentialHasher {
            inner: CommutativeHashBuilder::default(),
            scratch: Hash::default(),
            scratch_start: 0,
            scratch_size: 0,
            block_size: block_size.get(),
        }
    }

    /// Update the computed hash with some additional `data`.
    pub fn update(&mut self, data: &[u8]) {
        let remaining = self.block_size - self.scratch_size;
        split_first_then_chunks(data, remaining, self.block_size).for_each(|chunk| {
            self.scratch.update(chunk);
            self.scratch_size += chunk.len();
            if self.scratch_size == self.block_size {
                self.flush();
            }
        });
    }

    /// Finish the computation and return the final digest value.
    pub fn finalize(mut self) -> CommutativeHashDigest {
        self.flush();
        self.inner.finalize()
    }

    /// Compute a hash of the provided data.
    pub fn digest_from_bytes(block_size: NonZeroUsize, data: &[u8]) -> CommutativeHashDigest {
        let mut hasher = Self::new(block_size);
        hasher.update(data);
        hasher.finalize()
    }

    /// Flush the scratch data to the `CommutativeHashBuilder`.
    fn flush(&mut self) {
        if self.scratch_size == 0 {
            return;
        }
        let mut scratch = std::mem::take(&mut self.scratch);

        // The start and length are added to ensure that blocks are not reordered.
        scratch.update(to_padded_bytes(self.scratch_start));
        scratch.update(to_padded_bytes(self.scratch_size));
        self.inner.add_hash(scratch);

        self.scratch_start += self.scratch_size;
        self.scratch_size = 0;
    }
}

// Split the data into a certain size for the first chunk, and then equal sized subsequent chunks.
fn split_first_then_chunks(
    data: &[u8],
    first_size: usize,
    block_size: usize,
) -> impl Iterator<Item = &[u8]> {
    let (first, rest) = data.split_at(data.len().min(first_size));
    std::iter::once(first).chain(rest.chunks(block_size))
}

// Pad the bytes to ensure cross-platform compatibility, as some platforms use <8 bytes
// for a length.
fn to_padded_bytes(v: usize) -> [u8; 8] {
    (v as u64).to_le_bytes()
}

#[cfg(test)]
mod tests {
    use crate::{
        CommutativeHashDigest, CommutativeHashDigestError, ParallelHasher, ParallelHasherError,
        Result, SequentialHasher, to_padded_bytes,
    };
    use itertools::Itertools;
    use rand::prelude::*;
    use rayon::prelude::*;
    use rstest::rstest;
    use rstest_reuse::{apply, template};
    use sha2::Sha512;
    use std::assert_matches;
    use std::marker::PhantomData;
    use std::num::NonZeroUsize;

    trait TestHasher {
        fn new(block_size: NonZeroUsize) -> Self
        where
            Self: Sized;
        fn update(&mut self, start_byte: usize, data: &[u8]) -> Result<()>;
        fn finalize(self) -> CommutativeHashDigest;
    }

    impl<Hash> TestHasher for ParallelHasher<Hash>
    where
        Hash: digest::Digest<OutputSize = digest::consts::U64> + Default,
    {
        fn new(block_size: NonZeroUsize) -> Self {
            ParallelHasher::<Hash>::new(block_size)
        }

        fn update(&mut self, start_byte: usize, data: &[u8]) -> Result<()> {
            ParallelHasher::<Hash>::update(self, start_byte, data)
        }

        fn finalize(self) -> CommutativeHashDigest {
            ParallelHasher::<Hash>::finalize(self)
        }
    }

    impl<Hash> TestHasher for SequentialHasher<Hash>
    where
        Hash: digest::Digest<OutputSize = digest::consts::U64> + Default,
    {
        fn new(block_size: NonZeroUsize) -> Self {
            SequentialHasher::<Hash>::new(block_size)
        }

        fn update(&mut self, _start_byte: usize, data: &[u8]) -> Result<()> {
            SequentialHasher::<Hash>::update(self, data);
            Ok(())
        }

        fn finalize(self) -> CommutativeHashDigest {
            SequentialHasher::<Hash>::finalize(self)
        }
    }

    #[test]
    fn test_digest_from_bytes() {
        let digest = SequentialHasher::<Sha512>::digest_from_bytes(
            NonZeroUsize::new(4).expect("not zero"),
            "a short string".as_bytes(),
        );
        let expected_digest = "ac533343d9d346805bfea710f4ff5d46024d25e46b1286ab29c95b7de12ad877";
        assert_eq!(digest.hex_digest(), expected_digest);
    }

    #[template]
    #[rstest]
    #[case::parallel(PhantomData::<ParallelHasher::<Sha512>>)]
    #[case::sequential(PhantomData::<SequentialHasher::<Sha512>>)]
    fn hashers<T: TestHasher>(#[case] _t: PhantomData<T>) {}

    #[apply(hashers)]
    fn test_one_block<T: TestHasher>(#[case] _t: PhantomData<T>) {
        let mut hasher = T::new(NonZeroUsize::new(16).expect("not zero"));
        assert_matches!(hasher.update(0, "a short string".as_bytes()), Ok(()));
        let expected_digest = "d637c375199003362b880b43c818c8e8ece54ef9e3725d5a7d6e95462f4d1915";
        assert_eq!(hasher.finalize().hex_digest(), expected_digest);
    }

    #[apply(hashers)]
    fn test_multiple_blocks<T: TestHasher>(#[case] _t: PhantomData<T>) {
        let mut hasher = T::new(NonZeroUsize::new(6).expect("not zero"));
        assert_matches!(hasher.update(0, "a short string".as_bytes()), Ok(()));
        let expected_digest = "10b93d1d1f852fad1927388fd786135ef3f0e1a8a9e14c6fc1c7c7097fc97c5a";
        assert_eq!(hasher.finalize().hex_digest(), expected_digest);
    }

    #[apply(hashers)]
    fn test_multiple_updates<T: TestHasher>(#[case] _t: PhantomData<T>) {
        let mut hasher = T::new(NonZeroUsize::new(4).expect("not zero"));
        assert_matches!(hasher.update(0, "this".as_bytes()), Ok(()));
        assert_matches!(hasher.update(4, "that".as_bytes()), Ok(()));
        assert_matches!(hasher.update(8, "more".as_bytes()), Ok(()));
        assert_matches!(hasher.update(12, "very".as_bytes()), Ok(()));
        assert_matches!(hasher.update(16, "long".as_bytes()), Ok(()));
        let expected_digest = "9258f9de43401cd6e8f55545754b84ac58257ec7779723c9790a986daab18206";
        assert_eq!(hasher.finalize().hex_digest(), expected_digest);
    }

    #[apply(hashers)]
    fn test_providing_data_chunks_different_hashes_same<T: TestHasher>(#[case] _t: PhantomData<T>) {
        let mut hasher1 = T::new(NonZeroUsize::new(4).expect("not zero"));
        assert_matches!(hasher1.update(0, "this".as_bytes()), Ok(()));
        assert_matches!(hasher1.update(4, "that".as_bytes()), Ok(()));
        assert_matches!(hasher1.update(8, "more".as_bytes()), Ok(()));
        assert_matches!(hasher1.update(12, "very".as_bytes()), Ok(()));
        assert_matches!(hasher1.update(16, "long".as_bytes()), Ok(()));
        let digest1 = hasher1.finalize();
        let expected_digest = "9258f9de43401cd6e8f55545754b84ac58257ec7779723c9790a986daab18206";
        assert_eq!(digest1.hex_digest(), expected_digest);

        let mut hasher2 = T::new(NonZeroUsize::new(4).expect("not zero"));
        assert_matches!(hasher2.update(0, "thisthat".as_bytes()), Ok(()));
        assert_matches!(hasher2.update(8, "moreverylong".as_bytes()), Ok(()));
        let digest2 = hasher2.finalize();
        assert_eq!(digest2.hex_digest(), expected_digest);

        assert_eq!(digest1.to_bytes(), digest2.to_bytes());
    }

    #[apply(hashers)]
    fn test_no_blocks<T: TestHasher>(#[case] _t: PhantomData<T>) {
        let hasher = T::new(NonZeroUsize::new(16).expect("not zero"));
        let expected_digest = "0000000000000000000000000000000000000000000000000000000000000000";
        assert_eq!(hasher.finalize().hex_digest(), expected_digest);
    }

    #[apply(hashers)]
    fn test_different_block_sizes_hash_differently<T: TestHasher>(#[case] _t: PhantomData<T>) {
        let mut hasher1 = T::new(NonZeroUsize::new(16).expect("not zero"));
        assert_matches!(hasher1.update(0, "a short string".as_bytes()), Ok(()));

        let mut hasher2 = T::new(NonZeroUsize::new(4).expect("not zero"));
        assert_matches!(hasher2.update(0, "a short string".as_bytes()), Ok(()));
        let digest1 = hasher1.finalize();
        let digest2 = hasher2.finalize();
        assert_ne!(digest1.to_bytes(), digest2.to_bytes(),);
        assert_ne!(digest1.hex_digest(), digest2.hex_digest(),);
    }

    #[apply(hashers)]
    fn test_digest<T: TestHasher>(#[case] _t: PhantomData<T>) {
        let mut hasher = T::new(NonZeroUsize::new(16).expect("not zero"));
        assert_matches!(hasher.update(0, "a short string".as_bytes()), Ok(()));
        let digest = hasher.finalize();
        let hex_digest = digest.hex_digest();
        let new_digest =
            CommutativeHashDigest::decode_hex_digest(hex_digest.as_str()).expect("hex to decode");
        assert_eq!(new_digest.to_bytes(), digest.to_bytes());
    }

    #[test]
    fn test_misaligned_data() {
        let hasher = ParallelHasher::<Sha512>::new(NonZeroUsize::new(16).expect("not zero"));
        assert_matches!(
            hasher.update(2, "a short string".as_bytes()),
            Err(ParallelHasherError::MisalignedData)
        );
    }

    #[test]
    fn test_hasher_is_commutative() {
        let chunks = ["this", "that", "and more"];
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let expected_digest = "9874cae284d8b2e71dea634e932be1049b5e0740964d1bb5db32a03ca81bec28";
        for permutation in permutations {
            let hasher = ParallelHasher::<Sha512>::new(NonZeroUsize::new(4).expect("not zero"));
            for i in permutation {
                assert_matches!(hasher.update(i * 4, chunks[i].as_bytes()), Ok(()));
            }
            assert_eq!(hasher.finalize().hex_digest(), expected_digest);
        }
    }

    fn generate_data(size: usize) -> Vec<u8> {
        let seed_value: u64 = 42;
        let mut rng = StdRng::seed_from_u64(seed_value);
        let mut bytes = vec![0u8; size];
        rng.fill_bytes(&mut bytes);
        bytes
    }

    #[test]
    fn test_hash_large_value_parallel() {
        let expected_digest = "ee7338897f6eaf9b4f26b6ec33d3192bca8231485ac70d64fc81128397af0755";
        let data = generate_data(1 << 27); // 128 MiB
        let data_size = data.len();
        let chunk_size = 1 << 24; // 16 MiB
        let block_size = 1 << 20; // 1 MiB
        let hasher =
            ParallelHasher::<Sha512>::new(NonZeroUsize::new(block_size).expect("not zero"));
        (0..data_size)
            .step_by(1 << 24)
            .collect_vec()
            .into_par_iter()
            .try_for_each(|start| -> Result<()> {
                let buf = &data[start..(start + chunk_size.min(data_size - start))];
                hasher.update(start, buf)?;
                Ok(())
            })
            .expect("hashing to succeed");
        assert_eq!(hasher.finalize().hex_digest(), expected_digest);
    }

    #[rstest]
    fn test_hash_large_value_sequential(#[values(1, 2, 8, 10, 60, 200)] num_chunks: usize) {
        let expected_digest = "ee7338897f6eaf9b4f26b6ec33d3192bca8231485ac70d64fc81128397af0755";
        let data = generate_data(1 << 27); // 128 MiB
        let data_size = data.len();
        let chunk_size = data_size / num_chunks;
        let block_size = 1 << 20; // 1 MiB
        let mut hasher =
            SequentialHasher::<Sha512>::new(NonZeroUsize::new(block_size).expect("not zero"));
        data.chunks(chunk_size).for_each(|chunk| {
            hasher.update(chunk);
        });
        assert_eq!(hasher.finalize().hex_digest(), expected_digest);
    }

    #[test]
    fn test_decode_hex_digest() {
        let digest = "fecf448671b40c18276fc45c7c600be62dd514c87bb614591d151f4db668272b";
        assert_eq!(
            CommutativeHashDigest::decode_hex_digest(digest)
                .expect("to decode")
                .hex_digest(),
            digest
        );
        assert_matches!(
            CommutativeHashDigest::decode_hex_digest(""),
            Err(CommutativeHashDigestError::Not32Bytes)
        );
        assert_matches!(
            CommutativeHashDigest::decode_hex_digest("abc"),
            Err(CommutativeHashDigestError::OddLength)
        );
        assert_matches!(
            CommutativeHashDigest::decode_hex_digest("ZYXW"),
            Err(CommutativeHashDigestError::InvalidHexCharacter { c: _, index: _ })
        );
    }

    #[test]
    fn test_decode_invalid_length_hex_digest() {
        assert_matches!(
            CommutativeHashDigest::decode_hex_digest("abcd"),
            Err(CommutativeHashDigestError::Not32Bytes)
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialization() {
        let digest = SequentialHasher::<Sha512>::digest_from_bytes(
            NonZeroUsize::new(16).expect("not zero"),
            "a short string".as_bytes(),
        );
        let serialized = serde_json::to_string(&digest).unwrap();
        let deserialized: CommutativeHashDigest = serde_json::from_str(&serialized).unwrap();
        assert_eq!(digest, deserialized);
    }

    #[test]
    fn test_to_padded_bytes() {
        assert_eq!(to_padded_bytes(16), [16, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(to_padded_bytes(2_usize.pow(31)), [0, 0, 0, 128, 0, 0, 0, 0]);
    }
}
