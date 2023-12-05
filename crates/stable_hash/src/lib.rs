use std::hash::Hasher;

use fnv::FnvHasher;

/// Trait for values that have a stable 64-bit hash code.
///
/// Unlike ordinary hash codes in Rust, stable hash codes are consistent across
/// process boundaries and Rust toolchain versions.
///
/// If the stable hash of *any* type changes, it's a breaking change, so
/// bump [`INDEX_VERSION`].
pub trait StableHash {
    fn stable_hash(&self) -> u64;
}

impl StableHash for u16 {
    fn stable_hash(&self) -> u64 {
        let mut hasher = FnvHasher::default();
        hasher.write_u16(*self);
        hasher.finish()
    }
}

impl StableHash for u32 {
    fn stable_hash(&self) -> u64 {
        let mut hasher = FnvHasher::default();
        hasher.write_u32(*self);
        hasher.finish()
    }
}

impl StableHash for u64 {
    fn stable_hash(&self) -> u64 {
        let mut hasher = FnvHasher::default();
        hasher.write_u64(*self);
        hasher.finish()
    }
}

impl StableHash for &str {
    fn stable_hash(&self) -> u64 {
        let mut hasher = FnvHasher::default();
        hasher.write(self.as_bytes());
        hasher.finish()
    }
}

impl StableHash for String {
    fn stable_hash(&self) -> u64 {
        let mut hasher = FnvHasher::default();
        hasher.write(self.as_bytes());
        hasher.finish()
    }
}

impl StableHash for &[u8] {
    fn stable_hash(&self) -> u64 {
        let mut hasher = FnvHasher::default();
        hasher.write(self);
        hasher.finish()
    }
}

impl StableHash for [u8] {
    fn stable_hash(&self) -> u64 {
        let mut hasher = FnvHasher::default();
        hasher.write(self);
        hasher.finish()
    }
}

impl StableHash for Vec<u8> {
    fn stable_hash(&self) -> u64 {
        self.as_slice().stable_hash()
    }
}
