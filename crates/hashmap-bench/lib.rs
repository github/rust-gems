pub mod prefix_map;
pub mod prefix_map_simd;

use rand::Rng;
use std::hash::{BuildHasherDefault, Hasher};

/// A hasher that returns the input unchanged. Only valid for u32 keys
/// that are already well-distributed hashes.
#[derive(Default)]
pub struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn write(&mut self, _bytes: &[u8]) {
        unimplemented!("IdentityHasher only supports write_u32");
    }
    fn write_u32(&mut self, i: u32) {
        self.0 = i as u64;
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

pub type IdentityBuildHasher = BuildHasherDefault<IdentityHasher>;

/// Generate `n` random trigrams as well-distributed u32 hashes.
/// Each trigram is packed into a u32, then scrambled with a murmur3 finalizer.
pub fn random_trigram_hashes(n: usize) -> Vec<u32> {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| {
            let a = rng.random_range(b'a'..=b'z') as u32;
            let b = rng.random_range(b'a'..=b'z') as u32;
            let c = rng.random_range(b'a'..=b'z') as u32;
            let packed = a | (b << 8) | (c << 16);
            let mut h = packed;
            h ^= h >> 16;
            h = h.wrapping_mul(0x85ebca6b);
            h ^= h >> 13;
            h = h.wrapping_mul(0xc2b2ae35);
            h ^= h >> 16;
            h
        })
        .collect()
}
