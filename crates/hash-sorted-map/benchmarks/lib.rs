use std::hash::{BuildHasherDefault, Hasher};

use rand::Rng;

const ARBITRARY0: u64 = 0x243f6a8885a308d3;

/// Folded multiply: full u64×u64→u128, then XOR the two halves.
#[inline(always)]
pub fn folded_multiply(x: u64, y: u64) -> u64 {
    let full = (x as u128).wrapping_mul(y as u128);
    (full as u64) ^ ((full >> 64) as u64)
}

/// A hasher that passes through u32 keys without hashing, suitable for
/// keys that are already well-distributed.
#[derive(Default)]
pub struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn write(&mut self, _bytes: &[u8]) {
        unimplemented!("IdentityHasher only supports write_u32");
    }
    fn write_u32(&mut self, i: u32) {
        self.0 = (i as u64) | ((i as u64) << 32);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

pub type IdentityBuildHasher = BuildHasherDefault<IdentityHasher>;

/// Generate `n` random trigrams as well-distributed u32 hashes.
/// Each trigram is packed into a u32, then scrambled with folded_multiply.
pub fn random_trigram_hashes(n: usize) -> Vec<u32> {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| {
            let a = rng.random_range(b'a'..=b'z') as u32;
            let b = rng.random_range(b'a'..=b'z') as u32;
            let c = rng.random_range(b'a'..=b'z') as u32;
            let packed = a | (b << 8) | (c << 16);
            folded_multiply(packed as u64, ARBITRARY0) as u32
        })
        .collect()
}
