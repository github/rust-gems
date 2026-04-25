//! Murmur1 hash function (ported from blackbird_core).

use std::hash::Hasher;

trait XorRsh {
    fn xor_rsh(self, nbits: u32) -> Self;
}

impl XorRsh for u32 {
    #[inline]
    fn xor_rsh(self, nbits: u32) -> u32 {
        self ^ (self >> nbits)
    }
}

fn murmur1_hash(bytes: &[u8], seed: u32) -> u32 {
    const M: u32 = 0xc6a4_a793;
    const R: u32 = 16;

    let mut h = seed ^ (bytes.len() as u32).wrapping_mul(M);

    let chunks_len = bytes.len() / 4 * 4;
    for chunk in bytes[..chunks_len].chunks_exact(4) {
        let ptr = chunk.as_ptr() as *const u32;
        let k = unsafe { ptr.read_unaligned() };
        h = h.wrapping_add(k).wrapping_mul(M).xor_rsh(R);
    }

    let mut tail_bytes = [0u8; 4];
    tail_bytes[..bytes.len() - chunks_len].copy_from_slice(&bytes[chunks_len..]);
    h = h
        .wrapping_add(u32::from_le_bytes(tail_bytes))
        .wrapping_mul(M)
        .xor_rsh(R);

    h.wrapping_mul(M).xor_rsh(10).wrapping_mul(M).xor_rsh(17)
}

/// Hasher implementing the Murmur1 hash function.
struct Murmur1Hasher {
    bytes: Vec<u8>,
}

impl Default for Murmur1Hasher {
    fn default() -> Self {
        Self {
            bytes: Vec::with_capacity(64),
        }
    }
}

impl Hasher for Murmur1Hasher {
    fn write(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    fn finish(&self) -> u64 {
        murmur1_hash(&self.bytes, 0).into()
    }
}

pub(crate) fn hash_bigram(gram: (u8, u8)) -> u32 {
    use std::hash::Hash;
    let mut h = Murmur1Hasher::default();
    gram.hash(&mut h);
    h.finish() as u32
}
