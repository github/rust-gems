//! Rough throughput benchmark for `RibbonMap`.
//!
//! Run with e.g. `cargo run --release --example bench -- 10000000`.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::time::Instant;

use ribbon_map::RibbonMap;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn murmur64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    h
}

#[derive(Default)]
struct RibbonHasher {
    hash: u64,
}

impl Hasher for RibbonHasher {
    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, bytes: &[u8]) {
        // Keys in this benchmark are always u64. Keep a deterministic fallback for safety.
        let mut h = 0u64;
        for &b in bytes {
            h = h.wrapping_mul(0x100_0000_01B3).wrapping_add(b as u64);
        }
        self.hash = murmur64(h);
    }

    fn write_u64(&mut self, i: u64) {
        self.hash = murmur64(i);
    }
}

fn bench(num_keys: usize) {
    let mut state = 0x1234_5678_9abc_def0 ^ num_keys as u64;
    let keys: Vec<u64> = (0..num_keys).map(|_| splitmix64(&mut state)).collect();
    let values: Vec<u32> = (0..num_keys as u32)
        .map(|i| i.wrapping_mul(31).wrapping_add(7))
        .collect();

    let t = Instant::now();
    let ribbon = RibbonMap::try_construct(&keys, &values).expect("construction succeeds");
    let ribbon_build_s = t.elapsed().as_secs_f64();

    // Positive lookups (all keys present) for RibbonMap.
    let t = Instant::now();
    let mut ribbon_checksum = 0u64;
    for &k in &keys {
        if let Some(v) = ribbon.get(k) {
            ribbon_checksum = ribbon_checksum.wrapping_add(*v as u64);
        }
    }
    let ribbon_pos_s = t.elapsed().as_secs_f64();

    // Negative lookups (disjoint key set) for RibbonMap.
    let absent: Vec<u64> = (0..num_keys).map(|_| splitmix64(&mut state)).collect();
    let t = Instant::now();
    let mut ribbon_hits = 0u64;
    for &k in &absent {
        if ribbon.get(k).is_some() {
            ribbon_hits += 1;
        }
    }
    let ribbon_neg_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut hash: HashMap<u64, u32, BuildHasherDefault<RibbonHasher>> =
        HashMap::with_capacity_and_hasher(num_keys, BuildHasherDefault::default());
    for (&k, &v) in keys.iter().zip(&values) {
        hash.insert(k, v);
    }
    let hash_build_s = t.elapsed().as_secs_f64();

    // Positive lookups (all keys present) for HashMap.
    let t = Instant::now();
    let mut hash_checksum = 0u64;
    for &k in &keys {
        if let Some(v) = hash.get(&k) {
            hash_checksum = hash_checksum.wrapping_add(*v as u64);
        }
    }
    let hash_pos_s = t.elapsed().as_secs_f64();

    // Negative lookups (disjoint key set) for HashMap.
    let t = Instant::now();
    let mut hash_hits = 0u64;
    for &k in &absent {
        if hash.get(&k).is_some() {
            hash_hits += 1;
        }
    }
    let hash_neg_s = t.elapsed().as_secs_f64();

    let keys_f64 = num_keys as f64;
    let ribbon_slots_per_key = if num_keys == 0 {
        0.0
    } else {
        ribbon.slot_count() as f64 / keys_f64
    };

    println!("u32 keys={:>10}", num_keys,);
    println!(
        "  RibbonMap: build {:>6.2} Ms/s ({:>5.2}s)  pos {:>6.1} M/s  neg {:>6.1} M/s  \
         {:>4.1} bits/key  slots/key {:.3}  (checksum {}, neg_hit {:.3})",
        keys_f64 / ribbon_build_s / 1e6,
        ribbon_build_s,
        keys_f64 / ribbon_pos_s / 1e6,
        keys_f64 / ribbon_neg_s / 1e6,
        ribbon.bits_per_key(),
        ribbon_slots_per_key,
        ribbon_checksum,
        if num_keys == 0 {
            0.0
        } else {
            ribbon_hits as f64 / keys_f64
        },
    );
    println!(
        "  HashMap:  build {:>6.2} Ms/s ({:>5.2}s)  pos {:>6.1} M/s  neg {:>6.1} M/s  \
         {:>4} bits/key  slots/key {:>5}  (checksum {}, neg_hit {:.3})",
        keys_f64 / hash_build_s / 1e6,
        hash_build_s,
        keys_f64 / hash_pos_s / 1e6,
        keys_f64 / hash_neg_s / 1e6,
        "n/a",
        "n/a",
        hash_checksum,
        if num_keys == 0 {
            0.0
        } else {
            hash_hits as f64 / keys_f64
        },
    );
}

fn main() {
    let num_keys: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000);

    bench(num_keys);
}
