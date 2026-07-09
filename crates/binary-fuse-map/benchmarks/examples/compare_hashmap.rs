//! One-shot construction/lookup comparison of [`BinaryFuseMap`] versus a [`foldhash`]-backed
//! `std::collections::HashMap` at large sizes (criterion's repeated sampling is impractical here).
//!
//! Run with, e.g.:
//! ```text
//! cargo run --release -p binary-fuse-map-benchmarks --example compare_hashmap -- 100000000
//! ```
//! The optional argument is the number of entries (default 100,000,000); values are 5 bytes.

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

use binary_fuse_map::BinaryFuseMap;
use foldhash::fast::RandomState;

const VALUE_BYTES: usize = 5;

/// SplitMix64: a bijection on `u64`, so successive outputs are distinct.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn secs(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(100_000_000);
    println!("comparison at {n} entries, {VALUE_BYTES}-byte values\n");

    // Distinct keys and values.
    let mut state = 0x0123_4567_89ab_cdef_u64;
    let keys: Vec<u64> = (0..n).map(|_| splitmix64(&mut state)).collect();
    let values: Vec<[u8; VALUE_BYTES]> = (0..n)
        .map(|i| {
            let mut v = [0u8; VALUE_BYTES];
            v.copy_from_slice(&(i as u64).to_le_bytes()[..VALUE_BYTES]);
            v
        })
        .collect();

    // --- binary fuse map ---
    let t = Instant::now();
    let fuse = BinaryFuseMap::<VALUE_BYTES>::try_construct(&keys, &values).expect("distinct keys");
    let fuse_build = secs(t);

    let sample = n.min(50_000_000);
    let t = Instant::now();
    let mut acc = 0u8;
    for &k in keys.iter().take(sample) {
        acc ^= fuse.get(k).expect("present")[0];
    }
    let fuse_lookup = secs(t);
    black_box(acc);
    let fuse_mib = fuse.memory_usage() as f64 / (1u64 << 20) as f64;

    // --- foldhash HashMap ---
    let t = Instant::now();
    let mut hash: HashMap<u64, [u8; VALUE_BYTES], RandomState> =
        HashMap::with_capacity_and_hasher(n, RandomState::default());
    hash.extend(keys.iter().copied().zip(values.iter().copied()));
    let hash_build = secs(t);

    let t = Instant::now();
    let mut acc = 0u8;
    for &k in keys.iter().take(sample) {
        acc ^= hash.get(&k).expect("present")[0];
    }
    let hash_lookup = secs(t);
    black_box(acc);
    // Allocated buckets: capacity rounded up to a power of two, 14 bytes each (8 key + 5 val + 1
    // control byte). `capacity()` reports usable slots (~7/8 of the bucket count).
    let buckets = (hash.capacity() as f64 / 0.875).ceil() as u64;
    let hash_mib =
        (buckets.next_power_of_two() * (8 + VALUE_BYTES as u64 + 1)) as f64 / (1u64 << 20) as f64;

    let mlookups = |s: f64| sample as f64 / s / 1e6;
    let mbuild = |s: f64| n as f64 / s / 1e6;
    println!("                     build            lookup ({sample} keys)      memory");
    println!(
        "  fuse map     {:>7.2}s ({:>5.1} M/s)   {:>7.2}s ({:>5.1} M/s)   {:>7.0} MiB ({:.0} bits/key)",
        fuse_build,
        mbuild(fuse_build),
        fuse_lookup,
        mlookups(fuse_lookup),
        fuse_mib,
        fuse.bits_per_key(),
    );
    println!(
        "  foldhash map {:>7.2}s ({:>5.1} M/s)   {:>7.2}s ({:>5.1} M/s)   {:>7.0} MiB ({:.0} bits/key)",
        hash_build,
        mbuild(hash_build),
        hash_lookup,
        mlookups(hash_lookup),
        hash_mib,
        hash_mib * (1u64 << 20) as f64 * 8.0 / n as f64,
    );
}
