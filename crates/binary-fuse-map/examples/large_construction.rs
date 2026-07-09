//! One-shot construction/lookup benchmark for large inputs (criterion's repeated sampling is
//! impractical at hundreds of millions of keys).
//!
//! Run with, e.g.:
//! ```text
//! cargo run --release --example large_construction -- 400000000
//! ```
//! The optional argument is the number of keys (default 400,000,000). Values are 8 bytes each.

use std::time::Instant;

use binary_fuse_map::BinaryFuseMap;

const VALUE_BYTES: usize = 5;

/// SplitMix64: a bijection on `u64`, so successive outputs are distinct — handy for generating a
/// large set of distinct keys cheaply and deterministically.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn human_duration(secs: f64) -> String {
    if secs >= 1.0 {
        format!("{secs:.2} s")
    } else {
        format!("{:.1} ms", secs * 1000.0)
    }
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(400_000_000);

    println!("binary fuse map: {n} keys, {VALUE_BYTES}-byte values");

    // Generate distinct keys and values.
    let t = Instant::now();
    let mut state = 0x0123_4567_89ab_cdef_u64;
    let mut keys = vec![0u64; n];
    for k in keys.iter_mut() {
        *k = splitmix64(&mut state);
    }
    let mut values = vec![[0u8; VALUE_BYTES]; n];
    for (i, v) in values.iter_mut().enumerate() {
        v.copy_from_slice(&(i as u64).to_le_bytes()[..VALUE_BYTES]);
    }
    println!(
        "  generated inputs in {}",
        human_duration(t.elapsed().as_secs_f64())
    );

    // Construct.
    let t = Instant::now();
    let map = BinaryFuseMap::<VALUE_BYTES>::try_construct(&keys, &values)
        .expect("construction failed (are all keys distinct?)");
    let build = t.elapsed().as_secs_f64();
    println!(
        "  built in {} ({:.2} M keys/s)",
        human_duration(build),
        n as f64 / build / 1e6
    );
    println!(
        "  {} slots, {:.3} bits/key, {:.2} GiB",
        map.slot_count(),
        map.bits_per_key(),
        map.memory_usage() as f64 / (1u64 << 30) as f64
    );

    // Verify + time lookups on a sample (full verification of 400M is itself expensive).
    let sample = n.min(50_000_000);
    let t = Instant::now();
    let mut checksum = 0u64;
    for &k in keys.iter().take(sample) {
        let v = map.get(k).expect("inserted key must be found");
        let mut bytes = [0u8; 8];
        bytes[..VALUE_BYTES].copy_from_slice(&v);
        checksum ^= u64::from_le_bytes(bytes);
    }
    let lookup = t.elapsed().as_secs_f64();
    // Fully validate correctness on the sample (cheap relative to construction).
    for (k, v) in keys.iter().zip(&values).take(sample) {
        assert_eq!(map.get(*k), Some(*v), "value mismatch");
    }
    println!(
        "  {sample} lookups in {} ({:.2} M lookups/s), checksum={checksum:#x}",
        human_duration(lookup),
        sample as f64 / lookup / 1e6
    );
    println!("  verified {sample} round-trips OK");
}
