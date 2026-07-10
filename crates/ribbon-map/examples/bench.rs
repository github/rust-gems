//! Rough throughput benchmark for `RibbonMap`.
//!
//! Run with e.g. `cargo run --release --example bench -- 10000000`.

use std::time::Instant;

use ribbon_map::RibbonMap;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn bench(n: usize) {
    let mut state = 0x1234_5678_9abc_def0 ^ n as u64;
    let keys: Vec<u64> = (0..n).map(|_| splitmix64(&mut state)).collect();
    let values: Vec<u32> = (0..n as u32)
        .map(|i| i.wrapping_mul(31).wrapping_add(7))
        .collect();

    let t = Instant::now();
    let map = RibbonMap::try_construct(&keys, &values).expect("construction succeeds");
    let build_s = t.elapsed().as_secs_f64();

    // Positive lookups (all keys present).
    let t = Instant::now();
    let mut checksum = 0u64;
    for &k in &keys {
        if let Some(v) = map.get(k) {
            checksum = checksum.wrapping_add(*v as u64);
        }
    }
    let pos_s = t.elapsed().as_secs_f64();

    // Negative lookups (disjoint key set).
    let absent: Vec<u64> = (0..n).map(|_| splitmix64(&mut state)).collect();
    let t = Instant::now();
    let mut hits = 0u64;
    for &k in &absent {
        if map.get(k).is_some() {
            hits += 1;
        }
    }
    let neg_s = t.elapsed().as_secs_f64();

    println!(
        "u32 n={:>10}  build {:>6.2} Ms/s ({:>5.2}s)  pos {:>6.1} M/s  neg {:>6.1} M/s  \
         {:>4.1} bits/key  slots/key {:.3}  (checksum {}, neg_fp {:.3})",
        n,
        n as f64 / build_s / 1e6,
        build_s,
        n as f64 / pos_s / 1e6,
        n as f64 / neg_s / 1e6,
        map.bits_per_key(),
        map.slot_count() as f64 / n as f64,
        checksum,
        hits as f64 / n as f64,
    );
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000);

    bench(n);
}
