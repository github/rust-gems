//! Construction and lookup throughput of the windowed [`CuckooFilter`] as a **value map**, compared
//! with the [`BinaryFuseMap`] and a `std::HashMap` — all storing an `8`-byte value per key and using
//! the **same** `murmur64` hash.
//!
//! The three make different trade-offs. The cuckoo map probes two local windows (its fingerprints
//! sit in two cache lines) then fetches the value from a parallel array; the fuse map probes three
//! scattered slots but fuses each value into the word it already reads; the `HashMap` chases one
//! bucket and compares the full key. The cuckoo and fuse maps are approximate (a ~1-byte fingerprint
//! per key), while the `HashMap` is exact.
//!
//! The cuckoo map's XOR addressing rounds the window count up to a power of two, so to compare it at
//! a fair memory point (not a wasteful rounding boundary) the sizes here are chosen as **power-of-two
//! window counts filled to [`FILL`]** — i.e. the power-of-two table at its maximal occupancy.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::hint::black_box;
use std::time::Duration;

use binary_fuse_map::BinaryFuseMap;
use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use cuckoo_filter::{CuckooFilter, WINDOW};
use cuckoofilter::CuckooFilter as CuckooCrate;
use rand::{rngs::StdRng, RngExt, SeedableRng};

/// Value size (bytes) stored per key by all three maps.
const VAL: usize = 4;

/// The same 64-bit finalizer (`murmur64`) the cuckoo filter hashes keys with, so the `HashMap`
/// baseline uses an identical hash function rather than the standard library's SipHash.
fn murmur64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    h
}

/// A `Hasher` that maps a single `u64` key through `murmur64` (u64 keys go through `write_u64`).
#[derive(Default)]
struct MurmurHasher(u64);

impl Hasher for MurmurHasher {
    fn finish(&self) -> u64 {
        murmur64(self.0)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(b);
        }
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
}

/// A `HashMap<u64, [u8; VAL]>` that hashes with the filter's `murmur64` — the "same hash" baseline.
type MurmurMap = HashMap<u64, [u8; VAL], BuildHasherDefault<MurmurHasher>>;

fn build_hashmap(keys: &[u64], values: &[[u8; VAL]]) -> MurmurMap {
    let mut map = MurmurMap::with_capacity_and_hasher(keys.len(), Default::default());
    map.extend(keys.iter().copied().zip(values.iter().copied()));
    map
}

/// Occupancy each power-of-two window table is filled to: the highest load the random-walk build
/// reaches reliably in a single attempt, so the power-of-two filter is measured at its densest.
/// `(2,2)` (`WINDOW == 2`) has a lower threshold than `(2,4)`, so it is filled less densely.
const FILL: f64 = if WINDOW <= 2 { 0.88 } else { 0.92 };

/// `log2` of the window counts to benchmark (~16 K, 128 K, 1 M, 16 M windows).
const LOG2_WINDOWS: [u32; 4] = [14, 17, 20, 24];

/// Key count and slot count that fill a `2^b`-window table to `FILL`.
fn dims(b: u32) -> (usize, usize) {
    let num_windows = 1usize << b;
    (
        (FILL * num_windows as f64) as usize,
        num_windows + WINDOW - 1,
    )
}

/// Distinct random keys derived from a seeded RNG (distinct with overwhelming probability).
fn keys(n: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(0xB1A2_F03E_5151_7C0D);
    (0..n).map(|_| rng.random()).collect()
}

/// A `VAL`-byte value per key (the low bytes of the key).
fn values(keys: &[u64]) -> Vec<[u8; VAL]> {
    keys.iter()
        .map(|&k| {
            let mut v = [0u8; VAL];
            let bytes = k.to_le_bytes();
            v.copy_from_slice(&bytes[..VAL]);
            v
        })
        .collect()
}

/// Builds a cuckoo map at the exact `slots` size (retrying a few seeds).
fn build_cuckoo(keys: &[u64], vals: &[[u8; VAL]], slots: usize) -> CuckooFilter<VAL> {
    (0..16)
        .find_map(|s| CuckooFilter::try_construct(keys, vals, slots, s))
        .expect("construction succeeds")
}

/// Builds the reference `cuckoofilter` crate's filter — a classic `(2, 4)` cuckoo filter with a
/// 1-byte fingerprint (membership only, no values) — from the same keys, hashing with the same
/// `murmur64`. It is dynamic (one `add` at a time) and sizes to the same power-of-two load as ours.
fn build_cuckoo_crate(keys: &[u64]) -> CuckooCrate<MurmurHasher> {
    let mut cf = CuckooCrate::<MurmurHasher>::with_capacity(keys.len());
    for &k in keys {
        let _ = cf.add(&k);
    }
    cf
}

fn construct(c: &mut Criterion) {
    let mut group = c.benchmark_group("construct");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    for b in LOG2_WINDOWS {
        let (n, slots) = dims(b);
        let ks = keys(n);
        let vs = values(&ks);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("cuckoo", n), &n, |bn, _| {
            bn.iter(|| black_box(build_cuckoo(black_box(&ks), black_box(&vs), slots)))
        });
        group.bench_with_input(BenchmarkId::new("fuse_map", n), &n, |bn, _| {
            bn.iter(|| {
                black_box(
                    BinaryFuseMap::<VAL>::try_construct(black_box(&ks), black_box(&vs))
                        .expect("construction succeeds"),
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("hashmap", n), &n, |bn, _| {
            bn.iter(|| black_box(build_hashmap(black_box(&ks), black_box(&vs))))
        });
        group.bench_with_input(BenchmarkId::new("cuckoo_crate", n), &n, |bn, _| {
            bn.iter(|| black_box(build_cuckoo_crate(black_box(&ks))))
        });
    }
    group.finish();
}

fn get(c: &mut Criterion) {
    let mut group = c.benchmark_group("get");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for b in [14u32, 20, 24] {
        let (n, slots) = dims(b);
        let ks = keys(n);
        let vs = values(&ks);
        let cuckoo = build_cuckoo(&ks, &vs, slots);
        let fuse = BinaryFuseMap::<VAL>::try_construct(&ks, &vs).unwrap();
        let map = build_hashmap(&ks, &vs);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("cuckoo", n), &n, |bn, _| {
            bn.iter(|| {
                let mut acc = 0u8;
                for &k in &ks {
                    if let Some(v) = cuckoo.get(black_box(k)) {
                        acc ^= v[0];
                    }
                }
                black_box(acc)
            })
        });
        group.bench_with_input(BenchmarkId::new("fuse_map", n), &n, |bn, _| {
            bn.iter(|| {
                let mut acc = 0u8;
                for &k in &ks {
                    if let Some(v) = fuse.get(black_box(k)) {
                        acc ^= v[0];
                    }
                }
                black_box(acc)
            })
        });
        group.bench_with_input(BenchmarkId::new("hashmap", n), &n, |bn, _| {
            bn.iter(|| {
                let mut acc = 0u8;
                for &k in &ks {
                    if let Some(v) = map.get(black_box(&k)) {
                        acc ^= v[0];
                    }
                }
                black_box(acc)
            })
        });
    }
    group.finish();
}

/// Membership-only comparison against the `cuckoofilter` crate (which stores no values), using our
/// [`CuckooFilter::contains`]. Both hold the same keys hashed with the same `murmur64`.
fn contains(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for b in [14u32, 20, 24] {
        let (n, slots) = dims(b);
        let ks = keys(n);
        let vs = values(&ks);
        let cuckoo = build_cuckoo(&ks, &vs, slots);
        let cuckoo_crate = build_cuckoo_crate(&ks);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("cuckoo", n), &n, |bn, _| {
            bn.iter(|| {
                let mut acc = 0u64;
                for &k in &ks {
                    acc += cuckoo.contains(black_box(k)) as u64;
                }
                black_box(acc)
            })
        });
        group.bench_with_input(BenchmarkId::new("cuckoo_crate", n), &n, |bn, _| {
            bn.iter(|| {
                let mut acc = 0u64;
                for &k in &ks {
                    acc += cuckoo_crate.contains(black_box(&k)) as u64;
                }
                black_box(acc)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, construct, get, contains);
criterion_main!(benches);
