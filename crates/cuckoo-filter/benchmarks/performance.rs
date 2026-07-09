//! Construction and lookup throughput of the windowed [`CuckooFilter`] compared with the
//! [`BinaryFuseMap`] (a binary fuse filter), on the same random `u64` keys.
//!
//! Both are approximate-membership structures with a ~1-byte fingerprint per key. They make
//! different trade-offs: the cuckoo filter probes two local windows (two cache lines) but relocates
//! on insert, while the fuse filter probes three scattered slots (three cache lines) but is built by
//! a single peeling pass.
//!
//! The cuckoo filter's XOR addressing rounds the window count up to a power of two, so to compare it
//! at a fair memory point (not a wasteful rounding boundary) the sizes here are chosen as
//! **power-of-two window counts filled to [`FILL`]** — i.e. the power-of-two table at its maximal
//! occupancy (~8.7 bits/key, matching an arbitrary-sized filter at the same load).
//!
//! The fuse map is instantiated with a 1-byte value (`BinaryFuseMap<1>`) so both structures store a
//! comparable amount per key; `get(k).is_some()` is used as the membership test.

use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hasher};
use std::hint::black_box;
use std::time::Duration;

use binary_fuse_map::BinaryFuseMap;
use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use cuckoo_filter::{CuckooFilter, WINDOW};
use rand::{rngs::StdRng, RngExt, SeedableRng};

/// The same 64-bit finalizer (`murmur64`) the cuckoo filter hashes keys with, so the `HashSet`
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

/// A `HashSet<u64>` that hashes with the filter's `murmur64` — the "same hash function" baseline.
type MurmurSet = HashSet<u64, BuildHasherDefault<MurmurHasher>>;

fn build_hashset(keys: &[u64]) -> MurmurSet {
    let mut set = MurmurSet::with_capacity_and_hasher(keys.len(), Default::default());
    set.extend(keys.iter().copied());
    set
}

/// Occupancy each power-of-two window table is filled to: the highest load the random-walk build
/// reaches reliably in a single attempt, so the power-of-two filter is measured at its densest.
const FILL: f64 = 0.92;

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

/// Builds a cuckoo filter at the exact `slots` size (retrying a few seeds).
fn build_cuckoo(keys: &[u64], slots: usize) -> CuckooFilter {
    (0..16)
        .find_map(|s| CuckooFilter::try_construct(keys, slots, s))
        .expect("construction succeeds")
}

fn construct(c: &mut Criterion) {
    let mut group = c.benchmark_group("construct");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    for b in LOG2_WINDOWS {
        let (n, slots) = dims(b);
        let ks = keys(n);
        let values = vec![[0u8; 1]; n];
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("cuckoo", n), &n, |bn, _| {
            bn.iter(|| black_box(build_cuckoo(black_box(&ks), slots)))
        });
        group.bench_with_input(BenchmarkId::new("fuse_map", n), &n, |bn, _| {
            bn.iter(|| {
                black_box(
                    BinaryFuseMap::<1>::try_construct(black_box(&ks), black_box(&values))
                        .expect("construction succeeds"),
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("hashset", n), &n, |bn, _| {
            bn.iter(|| black_box(build_hashset(black_box(&ks))))
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
        let values = vec![[0u8; 1]; n];
        let cuckoo = build_cuckoo(&ks, slots);
        let fuse = BinaryFuseMap::<1>::try_construct(&ks, &values).unwrap();
        let set = build_hashset(&ks);
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
        group.bench_with_input(BenchmarkId::new("hashset", n), &n, |bn, _| {
            bn.iter(|| {
                let mut acc = 0u64;
                for &k in &ks {
                    acc += set.contains(black_box(&k)) as u64;
                }
                black_box(acc)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, construct, get);
criterion_main!(benches);
