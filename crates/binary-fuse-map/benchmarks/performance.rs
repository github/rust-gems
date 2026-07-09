//! Construction and lookup throughput for [`BinaryFuseMap`] across a range of sizes.
//!
//! Each size is also measured against a `std::collections::HashMap` keyed by the same `u64`s and
//! hashed with [`foldhash`], as an exact (no false positives) baseline. The fuse map trades a small
//! false-positive rate for far lower memory and, at large sizes, faster lookups; the hash map is the
//! natural exact alternative.
//!
//! The headline "insert 400 million values" measurement lives in the `large_construction` example
//! of the `binary-fuse-map` crate, since criterion's repeated sampling is impractical at that
//! scale. These benches cover the sizes where statistical sampling is cheap.

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

use binary_fuse_map::BinaryFuseMap;
use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use foldhash::fast::RandomState;
use rand::{rngs::StdRng, RngExt, SeedableRng};

/// A `HashMap` from `u64` keys to the fuse map's value type, hashed with foldhash.
type FoldHashMap = HashMap<u64, [u8; VALUE_BYTES], RandomState>;

/// Builds a foldhash-backed `HashMap` from the given keys and values.
fn build_hashmap(keys: &[u64], values: &[[u8; VALUE_BYTES]]) -> FoldHashMap {
    let mut map = HashMap::with_capacity_and_hasher(keys.len(), RandomState::default());
    map.extend(keys.iter().copied().zip(values.iter().copied()));
    map
}

const VALUE_BYTES: usize = 5;

/// Distinct random keys plus `VALUE_BYTES`-byte values derived from a seeded RNG.
fn inputs(n: usize) -> (Vec<u64>, Vec<[u8; VALUE_BYTES]>) {
    let mut rng = StdRng::seed_from_u64(0xB1A2_F03E_5151_7C0D);
    // Random u64s are distinct with overwhelming probability at these sizes.
    let keys: Vec<u64> = (0..n).map(|_| rng.random()).collect();
    let values: Vec<[u8; VALUE_BYTES]> = (0..n)
        .map(|_| {
            let mut v = [0u8; VALUE_BYTES];
            v.copy_from_slice(&rng.random::<u64>().to_le_bytes()[..VALUE_BYTES]);
            v
        })
        .collect();
    (keys, values)
}

fn construct(c: &mut Criterion) {
    let mut group = c.benchmark_group("construct");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    for n in [10_000usize, 100_000, 1_000_000, 10_000_000] {
        let (keys, values) = inputs(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("fuse_map", n), &n, |b, _| {
            b.iter(|| {
                black_box(
                    BinaryFuseMap::<VALUE_BYTES>::try_construct(
                        black_box(&keys),
                        black_box(&values),
                    )
                    .expect("construction succeeds"),
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("foldhash_map", n), &n, |b, _| {
            b.iter(|| black_box(build_hashmap(black_box(&keys), black_box(&values))))
        });
    }
    group.finish();
}

fn get(c: &mut Criterion) {
    let mut group = c.benchmark_group("get");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for n in [10_000usize, 1_000_000, 10_000_000] {
        let (keys, values) = inputs(n);
        let fuse = BinaryFuseMap::<VALUE_BYTES>::try_construct(&keys, &values).unwrap();
        let hash = build_hashmap(&keys, &values);
        group.throughput(Throughput::Elements(keys.len() as u64));
        group.bench_with_input(BenchmarkId::new("fuse_map", n), &n, |b, _| {
            b.iter(|| {
                let mut acc = 0u8;
                for &k in &keys {
                    if let Some(v) = fuse.get(k) {
                        acc ^= v[0];
                    }
                }
                black_box(acc)
            })
        });
        group.bench_with_input(BenchmarkId::new("foldhash_map", n), &n, |b, _| {
            b.iter(|| {
                let mut acc = 0u8;
                for &k in &keys {
                    if let Some(v) = hash.get(&k) {
                        acc ^= v[0];
                    }
                }
                black_box(acc)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, construct, get);
criterion_main!(benches);
