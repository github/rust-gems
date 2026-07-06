use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use geo_filters::config::GeoConfig;
use geo_filters::diff_count::{GeoDiffConfig13, GeoDiffConfig7, GeoDiffCount};
use geo_filters::{Count, Diff};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Repeated 20-bit mask with 3 bits set, applied to the filters before comparison.
const MASK: u64 = 0b0000_0100_0000_1000_0001;
const MASK_SIZE: usize = 20;

/// Number of filters that are compared / sorted.
const N: usize = 1000;
/// Number of items inserted into each filter.
const ITEMS: usize = 1000;

fn random_filters<C: GeoConfig<Diff> + Default>(
    n: usize,
    items: usize,
    seed: u64,
) -> Vec<GeoDiffCount<'static, C>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let mut f = GeoDiffCount::<C>::new(C::default());
            for _ in 0..items {
                f.push_hash(rng.next_u64());
            }
            f
        })
        .collect()
}

fn bench_config<C: GeoConfig<Diff> + Default>(c: &mut Criterion, name: &str) {
    let filters = random_filters::<C>(N, ITEMS, 42);

    let mut group = c.benchmark_group(format!("masked_sort/{name}"));
    // Baseline: sort the filters by comparing them directly with the exact masked comparison.
    group.bench_function("cmp_masked", |b| {
        b.iter(|| {
            let mut idx: Vec<u32> = (0..N as u32).collect();
            idx.sort_unstable_by(|&i, &j| {
                filters[i as usize].cmp_masked(&filters[j as usize], MASK, MASK_SIZE)
            });
            black_box(idx)
        })
    });
    // Sort by building the sort keys first (construction is included in the measurement), then
    // comparing them numerically, falling back to `cmp_masked` only on ties.
    group.bench_function("sort_key", |b| {
        b.iter(|| {
            let keys: Vec<u64> = filters
                .iter()
                .map(|f| f.masked_sort_key(MASK, MASK_SIZE))
                .collect();
            let mut idx: Vec<u32> = (0..N as u32).collect();
            idx.sort_unstable_by(|&i, &j| {
                let (i, j) = (i as usize, j as usize);
                keys[i]
                    .cmp(&keys[j])
                    .then_with(|| filters[i].cmp_masked(&filters[j], MASK, MASK_SIZE))
            });
            black_box(idx)
        })
    });
    group.finish();

    // Key construction on its own, i.e. the extra work the `sort_key` variant pays up front.
    c.benchmark_group(format!("build_keys/{name}"))
        .bench_function("masked_sort_key", |b| {
            b.iter(|| {
                let keys: Vec<u64> = filters
                    .iter()
                    .map(|f| f.masked_sort_key(MASK, MASK_SIZE))
                    .collect();
                black_box(keys)
            })
        });
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_config::<GeoDiffConfig7>(c, "geo_diff_count_7");
    bench_config::<GeoDiffConfig13>(c, "geo_diff_count_13");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
