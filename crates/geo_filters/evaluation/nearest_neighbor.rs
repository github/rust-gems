//! Benchmarks the different ways to measure (dis)similarity of `GeoDiffCount` filters, in a
//! nearest-neighbor setting where a `query` is compared against many candidates.
//!
//! Two families are compared:
//!
//! * The calibrated *size estimate* ([`Count::size`] / [`Count::size_with_sketch`]), which scans
//!   only a bounded window and upscales.
//! * The exact *bit count* via [`GeoDiffMetric`], a simple, uncalibrated metric based on the number
//!   of differing one-bits (Hamming distance). It caches each filter's one-bit count (`size`) and
//!   exposes `symmetric_diff_size`, the exact distance, abandoned once a given bound is reached.
//!
//! Both are measured for a `far` candidate (disjoint, rejected by the bound) and a `kept` candidate
//! (a near-duplicate, roughly at the bound and scanned in full), to contrast pruning vs. keeping.
//!
//! Groups:
//! * `size:*` - single filter: `estimate` (calibrated) vs. `metric` (exact one-bit count).
//! * `diff:*` - pairwise: `estimate`/`symmetric_diff_size`/`symmetric_diff_size_capped` for the
//!   `far` and `kept` candidates.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use geo_filters::config::GeoConfig;
use geo_filters::diff_count::{
    GeoDiffConfig10, GeoDiffConfig13, GeoDiffConfig7, GeoDiffCount, GeoDiffMetric, OnesMetric,
};
use geo_filters::{Count, Diff, Metric, MetricSpace};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

fn build<C: GeoConfig<Diff> + Default>(
    rng: &mut ChaCha12Rng,
    n: usize,
) -> GeoDiffCount<'static, C> {
    let mut f = GeoDiffCount::<C>::default();
    for _ in 0..n {
        f.push_hash(rng.next_u64());
    }
    f
}

/// Returns a near-duplicate of `base` that differs from it by roughly `diff` items.
fn near<C: GeoConfig<Diff> + Default>(
    rng: &mut ChaCha12Rng,
    base: &GeoDiffCount<'static, C>,
    diff: usize,
) -> GeoDiffCount<'static, C> {
    let mut f = base.clone();
    for _ in 0..diff {
        f.push_hash(rng.next_u64());
    }
    f
}

fn bench_config<C: GeoConfig<Diff> + Default>(c: &mut Criterion, name: &str, items: usize) {
    let mut rng = ChaCha12Rng::seed_from_u64(42);

    // Each filter is owned by its metric, which caches the one-bit count, as a real index would.
    let query_m = GeoDiffMetric::new(build::<C>(&mut rng, items));
    // A far-away candidate (disjoint set), and a near neighbor differing by ~10% of the base filter
    // whose distance provides the prior bound.
    let far_m = GeoDiffMetric::new(build::<C>(&mut rng, items));
    let bound_m = GeoDiffMetric::new(near(&mut rng, query_m.filter(), (items / 10).max(1)));
    // A kept candidate: a closer ~5% near-duplicate, i.e. below the bound and hence scanned in full.
    let kept_m = GeoDiffMetric::new(near(&mut rng, query_m.filter(), (items / 20).max(1)));
    let ones_bound = query_m.symmetric_diff_size(&bound_m, OnesMetric::infinite());

    // Single-filter size metric: the estimate vs. the exact bit count computed when constructing a
    // `GeoDiffMetric` (the filter is cloned in the untimed setup so only the count is measured).
    let mut group = c.benchmark_group(format!("size:{name}/{items}"));
    group.bench_function("estimate", |b| {
        b.iter(|| black_box(query_m.filter().size()))
    });
    group.bench_function("metric", |b| {
        b.iter_batched(
            || query_m.filter().clone(),
            |f| black_box(GeoDiffMetric::new(f).size()),
            BatchSize::SmallInput,
        )
    });
    group.finish();

    // Diff between two filters, for a `far` candidate (disjoint, pruned by the bound) and a `kept`
    // candidate (a ~10% near-duplicate, roughly at the bound and scanned in full):
    // * `estimate`: calibrated size of the symmetric difference (`Count::size_with_sketch`);
    // * `symmetric_diff_size`: exact one-bit distance (`MetricSpace::symmetric_diff_size` with an
    //   infinite bound), scanning both filters in full;
    // * `symmetric_diff_size_capped`: same, but abandons once `ones_bound` differing bits are reached.
    let mut group = c.benchmark_group(format!("diff:{name}/{items}"));
    group.bench_function("estimate_far", |b| {
        b.iter(|| black_box(query_m.filter().size_with_sketch(black_box(far_m.filter()))))
    });
    group.bench_function("symmetric_diff_size_far", |b| {
        b.iter(|| black_box(query_m.symmetric_diff_size(black_box(&far_m), OnesMetric::infinite())))
    });
    group.bench_function("symmetric_diff_size_capped_far", |b| {
        b.iter(|| black_box(query_m.symmetric_diff_size(black_box(&far_m), ones_bound)))
    });
    group.bench_function("estimate_kept", |b| {
        b.iter(|| {
            black_box(
                query_m
                    .filter()
                    .size_with_sketch(black_box(kept_m.filter())),
            )
        })
    });
    group.bench_function("symmetric_diff_size_kept", |b| {
        b.iter(|| {
            black_box(query_m.symmetric_diff_size(black_box(&kept_m), OnesMetric::infinite()))
        })
    });
    group.bench_function("symmetric_diff_size_capped_kept", |b| {
        b.iter(|| black_box(query_m.symmetric_diff_size(black_box(&kept_m), ones_bound)))
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    for items in [1_000, 10_000, 100_000, 1_000_000] {
        bench_config::<GeoDiffConfig7>(c, "config7", items);
        bench_config::<GeoDiffConfig10>(c, "config10", items);
        bench_config::<GeoDiffConfig13>(c, "config13", items);
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
