use std::hash::BuildHasher;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use geo_filters::build_hasher::UnstableDefaultBuildHasher;
use geo_filters::config::VariableConfig;
use geo_filters::diff_count::{
    GeoDiffConfig13, GeoDiffCount, GeoDiffCount13, GeoDiffCount7, GeoDiffCountBuilder,
};
use geo_filters::distinct_count::GeoDistinctCount13;
use geo_filters::evaluation::hll::Hll14;
use geo_filters::Count;

fn criterion_benchmark(c: &mut Criterion) {
    let sizes = [100, 1000, 10000];

    for size in &sizes {
        let mut group = c.benchmark_group(format!("insert:{size}"));

        group.bench_function("geo_diff_count_13", |b| {
            b.iter(|| {
                let mut gc = GeoDiffCount13::default();
                for i in 0..*size {
                    gc.push(i);
                }
            })
        });
        group.bench_function("geo_diff_count_var_13", |b| {
            let c = VariableConfig::<_, u32, UnstableDefaultBuildHasher>::new(13, 7680, 256);
            b.iter(move || {
                let mut gc = GeoDiffCount::new(c.clone());
                for i in 0..*size {
                    gc.push(i);
                }
            })
        });
        group.bench_function("geo_distinct_count_13", |b| {
            b.iter(|| {
                let mut gc = GeoDistinctCount13::default();
                for i in 0..*size {
                    gc.push(i);
                }
            })
        });
        group.bench_function("hll_14", |b| {
            b.iter(|| {
                let mut hll = Hll14::default();
                for i in 0..*size {
                    hll.push(i);
                }
            })
        });
    }

    for size in &sizes {
        let mut group = c.benchmark_group(format!("estimate:{size}"));

        group.bench_function("geo_diff_count_13", |b| {
            b.iter(|| {
                let mut gc = GeoDiffCount13::default();
                for i in 0..*size {
                    gc.push(i);
                    black_box(gc.size());
                }
            })
        });
        group.bench_function("geo_diff_count_var_13", |b| {
            let c = VariableConfig::<_, u32, UnstableDefaultBuildHasher>::new(13, 7680, 256);
            b.iter(move || {
                let mut gc = GeoDiffCount::new(c.clone());
                for i in 0..*size {
                    gc.push(i);
                    black_box(gc.size());
                }
            })
        });
        group.bench_function("geo_distinct_count_13", |b| {
            b.iter(|| {
                let mut gc = GeoDistinctCount13::default();
                for i in 0..*size {
                    gc.push(i);
                    black_box(gc.size());
                }
            })
        });
        group.bench_function("hll_14", |b| {
            b.iter(|| {
                let mut hll = Hll14::default();
                for i in 0..*size {
                    hll.push(i);
                    black_box(hll.size());
                }
            })
        });
    }

    for size in &sizes {
        let mut group = c.benchmark_group(format!("estimate_with:{size}"));
        let size = size / 2; // so that the combined set will be the original size

        group.bench_function("geo_diff_count_13", |b| {
            b.iter(|| {
                let mut gc1 = GeoDiffCount13::default();
                let mut gc2 = GeoDiffCount13::default();
                for i in 0..size {
                    gc1.push(i * 2);
                    gc2.push(i * 2 + 1);
                    black_box(gc1.size_with_sketch(&gc2));
                }
            })
        });
        group.bench_function("geo_diff_count_var_13", |b| {
            let c = VariableConfig::<_, u32, UnstableDefaultBuildHasher>::new(13, 7680, 256);
            b.iter(move || {
                let mut gc1 = GeoDiffCount::new(c.clone());
                let mut gc2 = GeoDiffCount::new(c.clone());
                for i in 0..size {
                    gc1.push(i * 2);
                    gc2.push(i * 2 + 1);
                    black_box(gc1.size_with_sketch(&gc2));
                }
            })
        });
        group.bench_function("geo_distinct_count_13", |b| {
            b.iter(|| {
                let mut gc1 = GeoDistinctCount13::default();
                let mut gc2 = GeoDistinctCount13::default();
                for i in 0..size {
                    gc1.push(i * 2);
                    gc2.push(i * 2 + 1);
                    black_box(gc1.size_with_sketch(&gc2));
                }
            })
        });
    }

    // Compare building a diff filter from a precomputed slice of hashes one by one (`push_hash`)
    // versus via the incremental `GeoDiffCountBuilder` (per-hash, and the batched
    // `extend_by_hashes`). The hashes are precomputed so that only construction cost is measured.
    for size in [1000usize, 10000, 100000, 1000000] {
        let mut group = c.benchmark_group(format!("construct:{size}"));
        let build_hasher = UnstableDefaultBuildHasher::default();
        let hashes: Vec<u64> = (0..size).map(|i| build_hasher.hash_one(i)).collect();

        group.bench_function("geo_diff_count_7_push", |b| {
            b.iter(|| {
                let mut gc = GeoDiffCount7::default();
                for &hash in &hashes {
                    gc.push_hash(hash);
                }
                black_box(&gc);
            })
        });
        group.bench_function("geo_diff_count_13_push", |b| {
            b.iter(|| {
                let mut gc = GeoDiffCount13::default();
                for &hash in &hashes {
                    gc.push_hash(hash);
                }
                black_box(&gc);
            })
        });
        group.bench_function("geo_diff_count_13_builder_extend", |b| {
            b.iter(|| {
                let mut builder = GeoDiffCountBuilder::with_capacity(
                    GeoDiffConfig13::<UnstableDefaultBuildHasher>::default(),
                    0,
                );
                builder.extend_by_hashes(hashes.iter().copied());
                black_box(builder.build());
            })
        });
        group.bench_function("geo_diff_count_13_builder", |b| {
            b.iter(|| {
                let mut builder = GeoDiffCountBuilder::with_capacity(
                    GeoDiffConfig13::<UnstableDefaultBuildHasher>::default(),
                    size,
                );
                for &hash in &hashes {
                    builder.push_hash(hash);
                }
                black_box(builder.build());
            })
        });
        // Reserve nothing so the split starts at 0 and every bucket initially lands in `numbers`,
        // forcing the buffer to fill and compact (lazily flush) repeatedly as the split ramps up.
        // This isolates the cost of the lazy-flush path versus a well-positioned builder.
        group.bench_function("geo_diff_count_13_builder_unreserved", |b| {
            b.iter(|| {
                let mut builder = GeoDiffCountBuilder::with_capacity(
                    GeoDiffConfig13::<UnstableDefaultBuildHasher>::default(),
                    0,
                );
                for &hash in &hashes {
                    builder.push_hash(hash);
                }
                black_box(builder.build());
            })
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
