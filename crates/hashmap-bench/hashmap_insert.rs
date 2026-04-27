use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use hashmap_bench::random_trigram_hashes;

fn bench_hashmap_insert(c: &mut Criterion) {
    let trigrams = random_trigram_hashes(1000);

    let mut group = c.benchmark_group("hashmap_insert_1000_trigrams");

    group.bench_function("std::HashMap", |b| {
        b.iter_batched(
            || std::collections::HashMap::with_capacity(trigrams.len()),
            |mut map| {
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("hashbrown::HashMap", |b| {
        b.iter_batched(
            || hashbrown::HashMap::with_capacity(trigrams.len()),
            |mut map| {
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("FxHashMap", |b| {
        b.iter_batched(
            || rustc_hash::FxHashMap::with_capacity_and_hasher(trigrams.len(), Default::default()),
            |mut map| {
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("AHashMap", |b| {
        b.iter_batched(
            || ahash::AHashMap::with_capacity(trigrams.len()),
            |mut map| {
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("FoldHashMap", |b| {
        b.iter_batched(
            || hashbrown::HashMap::<u32, usize, foldhash::fast::FixedState>::with_capacity_and_hasher(
                trigrams.len(),
                foldhash::fast::FixedState::default(),
            ),
            |mut map| {
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("PrefixHashMap", |b| {
        b.iter_batched(
            || hashmap_bench::prefix_map::PrefixHashMap::with_capacity(trigrams.len()),
            |mut map| {
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("SimdPrefixHashMap", |b| {
        b.iter_batched(
            || hashmap_bench::prefix_map_simd::SimdPrefixHashMap::with_capacity(trigrams.len()),
            |mut map| {
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("NoHintScalar", |b| {
        b.iter_batched(
            || hashmap_bench::prefix_map::NoHintScalarPrefixHashMap::with_capacity(trigrams.len()),
            |mut map| {
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("NoHintSimd", |b| {
        b.iter_batched(
            || hashmap_bench::prefix_map_simd::NoHintPrefixHashMap::with_capacity(trigrams.len()),
            |mut map| {
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("GxHashMap", |b| {
        b.iter_batched(
            || gxhash::HashMap::with_capacity_and_hasher(trigrams.len(), Default::default()),
            |mut map| {
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("std::HashMap+FNV", |b| {
        b.iter_batched(
            || std::collections::HashMap::with_capacity_and_hasher(trigrams.len(), fnv::FnvBuildHasher::default()),
            |mut map| {
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("hashbrown+Identity", |b| {
        b.iter_batched(
            || hashbrown::HashMap::<u32, usize, hashmap_bench::IdentityBuildHasher>::with_capacity_and_hasher(
                trigrams.len(),
                Default::default(),
            ),
            |mut map| {
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_hashmap_insert);
criterion_main!(benches);
