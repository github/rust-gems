use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use hashmap_bench::random_trigram_hashes;

fn bench_hashmap_insert(c: &mut Criterion) {
    let trigrams = random_trigram_hashes(1000);

    // ── Main comparison: insert 1000 trigrams ───────────────────────────
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

    group.bench_function("PrefixHashMap", |b| {
        b.iter_batched(
            || hashmap_bench::prefix_map::PrefixHashMap::with_capacity_and_hasher(
                trigrams.len(),
                hashmap_bench::IdentityBuildHasher::default(),
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

    group.bench_function("SimdPrefixHashMap", |b| {
        b.iter_batched(
            || hashmap_bench::prefix_map_simd::SimdPrefixHashMap::with_capacity_and_hasher(
                trigrams.len(),
                hashmap_bench::IdentityBuildHasher::default(),
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

    // ── Re-insert: insert same keys twice (second pass = all overwrites) ─
    let mut group2 = c.benchmark_group("reinsert_1000_trigrams");

    group2.bench_function("hashbrown+Identity", |b| {
        b.iter_batched(
            || {
                let mut map = hashbrown::HashMap::<u32, usize, hashmap_bench::IdentityBuildHasher>::with_capacity_and_hasher(
                    trigrams.len(),
                    Default::default(),
                );
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            |mut map| {
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i + 1000);
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group2.bench_function("PrefixHashMap", |b| {
        b.iter_batched(
            || {
                let mut map = hashmap_bench::prefix_map::PrefixHashMap::with_capacity_and_hasher(
                    trigrams.len(),
                    hashmap_bench::IdentityBuildHasher::default(),
                );
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            |mut map| {
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i + 1000);
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group2.bench_function("SimdPrefixHashMap", |b| {
        b.iter_batched(
            || {
                let mut map = hashmap_bench::prefix_map_simd::SimdPrefixHashMap::with_capacity_and_hasher(
                    trigrams.len(),
                    hashmap_bench::IdentityBuildHasher::default(),
                );
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            |mut map| {
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i + 1000);
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group2.finish();

    // ── Growth penalty: start small (128), force 3 growths ──────────────
    let mut group3 = c.benchmark_group("grow_from_128_insert_1000_trigrams");

    group3.bench_function("hashbrown+Identity", |b| {
        b.iter_batched(
            || hashbrown::HashMap::<u32, usize, hashmap_bench::IdentityBuildHasher>::with_capacity_and_hasher(
                128,
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

    group3.bench_function("PrefixHashMap", |b| {
        b.iter_batched(
            || hashmap_bench::prefix_map::PrefixHashMap::with_capacity_and_hasher(
                128,
                hashmap_bench::IdentityBuildHasher::default(),
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

    group3.bench_function("SimdPrefixHashMap", |b| {
        b.iter_batched(
            || hashmap_bench::prefix_map_simd::SimdPrefixHashMap::with_capacity_and_hasher(
                128,
                hashmap_bench::IdentityBuildHasher::default(),
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

    group3.finish();
}

criterion_group!(benches, bench_hashmap_insert);
criterion_main!(benches);
