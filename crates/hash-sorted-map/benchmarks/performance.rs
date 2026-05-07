use std::hash::BuildHasher;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use hash_sorted_map::HashSortedMap;
use hash_sorted_map_benchmarks::{random_trigram_hashes, IdentityBuildHasher};

fn trigrams() -> Vec<u32> {
    random_trigram_hashes(1000)
}

fn bench_insert(c: &mut Criterion) {
    let trigrams = trigrams();
    let mut group = c.benchmark_group("presized_insert_1000_trigrams");

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

    group.bench_function("std::HashMap+FNV", |b| {
        b.iter_batched(
            || {
                std::collections::HashMap::with_capacity_and_hasher(
                    trigrams.len(),
                    fnv::FnvBuildHasher::default(),
                )
            },
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
            || {
                hashbrown::HashMap::<u32, usize, IdentityBuildHasher>::with_capacity_and_hasher(
                    trigrams.len(),
                    Default::default(),
                )
            },
            |mut map| {
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("HashSortedMap", |b| {
        b.iter_batched(
            || {
                HashSortedMap::with_capacity_and_hasher(
                    trigrams.len(),
                    IdentityBuildHasher::default(),
                )
            },
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

fn bench_reinsert(c: &mut Criterion) {
    let trigrams = trigrams();
    let mut group = c.benchmark_group("reinsert_1000_trigrams");

    group.bench_function("hashbrown+Identity", |b| {
        b.iter_batched(
            || {
                let mut map =
                    hashbrown::HashMap::<u32, usize, IdentityBuildHasher>::with_capacity_and_hasher(
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

    group.bench_function("HashSortedMap", |b| {
        b.iter_batched(
            || {
                let mut map = HashSortedMap::with_capacity_and_hasher(
                    trigrams.len(),
                    IdentityBuildHasher::default(),
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

    group.finish();
}

fn bench_grow(c: &mut Criterion) {
    let trigrams = trigrams();
    let mut group = c.benchmark_group("grow_from_128_insert_1000_trigrams");

    group.bench_function("hashbrown+Identity", |b| {
        b.iter_batched(
            || {
                hashbrown::HashMap::<u32, usize, IdentityBuildHasher>::with_capacity_and_hasher(
                    128,
                    Default::default(),
                )
            },
            |mut map| {
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("HashSortedMap", |b| {
        b.iter_batched(
            || HashSortedMap::with_capacity_and_hasher(128, IdentityBuildHasher::default()),
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

fn bench_count(c: &mut Criterion) {
    let trigrams = trigrams();
    let mut counted_trigrams = Vec::with_capacity(trigrams.len() * 4);
    for _ in 0..4 {
        counted_trigrams.extend_from_slice(&trigrams);
    }

    let mut group = c.benchmark_group("count_4000_trigrams_get_or_default");

    group.bench_function("hashbrown+Identity entry()", |b| {
        b.iter_batched(
            || {
                hashbrown::HashMap::<u32, u32, IdentityBuildHasher>::with_capacity_and_hasher(
                    trigrams.len(),
                    Default::default(),
                )
            },
            |mut map| {
                for &key in &counted_trigrams {
                    *map.entry(key).or_insert(0) += 1;
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("HashSortedMap get_or_default", |b| {
        b.iter_batched(
            || {
                HashSortedMap::<u32, u32, _>::with_capacity_and_hasher(
                    trigrams.len(),
                    IdentityBuildHasher::default(),
                )
            },
            |mut map| {
                for &key in &counted_trigrams {
                    *map.get_or_default(key) += 1;
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("HashSortedMap entry().or_default()", |b| {
        b.iter_batched(
            || {
                HashSortedMap::<u32, u32, _>::with_capacity_and_hasher(
                    trigrams.len(),
                    IdentityBuildHasher::default(),
                )
            },
            |mut map| {
                for &key in &counted_trigrams {
                    *map.entry(key).or_default() += 1;
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_iter(c: &mut Criterion) {
    let trigrams = trigrams();

    let mut group = c.benchmark_group("iter_1000_trigrams");

    group.bench_function("hashbrown+Identity iter()", |b| {
        b.iter_batched(
            || {
                let mut map =
                    hashbrown::HashMap::<u32, usize, IdentityBuildHasher>::with_capacity_and_hasher(
                        trigrams.len(),
                        Default::default(),
                    );
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            |map| {
                let mut sum = 0usize;
                for (&k, &v) in &map {
                    sum = sum.wrapping_add(v).wrapping_add(k as usize);
                }
                sum
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("HashSortedMap iter()", |b| {
        b.iter_batched(
            || {
                let mut map = HashSortedMap::with_capacity_and_hasher(
                    trigrams.len(),
                    IdentityBuildHasher::default(),
                );
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            |map| {
                let mut sum = 0usize;
                for (&k, &v) in &map {
                    sum = sum.wrapping_add(v).wrapping_add(k as usize);
                }
                sum
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("hashbrown+Identity into_iter()", |b| {
        b.iter_batched(
            || {
                let mut map =
                    hashbrown::HashMap::<u32, usize, IdentityBuildHasher>::with_capacity_and_hasher(
                        trigrams.len(),
                        Default::default(),
                    );
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            |map| {
                let mut sum = 0usize;
                for (k, v) in map {
                    sum = sum.wrapping_add(v).wrapping_add(k as usize);
                }
                sum
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("HashSortedMap into_iter()", |b| {
        b.iter_batched(
            || {
                let mut map = HashSortedMap::with_capacity_and_hasher(
                    trigrams.len(),
                    IdentityBuildHasher::default(),
                );
                for (i, &key) in trigrams.iter().enumerate() {
                    map.insert(key, i);
                }
                map
            },
            |map| {
                let mut sum = 0usize;
                for (k, v) in map {
                    sum = sum.wrapping_add(v).wrapping_add(k as usize);
                }
                sum
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_sort(c: &mut Criterion) {
    let keys = random_trigram_hashes(100_000);
    let hasher = IdentityBuildHasher::default();
    let mut group = c.benchmark_group("sort_100000_trigrams");

    group.bench_function("Vec::sort_unstable", |b| {
        b.iter(|| {
            let mut vec: Vec<_> = keys
                .iter()
                .enumerate()
                .map(|(i, &key)| (key, i))
                .collect();
            vec.sort_unstable_by_key(|&(key, _)| hasher.hash_one(key));
            vec
        });
    });

    group.bench_function("HashSortedMap sort_by_hash", |b| {
        b.iter(|| {
            let mut map = HashSortedMap::with_capacity_and_hasher(
                keys.len(),
                IdentityBuildHasher::default(),
            );
            for (i, &key) in keys.iter().enumerate() {
                map.insert(key, i);
            }
            map.sort_by_hash()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_insert,
    bench_reinsert,
    bench_grow,
    bench_count,
    bench_iter,
    bench_sort
);
criterion_main!(benches);
