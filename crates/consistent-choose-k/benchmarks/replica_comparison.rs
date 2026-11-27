//! Paired workloads only in the existing iterator's supported domain.
//! See ../docs/virtual-permutation-performance.md for methodology and results.

use std::{
    hash::{DefaultHasher, Hash, Hasher},
    hint::black_box,
    time::Duration,
};

use consistent_choose_k::{ConsistentPermutation, VirtualPermutation};
use criterion::{
    criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use rand::{rngs::StdRng, RngExt, SeedableRng};

const WORKLOAD_SEED: u64 = 0x7065_726d_7574_6531;
const KEY_COUNT: usize = 128;
const NODES: &[u32] = &[
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    9,
    14,
    15,
    16,
    17,
    30,
    31,
    32,
    33,
    254,
    255,
    256,
    257,
    1000,
    1022,
    1023,
    1024,
    1025,
    65534,
    65535,
    65536,
    65537,
    1_000_000,
    (1 << 30) - 2,
    (1 << 30) - 1,
    1 << 30,
];

fn hash_key(key: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

fn keys() -> Vec<u64> {
    StdRng::seed_from_u64(WORKLOAD_SEED)
        .random_iter()
        .take(KEY_COUNT)
        .collect()
}

fn counts(n: u32) -> Vec<usize> {
    let mut counts = vec![1, 2, 3, 8, 16];
    if n <= 1024 {
        counts.extend([n as usize / 4, n as usize]);
    }
    counts.retain(|&k| k > 0 && k <= n as usize);
    counts.sort_unstable();
    counts.dedup();
    counts
}

fn consume(iter: impl Iterator<Item = impl Into<u64>>, k: usize) {
    let sum = iter
        .take(k)
        .fold(0u64, |sum, node| sum.wrapping_add(node.into()));
    black_box(sum);
}

fn end_to_end(c: &mut Criterion) {
    let keys = keys();
    let seeds: Vec<_> = keys.iter().copied().map(hash_key).collect();
    for mode in ["fresh", "seeded"] {
        let mut group = c.benchmark_group(format!("sentinel_replicas/{mode}"));
        // Both execute exactly one complete query per key, including
        // construction, streaming consumption and state destruction.
        group.throughput(Throughput::Elements(KEY_COUNT as u64));
        group.sampling_mode(SamplingMode::Flat);
        for &n in NODES {
            for k in counts(n) {
                let input = if mode == "fresh" { &keys } else { &seeds };
                group.bench_function(BenchmarkId::new("layered", format!("n{n}_k{k}")), |b| {
                    b.iter(|| {
                        for &key in black_box(input) {
                            let seed = if mode == "fresh" { hash_key(key) } else { key };
                            consume(ConsistentPermutation::new(black_box(n), seed), black_box(k));
                        }
                    })
                });
                group.bench_function(BenchmarkId::new("sentinel", format!("n{n}_k{k}")), |b| {
                    b.iter(|| {
                        for &key in black_box(input) {
                            let seed = if mode == "fresh" { hash_key(key) } else { key };
                            consume(
                                VirtualPermutation::new(u64::from(black_box(n)), seed),
                                black_box(k),
                            );
                        }
                    })
                });
            }
        }
        group.finish();
    }
}

fn cost_components(c: &mut Criterion) {
    let keys = keys();
    let seeds: Vec<_> = keys.iter().copied().map(hash_key).collect();
    let mut setup = c.benchmark_group("sentinel_replicas/setup");
    setup.throughput(Throughput::Elements(KEY_COUNT as u64));
    setup.sampling_mode(SamplingMode::Flat);
    setup.bench_function("hash_u64", |b| {
        b.iter(|| {
            for &key in black_box(&keys) {
                black_box(hash_key(key));
            }
        })
    });
    for &n in &[17, 257, 1000, 65537, 1 << 30] {
        setup.bench_function(BenchmarkId::new("layered", n), |b| {
            b.iter(|| {
                for &seed in black_box(&seeds) {
                    black_box(ConsistentPermutation::new(black_box(n), seed));
                }
            })
        });
        setup.bench_function(BenchmarkId::new("sentinel", n), |b| {
            b.iter(|| {
                for &seed in black_box(&seeds) {
                    black_box(VirtualPermutation::new(u64::from(black_box(n)), seed));
                }
            })
        });
    }
    setup.finish();

    for mode in ["stream_only", "collect", "rank_replay"] {
        let mut group = c.benchmark_group(format!("sentinel_replicas/{mode}"));
        group.throughput(Throughput::Elements(KEY_COUNT as u64));
        group.sampling_mode(SamplingMode::Flat);
        for &n in &[17, 257, 1000, 65537, 1 << 30] {
            for k in counts(n) {
                group.bench_function(BenchmarkId::new("layered", format!("n{n}_k{k}")), |b| {
                    match mode {
                        "stream_only" => b.iter_batched_ref(
                            || {
                                seeds
                                    .iter()
                                    .map(|&seed| ConsistentPermutation::new(n, seed))
                                    .collect::<Vec<_>>()
                            },
                            |iterators| {
                                for iter in black_box(iterators) {
                                    consume(iter, black_box(k));
                                }
                            },
                            BatchSize::SmallInput,
                        ),
                        _ => b.iter(|| {
                            for &seed in black_box(&seeds) {
                                let mut iter = ConsistentPermutation::new(black_box(n), seed);
                                if mode == "collect" {
                                    // Same output width and allocation policy for both.
                                    let mut out = Vec::with_capacity(black_box(k));
                                    out.extend(iter.take(k).map(u64::from));
                                    black_box(out);
                                } else {
                                    // Both methods replay the prefix to answer a rank query.
                                    black_box(iter.nth(black_box(k - 1)));
                                }
                            }
                        }),
                    }
                });
                group.bench_function(BenchmarkId::new("sentinel", format!("n{n}_k{k}")), |b| {
                    match mode {
                        "stream_only" => b.iter_batched_ref(
                            || {
                                seeds
                                    .iter()
                                    .map(|&seed| VirtualPermutation::new(u64::from(n), seed))
                                    .collect::<Vec<_>>()
                            },
                            |iterators| {
                                for iter in black_box(iterators) {
                                    consume(iter, black_box(k));
                                }
                            },
                            BatchSize::SmallInput,
                        ),
                        _ => b.iter(|| {
                            for &seed in black_box(&seeds) {
                                let mut iter =
                                    VirtualPermutation::new(u64::from(black_box(n)), seed);
                                if mode == "collect" {
                                    let mut out = Vec::with_capacity(black_box(k));
                                    out.extend(iter.take(k));
                                    black_box(out);
                                } else {
                                    black_box(iter.nth(black_box(k - 1)));
                                }
                            }
                        }),
                    }
                });
            }
        }
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(300))
        .nresamples(1000)
        .without_plots();
    targets = end_to_end, cost_components
}
criterion_main!(benches);
