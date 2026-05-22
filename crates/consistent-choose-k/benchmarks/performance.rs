use std::{
    hash::{DefaultHasher, Hash},
    hint::black_box,
    time::Duration,
};

use consistent_choose_k::{
    ConsistentChooseKFastGrowHasher, ConsistentChooseKFastHasher, ConsistentChooseKHasher,
    ConsistentHasher,
    __bench_internals::{CompactMinSegTree, MinSegTree},
};
use criterion::{
    criterion_group, criterion_main, AxisScale, BatchSize, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use rand::{rng, RngExt};

fn throughput_benchmark(c: &mut Criterion) {
    let keys: Vec<u64> = rng().random_iter().take(1000).collect();

    let mut group = c.benchmark_group("choose");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for n in [1usize, 10, 100, 1000, 10000] {
        group.throughput(Throughput::Elements(keys.len() as u64));
        group.bench_with_input(BenchmarkId::new("1", n), &n, |b, n| {
            b.iter(|| {
                for key in &keys {
                    let mut h = DefaultHasher::default();
                    key.hash(&mut h);
                    black_box(ConsistentHasher::new(h).prev(*n + 1));
                }
            })
        });
        for k in [1, 2, 3, 10, 100] {
            group.throughput(Throughput::Elements((keys.len() * k) as u64));
            group.bench_with_input(BenchmarkId::new(format!("k_{k}"), n), &n, |b, n| {
                b.iter(|| {
                    for key in &keys {
                        let mut h = DefaultHasher::default();
                        key.hash(&mut h);
                        black_box(ConsistentChooseKHasher::new_with_k(h, *n + k, k));
                    }
                })
            });
        }
    }
    group.finish();
}

fn append_vs_new_with_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("append_vs_new_with_k");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for n in [10usize, 100, 1000, 10000] {
        for k in [2, 3, 10, 100] {
            group.bench_function(BenchmarkId::new(format!("new_with_k/k_{k}"), n), |b| {
                b.iter(|| {
                    let h = DefaultHasher::default();
                    black_box(ConsistentChooseKHasher::new_with_k(h, n + k, k));
                })
            });
            group.bench_function(BenchmarkId::new(format!("append/k_{k}"), n), |b| {
                b.iter(|| {
                    let h = DefaultHasher::default();
                    let mut iter = ConsistentChooseKHasher::new_with_capacity(h, n + k, k);
                    for _ in 0..k {
                        black_box(iter.grow_k());
                    }
                })
            });
            group.bench_function(BenchmarkId::new(format!("fast_new_with_k/k_{k}"), n), |b| {
                b.iter(|| {
                    let h = DefaultHasher::default();
                    black_box(ConsistentChooseKFastHasher::new_with_k(h, n + k, k));
                })
            });
            group.bench_function(BenchmarkId::new(format!("fast_append/k_{k}"), n), |b| {
                b.iter(|| {
                    let h = DefaultHasher::default();
                    let mut iter = ConsistentChooseKFastHasher::new_with_capacity(h, n + k, k);
                    for _ in 0..k {
                        black_box(iter.grow_k());
                    }
                })
            });
        }
    }
    group.finish();
}

fn shrink_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("shrink_n");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for n in [100usize, 1000, 10000, 100000] {
        for k in [2, 3, 10, 100] {
            group.throughput(Throughput::Elements((n * k) as u64));
            group.bench_function(BenchmarkId::new(format!("standard/k_{k}"), n), |b| {
                b.iter_batched(
                    || {
                        let h = DefaultHasher::default();
                        ConsistentChooseKHasher::new_with_k(h, n + k, k)
                    },
                    |mut iter| {
                        while iter.samples().last().copied().expect("k must be nonzero") > k {
                            black_box(iter.shrink_n());
                        }
                        black_box(iter);
                    },
                    BatchSize::SmallInput,
                )
            });
            group.bench_function(BenchmarkId::new(format!("fast/k_{k}"), n), |b| {
                b.iter_batched(
                    || {
                        let h = DefaultHasher::default();
                        ConsistentChooseKFastHasher::new_with_k(h, n + k, k)
                    },
                    |mut iter| {
                        while iter.samples().last().copied().expect("k must be nonzero") > k {
                            black_box(iter.shrink_n());
                        }
                        black_box(iter);
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

/// Reservoir-sampling comparison for `ConsistentChooseKFastGrowHasher`.
///
/// All variants ingest a stream of `n` items and maintain a `k`-sized
/// sample throughout. The point is to see whether fast-grow's `O(log k)`
/// per-displacement work is competitive against the standard reservoir
/// sampling algorithms:
///
/// * `fast_grow` — call `grow_n()` until `n` events have been processed.
/// * `reservoir_r` — Algorithm R (Vitter, 1985). Visits every item, does
///   one PRNG step + one comparison per item. `O(n)` total.
/// * `reservoir_l` — Algorithm L (Li, 1994). Skip-optimized: only
///   `O(k * log(n/k))` items are inspected, so it should be the closest
///   competitor to incremental hash-based grow.
///
/// All three produce a `k`-sample of `0..n`. Output distributions differ
/// (deterministic hash vs. uniform random) but the throughput question is
/// fair: per-item work cost as the stream grows.
fn grow_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("grow_n");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for k in [100usize, 1000] {
        for n in [10_000usize, 100_000, 1_000_000] {
            if n <= k {
                continue;
            }
            group.throughput(Throughput::Elements(n as u64));

            group.bench_function(BenchmarkId::new(format!("fast_grow/k_{k}"), n), |b| {
                b.iter(|| {
                    let h = DefaultHasher::default();
                    let mut iter = ConsistentChooseKFastGrowHasher::new(h, k);
                    while iter.n() < n {
                        black_box(iter.grow_n());
                    }
                    black_box(iter);
                })
            });

            group.bench_function(BenchmarkId::new(format!("reservoir_r/k_{k}"), n), |b| {
                b.iter(|| {
                    black_box(reservoir_r(n, k, 0x9E37_79B9_7F4A_7C15));
                })
            });

            group.bench_function(BenchmarkId::new(format!("reservoir_l/k_{k}"), n), |b| {
                b.iter(|| {
                    black_box(reservoir_l(n, k, 0x9E37_79B9_7F4A_7C15));
                })
            });
        }
    }
    group.finish();
}

/// SplitMix64 step. Cheap PRNG suitable for benchmark-grade randomness.
#[inline(always)]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline(always)]
fn next_f64(state: &mut u64) -> f64 {
    // Top 53 bits → uniform in `[0, 1)`.
    (splitmix64(state) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Algorithm R (Vitter 1985): linear-scan reservoir sampling.
fn reservoir_r(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut samples: Vec<usize> = (0..k).collect();
    let mut state = seed;
    for i in k..n {
        // Uniform integer in `0..=i` via the 64×64→128 multiply trick.
        let r = splitmix64(&mut state);
        let j = ((r as u128 * (i as u128 + 1)) >> 64) as usize;
        if j < k {
            samples[j] = i;
        }
    }
    samples
}

/// Algorithm L (Li 1994): skip-optimized reservoir sampling. Total work
/// is `O(k + k * log(n / k))` regardless of `n`.
fn reservoir_l(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut samples: Vec<usize> = (0..k).collect();
    let mut state = seed;
    let inv_k = 1.0 / k as f64;
    // `w` shrinks geometrically; each step's skip distance is drawn from a
    // geometric distribution parameterised by `w`.
    let mut w = (next_f64(&mut state).ln() * inv_k).exp();
    let mut i = k - 1;
    loop {
        // Skip `floor(ln(U) / ln(1 - w))` items, then visit the next one.
        let log_u = next_f64(&mut state).ln();
        let log_1mw = (1.0 - w).ln();
        let skip = (log_u / log_1mw) as usize + 1;
        i = match i.checked_add(skip) {
            Some(v) => v,
            None => break,
        };
        if i >= n {
            break;
        }
        // Replace a uniformly chosen slot with the freshly visited item.
        let r = splitmix64(&mut state);
        let j = ((r as u128 * k as u128) >> 64) as usize;
        samples[j] = i;
        w *= (next_f64(&mut state).ln() * inv_k).exp();
    }
    samples
}

/// Workload that mimics the segment-tree usage pattern inside `shrink_n`:
/// repeatedly find the right-most non-positive leaf, set it to a value,
/// and shift a suffix.
///
/// The op sequence is deterministic so the two trees process identical work.
fn seg_tree_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("seg_tree");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &size in &[64usize, 256, 1024, 4096, 16384] {
        let init: Vec<i64> = (0..size as i64)
            .map(|i| ((i.wrapping_mul(2654435761)) & 0xff) - 64)
            .collect();
        let ops: Vec<(usize, i64, usize, i64)> = (0..size)
            .map(|i| {
                let set_idx = (i * 5 + 3) % size;
                let set_val = ((i as i64) % 31) - 15;
                let suffix_lo = (i * 7) % size;
                let suffix_delta = if i & 1 == 0 { 1 } else { -1 };
                (set_idx, set_val, suffix_lo, suffix_delta)
            })
            .collect();

        group.throughput(Throughput::Elements(ops.len() as u64));

        group.bench_with_input(BenchmarkId::new("full", size), &size, |b, _| {
            b.iter_batched(
                || MinSegTree::new(&init, i64::MAX / 4),
                |mut t| {
                    for &(i, v, lo, d) in &ops {
                        black_box(t.rightmost_le_zero());
                        t.set(i, v);
                        t.suffix_add(lo, d);
                    }
                    black_box(t);
                },
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("compact", size), &size, |b, _| {
            b.iter_batched(
                || CompactMinSegTree::new(&init, i64::MAX / 4),
                |mut t| {
                    for &(i, v, lo, d) in &ops {
                        black_box(t.rightmost_le_zero());
                        t.set(i, v);
                        t.suffix_add(lo, d);
                    }
                    black_box(t);
                },
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("full_new", size), &size, |b, _| {
            b.iter(|| {
                black_box(MinSegTree::new(&init, i64::MAX / 4));
            })
        });
        group.bench_with_input(BenchmarkId::new("compact_new", size), &size, |b, _| {
            b.iter(|| {
                black_box(CompactMinSegTree::new(&init, i64::MAX / 4));
            })
        });
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
                .warm_up_time(Duration::from_millis(500))
                .measurement_time(Duration::from_millis(4000))
                .nresamples(1000);

    targets = throughput_benchmark, append_vs_new_with_k, shrink_n, grow_n, seg_tree_compare,
);
criterion_main!(benches);
