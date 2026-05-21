use std::{
    hash::{DefaultHasher, Hash},
    hint::black_box,
    time::Duration,
};

use consistent_choose_k::{
    ConsistentChooseKFastHasher, ConsistentChooseKHasher, ConsistentHasher,
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
                    let mut iter = ConsistentChooseKFastHasher::new(h, n + k);
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

    targets = throughput_benchmark, append_vs_new_with_k, shrink_n, seg_tree_compare,
);
criterion_main!(benches);
