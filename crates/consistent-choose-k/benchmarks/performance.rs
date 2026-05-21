use std::{
    hash::{DefaultHasher, Hash},
    hint::black_box,
    time::Duration,
};

use consistent_choose_k::{ConsistentChooseKHasher, ConsistentHasher};
use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
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
                    let mut iter = ConsistentChooseKHasher::new(h, n + k);
                    for _ in 0..k {
                        black_box(iter.grow_k());
                    }
                })
            });
        }
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
                .warm_up_time(Duration::from_millis(500))
                .measurement_time(Duration::from_millis(4000))
                .nresamples(1000);

    targets = throughput_benchmark, append_vs_new_with_k,
);
criterion_main!(benches);
