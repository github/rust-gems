use std::{
    hash::{DefaultHasher, Hash, Hasher},
    hint::black_box,
    time::Duration,
};

use consistent_hashing::{ConsistentChooseKHasher, ConsistentHasher};
use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use rand::{rng, Rng};

fn throughput_benchmark(c: &mut Criterion) {
    let keys: Vec<u64> = rng().random_iter().take(1000).collect();

    let mut group = c.benchmark_group(format!("choose"));
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for n in [1usize, 10, 100, 1000, 10000] {
        group.throughput(Throughput::Elements(keys.len() as u64));
        group.bench_with_input(BenchmarkId::new(format!("1"), n), &n, |b, n| {
            b.iter_batched(
                || &keys,
                |keys| {
                    for key in keys {
                        let mut h = DefaultHasher::default();
                        key.hash(&mut h);
                        black_box(ConsistentHasher::new(h).prev(*n + 1));
                    }
                },
                criterion::BatchSize::SmallInput,
            )
        });
        for k in [1, 2, 3, 10, 100] {
            group.bench_with_input(BenchmarkId::new(format!("k_{k}"), n), &n, |b, n| {
                b.iter_batched(
                    || &keys,
                    |keys| {
                        let mut res = Vec::with_capacity(k);
                        for key in keys {
                            let mut h = DefaultHasher::default();
                            key.hash(&mut h);
                            black_box(ConsistentChooseKHasher::new(h, k).prev_with_vec(*n + k, &mut res));
                        }
                    },
                    criterion::BatchSize::SmallInput,
                )
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

    targets = throughput_benchmark,
);
criterion_main!(benches);
