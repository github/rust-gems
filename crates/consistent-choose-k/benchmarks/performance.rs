use std::{
    hash::{DefaultHasher, Hash},
    hint::black_box,
    time::Duration,
};

use consistent_choose_k::{
    ConsistentChooseKHasher, ConsistentHasher, ConsistentPermutation, ConsistentReservoir,
};
use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
    Throughput,
};
use rand::{rng, rngs::StdRng, RngExt, SeedableRng};

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

fn grow_k_vs_permutation(c: &mut Criterion) {
    // Compare three ways to obtain `k` distinct samples out of `0..n`:
    //   * `ConsistentChooseKHasher::new_with_k` (pre-build the full set);
    //   * `ConsistentChooseKHasher::new` + `grow_k` k times (incremental);
    //   * `ConsistentPermutation` (per-layer Feistel permutation, take k).
    //
    // All three are driven from a per-key seed so the cost of building the
    // underlying permutation/hash state is included in each iteration.
    let keys: Vec<u64> = rng().random_iter().take(100).collect();

    let mut group = c.benchmark_group("grow_k_vs_permutation");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for n in [100usize, 1_000, 10_000, 100_000] {
        for k in [2usize, 10, 100, 1_000] {
            if k > n {
                continue;
            }
            group.throughput(Throughput::Elements((keys.len() * k) as u64));

            group.bench_with_input(BenchmarkId::new(format!("new_with_k/k_{k}"), n), &n, |b, n| {
                b.iter(|| {
                    for key in &keys {
                        let mut h = DefaultHasher::default();
                        key.hash(&mut h);
                        black_box(ConsistentChooseKHasher::new_with_k(h, *n, k));
                    }
                })
            });

            group.bench_with_input(BenchmarkId::new(format!("grow_k/k_{k}"), n), &n, |b, n| {
                b.iter(|| {
                    for key in &keys {
                        let mut h = DefaultHasher::default();
                        key.hash(&mut h);
                        let mut iter = ConsistentChooseKHasher::new(h, *n);
                        for _ in 0..k {
                            black_box(iter.grow_k());
                        }
                    }
                })
            });

            group.bench_with_input(BenchmarkId::new(format!("permutation/k_{k}"), n), &n, |b, n| {
                b.iter(|| {
                    for key in &keys {
                        let mut iter = ConsistentPermutation::new(*n as u32, *key);
                        for _ in 0..k {
                            black_box(iter.next());
                        }
                    }
                })
            });
        }
    }
    group.finish();
}

// A standard reservoir sampling (Algorithm R) implementation.
// It initializes a reservoir of size k, and then scans from k to n element-by-element,
// deciding for each step whether to admit that element.
fn standard_reservoir_r(k: u32, n: u32, seed: u64) -> Vec<u32> {
    let mut reservoir: Vec<u32> = (0..k).collect();
    if n <= k {
        return reservoir;
    }
    let mut rng = StdRng::seed_from_u64(seed);

    for i in k..n {
        let j = rng.random_range(0..=i);
        if j < k {
            reservoir[j as usize] = i;
        }
    }
    reservoir
}

// A standard skip-based reservoir sampling (Algorithm L / Vitter) implementation.
fn standard_reservoir_l(k: u32, n: u32, seed: u64) -> Vec<u32> {
    let mut reservoir: Vec<u32> = (0..k).collect();
    if n <= k {
        return reservoir;
    }
    let mut rng = StdRng::seed_from_u64(seed);

    let mut w = (rng.random::<f64>().ln() / (k as f64)).exp();
    let mut i = k;
    while i < n {
        let u: f64 = rng.random();
        let s = (u.ln() / (1.0 - w).ln()) as u32;
        i += s + 1;
        if i <= n {
            let j = rng.random_range(0..k);
            reservoir[j as usize] = i - 1;
            w *= (rng.random::<f64>().ln() / (k as f64)).exp();
        }
    }
    reservoir
}

fn consistent_reservoir_to_n(k: u32, target_n: u32, seed: u64) -> Vec<u32> {
    let mut r = ConsistentReservoir::new(k, k, seed);
    while r.n() < target_n {
        match r.next() {
            Some((added, _)) if added < target_n => {}
            _ => break,
        }
    }
    r.reservoir().collect()
}

fn reservoir_benchmarks(c: &mut Criterion) {
    let seed = 42u64;

    let mut group = c.benchmark_group("reservoir_computation_up_to_n");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(3));

    for &n in &[100_000, 10_000_000] {
        for &k in &[100, 1000] {
            // 1. ConsistentReservoir direct build (O(k))
            group.bench_function(BenchmarkId::new(format!("ConsistentReservoir_Direct/k_{k}"), n), |b| {
                b.iter(|| {
                    black_box(ConsistentReservoir::new(k, n, seed).reservoir().collect::<Vec<u32>>());
                })
            });

            // 2. ConsistentPermutation direct build (O(k))
            group.bench_function(BenchmarkId::new(format!("ConsistentPermutation_Direct/k_{k}"), n), |b| {
                b.iter(|| {
                    black_box(ConsistentPermutation::new(n, seed).take(k as usize).collect::<Vec<u32>>());
                })
            });

            // 3. ConsistentReservoir streaming iteration from k to n (O(k log(n/k)))
            group.bench_function(BenchmarkId::new(format!("ConsistentReservoir_Streaming/k_{k}"), n), |b| {
                b.iter(|| {
                    black_box(consistent_reservoir_to_n(k, n, seed));
                })
            });

            // 4. Standard Algorithm R (O(n) linear scan)
            group.bench_function(BenchmarkId::new(format!("Standard_Algorithm_R/k_{k}"), n), |b| {
                b.iter(|| {
                    black_box(standard_reservoir_r(k, n, seed));
                })
            });

            // 5. Standard Algorithm L / Vitter (O(k log(n/k)) skip based)
            group.bench_function(BenchmarkId::new(format!("Standard_Algorithm_L/k_{k}"), n), |b| {
                b.iter(|| {
                    black_box(standard_reservoir_l(k, n, seed));
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

    targets = throughput_benchmark, append_vs_new_with_k, grow_k_vs_permutation, reservoir_benchmarks,
);
criterion_main!(benches);
