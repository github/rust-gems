//! Compares the compact paged-bitmap simple-fold table to a `HashMap<u32, u32>`
//! baseline across a few representative workloads.

use casefold::simple_fold;
use casefold_benchmarks::{hashmap_fold, reference_map};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;

const SEED: u64 = 0xC0FF_EEFE_EDF0_0DAB;

/// Workload 1: every code point in U+0000..U+10000, in order (mostly ASCII /
/// BMP letters that have a defined fold).
fn workload_bmp_sequential() -> Vec<char> {
    (0..0x10000u32)
        .filter_map(char::from_u32) // skips surrogates
        .collect()
}

/// Workload 2: a random sample of BMP code points (mix of folding +
/// non-folding chars; simulates random text).
fn workload_bmp_random(n: usize) -> Vec<char> {
    let mut rng = StdRng::seed_from_u64(SEED);
    (0..n)
        .map(|_| loop {
            let cp = rng.random_range(0u32..0x10000u32);
            if let Some(c) = char::from_u32(cp) {
                return c;
            }
        })
        .collect()
}

/// Workload 3: random ASCII letters only (the common hot path).
fn workload_ascii(n: usize) -> Vec<char> {
    let mut rng = StdRng::seed_from_u64(SEED);
    (0..n)
        .map(|_| {
            let b = rng.random_range(b'A'..=b'z');
            b as char
        })
        .collect()
}

/// Workload 4: only code points that *do* fold (worst case for both
/// implementations, exercises every successful lookup path).
fn workload_only_folds() -> Vec<char> {
    let map = reference_map();
    let mut keys: Vec<u32> = map.keys().copied().collect();
    keys.sort();
    keys.into_iter().filter_map(char::from_u32).collect()
}

fn bench_workload(c: &mut Criterion, name: &str, chars: &[char]) {
    let map = reference_map();
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Elements(chars.len() as u64));

    group.bench_function(BenchmarkId::new("Casefold", chars.len()), |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for &ch in chars {
                acc = acc.wrapping_add(simple_fold(black_box(ch)) as u32);
            }
            acc
        });
    });

    group.bench_function(BenchmarkId::new("HashMap", chars.len()), |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for &ch in chars {
                acc = acc.wrapping_add(hashmap_fold(&map, black_box(ch)) as u32);
            }
            acc
        });
    });

    group.finish();
}

fn benches(c: &mut Criterion) {
    bench_workload(c, "bmp_sequential", &workload_bmp_sequential());
    bench_workload(c, "bmp_random_10k", &workload_bmp_random(10_000));
    bench_workload(c, "ascii_random_10k", &workload_ascii(10_000));
    bench_workload(c, "only_folds", &workload_only_folds());
}

criterion_group!(benches_group, benches);
criterion_main!(benches_group);
