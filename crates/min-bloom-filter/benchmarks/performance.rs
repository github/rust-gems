use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use min_bloom_filter::MinBloomFilter;
use sbbf_rs_safe::Filter as Sbbf;

const ENTRY_COUNT: usize = 3_000_000;
const RATES: [(&str, f64); 2] = [("1_percent", 0.01), ("0.1_percent", 0.001)];

fn hashes() -> Vec<u64> {
    (0..ENTRY_COUNT)
        .map(|index| splitmix64(index as u64))
        .collect()
}

fn benchmark_insert(c: &mut Criterion) {
    let hashes = hashes();
    let mut group = c.benchmark_group("insert_3m");
    group.sample_size(10);
    group.throughput(Throughput::Elements(ENTRY_COUNT as u64));

    for (name, rate) in RATES {
        group.bench_with_input(BenchmarkId::new("fpr", name), &rate, |b, rate| {
            b.iter_batched(
                || MinBloomFilter::new(ENTRY_COUNT, *rate),
                |mut filter| {
                    for (index, hash) in hashes.iter().copied().enumerate() {
                        filter.insert(black_box(hash), ((index % 15) + 1) as u8);
                    }
                    black_box(filter);
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn benchmark_get(c: &mut Criterion) {
    let hashes = hashes();
    let mut group = c.benchmark_group("get_3m");
    group.sample_size(10);
    group.throughput(Throughput::Elements(ENTRY_COUNT as u64));

    for (name, rate) in RATES {
        let mut filter = MinBloomFilter::new(ENTRY_COUNT, rate);
        for (index, hash) in hashes.iter().copied().enumerate() {
            filter.insert(hash, ((index % 15) + 1) as u8);
        }

        group.bench_with_input(BenchmarkId::new("fpr", name), &filter, |b, filter| {
            b.iter(|| {
                let mut sum = 0_u64;
                for hash in hashes.iter().copied() {
                    sum += u64::from(filter.get(black_box(hash)));
                }
                black_box(sum)
            });
        });
    }
    group.finish();
}

fn benchmark_sbbf_insert(c: &mut Criterion) {
    let hashes = hashes();
    let mut group = c.benchmark_group("sbbf_insert_3m");
    group.sample_size(10);
    group.throughput(Throughput::Elements(ENTRY_COUNT as u64));

    for (name, rate) in RATES {
        let bits_per_entry = bloom_bits_per_entry(rate);
        group.bench_with_input(
            BenchmarkId::new("fpr", name),
            &bits_per_entry,
            |b, bits_per_entry| {
                b.iter_batched(
                    || Sbbf::new(*bits_per_entry, ENTRY_COUNT),
                    |mut filter| {
                        for hash in hashes.iter().copied() {
                            black_box(filter.insert_hash(black_box(hash)));
                        }
                        black_box(filter);
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

fn benchmark_sbbf_contains(c: &mut Criterion) {
    let hashes = hashes();
    let mut group = c.benchmark_group("sbbf_contains_3m");
    group.sample_size(10);
    group.throughput(Throughput::Elements(ENTRY_COUNT as u64));

    for (name, rate) in RATES {
        let mut filter = Sbbf::new(bloom_bits_per_entry(rate), ENTRY_COUNT);
        for hash in hashes.iter().copied() {
            filter.insert_hash(hash);
        }

        group.bench_with_input(BenchmarkId::new("fpr", name), &filter, |b, filter| {
            b.iter(|| {
                let mut matches = 0_u64;
                for hash in hashes.iter().copied() {
                    matches += u64::from(filter.contains_hash(black_box(hash)));
                }
                black_box(matches)
            });
        });
    }
    group.finish();
}

fn bloom_bits_per_entry(false_positive_rate: f64) -> usize {
    let probes = 8.0;
    (-probes / (1.0 - false_positive_rate.powf(1.0 / probes)).ln()).ceil() as usize
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

criterion_group!(
    benches,
    benchmark_insert,
    benchmark_get,
    benchmark_sbbf_insert,
    benchmark_sbbf_contains
);
criterion_main!(benches);
