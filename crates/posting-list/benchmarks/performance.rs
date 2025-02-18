use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use posting_list::{
    encode, encode_bits_into_words, encode_ranges, encode_runs, BitReader, Decoder, RunDecoder, RunReader
};
use rand::{distr::Uniform, Rng};

fn create_input(density: u32) -> Vec<bool> {
    let runs = encode_ranges(
        rand::rng()
            .sample_iter(Uniform::new(0u32, 100).unwrap())
            .map(|v| v <= density)
            .take(10000),
    );
    encode(runs.into_iter())
}

fn decoding_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("decoding"));
    for density in [1, 10, 20, 50, 80, 90, 95, 99] {
        group.throughput(criterion::Throughput::Bytes(10000 * density as u64 / 100));
        group.bench_with_input(
            BenchmarkId::new("bit-decoder", density),
            &density,
            |b, &density| {
                b.iter_batched(
                    || create_input(density),
                    |bits| {
                        Decoder::new(bits.into_iter()).collect::<Vec<_>>()
                    },
                    criterion::BatchSize::LargeInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("bit-decoder-bit-reader", density),
            &density,
            |b, &density| {
                b.iter_batched(
                    || encode_bits_into_words(create_input(density).into_iter()),
                    |encoded_words| {
                        Decoder::new(BitReader::new(&encoded_words)).collect::<Vec<_>>()
                    },
                    criterion::BatchSize::LargeInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("run-decoder", density),
            &density,
            |b, &density| {
                b.iter_batched(
                    || encode_runs(create_input(density).into_iter()),
                    |encoded_runs| {
                        RunDecoder::new(encoded_runs.into_iter()).collect::<Vec<_>>()
                    },
                    criterion::BatchSize::LargeInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("run-decoder-word-reader", density),
            &density,
            |b, &density| {
                b.iter_batched(
                    || {
                        let input = create_input(density);
                        let len = input.len();
                        (encode_bits_into_words(input.into_iter()), len)
                    },
                    |(words, len)| {
                        RunDecoder::new(RunReader::new(&words, len as u32)).collect::<Vec<_>>()
                    },
                    criterion::BatchSize::LargeInput,
                )
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
                .warm_up_time(Duration::from_millis(500))
                .measurement_time(Duration::from_millis(4000))
                .nresamples(1000);
    targets = decoding_benchmark
);
criterion_main!(benches);
