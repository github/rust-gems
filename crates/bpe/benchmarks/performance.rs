use std::time::Duration;

use bpe::appendable_encoder::AppendableEncoder;
use bpe::byte_pair_encoding::{
    create_test_string, create_test_string_with_predicate, select_test_string,
};
use bpe::interval_encoding::IntervalEncoding;
use bpe_benchmarks::*;
use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
use rand::{thread_rng, Rng};

fn counting_benchmark(c: &mut Criterion) {
    for (name, bpe, _, _) in TOKENIZERS.iter() {
        let input = create_test_string(&bpe.bpe, 80_000);
        let fast = IntervalEncoding::new(&bpe.bpe, input.as_bytes());

        let mut group = c.benchmark_group(format!("counting-{name}"));
        group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
        for bytes in [10, 100, 1000, 10000] {
            group.throughput(criterion::Throughput::Bytes(bytes as u64));
            group.bench_with_input(BenchmarkId::new("interval", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || thread_rng().gen_range(0..input.len() - bytes),
                    |start| fast.count(start..start + bytes),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(
                BenchmarkId::new("backtracking", bytes),
                &bytes,
                |b, bytes| {
                    b.iter_batched(
                        || thread_rng().gen_range(0..input.len() - bytes),
                        |start| bpe.bpe.count(&input.as_bytes()[start..start + bytes]),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
        }
        group.finish();
    }
}

fn encoding_benchmark(c: &mut Criterion) {
    for (name, bpe, _, huggingface) in TOKENIZERS.iter() {
        let huggingface = without_pretokenizer(huggingface);

        let text = create_test_string(&bpe.bpe, 80_000);

        let mut group = c.benchmark_group(format!("encoding-{name}"));
        group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
        for bytes in [10, 100, 1000, 10000] {
            group.throughput(criterion::Throughput::Bytes(bytes as u64));
            group.bench_with_input(
                BenchmarkId::new("backtracking", bytes),
                &bytes,
                |b, bytes| {
                    b.iter_batched(
                        || select_test_string(&text, *bytes),
                        |text| bpe.bpe.encode_via_backtracking(text.as_bytes()),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
            group.bench_with_input(BenchmarkId::new("heap", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || select_test_string(&text, *bytes),
                    |text| bpe.bpe.encode_via_bitfield(text.as_bytes()),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(BenchmarkId::new("table", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || select_test_string(&text, *bytes),
                    |text| bpe.bpe.encode_via_table(text.as_bytes()),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(BenchmarkId::new("greedy", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || select_test_string(&text, *bytes),
                    |text| bpe.bpe.encode_greedy(text.as_bytes()),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(BenchmarkId::new("minimal", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || select_test_string(&text, *bytes),
                    |text| bpe.bpe.encode_minimal(text.as_bytes()),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(
                BenchmarkId::new("huggingface", bytes),
                &bytes,
                |b, bytes| {
                    b.iter_batched(
                        || select_test_string(&text, *bytes),
                        |text| huggingface.encode_fast(text, false).unwrap(),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
        }
        group.finish();
    }
}

fn appending_benchmark(c: &mut Criterion) {
    for (name, bpe, _, _) in TOKENIZERS.iter() {
        let text = create_test_string(&bpe.bpe, 80_000);

        let mut group = c.benchmark_group(format!("appending-{name}"));
        group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
        for bytes in [10, 100, 1000, 10000] {
            group.throughput(criterion::Throughput::Bytes(bytes as u64));
            group.bench_with_input(BenchmarkId::new("appending", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || {
                        (
                            AppendableEncoder::new(&bpe.bpe),
                            select_test_string(&text, *bytes),
                        )
                    },
                    |(mut enc, text)| enc.extend(text.as_bytes().iter().copied()),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(
                BenchmarkId::new("backtracking", bytes),
                &bytes,
                |b, bytes| {
                    b.iter_batched(
                        || select_test_string(&text, *bytes),
                        |text| bpe.bpe.count(text.as_bytes()),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
        }
        group.finish();
    }
}

fn comparison_benchmark(c: &mut Criterion) {
    for (name, bpe, tiktoken, huggingface) in TOKENIZERS.iter() {
        let text = create_test_string(&bpe.bpe, 80_000);

        let mut group = c.benchmark_group(format!("comparison-{name}"));
        group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
        for bytes in [10, 100, 1000, 10000] {
            group.throughput(criterion::Throughput::Bytes(bytes as u64));
            group.bench_with_input(
                BenchmarkId::new("backtracking", bytes),
                &bytes,
                |b, bytes| {
                    b.iter_batched(
                        || select_test_string(&text, *bytes),
                        |text| bpe.encode(text),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
            group.bench_with_input(BenchmarkId::new("tiktoken", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || select_test_string(&text, *bytes),
                    |text| tiktoken.encode_ordinary(text),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(
                BenchmarkId::new("huggingface", bytes),
                &bytes,
                |b, bytes| {
                    b.iter_batched(
                        || select_test_string(&text, *bytes),
                        |text| huggingface.encode_fast(text, false).unwrap(),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
        }
        group.finish();
    }
}

fn worstcase_comparison_benchmark(c: &mut Criterion) {
    for (name, tok, tiktoken, huggingface) in TOKENIZERS.iter() {
        let text = create_test_string_with_predicate(&tok.bpe, 100000, |text| {
            tok.split(text).nth(1).is_none()
        });

        let mut group = c.benchmark_group(format!("worstcase-{name}"));
        for bytes in [10, 100, 1000, 5000, 10000, 25000, 50000] {
            group.throughput(criterion::Throughput::Bytes(bytes as u64));
            group.bench_with_input(
                BenchmarkId::new("backtracking", bytes),
                &bytes,
                |b, bytes| {
                    b.iter_batched(
                        || select_test_string(&text, *bytes),
                        |text| tok.encode(text),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
            group.bench_with_input(BenchmarkId::new("tiktoken", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || select_test_string(&text, *bytes),
                    |text| tiktoken.encode_ordinary(text),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(
                BenchmarkId::new("huggingface", bytes),
                &bytes,
                |b, bytes| {
                    b.iter_batched(
                        || select_test_string(&text, *bytes),
                        |text| huggingface.encode_fast(text, false).unwrap(),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
        }
        group.finish();
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default()
                .warm_up_time(Duration::from_millis(500))
                .measurement_time(Duration::from_millis(4000))
                .nresamples(1000);
    targets = counting_benchmark, encoding_benchmark, appending_benchmark, comparison_benchmark, worstcase_comparison_benchmark
);
criterion_main!(benches);
