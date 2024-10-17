use std::time::Duration;

use bpe::appendable_encoder::AppendableEncoder;
use bpe::interval_encoding::IntervalEncoding;
use bpe_benchmarks::*;
use bpe_openai::{cl100k_base, Pretokenizer};
use bpe_tests::create_test_bytes;
use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
use rand::{thread_rng, Rng};

fn counting_benchmark(c: &mut Criterion) {
    for (name, bpe, _, _) in TOKENIZERS.iter() {
        let input = create_test_bytes(&bpe.bpe, 20000);
        let fast = IntervalEncoding::new(&bpe.bpe, &input);

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
                        |start| bpe.bpe.count(&input[start..start + bytes]),
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

        let text = create_test_string(&bpe.bpe, 20000);
        let input = text.as_bytes();

        let mut group = c.benchmark_group(format!("encoding-{name}"));
        group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
        for bytes in [10, 100, 1000, 10000] {
            group.throughput(criterion::Throughput::Bytes(bytes as u64));
            group.bench_with_input(
                BenchmarkId::new("backtracking", bytes),
                &bytes,
                |b, bytes| {
                    b.iter_batched(
                        || select_test_bytes(input, *bytes),
                        |input| bpe.bpe.encode_via_backtracking(input),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
            group.bench_with_input(BenchmarkId::new("heap", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || select_test_bytes(input, *bytes),
                    |input| bpe.bpe.encode_via_bitfield(input),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(BenchmarkId::new("table", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || select_test_bytes(input, *bytes),
                    |input| bpe.bpe.encode_via_table(input),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(BenchmarkId::new("greedy", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || select_test_bytes(input, *bytes),
                    |input| bpe.bpe.encode_greedy(input),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(BenchmarkId::new("minimal", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || select_test_bytes(input, *bytes),
                    |input| bpe.bpe.encode_minimal(input),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(
                BenchmarkId::new("huggingface", bytes),
                &bytes,
                |b, bytes| {
                    b.iter_batched(
                        || std::str::from_utf8(select_test_bytes(input, *bytes)).unwrap(),
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
        let input = create_test_bytes(&bpe.bpe, 20000);

        let mut group = c.benchmark_group(format!("appending-{name}"));
        group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
        for bytes in [10, 100, 1000, 10000] {
            group.throughput(criterion::Throughput::Bytes(bytes as u64));
            group.bench_with_input(BenchmarkId::new("appending", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || {
                        (
                            AppendableEncoder::new(&bpe.bpe),
                            select_test_bytes(&input, *bytes),
                        )
                    },
                    |(mut enc, input)| enc.extend(input.iter().copied()),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(
                BenchmarkId::new("backtracking", bytes),
                &bytes,
                |b, bytes| {
                    b.iter_batched(
                        || select_test_bytes(&input, *bytes),
                        |input| bpe.bpe.count(input),
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
        let text = create_test_string(&bpe.bpe, 20000);
        let input = text.as_bytes();

        let mut group = c.benchmark_group(format!("comparison-{name}"));
        group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
        for bytes in [10, 100, 1000, 10000] {
            group.throughput(criterion::Throughput::Bytes(bytes as u64));
            group.bench_with_input(
                BenchmarkId::new("backtracking", bytes),
                &bytes,
                |b, bytes| {
                    b.iter_batched(
                        || std::str::from_utf8(select_test_bytes(input, *bytes)).unwrap(),
                        |text| bpe.encode(text),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
            group.bench_with_input(BenchmarkId::new("tiktoken", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || std::str::from_utf8(select_test_bytes(input, *bytes)).unwrap(),
                    |text| tiktoken.encode_ordinary(text),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(
                BenchmarkId::new("huggingface", bytes),
                &bytes,
                |b, bytes| {
                    b.iter_batched(
                        || std::str::from_utf8(select_test_bytes(input, *bytes)).unwrap(),
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
    for (name, bpe, tiktoken, huggingface) in TOKENIZERS.iter() {
        let text: String = ('\0'..char::MAX).filter(|c| !c.is_whitespace()).collect();
        let input = text.as_bytes();

        let mut group = c.benchmark_group(format!("worstcase-{name}"));
        for bytes in [10, 100, 1000, 5000, 10000, 25000, 50000, 75000, 100000] {
            group.throughput(criterion::Throughput::Bytes(bytes as u64));
            group.bench_with_input(
                BenchmarkId::new("backtracking", bytes),
                &bytes,
                |b, bytes| {
                    b.iter_batched(
                        || std::str::from_utf8(select_test_bytes(input, *bytes)).unwrap(),
                        |text| bpe.encode(text),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
            group.bench_with_input(BenchmarkId::new("tiktoken", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || std::str::from_utf8(select_test_bytes(input, *bytes)).unwrap(),
                    |text| tiktoken.encode_ordinary(text),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(
                BenchmarkId::new("huggingface", bytes),
                &bytes,
                |b, bytes| {
                    b.iter_batched(
                        || std::str::from_utf8(select_test_bytes(input, *bytes)).unwrap(),
                        |text| huggingface.encode_fast(text, false).unwrap(),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
        }
        group.finish();
    }
}

fn pretok_benchmark(c: &mut Criterion) {
    let fast = cl100k_base().pre.as_ref().unwrap();

    let slow_pat = [
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)",
        "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+",
        "\\p{N}{1,3}",
        " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*",
        "(?:\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)",
    ]
    .join("|");
    let slow = Pretokenizer::from_pat(&slow_pat).unwrap();

    let text = create_test_string(&cl100k_base().bpe, 20000);
    let input = text.as_bytes();

    let mut group = c.benchmark_group(format!("pretok-cl100k"));
    for bytes in [10, 100, 1000, 5000, 10000, 25000, 50000, 75000, 100000] {
        group.throughput(criterion::Throughput::Bytes(bytes as u64));
        group.bench_with_input(BenchmarkId::new("fast", bytes), &bytes, |b, bytes| {
            b.iter_batched(
                || std::str::from_utf8(select_test_bytes(input, *bytes)).unwrap(),
                |text| fast.split(text).count(),
                criterion::BatchSize::SmallInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("slow", bytes), &bytes, |b, bytes| {
            b.iter_batched(
                || std::str::from_utf8(select_test_bytes(input, *bytes)).unwrap(),
                |text| slow.split(text).count(),
                criterion::BatchSize::SmallInput,
            )
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
    targets = counting_benchmark, encoding_benchmark, appending_benchmark, comparison_benchmark, worstcase_comparison_benchmark, pretok_benchmark
);
criterion_main!(benches);
