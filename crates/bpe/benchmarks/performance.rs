use std::path::Path;
use std::sync::LazyLock;
use std::time::Duration;

use bpe::appendable_encoder::AppendableEncoder;
use bpe::byte_pair_encoding::{create_test_bytes, BytePairEncoding};
use bpe::interval_encoding::IntervalEncoding;
use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
use rand::{thread_rng, Rng};
use tiktoken_rs::CoreBPE as TiktokenBPE;
use tokenizers::models::bpe::BPE as HuggingfaceBPE;
use tokenizers::Tokenizer as HuggingfaceTokenizer;

static TOKENIZERS: LazyLock<
    [(
        &'static str,
        &'static BytePairEncoding,
        TiktokenBPE,
        HuggingfaceTokenizer,
    ); 2],
> = LazyLock::new(|| {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    [
        (
            "cl100k",
            bpe_openai::cl100k(),
            tiktoken_rs::cl100k_base().unwrap(),
            {
                let bpe = HuggingfaceBPE::from_file(
                    data_dir.join("cl100k/vocab.json").to_str().unwrap(),
                    data_dir.join("cl100k/merges.txt").to_str().unwrap(),
                )
                .build()
                .unwrap();
                HuggingfaceTokenizer::new(bpe)
            },
        ),
        (
            "o200k",
            bpe_openai::o200k(),
            tiktoken_rs::o200k_base().unwrap(),
            {
                let bpe = HuggingfaceBPE::from_file(
                    data_dir.join("o200k/vocab.json").to_str().unwrap(),
                    data_dir.join("o200k/merges.txt").to_str().unwrap(),
                )
                .build()
                .unwrap();
                HuggingfaceTokenizer::new(bpe)
            },
        ),
    ]
});

fn counting_benchmark(c: &mut Criterion) {
    for (name, bpe, _, _) in TOKENIZERS.iter() {
        let input = create_test_bytes(bpe, 20000);
        let fast = IntervalEncoding::new(bpe, &input);

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
                        |start| bpe.count(&input[start..start + bytes]),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
        }
        group.finish();
    }
}

fn encoding_benchmark(c: &mut Criterion) {
    for (name, bpe, tiktoken, huggingface) in TOKENIZERS.iter() {
        let text = create_test_string(bpe, 20000);
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
                        |input| bpe.encode_via_backtracking(input),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
            group.bench_with_input(BenchmarkId::new("heap", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || select_test_bytes(input, *bytes),
                    |input| bpe.encode_via_bitfield(input),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(BenchmarkId::new("table", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || select_test_bytes(input, *bytes),
                    |input| bpe.encode_via_table(input),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(BenchmarkId::new("greedy", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || select_test_bytes(input, *bytes),
                    |input| bpe.encode_greedy(input),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(BenchmarkId::new("minimal", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || select_test_bytes(input, *bytes),
                    |input| bpe.encode_minimal(input),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(BenchmarkId::new("tiktoken", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || select_test_bytes(input, *bytes),
                    |input| tiktoken.encode_ordinary(std::str::from_utf8(input).unwrap()),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(
                BenchmarkId::new("huggingface", bytes),
                &bytes,
                |b, bytes| {
                    b.iter_batched(
                        || select_test_bytes(input, *bytes),
                        |input| huggingface.encode_fast(std::str::from_utf8(input).unwrap(), false),
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
        let input = create_test_bytes(bpe, 20000);

        let mut group = c.benchmark_group(format!("appending-{name}"));
        group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
        for bytes in [10, 100, 1000, 10000] {
            group.throughput(criterion::Throughput::Bytes(bytes as u64));
            group.bench_with_input(BenchmarkId::new("appending", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || {
                        (
                            AppendableEncoder::new(bpe),
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
                        |input| bpe.count(input),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
        }
        group.finish();
    }
}

fn worstcase_benchmark(c: &mut Criterion) {
    for (name, bpe, tiktoken, huggingface) in TOKENIZERS.iter() {
        let text: String = ('\0'..char::MAX).filter(|c| !c.is_whitespace()).collect();
        let input = text.as_bytes();

        let mut group = c.benchmark_group(format!("worstcase-{name}"));
        for bytes in [10, 100, 1000, 5000, 10000, 25000, 50000, 75000, 100000] {
            group.throughput(criterion::Throughput::Bytes(bytes as u64));
            group.bench_with_input(
                BenchmarkId::new("backtracking", bytes),
                &bytes,
                |b, bytes| b.iter(|| bpe.encode_via_backtracking(select_test_bytes(input, *bytes))),
            );
            group.bench_with_input(BenchmarkId::new("tiktoken", bytes), &bytes, |b, bytes| {
                b.iter_batched(
                    || select_test_bytes(input, *bytes),
                    |input| tiktoken.encode_ordinary(std::str::from_utf8(input).unwrap()),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(
                BenchmarkId::new("huggingface", bytes),
                &bytes,
                |b, bytes| {
                    b.iter_batched(
                        || select_test_bytes(input, *bytes),
                        |input| huggingface.encode_fast(std::str::from_utf8(input).unwrap(), false),
                        criterion::BatchSize::SmallInput,
                    )
                },
            );
        }
        group.finish();
    }
}

fn is_char_boundary(b: u8) -> bool {
    // Single byte encodings satisfy the bit pattern 0xxxxxxx, i.e. b < 128
    // Continuation bytes satisfy the bit pattern 10xxxxxx, i.e. b < 192
    // The rest are bytes belonging to the first byte of multi byte encodings (11xxxxxx): b >= 192
    // When interpreting the byte representation as signed integers, then numbers in the range 128..192
    // correspond to the smallest representable numbers. I.e. the two ranges [0, 128) and [192, 256) can
    // be tested with a single signed comparison.
    b as i8 >= -0x40 // NB: b < 128 || b >= 192
}

fn create_test_string(bpe: &BytePairEncoding, tokens: usize) -> String {
    use rand::{thread_rng, Rng};
    let mut text = String::new();
    for _ in 0..tokens {
        loop {
            let i = thread_rng().gen_range(0..bpe.num_tokens());
            let s = bpe.token_bytes(i as u32);
            if s.iter().all(|b| is_char_boundary(*b)) {
                if let Ok(s) = std::str::from_utf8(s) {
                    text.push_str(s);
                    break;
                }
            }
        }
    }
    text
}

fn select_test_bytes(input: &[u8], bytes: usize) -> &[u8] {
    let mut start = thread_rng().gen_range(0..input.len() - bytes);
    while start > 0 && !is_char_boundary(input[start]) {
        start -= 1;
    }
    let mut end = start + bytes;
    while end < input.len() && !is_char_boundary(input[end]) {
        end += 1;
    }
    &input[start..end]
}

criterion_group!(
    name = benches;
    config = Criterion::default()
                .warm_up_time(Duration::from_millis(500))
                .measurement_time(Duration::from_millis(4000))
                .nresamples(1000);
    targets = counting_benchmark, encoding_benchmark, appending_benchmark, worstcase_benchmark
);
criterion_main!(benches);
