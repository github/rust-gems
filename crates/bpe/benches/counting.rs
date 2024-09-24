use std::time::Duration;

use bpe::byte_pair_encoding::{create_test_bytes, BytePairEncoding};
use bpe::interval_encoding::IntervalEncoding;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::{thread_rng, Rng};

fn counting_benchmark(c: &mut Criterion) {
    for (name, bpe) in [
        ("cl100k", BytePairEncoding::cl100k()),
        ("o200k", BytePairEncoding::o200k()),
    ] {
        let text = create_test_bytes(&bpe, 20000);
        let fast = IntervalEncoding::new(&bpe, &text);

        for bytes in [10, 100, 1000, 10000] {
            let mut group = c.benchmark_group(format!("bpe-{name}-bytes-{bytes}"));
            group.bench_function("hybrid counting", |b| {
                b.iter_batched(
                    || thread_rng().gen_range(0..text.len() - bytes),
                    |start| fast.count(start..start + bytes),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_function("backtrack counting", |b| {
                b.iter_batched(
                    || thread_rng().gen_range(0..text.len() - bytes),
                    |start| bpe.count(&text[start..start + bytes]),
                    criterion::BatchSize::SmallInput,
                )
            });
        }
    }
}

fn encoding_benchmark(c: &mut Criterion) {
    for (name, bpe) in [
        ("cl100k", BytePairEncoding::cl100k()),
        ("o200k", BytePairEncoding::o200k()),
    ] {
        let tiktoken = tiktoken_rs::cl100k_base().unwrap();
        let text = create_test_string(&bpe, 20000);
        let input = text.as_bytes();

        for bytes in [10, 100, 1000, 10000] {
            let mut group = c.benchmark_group(format!("bpe-{name}-bytes-{bytes}"));
            group.bench_function("backtracking", |b| {
                b.iter_batched(
                    || thread_rng().gen_range(0..input.len() - bytes),
                    |start| bpe.encode_via_backtracking(&input[start..start + bytes]),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_function("heap", |b| {
                b.iter_batched(
                    || thread_rng().gen_range(0..input.len() - bytes),
                    |start| bpe.encode_via_bitfield(&input[start..start + bytes]),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_function("dynamic programming", |b| {
                b.iter_batched(
                    || thread_rng().gen_range(0..input.len() - bytes),
                    |start| bpe.encode_via_table(&input[start..start + bytes]),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_function("greedy", |b| {
                b.iter_batched(
                    || thread_rng().gen_range(0..input.len() - bytes),
                    |start| bpe.encode_greedy(&input[start..start + bytes]),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_function("minimal", |b| {
                b.iter_batched(
                    || thread_rng().gen_range(0..input.len() - bytes),
                    |start| bpe.encode_minimal(&input[start..start + bytes]),
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_function("tiktoken", |b| {
                b.iter_batched(
                    || loop {
                        let start = thread_rng().gen_range(0..input.len() - bytes - 1);
                        if is_char_boundary(input[start]) && is_char_boundary(input[start + bytes])
                        {
                            return start;
                        }
                    },
                    |start| tiktoken.encode_ordinary(&text[start..start + bytes]),
                    criterion::BatchSize::SmallInput,
                )
            });
        }
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

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_millis(500)).nresamples(1000);
    targets = counting_benchmark, encoding_benchmark
);
criterion_main!(benches);
