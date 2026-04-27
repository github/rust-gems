use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use sparse_ngrams::{NGram, collect_sparse_grams_deque, collect_sparse_grams_scan, collect_sparse_grams_masked, collect_sparse_grams_masked_avx, collect_sparse_grams_wide, collect_sparse_grams_wide_avx, max_sparse_grams};

fn bench_collect(c: &mut Criterion) {
    let inputs: Vec<(&str, Vec<u8>)> = vec![
        ("small_11B", b"hello world".to_vec()),
        (
            "medium_900B",
            "the quick brown fox jumps over the lazy dog. "
                .repeat(20)
                .into_bytes(),
        ),
        ("large_15KB", include_str!("../src/lib.rs").as_bytes().to_vec()),
    ];

    let mut group = c.benchmark_group("collect");
    for (name, input) in &inputs {
        let mut buf = vec![NGram::from_bytes(b"xx"); max_sparse_grams(input.len())];
        group.throughput(Throughput::Bytes(input.len() as u64));

        group.bench_with_input(BenchmarkId::new("deque", name), input, |b, input| {
            b.iter(|| collect_sparse_grams_deque(black_box(input), &mut buf))
        });
        group.bench_with_input(BenchmarkId::new("scan", name), input, |b, input| {
            b.iter(|| collect_sparse_grams_scan(black_box(input), &mut buf))
        });
        group.bench_with_input(BenchmarkId::new("masked", name), input, |b, input| {
            b.iter(|| collect_sparse_grams_masked(black_box(input), &mut buf))
        });
        group.bench_with_input(BenchmarkId::new("masked_avx", name), input, |b, input| {
            b.iter(|| collect_sparse_grams_masked_avx(black_box(input), &mut buf))
        });
        group.bench_with_input(BenchmarkId::new("wide", name), input, |b, input| {
            b.iter(|| collect_sparse_grams_wide(black_box(input), &mut buf))
        });
        group.bench_with_input(BenchmarkId::new("wide_avx", name), input, |b, input| {
            b.iter(|| collect_sparse_grams_wide_avx(black_box(input), &mut buf))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_collect);
criterion_main!(benches);
