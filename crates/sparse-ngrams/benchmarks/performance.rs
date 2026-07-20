use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use sparse_ngrams::{
    NGram, collect_sparse_grams_deque, collect_sparse_grams_scan, max_sparse_grams,
};

fn bench_collect(c: &mut Criterion) {
    let inputs: Vec<(&str, Vec<u8>)> = vec![
        ("small_11B", b"hello world".to_vec()),
        (
            "medium_900B",
            "the quick brown fox jumps over the lazy dog. "
                .repeat(20)
                .into_bytes(),
        ),
        (
            "large_15KB",
            include_bytes!("fixtures/sample_code.txt").to_vec(),
        ),
    ];

    let mut group = c.benchmark_group("collect");
    for (name, input) in &inputs {
        let mut buf = vec![NGram::from_bytes(b"xx"); max_sparse_grams(input.len())];
        group.throughput(Throughput::Bytes(input.len() as u64));

        group.bench_with_input(BenchmarkId::new("deque", name), input, |b, input| {
            b.iter(|| {
                let mut w = 0usize;
                collect_sparse_grams_deque(black_box(input), |gram, _idx| {
                    buf[w] = gram;
                    w += 1;
                });
                w
            })
        });
        group.bench_with_input(BenchmarkId::new("scan", name), input, |b, input| {
            b.iter(|| {
                let mut w = 0usize;
                collect_sparse_grams_scan(black_box(input), |gram, _idx| {
                    buf[w] = gram;
                    w += 1;
                });
                w
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_collect);
criterion_main!(benches);
