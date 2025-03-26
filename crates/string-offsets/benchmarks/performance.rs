use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rng, Rng};
use string_offsets::{AllConfig, OnlyLines, StringOffsets};

fn only_lines_construction_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("only_lines_construction");
    for size in [1000, 10000, 100000] {
        let mut rng = rng();
        let random_input: String = (0..size)
            .map(|_| rng.random_range(32u8..128u8) as char)
            .collect();
        group.throughput(criterion::Throughput::Bytes(random_input.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &random_input,
            |b, input| b.iter(|| black_box(StringOffsets::<OnlyLines>::new(input))),
        );
    }
    group.finish();
}

fn full_construction_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_construction");
    for size in [1000, 10000, 100000] {
        let mut rng = rng();
        let random_input: String = (0..size)
            .map(|_| rng.random_range(32u8..128u8) as char)
            .collect();
        group.throughput(criterion::Throughput::Bytes(random_input.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &random_input,
            |b, input| b.iter(|| black_box(StringOffsets::<AllConfig>::new(input))),
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = only_lines_construction_benchmark, full_construction_benchmark
);
criterion_main!(benches);
