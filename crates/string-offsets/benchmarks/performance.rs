use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rng, Rng};
use string_offsets::StringOffsets;

fn construction_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");
    for size in [1000, 10000, 100000] {
        let mut rng = rng();
        // Generate random ascii input.
        let random_input: String = (0..size).map(|_| rng.random_range(32u8..128) as char).collect();
        group.throughput(criterion::Throughput::Bytes(random_input.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &random_input,
            |b, input| b.iter(|| black_box(StringOffsets::new(input))),
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = construction_benchmark
);
criterion_main!(benches);
