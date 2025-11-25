use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use pef::avx_batch_decoder::{new_decoder, BatchDecoder, AvxBatchDecoder};
use pef::{EFBatchDecoder, EliasFano};
use rand::prelude::*;

fn criterion_benchmark(c: &mut Criterion) {
    let size = 100_000;

    // Generate random sorted data with clustered gaps using a Markov chain
    let mut data = Vec::with_capacity(size as usize);
    let mut current = 0;
    let mut rng = StdRng::seed_from_u64(123456789);

    // State 0: Dense cluster (small gaps, mostly 1)
    // State 1: Sparse region (larger gaps)
    let mut state = 0;

    for _ in 0..size {
        let gap = if state == 0 {
            if rng.random_bool(0.1) {
                state = 1;
            } // Transition to sparse
            if rng.random_bool(0.9) {
                1
            } else {
                (rng.random::<u32>() % 5) + 1
            }
        } else {
            if rng.random_bool(0.1) {
                state = 0;
            } // Transition to dense
            (rng.random::<u32>() % 100) + 1
        };

        current += gap;
        data.push(current);
    }
    let max = current + 1;

    let elias_fano = EliasFano::new(data.iter().copied(), max, size);

    let mut group = c.benchmark_group("elias_fano");
    group.throughput(Throughput::Elements(size as u64));

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512vbmi2")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("popcnt")
    {
        group.bench_function("avx_batch_decode", |b| {
            b.iter(|| {
                //let mut decoder = new_decoder(&elias_fano);
                let mut decoder = AvxBatchDecoder::<4>::new(&elias_fano);
                let mut buffer = [0u32; 16];
                loop {
                    let count = decoder.decode_batch(&mut buffer);
                    if count == 0 {
                        break;
                    }
                    black_box(&buffer[..count]);
                }
            })
        });
    }

    group.bench_function("iter", |b| {
        b.iter(|| {
            for val in elias_fano.iter() {
                black_box(val);
            }
        })
    });

    group.bench_function("batch_decode", |b| {
        b.iter(|| {
            let mut batch_decoder = EFBatchDecoder::new(&elias_fano);
            loop {
                let batch = batch_decoder.decode_batch();
                if batch.is_empty() {
                    break;
                }
                black_box(batch);
            }
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
