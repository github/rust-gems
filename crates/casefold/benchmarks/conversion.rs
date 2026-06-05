//! Benchmarks for `casefold::fold_into_bytes`, comparing it against the
//! straightforward `str::to_lowercase` baseline (and the per-char
//! `c.to_lowercase()` flat-map variant) on several representative inputs.
//!
//! Note: `to_lowercase` performs Unicode *lowercasing*, not case folding —
//! the two agree on ASCII and on most BMP letters but diverge in a handful of
//! cases (e.g. `Σ` final-sigma context, `İ` → `i\u{0307}`). This benchmark is
//! about throughput on equivalent workloads, not output equality.

use casefold::fold_into_bytes;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

/// `str::to_lowercase` baseline: allocates a new `String` then converts to
/// bytes.
#[inline]
fn to_lowercase_into_bytes(s: String) -> Vec<u8> {
    s.to_lowercase().into_bytes()
}

/// Per-char `c.to_lowercase()` flat-map baseline. Same end-result as
/// `str::to_lowercase` but expressed in user code, so we know exactly what
/// optimizations are (not) being applied.
#[inline]
fn chars_to_lowercase_into_bytes(s: String) -> Vec<u8> {
    s.chars().flat_map(|c| c.to_lowercase()).collect::<String>().into_bytes()
}

/// `str::to_ascii_lowercase` baseline: byte-only, fast but ASCII-only
/// (leaves multibyte sequences untouched).
#[inline]
fn to_ascii_lowercase_into_bytes(s: String) -> Vec<u8> {
    s.to_ascii_lowercase().into_bytes()
}

/// `simd-normalizer::casefold` — SIMD-accelerated Unicode simple case
/// folding (C+S, Unicode 17.0). On aarch64 uses NEON; on x86_64 uses
/// SSE4.2/AVX2/AVX-512 via runtime CPUID dispatch.
#[inline]
fn simd_normalizer_casefold_into_bytes(s: String) -> Vec<u8> {
    let folded = simd_normalizer::casefold(&s, simd_normalizer::CaseFoldMode::Standard);
    folded.into_owned().into_bytes()
}

fn ascii_input() -> String {
    // A typical English sentence repeated to reach a meaningful working size.
    let unit = "The Quick BROWN fox JUMPS over THE lazy DOG. 0123456789. ";
    unit.repeat(100)
}

fn mixed_bmp_input() -> String {
    // Mixed Latin / Greek / Cyrillic — exercises the reallocating tier-2
    // UTF-8 fold path for every multibyte char (all same UTF-8 length).
    let unit = "Größe der ÜBERSICHT — ΣΟΦΙΑ καὶ ἈΛΉΘΕΙΑ — Привет, МИР! ";
    unit.repeat(100)
}

fn growing_input() -> String {
    // Contains both growing folds (U+023A → U+2C65 and U+023E → U+2C66),
    // mixed with shrinking and same-length folds.
    let unit = "abcdefg \u{023A}\u{023E} XYZ ";
    unit.repeat(100)
}

fn bench_conversion(c: &mut Criterion, name: &str, input: &str) {
    let mut group = c.benchmark_group(name);
    group.throughput(Throughput::Bytes(input.len() as u64));

    group.bench_function(BenchmarkId::new("Casefold::fold_into_bytes", input.len()), |b| {
        b.iter_batched(
            || input.to_string(),
            |s| fold_into_bytes(black_box(s)),
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function(BenchmarkId::new("str::to_lowercase", input.len()), |b| {
        b.iter_batched(
            || input.to_string(),
            |s| to_lowercase_into_bytes(black_box(s)),
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function(
        BenchmarkId::new("chars().flat_map(to_lowercase)", input.len()),
        |b| {
            b.iter_batched(
                || input.to_string(),
                |s| chars_to_lowercase_into_bytes(black_box(s)),
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.bench_function(
        BenchmarkId::new("str::to_ascii_lowercase", input.len()),
        |b| {
            b.iter_batched(
                || input.to_string(),
                |s| to_ascii_lowercase_into_bytes(black_box(s)),
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.bench_function(
        BenchmarkId::new("simd_normalizer::casefold", input.len()),
        |b| {
            b.iter_batched(
                || input.to_string(),
                |s| simd_normalizer_casefold_into_bytes(black_box(s)),
                criterion::BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

fn benches(c: &mut Criterion) {
    bench_conversion(c, "convert_ascii", &ascii_input());
    bench_conversion(c, "convert_mixed_bmp", &mixed_bmp_input());
    bench_conversion(c, "convert_growing", &growing_input());
}

criterion_group!(benches_group, benches);
criterion_main!(benches_group);
