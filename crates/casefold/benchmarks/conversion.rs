//! Benchmarks for `casefold::fold_into_bytes`, comparing it against the
//! straightforward `str::to_lowercase` baseline (and the per-char
//! `c.to_lowercase()` flat-map variant) on several representative inputs.
//!
//! Note: `to_lowercase` performs Unicode *lowercasing*, not case folding —
//! the two agree on ASCII and on most BMP letters but diverge in a handful of
//! cases (e.g. `Σ` final-sigma context, `İ` → `i\u{0307}`). This benchmark is
//! about throughput on equivalent workloads, not output equality.

use casefold::{fold_into_bytes, utf8_len};
use casefold_benchmarks::{hashmap_fold_utf8, reference_map_utf8, FoldHashMap};
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

/// Loads up to 4 bytes starting at `at` as a little-endian `u32`, zero-padding
/// when fewer than 4 bytes remain so the tail of the buffer never reads out of
/// bounds.
#[inline]
fn read_u32_le(bytes: &[u8], at: usize) -> u32 {
    let n = (bytes.len() - at).min(4);
    let mut word = 0u32;
    for j in 0..n {
        word |= (bytes[at + j] as u32) << (8 * j);
    }
    word
}

/// HashMap-based case fold operating directly on the raw UTF-8 bytes.
///
/// For each character it loads the next 4 bytes as a little-endian `u32`,
/// then shifts left by `4 - utf8_len` bytes so the bytes of the *following*
/// character (which the over-read pulled in) fall off the top and only the
/// current character's bytes remain — a left-aligned key, no masking needed.
/// [`hashmap_fold_utf8`] returns the fold in writable low-byte order, so the
/// full 4-byte word is stored and the cursor advances by just the folded
/// length.
fn hashmap_fold_into_bytes(map: &FoldHashMap, s: String) -> Vec<u8> {
    let bytes = s.into_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len() + 4);
    let mut read = 0usize;
    while read < bytes.len() {
        let word = read_u32_le(&bytes, read);
        let len = utf8_len((word & 0xFF) as u8);
        // Left-align: shift the over-read trailing bytes out of the word.
        let key = word << (8 * (4 - len) as u32);
        let folded = hashmap_fold_utf8(map, key);
        let dest_len = utf8_len((folded & 0xFF) as u8);
        out.reserve(4);
        let l = out.len();
        // SAFETY: `reserve(4)` guarantees ≥4 spare bytes at offset `l`; we store
        // 4 bytes there and expose only `dest_len` (≤ 4) of them.
        unsafe {
            out.as_mut_ptr()
                .add(l)
                .cast::<u32>()
                .write_unaligned(folded.to_le());
            out.set_len(l + dest_len);
        }
        read += len;
    }
    out
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

fn cjk_input() -> String {
    // Predominantly 3-byte CJK (fold-free lead bytes 0xE3..0xEE) with a few
    // ASCII spaces. Exercises the non-ASCII tail loop where every multibyte
    // char misses the fold table.
    let unit = "日本語のテキスト 漢字 ひらがな カタカナ 한국어 中文字符 ";
    unit.repeat(100)
}

fn symbols_input() -> String {
    // Punctuation / arrows / math / box-drawing from U+2000..2BFF plus Myanmar
    // (U+1000..109F). These live in blocks 0xE2 / 0xE1, whose *lead byte* does
    // contain folds elsewhere (Greek Ext, Latin Ext Additional, Roman
    // numerals…), so a lead-byte filter passes them — but every one of these
    // chars sits on a fold-free 64-cp page, so the page-precision probe rejects
    // them without decoding.
    let unit = "— “quote” … → ⇒ ∑ ∫ ≤ ≥ │ ┌ ┐ မြန်မာ စာ ◆ ★ ";
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

    let fold_map = reference_map_utf8();
    group.bench_function(
        BenchmarkId::new("HashMap::fold_into_bytes (UTF-8 u32)", input.len()),
        |b| {
            b.iter_batched(
                || input.to_string(),
                |s| hashmap_fold_into_bytes(&fold_map, black_box(s)),
                criterion::BatchSize::SmallInput,
            );
        },
    );

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
    bench_conversion(c, "convert_cjk", &cjk_input());
    bench_conversion(c, "convert_symbols", &symbols_input());
}

criterion_group!(benches_group, benches);
criterion_main!(benches_group);
