//! Build script: parses `data/CaseFolding.txt` and emits a compact
//! paged-bitmap + run-length encoded table to `$OUT_DIR/table.rs`.

use std::env;
use std::fs;
use std::path::PathBuf;

const DATA: &str = "data/CaseFolding.txt";

/// Maximum length of a single run (must fit in 7 bits so we can use the
/// 8th bit for the stride flag).
const MAX_RUN_LEN: u32 = 127;

/// Page = 64-cp block. After splitting each run is fully contained in one
/// page, so each run's `end` is uniquely identified by its 6-bit low part
/// within the page. With N=64 the densest page holds at most 30 runs, so a
/// linear scan within the page resolves the right run in a handful of
/// branch-predictable comparisons — never a global successor scan.
const PAGE_BITS: u32 = 6;
const PAGE_MASK: u32 = (1u32 << PAGE_BITS) - 1;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={DATA}");

    let folds = parse_folds(&fs::read_to_string(DATA).expect("read CaseFolding.txt"));
    let runs = build_runs(&folds);
    let runs = split_runs_at_page_boundary(&runs);
    let runs = split_runs_at_byte_delta(&runs);
    let out = emit_tables(&folds, &runs);

    let out_path: PathBuf = env::var_os("OUT_DIR")
        .expect("OUT_DIR is set during build")
        .into();
    fs::write(out_path.join("table.rs"), out).expect("write table.rs");
}

#[derive(Clone, Copy)]
struct Fold {
    cp: u32,
    fold: u32,
}

/// A run of consecutive (stride 1) or alternating (stride 2) code points
/// that all map with the same delta.
#[derive(Clone, Copy)]
struct Run {
    start: u32,
    stride: u8,
    length: u8,
    delta: i32,
}

fn parse_folds(text: &str) -> Vec<Fold> {
    let mut out = Vec::new();
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split(';').map(|s| s.trim());
        let cp_str = parts.next().expect("code point field");
        let cp = u32::from_str_radix(cp_str, 16).expect("code point is hex");
        let status = parts.next().expect("status field");
        let mapping = parts.next().expect("mapping field");
        // C = common (1:1), S = simple (1:1). Both are part of simple casefold.
        // F = full (1:N), T = Turkic (locale-specific) — both skipped.
        if status != "C" && status != "S" {
            continue;
        }
        let targets: Vec<u32> = mapping
            .split_whitespace()
            .map(|s| u32::from_str_radix(s, 16).expect("mapping is hex"))
            .collect();
        assert_eq!(targets.len(), 1, "C/S mappings are always 1:1");
        out.push(Fold {
            cp,
            fold: targets[0],
        });
    }
    out.sort_by_key(|f| f.cp);
    out
}

fn build_runs(folds: &[Fold]) -> Vec<Run> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < folds.len() {
        let cp0 = folds[i].cp;
        let delta0 = folds[i].fold as i64 - folds[i].cp as i64;
        // Greedy: try stride 1 and stride 2; pick whichever extends further.
        let extend = |stride: u32| -> u32 {
            let mut n: u32 = 1;
            loop {
                if n >= MAX_RUN_LEN {
                    break;
                }
                let j = i + n as usize;
                if j >= folds.len() {
                    break;
                }
                if folds[j].cp != cp0 + n * stride {
                    break;
                }
                if (folds[j].fold as i64 - folds[j].cp as i64) != delta0 {
                    break;
                }
                n += 1;
            }
            n
        };
        let len1 = extend(1);
        let len2 = extend(2);
        let (stride, length) = if len2 > len1 { (2, len2) } else { (1, len1) };
        runs.push(Run {
            start: cp0,
            stride: stride as u8,
            length: length as u8,
            delta: delta0 as i32,
        });
        i += length as usize;
    }
    runs
}

/// Split each run at page boundaries (64 cps) so its full `[start, end]`
/// range lives in a single page. After this transform the low 6 bits of `end`
/// are a unique-within-page identifier — which is what `PAGE_BITMAP`,
/// `PAGE_OFFSET` and `RUN_END_LOW` rely on.
///
/// Only stride-2 runs that happen to straddle a page boundary actually get
/// split (and only a handful in the real Unicode data). The split preserves
/// stride, length-per-piece, and per-piece delta — the delta is the same for
/// both halves of a split run since the underlying pattern is unchanged.
fn split_runs_at_page_boundary(runs: &[Run]) -> Vec<Run> {
    let mut out = Vec::new();
    for r in runs {
        let stride = r.stride as u32;
        let length = r.length as u32;
        let mut i = 0u32;
        while i < length {
            let sub_start = r.start + i * stride;
            let sub_page = sub_start >> PAGE_BITS;
            let mut j = i + 1;
            while j < length && (r.start + j * stride) >> PAGE_BITS == sub_page {
                j += 1;
            }
            out.push(Run {
                start: sub_start,
                stride: r.stride,
                length: (j - i) as u8,
                delta: r.delta,
            });
            i = j;
        }
    }
    out
}

/// Little-endian `u32` value of the UTF-8 encoding of `cp` (1–4 bytes, the
/// unused high bytes left zero). This is the number a little-endian load of
/// the character's bytes produces, and the basis for the raw byte-delta fold.
fn utf8_le(cp: u32) -> u32 {
    if cp < 0x80 {
        cp
    } else if cp < 0x800 {
        (0xC0 | (cp >> 6)) | ((0x80 | (cp & 0x3F)) << 8)
    } else if cp < 0x10000 {
        (0xE0 | (cp >> 12)) | ((0x80 | ((cp >> 6) & 0x3F)) << 8) | ((0x80 | (cp & 0x3F)) << 16)
    } else {
        (0xF0 | (cp >> 18))
            | ((0x80 | ((cp >> 12) & 0x3F)) << 8)
            | ((0x80 | ((cp >> 6) & 0x3F)) << 16)
            | ((0x80 | (cp & 0x3F)) << 24)
    }
}

/// The constant added to a character's little-endian byte word to fold it:
/// `utf8_le(cp + delta) - utf8_le(cp)`. Folding then becomes a masked load,
/// one `wrapping_add`, and a write of the result's bytes — no decode/encode.
fn byte_delta(cp: u32, delta: i32) -> u32 {
    let folded = (cp as i64 + delta as i64) as u32;
    utf8_le(folded).wrapping_sub(utf8_le(cp))
}

/// Split each (already page-split) run wherever the byte-delta changes between
/// consecutive elements, so every emitted run has a single constant byte-delta
/// (stored in `BYTE_DELTA`). This happens when a fold's destination crosses a
/// 64-cp boundary (the low-6-bit field wraps) or changes UTF-8 length; only a
/// handful of runs split. Codepoint delta, stride and per-piece length are
/// preserved, so the per-`char` lookup is unaffected.
fn split_runs_at_byte_delta(runs: &[Run]) -> Vec<Run> {
    let mut out = Vec::new();
    for r in runs {
        let stride = r.stride as u32;
        let length = r.length as u32;
        let mut i = 0u32;
        while i < length {
            let sub_start = r.start + i * stride;
            let bd = byte_delta(sub_start, r.delta);
            let mut j = i + 1;
            while j < length && byte_delta(r.start + j * stride, r.delta) == bd {
                j += 1;
            }
            out.push(Run {
                start: sub_start,
                stride: r.stride,
                length: (j - i) as u8,
                delta: r.delta,
            });
            i = j;
        }
    }
    out
}

fn emit_tables(folds: &[Fold], runs: &[Run]) -> String {
    let n = runs.len() as u32;

    // Each run's *inclusive end* code point. Runs are sorted by start and
    // non-overlapping, so ends are sorted ascending too.
    let ends: Vec<u32> = runs
        .iter()
        .map(|r| r.start + (r.length as u32 - 1) * (r.stride as u32))
        .collect();
    let last_covered = *ends.last().unwrap();

    // Pages are 64-cp blocks. After `split_runs_at_page_boundary` every run
    // lives in a single page, so the low 6 bits of `end` uniquely identify
    // a run within its page (stored as `RUN_END_LOW` below).
    let num_pages = (last_covered >> PAGE_BITS) as usize + 1;
    let num_bitmap_words = num_pages.div_ceil(64);
    let mut page_bitmap = vec![0u64; num_bitmap_words];
    let mut page_offset: Vec<u8> = vec![0];
    let mut prev_page: Option<u32> = None;
    let mut interval_count: u32 = 0;
    for &end in &ends {
        let page = end >> PAGE_BITS;
        page_bitmap[(page as usize) / 64] |= 1u64 << (page % 64);
        if Some(page) != prev_page {
            if prev_page.is_some() {
                page_offset.push(interval_count as u8);
            }
            prev_page = Some(page);
        }
        interval_count += 1;
    }
    page_offset.push(interval_count as u8);
    let num_populated_pages = page_offset.len() - 1;
    assert!(
        interval_count <= 255,
        "PAGE_OFFSET entries must fit in u8 (got {interval_count} intervals)",
    );

    // Cumulative popcount samples: `popcnt_samples[i]` is the number of
    // populated pages in `page_bitmap[0..i]`. This turns the per-query
    // popcount into a single load + one partial-word popcount.
    let mut popcnt_samples = vec![0u8; num_bitmap_words + 1];
    let mut cumul: u32 = 0;
    for (i, &w) in page_bitmap.iter().enumerate() {
        popcnt_samples[i] = cumul as u8;
        cumul += w.count_ones();
    }
    popcnt_samples[num_bitmap_words] = cumul as u8;
    assert_eq!(cumul as usize, num_populated_pages);
    assert!(cumul <= 255, "POPCNT_SAMPLES must fit in u8");

    // Per-run records, split into two byte arrays so the hot within-page scan
    // touches only clean `end_low` bytes (no masking, denser cache lines):
    //
    //   RUN_END_LOW[i]      = end   & PAGE_MASK            (0..=63, the scan key)
    //   RUN_START_STRIDE[i] = (start & PAGE_MASK)          (bits 0..6)
    //                       | ((stride - 1) << 6)          (bit 6)
    //
    // Every run is split to live inside one 64-cp page, so both ends fit in 6
    // bits and the membership test works directly on `cp & 0x3F` (`low_v`): the
    // scan finds the first run with `end_low >= low_v` by comparing raw bytes,
    // and a run then covers `low_v` iff `low_v >= start_low` (and, for stride 2,
    // `(low_v - start_low)` is even). The codepoint delta is *not* stored here —
    // both fold paths apply `BYTE_DELTA` instead.
    let mut run_end_low = Vec::<u8>::with_capacity(runs.len());
    let mut run_start_stride = Vec::<u8>::with_capacity(runs.len());
    let mut max_abs_delta: i32 = 0;
    for r in runs.iter() {
        assert!(r.length >= 1 && r.length <= 127);
        assert!(r.stride == 1 || r.stride == 2);
        let start_low = (r.start & PAGE_MASK) as u8;
        let end_low = ((r.start + (r.length as u32 - 1) * (r.stride as u32)) & PAGE_MASK) as u8;
        let stride_bit = r.stride - 1;
        max_abs_delta = max_abs_delta.max(r.delta.abs());
        run_end_low.push(end_low);
        run_start_stride.push(start_low | (stride_bit << 6));
    }
    // Pad `RUN_END_LOW` so the chunked 8-wide scan can always read a full
    // 8-byte chunk past any page start. `0xFF` padding reads as "≥ low_v" but
    // lands past the page's run count, so the `j < n` check discards it.
    let num_runs = run_end_low.len();
    run_end_low.resize(num_runs + 8, 0xFF);

    // Parallel little-endian byte deltas, one per run. Each run was split so
    // its byte-delta is constant, so we read it off the run's start. The byte
    // path folds with `from_le_bytes(masked) + BYTE_DELTA[idx]` — no decode.
    let byte_deltas: Vec<u32> = runs.iter().map(|r| byte_delta(r.start, r.delta)).collect();
    let max_abs_byte_delta = byte_deltas
        .iter()
        .map(|&b| (b as i32).unsigned_abs())
        .max()
        .unwrap_or(0);

    // Sanity: size accounting (printed as build warnings for visibility).
    let index_bytes = page_bitmap.len() * 8 + popcnt_samples.len() + page_offset.len();
    let total = index_bytes + run_end_low.len() + run_start_stride.len() + byte_deltas.len() * 4;
    if env::var_os("CASEFOLD_BUILD_INFO").is_some() {
        println!(
            "cargo:warning=casefold table: {} fold entries, {} runs, {} populated pages, {} bytes total ({:.2} bits/entry), max |delta| = {}, max |byte_delta| = {}",
            folds.len(),
            n,
            num_populated_pages,
            total,
            total as f64 * 8.0 / folds.len() as f64,
            max_abs_delta,
            max_abs_byte_delta,
        );
    }

    // Emit Rust source.
    let mut s = String::new();
    s.push_str("// AUTO-GENERATED by build.rs from data/CaseFolding.txt. Do not edit.\n\n");
    s.push_str("#[cfg(test)]\n");
    s.push_str(&format!(
        "pub(crate) const NUM_FOLD_ENTRIES: u32 = {};\n\n",
        folds.len()
    ));

    emit_u64_array(&mut s, "PAGE_BITMAP", &page_bitmap);
    emit_u8_array(&mut s, "POPCNT_SAMPLES", &popcnt_samples);
    emit_u8_array(&mut s, "PAGE_OFFSET", &page_offset);
    emit_u8_array(&mut s, "RUN_END_LOW", &run_end_low);
    emit_u8_array(&mut s, "RUN_START_STRIDE", &run_start_stride);
    emit_u32_array(&mut s, "BYTE_DELTA", &byte_deltas);

    s
}

fn emit_u64_array(s: &mut String, name: &str, data: &[u64]) {
    s.push_str(&format!(
        "pub(crate) static {name}: [u64; {}] = [\n",
        data.len()
    ));
    for chunk in data.chunks(4) {
        s.push_str("    ");
        for v in chunk {
            s.push_str(&format!("0x{:016x}, ", v));
        }
        s.push('\n');
    }
    s.push_str("];\n\n");
}

fn emit_u8_array(s: &mut String, name: &str, data: &[u8]) {
    s.push_str(&format!(
        "pub(crate) static {name}: [u8; {}] = [\n",
        data.len()
    ));
    for chunk in data.chunks(16) {
        s.push_str("    ");
        for v in chunk {
            s.push_str(&format!("0x{:02x}, ", v));
        }
        s.push('\n');
    }
    s.push_str("];\n\n");
}

fn emit_u32_array(s: &mut String, name: &str, data: &[u32]) {
    s.push_str(&format!(
        "pub(crate) static {name}: [u32; {}] = [\n",
        data.len()
    ));
    for chunk in data.chunks(8) {
        s.push_str("    ");
        for v in chunk {
            s.push_str(&format!("0x{:08x}, ", v));
        }
        s.push('\n');
    }
    s.push_str("];\n\n");
}
