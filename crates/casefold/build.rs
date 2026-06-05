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
    let out = emit_tables(&folds, &runs);

    let out_path: PathBuf = env::var_os("OUT_DIR").expect("OUT_DIR is set during build").into();
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
/// `PAGE_OFFSET` and the low bits of `RUN_DATA` rely on.
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
    // a run within its page (stored in the low bits of RUN_DATA below).
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

    // Per-run packed u32 layout (one entry per run, parallel to PAGE_OFFSET
    // which slices RUN_DATA into per-page groups):
    //
    //   bits  0..6   end_low = end & PAGE_MASK                 (6 bits)
    //   bit   6      stride - 1                                (1 bit, 0 or 1)
    //   bits  7..14  length                                    (7 bits, 1..=127)
    //   bits 14..32  delta, sign-extended                      (18 bits, ±131072)
    //
    // 18-bit signed delta range easily covers the largest Unicode simple-fold
    // delta (max |δ| in the data is 42561). The single-array layout lets the
    // within-page linear scan read the next entry's `end_low` with the same
    // indexed load that decodes the run on a hit — one 64-byte cache line
    // covers 16 packed runs.
    let mut run_data = Vec::<u32>::with_capacity(runs.len());
    let mut max_abs_delta: i32 = 0;
    for (i, r) in runs.iter().enumerate() {
        assert!(r.length >= 1 && r.length <= 127);
        assert!(r.stride == 1 || r.stride == 2);
        let end_low = ends[i] & PAGE_MASK;
        let stride_bit = (r.stride as u32) - 1;
        let length = r.length as u32;
        assert!(
            (-131072..=131071).contains(&r.delta),
            "delta {} for run {} does not fit in 18 signed bits",
            r.delta,
            i,
        );
        max_abs_delta = max_abs_delta.max(r.delta.abs());
        let delta_field = (r.delta as u32) & 0x3FFFF; // low 18 bits
        let packed = end_low | (stride_bit << 6) | (length << 7) | (delta_field << 14);
        run_data.push(packed);
    }

    // Sanity: size accounting (printed as build warnings for visibility).
    let index_bytes = page_bitmap.len() * 8 + popcnt_samples.len() + page_offset.len();
    let total = index_bytes + run_data.len() * 4;
    if env::var_os("CASEFOLD_BUILD_INFO").is_some() {
        println!(
            "cargo:warning=casefold table: {} fold entries, {} runs, {} populated pages, {} bytes total ({:.2} bits/entry), max |delta| = {}",
            folds.len(),
            n,
            num_populated_pages,
            total,
            total as f64 * 8.0 / folds.len() as f64,
            max_abs_delta,
        );
    }

    // Emit Rust source.
    let mut s = String::new();
    s.push_str("// AUTO-GENERATED by build.rs from data/CaseFolding.txt. Do not edit.\n\n");
    s.push_str("#[allow(dead_code)]\n");
    s.push_str(&format!("pub(crate) const N: u32 = {n};\n"));
    s.push_str(&format!(
        "pub(crate) const LAST_COVERED: u32 = {last_covered};\n"
    ));
    s.push_str(&format!(
        "pub(crate) const NUM_FOLD_ENTRIES: u32 = {};\n\n",
        folds.len()
    ));

    emit_u64_array(&mut s, "PAGE_BITMAP", &page_bitmap);
    emit_u8_array(&mut s, "POPCNT_SAMPLES", &popcnt_samples);
    emit_u8_array(&mut s, "PAGE_OFFSET", &page_offset);
    emit_u32_array(&mut s, "RUN_DATA", &run_data);

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
