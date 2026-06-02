//! Statistical comparison of consistent-permutation constructions
//! against `ConsistentChooseKHasher` (the gold-standard reference).
//!
//! Each generator is asked to produce a uniformly-random ordering of
//! `0..n` and we run four chi-squared tests at alpha = 0.001:
//!
//! 1. Marginal at position 0 (first emitted element should be uniform).
//! 2. Marginal at position `k-1` (last sampled element should be uniform).
//! 3. Choose-2 unordered-pair uniformity over the `(n choose 2)` pairs.
//! 4. Per-position bit balance (each bit should be 1 ~half the time).
//!
//! Run with:  cargo run --release --example statistical_comparison

use std::hash::{DefaultHasher, Hash};

use consistent_choose_k::{ConsistentChooseKHasher, ConsistentPermutation};

const SAMPLES: usize = 200_000;

/// 99.9% upper-tail critical value for chi-squared with `df` degrees of
/// freedom, computed by Wilson-Hilferty:
///   `chi2_{df, 1-alpha} ≈ df * (1 - 2/(9 df) + z * sqrt(2/(9 df)))^3`
/// where `z = Phi^{-1}(1 - alpha)`. For alpha=0.001, z ≈ 3.0902.
fn chi2_critical_999(df: usize) -> f64 {
    let df = df as f64;
    let a = 2.0 / (9.0 * df);
    df * (1.0 - a + 3.0902 * a.sqrt()).powi(3)
}

/// Pearson chi-squared statistic for `observed` against the uniform
/// distribution; returns `(chi2, df)`.
fn chi2_uniform(observed: &[u64]) -> (f64, usize) {
    let total: u64 = observed.iter().sum();
    let cells = observed.len();
    let expected = total as f64 / cells as f64;
    let mut chi2 = 0.0;
    for &c in observed {
        let d = c as f64 - expected;
        chi2 += d * d / expected;
    }
    (chi2, cells - 1)
}

// ---------- Generators ----------

fn chooseh_take(key: u64, n: usize, k: usize) -> Vec<u32> {
    let mut h = DefaultHasher::default();
    key.hash(&mut h);
    ConsistentChooseKHasher::new(h, n)
        .map(|v| v as u32)
        .take(k)
        .collect()
}

fn perm_take(key: u64, n: usize, k: usize) -> Vec<u32> {
    ConsistentPermutation::new(n as u32, key).take(k).collect()
}

// ---------- Per-method accumulator ----------

struct Acc {
    name: String,
    first: Vec<u64>,
    kth: Vec<u64>,
    pair: Vec<u64>, // n*n, indexed by lo*n + hi (lo <= hi)
    bits: Vec<Vec<u64>>, // bits[pos][bit]
}

impl Acc {
    fn new(name: &str, n: usize, kk: usize, n_bits: usize) -> Self {
        Self {
            name: name.to_string(),
            first: vec![0u64; n],
            kth: vec![0u64; n],
            pair: vec![0u64; n * n],
            bits: vec![vec![0u64; n_bits]; kk],
        }
    }

    fn record(&mut self, v: &[u32], n: usize, n_bits: usize) {
        let kk = self.bits.len();
        self.first[v[0] as usize] += 1;
        self.kth[v[kk - 1] as usize] += 1;
        let (lo, hi) = if v[0] <= v[1] {
            (v[0], v[1])
        } else {
            (v[1], v[0])
        };
        self.pair[(lo as usize) * n + hi as usize] += 1;
        for (pos, &val) in v.iter().enumerate().take(kk) {
            for b in 0..n_bits {
                if (val >> b) & 1 == 1 {
                    self.bits[pos][b] += 1;
                }
            }
        }
    }
}

fn run_for_n(n: usize) {
    let kk = 2usize.min(n - 1).max(1);
    let n_bits = ((n - 1).next_power_of_two().trailing_zeros() as usize).max(1);

    type GenFn = Box<dyn Fn(u64, usize, usize) -> Vec<u32>>;

    // Build the list of methods to test.
    let methods: Vec<(&'static str, GenFn)> = vec![
        ("ChooseKHasher    ", Box::new(chooseh_take)),
        ("Permutation      ", Box::new(perm_take)),
    ];

    let mut accs: Vec<Acc> = methods
        .iter()
        .map(|(name, _)| Acc::new(name, n, kk, n_bits))
        .collect();

    for seed in 0..SAMPLES as u64 {
        // Avalanche each sample's seed so we don't pass small
        // sequential integers into the (single-pass) hashers, which
        // would correlate strongly. `ConsistentPermutation::new`
        // does not avalanche its key argument, so we have to do it
        // here.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let key = z ^ (z >> 31);
        for ((_, gen), acc) in methods.iter().zip(accs.iter_mut()) {
            let v = gen(key, n, kk);
            assert_eq!(v.len(), kk);
            acc.record(&v, n, n_bits);
        }
    }

    println!("\n=== n = {n}  (samples = {SAMPLES}) ===");
    println!(
        "{:<46} {:>14} {:>6} {:>14} {:>7}",
        "test", "chi2", "df", "crit(0.001)", "verdict"
    );

    let print_row = |label: &str, chi2: f64, df: usize| {
        let crit = chi2_critical_999(df);
        let verdict = if chi2 <= crit { "PASS" } else { "FAIL" };
        println!("{label:<46} {chi2:>14.2} {df:>6} {crit:>14.2} {verdict:>7}");
    };

    // 1) Marginal @ pos 0.
    for acc in &accs {
        let (c, d) = chi2_uniform(&acc.first);
        print_row(&format!("{}: marginal @ pos 0", acc.name), c, d);
    }
    // 2) Marginal @ pos kk-1.
    for acc in &accs {
        let (c, d) = chi2_uniform(&acc.kth);
        print_row(&format!("{}: marginal @ pos {}", acc.name, kk - 1), c, d);
    }
    // 3) Choose-2 set uniformity over the (n choose 2) cells with i<j.
    for acc in &accs {
        let mut cells = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                cells.push(acc.pair[i * n + j]);
            }
        }
        let (c, d) = chi2_uniform(&cells);
        print_row(&format!("{}: choose-2 set uniformity", acc.name), c, d);
    }
    // 4) Per-(pos, bit) balance against the expected per-bit fraction.
    for acc in &accs {
        let mut chi2 = 0.0;
        let mut df = 0;
        for b in 0..n_bits {
            let pi = (0..n).filter(|v| (v >> b) & 1 == 1).count() as f64 / n as f64;
            if pi == 0.0 || pi == 1.0 {
                continue;
            }
            let exp_set = SAMPLES as f64 * pi;
            let exp_unset = SAMPLES as f64 * (1.0 - pi);
            for pos in 0..kk {
                let set = acc.bits[pos][b] as f64;
                let unset = SAMPLES as f64 - set;
                chi2 += (set - exp_set).powi(2) / exp_set
                    + (unset - exp_unset).powi(2) / exp_unset;
                df += 1;
            }
        }
        print_row(&format!("{}: per-(pos,bit) balance", acc.name), chi2, df);
    }
}

fn main() {
    println!("Statistical comparison of consistent-permutation generators");
    println!("(Chi-squared tests, alpha = 0.001 — PASS means we cannot reject uniformity.)");
    for n in [8usize, 16, 32, 64, 128, 256, 1024] {
        run_for_n(n);
    }
}
