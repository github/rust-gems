//! Deterministic distribution diagnostics, not probabilistic CI gates.
//! cargo run --release -p consistent-choose-k --example permutation_diagnostics

use std::hash::{DefaultHasher, Hash, Hasher};

use consistent_choose_k::{ConsistentPermutation, VirtualPermutation};

const SAMPLES: u64 = 200_000;
fn key_seed(key: u64, workload_seed: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    (workload_seed ^ key).hash(&mut hasher);
    hasher.finish()
}

fn report(algorithm: &str, n: usize, metric: &str, cells: &[u64]) {
    let expected = cells.iter().sum::<u64>() as f64 / cells.len() as f64;
    let chi2: f64 = cells
        .iter()
        .map(|&v| (v as f64 - expected).powi(2) / expected)
        .sum();
    println!(
        "{algorithm},{n},{metric},{},{expected:.3},{chi2:.3},{},{}",
        cells.len() - 1,
        cells.iter().min().expect("nonempty histogram"),
        cells.iter().max().expect("nonempty histogram"),
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let workload_seed = match args.next().as_deref() {
        None => 0x7065_726d_7574_6531,
        Some("--held-out") => 0x686f_6c64_6f75_7431,
        Some(_) => panic!("usage: permutation_diagnostics [--held-out]"),
    };
    assert!(
        args.next().is_none(),
        "usage: permutation_diagnostics [--held-out]"
    );
    println!("# samples={SAMPLES}, key_seed={workload_seed:#x}");
    println!(
        "# iterator_bytes: layered={}, virtual={}; layered also owns heap counters",
        std::mem::size_of::<ConsistentPermutation>(),
        std::mem::size_of::<VirtualPermutation>(),
    );
    println!("algorithm,n,metric,df,expected_per_cell,chi2,min,max");
    for n in [3, 4, 5, 7, 8, 9, 16, 17, 32, 64] {
        for algorithm in ["layered", "virtual"] {
            let slots = n.min(8);
            let mut marginal = vec![vec![0u64; n]; slots];
            let mut pairs = vec![0; n * n];
            let mut distant_pairs = vec![0; n * n];
            let mut triples = vec![0; n * (n - 1) * (n - 2) / 6];
            let mut consecutive_keys = vec![0; n * n];
            let mut previous_primary = None;
            for key in 0..SAMPLES {
                let seed = key_seed(key, workload_seed);
                let values: Vec<usize> = if algorithm == "layered" {
                    ConsistentPermutation::new(n as u32, seed)
                        .take(slots)
                        .map(|v| v as usize)
                        .collect()
                } else {
                    VirtualPermutation::new(n as u64, seed)
                        .take(slots)
                        .map(|v| v as usize)
                        .collect()
                };
                for (histogram, &v) in marginal.iter_mut().zip(&values) {
                    histogram[v] += 1;
                }
                pairs[values[0] * n + values[1]] += 1;
                distant_pairs[values[0] * n + values[slots - 1]] += 1;
                let mut triple = [values[0], values[1], values[2]];
                triple.sort_unstable();
                let [a, b, c] = triple;
                triples[c * (c - 1) * (c - 2) / 6 + b * (b - 1) / 2 + a] += 1;
                if let Some(previous) = previous_primary {
                    consecutive_keys[previous * n + values[0]] += 1;
                }
                previous_primary = Some(values[0]);
            }
            for (slot, cells) in marginal.iter().enumerate() {
                report(algorithm, n, &format!("slot_{slot}"), cells);
            }
            for (metric, cells) in [("ordered_0_1", pairs), ("ordered_0_last", distant_pairs)] {
                assert!((0..n).all(|v| cells[v * n + v] == 0));
                let off_diagonal: Vec<_> = cells
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, count)| (i / n != i % n).then_some(count))
                    .collect();
                report(algorithm, n, metric, &off_diagonal);
            }
            report(algorithm, n, "choose_3", &triples);
            report(algorithm, n, "consecutive_key_primaries", &consecutive_keys);
        }
    }
}
