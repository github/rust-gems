//! Deterministic distribution diagnostics, not statistical CI gates.
//! Run with no arguments for the primary corpus, or --held-out.

use std::hash::{DefaultHasher, Hash, Hasher};

use consistent_choose_k::{ConsistentPermutation, VirtualPermutation};

fn key_seed(key: u64, corpus: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    (corpus ^ key).hash(&mut hasher);
    hasher.finish()
}

struct Histogram {
    counts: Vec<u64>,
    probabilities: Vec<f64>,
}

impl Histogram {
    fn uniform(cells: usize) -> Self {
        Self {
            counts: vec![0; cells],
            probabilities: vec![1.0 / cells as f64; cells],
        }
    }

    fn pairs(n: usize, buckets: usize, distinct: bool) -> Self {
        let mut sizes = vec![0; buckets];
        for node in 0..n {
            sizes[node * buckets / n] += 1;
        }
        let mut probabilities = vec![];
        for a in 0..buckets {
            for b in 0..buckets {
                let available = sizes[b] - usize::from(distinct && a == b);
                probabilities
                    .push((sizes[a] * available) as f64 / (n * (n - usize::from(distinct))) as f64);
            }
        }
        Self {
            counts: vec![0; buckets * buckets],
            probabilities,
        }
    }

    fn record(&mut self, cell: usize) {
        self.counts[cell] += 1;
    }

    fn report(&self, algorithm: &str, n: usize, metric: &str) {
        let total: u64 = self.counts.iter().sum();
        let mut chi2 = 0.0;
        let mut tv = 0.0;
        let mut max_relative: f64 = 0.0;
        let mut min_expected = f64::INFINITY;
        let mut cells = 0;
        for (&count, &probability) in self.counts.iter().zip(&self.probabilities) {
            if probability == 0.0 {
                assert_eq!(count, 0, "impossible pair observed");
                continue;
            }
            let expected = total as f64 * probability;
            assert!(expected >= 9.99, "inadequate expected cell count");
            let delta = (count as f64 - expected).abs();
            chi2 += delta * delta / expected;
            tv += delta / total as f64 / 2.0;
            max_relative = max_relative.max(delta / expected);
            min_expected = min_expected.min(expected);
            cells += 1;
        }
        println!(
            "{algorithm},{n},{metric},{total},{},{min_expected:.3},{chi2:.3},{:.3},{tv:.6}",
            cells - 1,
            max_relative * 100.0
        );
    }
}

fn order(algorithm: &str, n: usize, seed: u64) -> Vec<usize> {
    match algorithm {
        "layered" => ConsistentPermutation::new(n as u32, seed)
            .map(|v| v as usize)
            .collect(),
        "sentinel" => VirtualPermutation::new(n as u64, seed)
            .map(|v| v as usize)
            .collect(),
        _ => unreachable!(),
    }
}

fn first(algorithm: &str, n: usize, seed: u64) -> usize {
    match algorithm {
        "layered" => ConsistentPermutation::new(n as u32, seed)
            .next()
            .expect("nonempty") as usize,
        "sentinel" => VirtualPermutation::new(n as u64, seed)
            .next()
            .expect("nonempty") as usize,
        _ => unreachable!(),
    }
}

fn permutation_index(values: &[usize]) -> usize {
    let mut index = 0;
    for (i, &value) in values.iter().enumerate() {
        index = index * (values.len() - i) + values[i + 1..].iter().filter(|&&v| v < value).count();
    }
    index
}

fn main() {
    let mut args = std::env::args().skip(1);
    let corpus = match args.next().as_deref() {
        None => 0x7065_726d_7574_6531,
        Some("--held-out") => 0x686f_6c64_6f75_7431,
        Some(_) => panic!("usage: permutation_diagnostics [--held-out]"),
    };
    assert!(
        args.next().is_none(),
        "usage: permutation_diagnostics [--held-out]"
    );
    println!("# corpus={corpus:#x}; exact bins require >=10 expected observations");
    println!(
        "# iterator_bytes: layered={}, sentinel={}; layered also owns heap counters",
        std::mem::size_of::<ConsistentPermutation>(),
        std::mem::size_of::<VirtualPermutation>()
    );
    println!(
        "algorithm,n,metric,observations,df,min_expected,chi2,max_relative_percent,total_variation"
    );
    for n in [
        1usize, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 30, 31, 32, 33, 62, 63, 64, 65, 126, 127,
        128, 129, 254, 255, 256, 257,
    ] {
        let samples = if n <= 33 {
            200_000
        } else if n <= 65 {
            50_000
        } else {
            20_000
        };
        let buckets = if samples / (n * n) >= 10 { n } else { 8 };
        let pair_kind = if buckets == n { "exact" } else { "bucket8" };
        let mut ranks = vec![
            0,
            1.min(n - 1),
            2.min(n - 1),
            n / 2,
            n.saturating_sub(2),
            n - 1,
        ];
        ranks.sort_unstable();
        ranks.dedup();
        let mut rank_pairs = vec![
            (0, 1.min(n - 1)),
            (n / 2, (n / 2 + 1).min(n - 1)),
            (0, n / 2),
            (0, n - 1),
        ];
        rank_pairs.retain(|(a, b)| a != b);
        rank_pairs.sort_unstable();
        rank_pairs.dedup();
        let triple_cells = if n >= 3 { n * (n - 1) * (n - 2) / 6 } else { 0 };
        let triple_ok = triple_cells > 0 && samples / triple_cells >= 10;
        if n >= 3 && !triple_ok {
            println!(
                "# n={n}: skipping exact choose_3; expected={:.3}",
                samples as f64 / triple_cells as f64
            );
        }
        for algorithm in ["layered", "sentinel"] {
            let mut marginals: Vec<_> = ranks.iter().map(|_| Histogram::uniform(n)).collect();
            let mut pairs: Vec<_> = rank_pairs
                .iter()
                .map(|_| Histogram::pairs(n, buckets, true))
                .collect();
            let mut triples = triple_ok.then(|| Histogram::uniform(triple_cells));
            let mut full = (n <= 7).then(|| Histogram::uniform((1..=n).product()));
            let mut adjacent_keys = Histogram::pairs(n, buckets, false);
            let mut related_seeds = Histogram::pairs(n, buckets, false);
            let mut previous = None;
            for key in 0..samples as u64 {
                let seed = key_seed(key, corpus);
                let values = order(algorithm, n, seed);
                assert_eq!(values.len(), n);
                for (hist, &rank) in marginals.iter_mut().zip(&ranks) {
                    hist.record(values[rank]);
                }
                for (hist, &(a, b)) in pairs.iter_mut().zip(&rank_pairs) {
                    assert_ne!(values[a], values[b]);
                    hist.record((values[a] * buckets / n) * buckets + values[b] * buckets / n);
                }
                if let Some(hist) = &mut triples {
                    let mut triple = [values[0], values[1], values[2]];
                    triple.sort_unstable();
                    let [a, b, c] = triple;
                    hist.record(c * (c - 1) * (c - 2) / 6 + b * (b - 1) / 2 + a);
                }
                if let Some(hist) = &mut full {
                    hist.record(permutation_index(&values));
                }
                let primary = values[0] * buckets / n;
                if let Some(old) = previous {
                    adjacent_keys.record(old * buckets + primary);
                }
                previous = Some(primary);
                let related = first(algorithm, n, seed ^ (1u64 << 63)) * buckets / n;
                related_seeds.record(primary * buckets + related);
            }
            for (hist, rank) in marginals.iter().zip(&ranks) {
                hist.report(algorithm, n, &format!("rank_{rank}"));
            }
            for (hist, (a, b)) in pairs.iter().zip(&rank_pairs) {
                hist.report(algorithm, n, &format!("ordered_{a}_{b}_{pair_kind}"));
            }
            if let Some(hist) = triples {
                hist.report(algorithm, n, "choose_3");
            }
            if let Some(hist) = full {
                hist.report(algorithm, n, "full_order");
            }
            adjacent_keys.report(
                algorithm,
                n,
                &format!("consecutive_key_primaries_{pair_kind}"),
            );
            related_seeds.report(algorithm, n, &format!("seed_bitflip_primaries_{pair_kind}"));
        }
    }
}
