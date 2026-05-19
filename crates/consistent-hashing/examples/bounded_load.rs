//! Bounded-load consistent hashing example.
//!
//! Compares unbounded vs bounded-load assignment across many random seeds,
//! reporting average and standard deviation of load spread and consistency
//! (fraction of assignments that change when a node is added).
//!
//! Bounded assignment iterates over tokens sequentially, greedily assigning
//! each token its k most-preferred nodes that still have capacity. Using
//! round-robin (all tokens claim one replica per round) yields nearly
//! identical churn numbers with marginally better load spread.
//!
//! Run with:  cargo run --example bounded_load

use std::hash::{DefaultHasher, Hash, Hasher};

use consistent_hashing::ConsistentChooseKHasher;

/// Bounded-load assignment.
///
/// Each token claims all k replicas before moving to the next token,
/// skipping any node that has reached `max_load`.
fn bounded_load_assign(
    iters: impl IntoIterator<Item = ConsistentChooseKHasher<DefaultHasher>>,
    k: usize,
    n: usize,
    max_load: usize,
) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut load = vec![0usize; n];
    let mut assignments = Vec::new();

    for mut iter in iters {
        let mut assigned = Vec::with_capacity(k);
        for node in iter.by_ref() {
            if load[node] < max_load {
                load[node] += 1;
                assigned.push(node);
                if assigned.len() == k {
                    break;
                }
            }
        }
        assignments.push(assigned);
    }
    (assignments, load)
}

fn hasher_for_seed_and_key(seed: u64, key: u64) -> DefaultHasher {
    let mut h = DefaultHasher::default();
    seed.hash(&mut h);
    let seed_state = h.finish();
    let mut h2 = DefaultHasher::default();
    seed_state.hash(&mut h2);
    key.hash(&mut h2);
    h2
}

struct Stats {
    sum: f64,
    sum_sq: f64,
    count: f64,
}

impl Stats {
    fn new() -> Self {
        Self {
            sum: 0.0,
            sum_sq: 0.0,
            count: 0.0,
        }
    }

    fn push(&mut self, x: f64) {
        self.sum += x;
        self.sum_sq += x * x;
        self.count += 1.0;
    }

    fn mean(&self) -> f64 {
        self.sum / self.count
    }

    fn stddev(&self) -> f64 {
        (self.sum_sq / self.count - self.mean().powi(2))
            .max(0.0)
            .sqrt()
    }
}

fn run(num_tokens: usize, k: usize, n: usize, num_seeds: u64) {
    let total = num_tokens * k;
    let cap = total.div_ceil(n);

    println!("Parameters: {num_tokens} tokens, k={k} replicas, {n} machines, {num_seeds} seeds");
    println!("Total assignments: {total},  capacity cap per machine: {cap}");
    println!(
        "Perfect balance: {}×{} + {}×{}",
        n - total % n,
        total / n,
        total % n,
        total / n + 1
    );
    println!();

    let mut ub_spread = Stats::new();
    let mut b_spread = Stats::new();
    let mut ub_changes = Stats::new();
    let mut b_changes = Stats::new();

    for seed in 0..num_seeds {
        // ── Unbounded ────────────────────────────────────────────────────
        let make_iters = |n| {
            (0..num_tokens as u64)
                .map(move |key| ConsistentChooseKHasher::new(hasher_for_seed_and_key(seed, key), n))
        };
        let (unbounded, ub_load) = bounded_load_assign(make_iters(n), k, n, usize::MAX);
        let ub_min = *ub_load.iter().min().unwrap();
        let ub_max = *ub_load.iter().max().unwrap();
        ub_spread.push((ub_max - ub_min) as f64);

        // ── Bounded ──────────────────────────────────────────────────────
        let (bounded, b_load) = bounded_load_assign(make_iters(n), k, n, cap);
        let b_min = *b_load.iter().min().unwrap();
        let b_max = *b_load.iter().max().unwrap();
        b_spread.push((b_max - b_min) as f64);

        // ── Consistency: add one machine ─────────────────────────────────
        let n2 = n + 1;
        let cap2 = total.div_ceil(n2);

        let (unbounded2, _) = bounded_load_assign(make_iters(n2), k, n2, usize::MAX);
        let mut ub_chg = 0usize;
        for (before, after) in unbounded.iter().zip(unbounded2.iter()) {
            for node in before {
                if !after.contains(node) {
                    ub_chg += 1;
                }
            }
        }
        ub_changes.push(ub_chg as f64 / total as f64 * 100.0);

        let (bounded2, _) = bounded_load_assign(make_iters(n2), k, n2, cap2);
        let mut b_chg = 0usize;
        for (before, after) in bounded.iter().zip(bounded2.iter()) {
            for node in before {
                if !after.contains(node) {
                    b_chg += 1;
                }
            }
        }
        b_changes.push(b_chg as f64 / total as f64 * 100.0);
    }

    println!(
        "{:<24} {:>16} {:>16}",
        "", "Unbounded", "Bounded"
    );
    println!("{:-<24} {:->16} {:->16}", "", "", "");
    println!(
        "{:<24} {:>11.2} ± {:<5.2} {:>10.2} ± {:<5.2}",
        "Load spread (max-min)",
        ub_spread.mean(),
        ub_spread.stddev(),
        b_spread.mean(),
        b_spread.stddev(),
    );
    println!(
        "{:<24} {:>10.2}% ± {:<5.2} {:>9.2}% ± {:<5.2}",
        "Churn on n→n+1",
        ub_changes.mean(),
        ub_changes.stddev(),
        b_changes.mean(),
        b_changes.stddev(),
    );
    println!(
        "\n  ideal churn: {:.2}%",
        1.0 / (n + 1) as f64 * 100.0
    );
}

fn main() {
    let configs: &[(usize, usize, usize)] = &[
        // (num_tokens, k, n)
        (64, 3, 24),  // original
        (256, 3, 24), // more tokens, same k and n
        (64, 1, 24),  // k=1 (no replication)
        (64, 5, 24),  // higher replication
        (64, 3, 8),   // fewer machines
        (64, 3, 60),  // many machines (sparse)
    ];
    let num_seeds = 1000;

    for (i, &(num_tokens, k, n)) in configs.iter().enumerate() {
        if i > 0 {
            println!("\n{}\n", "=".repeat(76));
        }
        run(num_tokens, k, n, num_seeds);
    }
}
