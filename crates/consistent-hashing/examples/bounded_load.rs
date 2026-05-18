//! Bounded-load consistent hashing example.
//!
//! Pure consistent hashing selects each node with equal probability, but for
//! small workloads (e.g. 64 tokens across 24 machines) random variance causes
//! highly skewed assignments. This example layers a capacity cap on top of
//! ConsistentChooseK to enforce near-perfect balance.
//!
//! Assignment uses round-robin over replicas: first assign every token's
//! most-preferred machine, then every token's second-preferred, etc. This
//! ensures all tokens compete fairly for each replica round.
//!
//! Run with:  cargo run --example bounded_load

use std::hash::{DefaultHasher, Hash};

use consistent_hashing::ConsistentChooseKHasher;

/// Round-robin bounded-load assignment.
///
/// For each replica round r = 0..k, iterate over all tokens and assign each
/// to its next most-preferred node that still has capacity. This gives every
/// token equal priority within each round.
fn bounded_load_assign(
    rankings: &[Vec<usize>],
    k: usize,
    n: usize,
    max_load: usize,
) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut load = vec![0usize; n];
    let num_tokens = rankings.len();
    let mut assignments = vec![Vec::with_capacity(k); num_tokens];
    let mut cursors = vec![0usize; num_tokens];

    for _round in 0..k {
        for (token, ranking) in rankings.iter().enumerate() {
            while cursors[token] < ranking.len() {
                let node = ranking[cursors[token]];
                cursors[token] += 1;
                if load[node] < max_load {
                    load[node] += 1;
                    assignments[token].push(node);
                    break;
                }
            }
        }
    }
    (assignments, load)
}

fn main() {
    let num_tokens: usize = 64;
    let k: usize = 2; // replicas per token
    let n: usize = 24; // machines
    let total = num_tokens * k;
    let cap = total.div_ceil(n); // ceil(128/24) = 6

    println!("Parameters: {num_tokens} tokens, k={k} replicas, {n} machines");
    println!("Total assignments: {total},  capacity cap per machine: {cap}");
    println!(
        "Perfect balance: {}×{} + {}×{}\n",
        n - total % n,
        total / n,
        total % n,
        total / n + 1
    );

    // ── Unbounded ────────────────────────────────────────────────────────
    let unbounded: Vec<Vec<usize>> = (0..num_tokens as u64)
        .map(|key| {
            let mut h = DefaultHasher::default();
            key.hash(&mut h);
            ConsistentChooseKHasher::new(h, n).take(k).collect()
        })
        .collect();
    let mut unbounded_load = vec![0usize; n];
    for a in &unbounded {
        for &node in a {
            unbounded_load[node] += 1;
        }
    }

    // ── Bounded (round-robin) ────────────────────────────────────────────
    let rankings: Vec<Vec<usize>> = (0..num_tokens as u64)
        .map(|key| {
            let mut h = DefaultHasher::default();
            key.hash(&mut h);
            ConsistentChooseKHasher::new(h, n).collect()
        })
        .collect();
    let (bounded, bounded_load) = bounded_load_assign(&rankings, k, n, cap);

    // ── Display ──────────────────────────────────────────────────────────
    println!("{:<12} {:>10} {:>10}", "Machine", "Unbounded", "Bounded");
    println!("{:-<12} {:->10} {:->10}", "", "", "");
    for i in 0..n {
        println!(
            "{:<12} {:>10} {:>10}",
            i, unbounded_load[i], bounded_load[i]
        );
    }

    let ub_min = *unbounded_load.iter().min().unwrap();
    let ub_max = *unbounded_load.iter().max().unwrap();
    let b_min = *bounded_load.iter().min().unwrap();
    let b_max = *bounded_load.iter().max().unwrap();
    println!("{:-<12} {:->10} {:->10}", "", "", "");
    println!(
        "{:<12} {:>10} {:>10}",
        "spread",
        ub_max - ub_min,
        b_max - b_min
    );

    // ── Consistency check: what happens when we add one machine? ─────────
    let n2 = n + 1;
    let cap2 = (num_tokens * k).div_ceil(n2);
    let rankings2: Vec<Vec<usize>> = (0..num_tokens as u64)
        .map(|key| {
            let mut h = DefaultHasher::default();
            key.hash(&mut h);
            ConsistentChooseKHasher::new(h, n2).collect()
        })
        .collect();
    let (bounded2, _) = bounded_load_assign(&rankings2, k, n2, cap2);

    let mut changes = 0;
    for (before, after) in bounded.iter().zip(bounded2.iter()) {
        for node in before {
            if !after.contains(node) {
                changes += 1;
            }
        }
    }
    println!("\nConsistency: adding machine {n} → {n2}");
    println!(
        "  {changes}/{total} assignments changed ({:.1}%),  ideal ≈ {:.1}%",
        changes as f64 / total as f64 * 100.0,
        k as f64 / n2 as f64 * 100.0
    );
}
