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
use std::rc::Rc;

use consistent_hashing::ConsistentChooseKHasher;

/// Bounded-load assignment.
///
/// Each token claims all k replicas before moving to the next token,
/// skipping any node that has reached `max_load`.
fn bounded_load_assign<I: Iterator<Item = usize>>(
    iters: impl IntoIterator<Item = I>,
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

/// Count the number of assignments that changed between two runs.
fn count_churn(before: &[Vec<usize>], after: &[Vec<usize>]) -> usize {
    before
        .iter()
        .zip(after.iter())
        .map(|(b, a)| b.iter().filter(|node| !a.contains(node)).count())
        .sum()
}

/// Load spread: difference between max and min loaded nodes.
fn load_spread(load: &[usize]) -> usize {
    load.iter().max().unwrap() - load.iter().min().unwrap()
}

/// A hash ring with `v` virtual nodes per physical node.
struct HashRing {
    ring: Rc<Vec<(u64, usize)>>,
}

impl HashRing {
    fn new(seed: u64, n: usize, v: usize) -> Self {
        let mut ring: Vec<(u64, usize)> = (0..n)
            .flat_map(|node| {
                (0..v).map(move |vi| {
                    let mut h = DefaultHasher::default();
                    seed.hash(&mut h);
                    node.hash(&mut h);
                    vi.hash(&mut h);
                    (h.finish(), node)
                })
            })
            .collect();
        ring.sort_unstable_by_key(|&(pos, _)| pos);
        Self { ring: Rc::new(ring) }
    }

    /// Return an iterator over distinct physical nodes for the given token hash,
    /// walking clockwise from the token's position on the ring.
    fn iter(&self, token_hash: u64) -> HashRingIter {
        let start = self.ring.partition_point(|&(pos, _)| pos < token_hash);
        HashRingIter {
            ring: Rc::clone(&self.ring),
            start,
            offset: 0,
            seen: Vec::new(),
        }
    }
}

/// Iterator that walks a hash ring clockwise, yielding distinct physical nodes.
struct HashRingIter {
    ring: Rc<Vec<(u64, usize)>>,
    start: usize,
    offset: usize,
    seen: Vec<usize>,
}

impl Iterator for HashRingIter {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        while self.offset < self.ring.len() {
            let (_, node) = self.ring[(self.start + self.offset) % self.ring.len()];
            self.offset += 1;
            if !self.seen.contains(&node) {
                self.seen.push(node);
                return Some(node);
            }
        }
        None
    }
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

const VIRTUAL_NODES: usize = 200;

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
    let mut ring_spread = Stats::new();
    let mut ub_changes = Stats::new();
    let mut b_changes = Stats::new();
    let mut ring_changes = Stats::new();

    for seed in 0..num_seeds {
        // ── Choose-k (unbounded) ─────────────────────────────────────────
        let make_iters = |n| {
            (0..num_tokens as u64)
                .map(move |key| ConsistentChooseKHasher::new(hasher_for_seed_and_key(seed, key), n))
        };
        let (unbounded, ub_load) = bounded_load_assign(make_iters(n), k, n, usize::MAX);
        ub_spread.push(load_spread(&ub_load) as f64);

        // ── Choose-k (bounded) ───────────────────────────────────────────
        let (bounded, b_load) = bounded_load_assign(make_iters(n), k, n, cap);
        b_spread.push(load_spread(&b_load) as f64);

        // ── Hash ring (bounded) ──────────────────────────────────────────
        let ring = HashRing::new(seed, n, VIRTUAL_NODES);
        let (ring_assign, r_load) = bounded_load_assign(
            (0..num_tokens as u64)
                .map(|key| ring.iter(hasher_for_seed_and_key(seed, key).finish())),
            k,
            n,
            cap,
        );
        ring_spread.push(load_spread(&r_load) as f64);

        // ── Consistency: add one machine ─────────────────────────────────
        let n2 = n + 1;
        let cap2 = total.div_ceil(n2);

        let (unbounded2, _) = bounded_load_assign(make_iters(n2), k, n2, usize::MAX);
        ub_changes.push(count_churn(&unbounded, &unbounded2) as f64 / total as f64 * 100.0);

        let (bounded2, _) = bounded_load_assign(make_iters(n2), k, n2, cap2);
        b_changes.push(count_churn(&bounded, &bounded2) as f64 / total as f64 * 100.0);

        let ring2 = HashRing::new(seed, n2, VIRTUAL_NODES);
        let (ring_assign2, _) = bounded_load_assign(
            (0..num_tokens as u64)
                .map(|key| ring2.iter(hasher_for_seed_and_key(seed, key).finish())),
            k,
            n2,
            cap2,
        );
        ring_changes.push(count_churn(&ring_assign, &ring_assign2) as f64 / total as f64 * 100.0);
    }

    println!(
        "{:<24} {:>16} {:>16} {:>16}",
        "", "Choose-k", "Bounded", "Ring Bounded"
    );
    println!("{:-<24} {:->16} {:->16} {:->16}", "", "", "", "");
    println!(
        "{:<24} {:>11.2} ± {:<5.2} {:>10.2} ± {:<5.2} {:>10.2} ± {:<5.2}",
        "Load spread (max-min)",
        ub_spread.mean(),
        ub_spread.stddev(),
        b_spread.mean(),
        b_spread.stddev(),
        ring_spread.mean(),
        ring_spread.stddev(),
    );
    println!(
        "{:<24} {:>10.2}% ± {:<5.2} {:>9.2}% ± {:<5.2} {:>9.2}% ± {:<5.2}",
        "Churn on n→n+1",
        ub_changes.mean(),
        ub_changes.stddev(),
        b_changes.mean(),
        b_changes.stddev(),
        ring_changes.mean(),
        ring_changes.stddev(),
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
