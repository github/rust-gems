//! NN-Descent algorithm for approximate k-NN graph construction
//!
//! Based on: "Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures"
//! by Wei Dong, Charikar Moses, and Kai Li (2011)
//!
//! Uses the "new/old" optimization from pynndescent: each neighbor is marked as "new"
//! when first added. Only pairs where at least one is "new" need to be compared.
//! The LSB of NodeId is used as the "new" flag.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashSet;

use crate::{AnnGraph, FamstConfig, Neighbor, NodeId};

/// Reverse neighbor lists with bounded size k per node.
/// Uses flat storage: data[i*k..(i+1)*k] contains up to k reverse neighbors of node i.
/// Uses reservoir sampling when more than k reverse edges exist.
struct ReverseNeighbors {
    /// Flat storage of reverse neighbor IDs (with new flag preserved)
    data: Vec<NodeId>,
    /// Count of reverse neighbors seen so far (for reservoir sampling)
    counts: Vec<u32>,
    /// Max reverse neighbors per node
    k: usize,
}

impl ReverseNeighbors {
    fn new(n: usize, k: usize) -> Self {
        ReverseNeighbors {
            data: vec![NodeId::new(0); n * k],
            counts: vec![0; n],
            k,
        }
    }

    /// Add a reverse edge: node `from` is a neighbor of node `to`, so `to` has reverse edge to `from`.
    /// Uses reservoir sampling to maintain at most k reverse neighbors.
    #[inline]
    fn add(&mut self, to: usize, from: NodeId, rng: &mut impl Rng) {
        let count = self.counts[to] as usize;
        let start = to * self.k;

        if count < self.k {
            // Still have room, just append
            self.data[start + count] = from;
        } else {
            // Reservoir sampling: replace with probability k / (count + 1)
            let j = rng.gen_range(0..=count);
            if j < self.k {
                self.data[start + j] = from;
            }
        }
        self.counts[to] += 1;
    }

    /// Get the reverse neighbors of node i (only the filled slots)
    #[inline]
    fn get(&self, i: usize) -> &[NodeId] {
        let start = i * self.k;
        let count = (self.counts[i] as usize).min(self.k);
        &self.data[start..start + count]
    }
}

/// Build reverse neighbor lists with reservoir sampling.
/// Returns separate old and new reverse neighbor structures.
fn build_reverse_lists(graph: &AnnGraph, rng: &mut impl Rng) -> (ReverseNeighbors, ReverseNeighbors) {
    let n = graph.n();
    let k = graph.k();
    let mut old_reverse = ReverseNeighbors::new(n, k);
    let mut new_reverse = ReverseNeighbors::new(n, k);

    for i in 0..n {
        let i_id = NodeId::new(i as u32);
        for neighbor in graph.neighbors(i) {
            let target = neighbor.index.index() as usize;
            if neighbor.index.is_new() {
                new_reverse.add(target, i_id, rng);
            } else {
                old_reverse.add(target, i_id, rng);
            }
        }
    }
    (old_reverse, new_reverse)
}

/// Initialize ANN graph with random neighbors
fn init_random_graph<T, D, R>(data: &[T], k: usize, distance_fn: &D, rng: &mut R) -> AnnGraph
where
    T: Sync,
    D: Fn(&T, &T) -> f32 + Sync,
    R: Rng,
{
    let n = data.len();

    // Generate seeds for per-thread RNGs
    let seeds: Vec<u64> = (0..n).map(|_| rng.r#gen()).collect();

    let graph_data: Vec<Neighbor> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let mut local_rng = SmallRng::seed_from_u64(seeds[i]);
            let mut neighbors: Vec<Neighbor> = Vec::with_capacity(k);
            let mut seen: HashSet<u32> = HashSet::with_capacity(k);

            // Sample k random neighbors using Floyd's algorithm - guaranteed O(k)
            let effective_n = n - 1; // exclude self
            let range_start = effective_n.saturating_sub(k);
            for t in range_start..effective_n {
                let j = local_rng.gen_range(0..=t);
                // Map j to actual index, skipping i
                let actual_j = (if j >= i { j + 1 } else { j }) as u32;

                let selected = if seen.insert(actual_j) {
                    actual_j
                } else {
                    // j was already selected, so add t instead
                    let actual_t = (if t >= i { t + 1 } else { t }) as u32;
                    seen.insert(actual_t);
                    actual_t
                };

                let d = distance_fn(&data[i], &data[selected as usize]);
                // Mark as new (not yet used for candidate generation)
                neighbors.push(Neighbor {
                    index: NodeId::new(selected).as_new(),
                    distance: d,
                });
            }

            // Sort by (distance, index) for total ordering
            neighbors.sort();
            neighbors
        })
        .collect();

    AnnGraph::new(n, k, graph_data)
}

/// Try to insert a new neighbor into a sorted neighbor slice.
/// Returns true if the neighbor was inserted (better than the worst).
/// Assumes neighbors are sorted by (distance, index) for total ordering.
/// The new flag (LSB) doesn't affect ordering since NodeId::cmp ignores it.
fn insert_neighbor(neighbors: &mut [Neighbor], new_index: NodeId, new_distance: f32) -> bool {
    // Create a search key with the new flag set (new insertions are always "new")
    let new_neighbor = Neighbor {
        index: new_index.as_new(),
        distance: new_distance,
    };

    // Binary search by (distance, index) - NodeId comparison ignores new flag
    // so this will find the node regardless of its new/old status
    match neighbors.binary_search(&new_neighbor) {
        Ok(_) => false, // Already exists
        Err(insert_pos) => {
            // Check if better than worst (last element)
            if insert_pos >= neighbors.len() {
                return false;
            }

            // Shift elements to make room (dropping the last/worst)
            for j in (insert_pos + 1..neighbors.len()).rev() {
                neighbors[j] = neighbors[j - 1];
            }

            neighbors[insert_pos] = new_neighbor;
            true
        }
    }
}

/// NN-Descent algorithm for approximate k-NN graph construction
///
/// Based on: "Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures"
/// by Wei Dong, Charikar Moses, and Kai Li (2011)
///
/// Uses the "new/old" optimization: only compare pairs where at least one neighbor is "new".
pub(crate) fn nn_descent<T, D, R>(
    data: &[T],
    distance_fn: &D,
    config: &FamstConfig,
    rng: &mut R,
) -> AnnGraph
where
    T: Sync,
    D: Fn(&T, &T) -> f32 + Sync,
    R: Rng,
{
    let n = data.len();
    let k = config.k.min(n - 1);

    if k == 0 || n <= 1 {
        return AnnGraph::new(n, 0, vec![]);
    }

    // Initialize ANN graph with random neighbors (all marked as "new")
    println!("Initializing random graph");
    let mut graph = init_random_graph(data, k, distance_fn, rng);

    // NN-Descent iterations
    for iter in 0..config.nn_descent_iterations {
        // Build reverse neighbor lists, separating old and new (with reservoir sampling)
        let (old_reverse, new_reverse) = build_reverse_lists(&graph, rng);

        // For each point, collect old and new forward neighbors
        let mut old_neighbors: Vec<Vec<NodeId>> = vec![Vec::new(); n];
        let mut new_neighbors: Vec<Vec<NodeId>> = vec![Vec::new(); n];

        for i in 0..n {
            for nb in graph.neighbors(i) {
                let idx = nb.index.as_old(); // Strip new flag for storage
                if nb.index.is_new() {
                    new_neighbors[i].push(idx);
                } else {
                    old_neighbors[i].push(idx);
                }
            }
        }

        // Mark all neighbors as old for next iteration
        for i in 0..n {
            for nb in graph.neighbors_mut(i) {
                nb.index = nb.index.as_old();
            }
        }

        let mut updates = 0;

        // For each point, generate candidates from neighbors of neighbors
        // Key optimization: only consider pairs where at least one is "new"
        for i in 0..n {
            let i_id = NodeId::new(i as u32);

            // Combine forward and reverse neighbors
            let old_i: Vec<NodeId> = old_neighbors[i]
                .iter()
                .chain(old_reverse.get(i).iter())
                .copied()
                .collect();
            let new_i: Vec<NodeId> = new_neighbors[i]
                .iter()
                .chain(new_reverse.get(i).iter())
                .copied()
                .collect();

            // Skip if no new neighbors
            if new_i.is_empty() {
                continue;
            }

            // Build set of current neighbors for O(1) lookup
            let current_neighbors: HashSet<NodeId> = graph
                .neighbors(i)
                .iter()
                .map(|nb| nb.index.as_old())
                .collect();

            let mut candidates: HashSet<NodeId> = HashSet::new();

            // new-new pairs: for each new neighbor, look at their new neighbors
            for &u in &new_i {
                let u_idx = u.index() as usize;
                for &v in &new_neighbors[u_idx] {
                    if v != i_id && !current_neighbors.contains(&v) {
                        candidates.insert(v);
                    }
                }
                for v in new_reverse.get(u_idx) {
                    if *v != i_id && !current_neighbors.contains(v) {
                        candidates.insert(*v);
                    }
                }
            }

            // new-old pairs: for each new neighbor, look at their old neighbors
            for &u in &new_i {
                let u_idx = u.index() as usize;
                for &v in &old_neighbors[u_idx] {
                    if v != i_id && !current_neighbors.contains(&v) {
                        candidates.insert(v);
                    }
                }
                for v in old_reverse.get(u_idx) {
                    if *v != i_id && !current_neighbors.contains(v) {
                        candidates.insert(*v);
                    }
                }
            }

            // old-new pairs: for each old neighbor, look at their new neighbors
            for &u in &old_i {
                let u_idx = u.index() as usize;
                for &v in &new_neighbors[u_idx] {
                    if v != i_id && !current_neighbors.contains(&v) {
                        candidates.insert(v);
                    }
                }
                for v in new_reverse.get(u_idx) {
                    if *v != i_id && !current_neighbors.contains(v) {
                        candidates.insert(*v);
                    }
                }
            }

            // Try to improve neighbors with candidates
            for c in candidates {
                let d = distance_fn(&data[i], &data[c.index() as usize]);
                if insert_neighbor(graph.neighbors_mut(i), c, d) {
                    updates += 1;
                }
            }
        }

        println!("NN-Descent iteration {iter}: {updates} updates");

        // Early termination if no updates
        if updates == 0 {
            break;
        }
    }

    // Strip the new flag from all neighbors before returning
    for i in 0..n {
        for nb in graph.neighbors_mut(i) {
            nb.index = nb.index.as_old();
        }
    }

    graph
}
