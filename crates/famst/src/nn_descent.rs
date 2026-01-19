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

/// Neighbor lists with bounded size k per node.
/// Uses flat storage: data[i*k..(i+1)*k] contains up to k neighbors of node i.
/// Uses reservoir sampling when more than k neighbors exist.
struct Neighbors {
    /// Flat storage of neighbor IDs
    data: Vec<u32>,
    /// Count of neighbors seen so far (for reservoir sampling)
    counts: Vec<u32>,
    /// Max neighbors per node
    k: usize,
}

impl Neighbors {
    fn new(n: usize, k: usize) -> Self {
        Neighbors {
            data: vec![0; n * k],
            counts: vec![0; n],
            k,
        }
    }

    /// Add a neighbor using reservoir sampling to maintain at most k neighbors.
    /// Skips if the neighbor is already present.
    #[inline]
    fn add(&mut self, node: usize, neighbor: u32, rng: &mut impl Rng) {
        let count = self.counts[node] as usize;
        let start = node * self.k;
        let filled = count.min(self.k);

        // Check if neighbor already exists in the filled portion
        if self.data[start..start + filled].contains(&neighbor) {
            return;
        }

        if count < self.k {
            // Still have room, just append
            self.data[start + count] = neighbor;
        } else {
            // Reservoir sampling: replace with probability k / (count + 1)
            let j = rng.gen_range(0..=count);
            if j < self.k {
                self.data[start + j] = neighbor;
            }
        }
        self.counts[node] += 1;
    }

    /// Get the neighbors of node i (only the filled slots)
    #[inline]
    fn get(&self, i: usize) -> &[u32] {
        let start = i * self.k;
        let count = (self.counts[i] as usize).min(self.k);
        &self.data[start..start + count]
    }
}

/// Build combined neighbor lists (forward + reverse) with reservoir sampling.
/// Returns (old_neighbors, new_neighbors), each with 2*k capacity per node.
/// Only marks neighbors that were selected into new_neighbors as old.
fn build_neighbor_lists(graph: &mut AnnGraph, rng: &mut impl Rng) -> (Neighbors, Neighbors) {
    let n = graph.n();
    let k = graph.k();
    // 2*k capacity: k for forward + k for reverse
    let mut old_neighbors = Neighbors::new(n, k * 2);
    let mut new_neighbors = Neighbors::new(n, k * 2);

    for i in 0..n {
        for neighbor in graph.neighbors(i) {
            let target = neighbor.index.index();
            if neighbor.index.is_new() {
                // Forward: i -> target, Reverse: target <- i
                new_neighbors.add(i, target, rng);
                new_neighbors.add(target as usize, i as u32, rng);
            } else {
                old_neighbors.add(i, target, rng);
                old_neighbors.add(target as usize, i as u32, rng);
            }
        }
    }

    // Only mark neighbors as old if they were selected into new_neighbors
    graph
        .neighbors_chunks_mut()
        .enumerate()
        .for_each(|(i, neighbors)| {
            for &selected_id in new_neighbors.get(i) {
                // Find this neighbor in the graph and mark as old if it's still new
                for nb in neighbors.iter_mut() {
                    if nb.index.index() == selected_id {
                        nb.index = nb.index.as_old();
                        break;
                    }
                }
            }
        });

    (old_neighbors, new_neighbors)
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
        println!("NN-Descent iteration {iter}...");
        // Build combined neighbor lists (forward + reverse, with reservoir sampling)
        // Also marks all neighbors as old for next iteration
        let (old_neighbors, new_neighbors) = build_neighbor_lists(&mut graph, rng);

        // For each point, generate candidates from neighbors of neighbors
        // Key optimization: only consider pairs where at least one is "new"
        let candidates: HashSet<(u32, u32)> = (0..n)
            .into_par_iter()
            .fold(
                HashSet::new,
                |mut local_candidates, i| {
                    let old_i = old_neighbors.get(i);
                    let new_i = new_neighbors.get(i);

                    // Skip if no new neighbors
                    if !new_i.is_empty() {
                        for &u in new_i {
                            for &v in new_i {
                                if u < v {
                                    local_candidates.insert((u, v));
                                }
                            }
                            for &v in old_i {
                                if u != v {
                                    local_candidates.insert((u.min(v), u.max(v)));
                                }
                            }
                        }
                    }
                    local_candidates
                },
            )
            .reduce(HashSet::new, |mut a, b| {
                a.extend(b);
                a
            });

        // Try to improve neighbors with candidates
        let mut updates = 0;
        for &(u, v) in &candidates {
            let d = distance_fn(&data[u as usize], &data[v as usize]);
            if insert_neighbor(graph.neighbors_mut(u as usize), NodeId::new(v), d) {
                updates += 1;
            }
            if insert_neighbor(graph.neighbors_mut(v as usize), NodeId::new(u), d) {
                updates += 1;
            }
        }

        println!("NN-Descent iteration {iter}: {updates} updates");

        // Early termination if no updates
        if updates == 0 {
            break;
        }
    }
    graph
}
