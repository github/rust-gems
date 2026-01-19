//! NN-Descent algorithm for approximate k-NN graph construction
//!
//! Based on: "Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures"
//! by Wei Dong, Charikar Moses, and Kai Li (2011)

use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashSet;

use crate::{AnnGraph, FamstConfig, Neighbor, NodeId};

/// Build reverse neighbor lists (who has me as a neighbor)
fn build_reverse(graph: &AnnGraph) -> Vec<Vec<NodeId>> {
    let n = graph.n();
    let mut reverse: Vec<Vec<NodeId>> = vec![Vec::new(); n];
    for i in 0..n {
        for neighbor in graph.neighbors(i) {
            reverse[neighbor.index as usize].push(i as NodeId);
        }
    }
    reverse
}

/// Initialize ANN graph with random neighbors
fn init_random_graph<T, D, R>(data: &[T], k: usize, distance_fn: &D, rng: &mut R) -> AnnGraph
where
    D: Fn(&T, &T) -> f32,
    R: Rng,
{
    let n = data.len();
    let mut graph_data: Vec<Neighbor> = Vec::with_capacity(n * k);

    for i in 0..n {
        let mut neighbors: Vec<Neighbor> = Vec::with_capacity(k);
        let mut seen: HashSet<NodeId> = HashSet::with_capacity(k);

        // Sample k random neighbors using Floyd's algorithm - guaranteed O(k)
        let effective_n = n - 1; // exclude self
        let range_start = effective_n.saturating_sub(k);
        for t in range_start..effective_n {
            let j = rng.gen_range(0..=t);
            // Map j to actual index, skipping i
            let actual_j = (if j >= i { j + 1 } else { j }) as NodeId;

            let selected = if seen.insert(actual_j) {
                actual_j
            } else {
                // j was already selected, so add t instead
                let actual_t = (if t >= i { t + 1 } else { t }) as NodeId;
                seen.insert(actual_t);
                actual_t
            };

            let d = distance_fn(&data[i], &data[selected as usize]);
            neighbors.push(Neighbor {
                index: selected,
                distance: d,
            });
        }

        // Sort by (distance, index) for total ordering
        neighbors.sort();
        graph_data.extend(neighbors);
    }

    AnnGraph::new(n, k, graph_data)
}

/// Try to insert a new neighbor into a sorted neighbor slice.
/// Returns true if the neighbor was inserted (better than the worst).
/// Assumes neighbors are sorted by (distance, index) for total ordering.
fn insert_neighbor(neighbors: &mut [Neighbor], new_index: NodeId, new_distance: f32) -> bool {
    let new_neighbor = Neighbor {
        index: new_index,
        distance: new_distance,
    };

    // Binary search using total ordering - also serves as existence check
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
pub(crate) fn nn_descent<T, D, R>(
    data: &[T],
    distance_fn: &D,
    config: &FamstConfig,
    rng: &mut R,
) -> AnnGraph
where
    D: Fn(&T, &T) -> f32,
    R: Rng,
{
    let n = data.len();
    let k = config.k.min(n - 1);

    if k == 0 || n <= 1 {
        return AnnGraph::new(n, 0, vec![]);
    }

    // Initialize ANN graph with random neighbors
    println!("Initializing random graph");
    let mut graph = init_random_graph(data, k, distance_fn, rng);

    // NN-Descent iterations
    for iter in 0..config.nn_descent_iterations {
        println!("NN-Descent iteration {iter}");
        let mut updates = 0;
        let reverse_neighbors = build_reverse(&graph);

        // For each point, explore neighbors of neighbors
        for i in 0..n {
            // Build set of current neighbors for O(1) lookup
            let current_neighbors: HashSet<NodeId> =
                graph.neighbors(i).iter().map(|nb| nb.index).collect();

            // Collect candidates: neighbors and reverse neighbors
            let mut candidates: Vec<NodeId> = Vec::new();

            // Sample from forward neighbors
            let mut sampled_forward: Vec<NodeId> =
                graph.neighbors(i).iter().map(|nb| nb.index).collect();
            let sample_size =
                ((sampled_forward.len() as f64 * config.nn_descent_sample_rate).ceil() as usize)
                    .max(1);
            sampled_forward.shuffle(rng);
            sampled_forward.truncate(sample_size);

            // Sample from reverse neighbors
            let mut sampled_reverse = reverse_neighbors[i].clone();
            let sample_size =
                ((sampled_reverse.len() as f64 * config.nn_descent_sample_rate).ceil() as usize)
                    .max(1);
            sampled_reverse.shuffle(rng);
            sampled_reverse.truncate(sample_size);

            // Neighbors of neighbors
            let i_id = i as NodeId;
            for &neighbor in sampled_forward.iter().chain(sampled_reverse.iter()) {
                for nb in graph.neighbors(neighbor as usize) {
                    if nb.index != i_id && !current_neighbors.contains(&nb.index) {
                        candidates.push(nb.index);
                    }
                }
                // Also check reverse neighbors of neighbors
                for &rn in &reverse_neighbors[neighbor as usize] {
                    if rn != i_id && !current_neighbors.contains(&rn) {
                        candidates.push(rn);
                    }
                }
            }

            // Deduplicate candidates
            candidates.sort_unstable();
            candidates.dedup();

            // Try to improve neighbors
            for c in candidates {
                let d = distance_fn(&data[i], &data[c as usize]);

                if insert_neighbor(graph.neighbors_mut(i), c, d) {
                    updates += 1;
                }
            }
        }

        // Early termination if no updates
        if updates == 0 {
            break;
        }
    }

    graph
}
