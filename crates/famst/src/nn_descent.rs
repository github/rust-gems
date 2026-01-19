//! NN-Descent algorithm for approximate k-NN graph construction
//!
//! Based on: "Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures"
//! by Wei Dong, Charikar Moses, and Kai Li (2011)

use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::BinaryHeap;

use crate::{AnnGraph, FamstConfig, Neighbor, NodeId};

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

    // Helper: check if sorted vec contains value
    fn sorted_contains(v: &[NodeId], x: NodeId) -> bool {
        v.binary_search(&x).is_ok()
    }

    // Helper: insert into sorted vec, returns true if inserted (was not present)
    fn sorted_insert(v: &mut Vec<NodeId>, x: NodeId) -> bool {
        match v.binary_search(&x) {
            Ok(_) => false,
            Err(pos) => {
                v.insert(pos, x);
                true
            }
        }
    }

    // Helper: remove from sorted vec
    fn sorted_remove(v: &mut Vec<NodeId>, x: NodeId) {
        if let Ok(pos) = v.binary_search(&x) {
            v.remove(pos);
        }
    }

    // Initialize with random neighbors using max-heap for each point
    // neighbor_lists[i] is kept sorted by index for O(log k) membership tests
    let mut heaps: Vec<BinaryHeap<Neighbor>> = Vec::with_capacity(n);
    let mut neighbor_lists: Vec<Vec<NodeId>> = vec![Vec::with_capacity(k); n];

    for i in 0..n {
        let mut heap = BinaryHeap::with_capacity(k);

        // Sample k random neighbors using Floyd's algorithm - guaranteed O(k)
        // https://fermatslibrary.com/s/a-sample-of-brilliance
        // This selects k distinct elements from 0..n, excluding i
        let effective_n = n - 1; // exclude self
        let range_start = effective_n.saturating_sub(k);
        for t in range_start..effective_n {
            let j = rng.gen_range(0..=t);
            // Map j to actual index, skipping i
            let actual_j = (if j >= i { j + 1 } else { j }) as NodeId;

            if !sorted_insert(&mut neighbor_lists[i], actual_j) {
                // j was already selected, so add t instead
                let actual_t = (if t >= i { t + 1 } else { t }) as NodeId;
                sorted_insert(&mut neighbor_lists[i], actual_t);
                let d = distance_fn(&data[i], &data[actual_t as usize]);
                heap.push(Neighbor {
                    index: actual_t,
                    distance: d,
                });
            } else {
                let d = distance_fn(&data[i], &data[actual_j as usize]);
                heap.push(Neighbor {
                    index: actual_j,
                    distance: d,
                });
            }
        }
        heaps.push(heap);
    }

    // Build reverse neighbor lists (who has me as a neighbor)
    // Returns sorted vecs for each point
    let build_reverse = |neighbor_lists: &[Vec<NodeId>]| -> Vec<Vec<NodeId>> {
        let mut reverse: Vec<Vec<NodeId>> = vec![Vec::new(); n];
        for (i, neighbors) in neighbor_lists.iter().enumerate() {
            for &j in neighbors {
                reverse[j as usize].push(i as NodeId);
            }
        }
        // Sort each reverse list (they're built in order of i, so already sorted)
        reverse
    };

    // NN-Descent iterations
    for _ in 0..config.nn_descent_iterations {
        let mut updates = 0;
        let reverse_neighbors = build_reverse(&neighbor_lists);

        // For each point, explore neighbors of neighbors
        for i in 0..n {
            // Collect candidates: neighbors and reverse neighbors
            let mut candidates: Vec<NodeId> = Vec::new();

            // Sample from forward neighbors
            let mut sampled_forward = neighbor_lists[i].clone();
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
                for &nn in &neighbor_lists[neighbor as usize] {
                    if nn != i_id && !sorted_contains(&neighbor_lists[i], nn) {
                        candidates.push(nn);
                    }
                }
                // Also check reverse neighbors of neighbors
                for &rn in &reverse_neighbors[neighbor as usize] {
                    if rn != i_id && !sorted_contains(&neighbor_lists[i], rn) {
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

                // Check if this is better than the worst current neighbor
                if let Some(worst) = heaps[i].peek() {
                    if d < worst.distance {
                        // Remove worst and add new neighbor
                        let removed = heaps[i].pop().unwrap();
                        sorted_remove(&mut neighbor_lists[i], removed.index);

                        heaps[i].push(Neighbor { index: c, distance: d });
                        sorted_insert(&mut neighbor_lists[i], c);
                        updates += 1;
                    }
                }
            }
        }

        // Early termination if no updates
        if updates == 0 {
            break;
        }
    }

    // Convert heaps to flat neighbor array sorted by distance
    let mut result_data = Vec::with_capacity(n * k);

    for heap in heaps {
        let mut entries: Vec<Neighbor> = heap.into_vec();
        entries.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        result_data.extend(entries);
    }

    AnnGraph::new(n, k, result_data)
}
