//! FAMST: Fast Approximate Minimum Spanning Tree
//!
//! Implementation of the FAMST algorithm from:
//! "FAMST: Fast Approximate Minimum Spanning Tree Construction for Large-Scale
//! and High-Dimensional Data" (Almansoori & Telek, 2025)
//!
//! The algorithm uses three phases:
//! 1. ANN graph construction using NN-Descent
//! 2. Component analysis and connection with random edges
//! 3. Iterative edge refinement
//!
//! Generic over data type `T` and distance function.

use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// An edge in the MST, represented as (node_a, node_b, distance)
#[derive(Debug, Clone)]
pub struct Edge {
    pub u: usize,
    pub v: usize,
    pub distance: f64,
}

impl Edge {
    pub fn new(u: usize, v: usize, distance: f64) -> Self {
        Edge { u, v, distance }
    }
}

/// Union-Find (Disjoint Set Union) data structure for Kruskal's algorithm
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // Path compression
        }
        self.parent[x]
    }

    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let px = self.find(x);
        let py = self.find(y);
        if px == py {
            return false;
        }
        // Union by rank
        match self.rank[px].cmp(&self.rank[py]) {
            std::cmp::Ordering::Less => self.parent[px] = py,
            std::cmp::Ordering::Greater => self.parent[py] = px,
            std::cmp::Ordering::Equal => {
                self.parent[py] = px;
                self.rank[px] += 1;
            }
        }
        true
    }
}

/// Approximate Nearest Neighbors graph representation
/// Contains neighbor indices and distances for each point
pub struct AnnGraph {
    /// neighbors[i] contains the indices of k nearest neighbors of point i
    pub neighbors: Vec<Vec<usize>>,
    /// distances[i] contains the distances to k nearest neighbors of point i
    pub distances: Vec<Vec<f64>>,
}

impl AnnGraph {
    pub fn new(neighbors: Vec<Vec<usize>>, distances: Vec<Vec<f64>>) -> Self {
        assert_eq!(neighbors.len(), distances.len());
        AnnGraph {
            neighbors,
            distances,
        }
    }

    pub fn n(&self) -> usize {
        self.neighbors.len()
    }
}

/// FAMST algorithm configuration
pub struct FamstConfig {
    /// Number of nearest neighbors (k in k-NN graph)
    pub k: usize,
    /// Number of random edges per component pair (λ in the paper)
    pub lambda: usize,
    /// Maximum refinement iterations (0 for unlimited until convergence)
    pub max_iterations: usize,
    /// Maximum NN-Descent iterations
    pub nn_descent_iterations: usize,
    /// Sample rate for NN-Descent (fraction of neighbors to sample)
    pub nn_descent_sample_rate: f64,
}

impl Default for FamstConfig {
    fn default() -> Self {
        FamstConfig {
            k: 20,
            lambda: 5,
            max_iterations: 100,
            nn_descent_iterations: 10,
            nn_descent_sample_rate: 0.5,
        }
    }
}

/// Result of FAMST algorithm
pub struct FamstResult {
    /// MST edges
    pub edges: Vec<Edge>,
    /// Total weight of the MST
    pub total_weight: f64,
}

/// Main FAMST algorithm implementation
///
/// Generic over:
/// - `T`: The data type stored at each point
/// - `D`: Distance function `Fn(&T, &T) -> f64`
///
/// # Arguments
/// * `data` - Slice of data points
/// * `distance_fn` - Function to compute distance between two points
/// * `config` - Algorithm configuration
///
/// # Returns
/// The approximate MST as a list of edges
pub fn famst<T, D>(data: &[T], distance_fn: D, config: &FamstConfig) -> FamstResult
where
    D: Fn(&T, &T) -> f64,
{
    famst_with_rng(data, distance_fn, config, &mut rand::thread_rng())
}

/// FAMST with custom RNG for reproducibility
pub fn famst_with_rng<T, D, R>(
    data: &[T],
    distance_fn: D,
    config: &FamstConfig,
    rng: &mut R,
) -> FamstResult
where
    D: Fn(&T, &T) -> f64,
    R: Rng,
{
    let n = data.len();
    if n == 0 {
        return FamstResult {
            edges: vec![],
            total_weight: 0.0,
        };
    }
    if n == 1 {
        return FamstResult {
            edges: vec![],
            total_weight: 0.0,
        };
    }

    // Phase 1: Build ANN graph using NN-Descent
    let ann_graph = nn_descent(data, &distance_fn, config, rng);

    // Phase 2: Build undirected graph and find connected components
    let (undirected_graph, components) = find_components(&ann_graph);

    // If only one component, skip inter-component edge logic
    println!("components {}", components.len());
    if components.len() <= 1 {
        let edges = extract_mst_from_ann(&ann_graph, n);
        let total_weight = edges.iter().map(|e| e.distance).sum();
        return FamstResult {
            edges,
            total_weight,
        };
    }

    // Phase 2 continued: Add random edges between components
    let (mut inter_edges, edge_components) =
        add_random_edges(data, &components, config.lambda, &distance_fn, rng);

    // Phase 3: Iterative edge refinement
    let mut iterations = 0;
    loop {
        let (refined_edges, changes) = refine_edges(
            data,
            &undirected_graph,
            &components,
            &inter_edges,
            &edge_components,
            &distance_fn,
        );
        inter_edges = refined_edges;

        if changes == 0 {
            break;
        }

        iterations += 1;
        if config.max_iterations > 0 && iterations >= config.max_iterations {
            break;
        }
    }

    // Phase 4: Extract MST using Kruskal's algorithm
    let edges = extract_mst(&ann_graph, &inter_edges, n);
    let total_weight = edges.iter().map(|e| e.distance).sum();

    FamstResult {
        edges,
        total_weight,
    }
}

/// A neighbor entry in the k-NN heap (max-heap by distance for easy replacement of farthest)
#[derive(Clone, Copy)]
struct NeighborEntry {
    index: usize,
    distance: f64,
}

impl PartialEq for NeighborEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for NeighborEntry {}

impl PartialOrd for NeighborEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NeighborEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: larger distances have higher priority
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// NN-Descent algorithm for approximate k-NN graph construction
///
/// Based on: "Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures"
/// by Wei Dong, Charikar Moses, and Kai Li (2011)
fn nn_descent<T, D, R>(data: &[T], distance_fn: &D, config: &FamstConfig, rng: &mut R) -> AnnGraph
where
    D: Fn(&T, &T) -> f64,
    R: Rng,
{
    let n = data.len();
    let k = config.k.min(n - 1);

    if k == 0 || n <= 1 {
        return AnnGraph::new(vec![vec![]; n], vec![vec![]; n]);
    }

    // Initialize with random neighbors using max-heap for each point
    let mut heaps: Vec<BinaryHeap<NeighborEntry>> = Vec::with_capacity(n);
    let mut neighbor_sets: Vec<HashSet<usize>> = vec![HashSet::with_capacity(k); n];

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
            let actual_j = if j >= i { j + 1 } else { j };

            if !neighbor_sets[i].insert(actual_j) {
                // j was already selected, so add t instead
                let actual_t = if t >= i { t + 1 } else { t };
                neighbor_sets[i].insert(actual_t);
                let d = distance_fn(&data[i], &data[actual_t]);
                heap.push(NeighborEntry {
                    index: actual_t,
                    distance: d,
                });
            } else {
                let d = distance_fn(&data[i], &data[actual_j]);
                heap.push(NeighborEntry {
                    index: actual_j,
                    distance: d,
                });
            }
        }
        heaps.push(heap);
    }

    // Build reverse neighbor lists (who has me as a neighbor)
    let build_reverse = |neighbor_sets: &[HashSet<usize>]| -> Vec<HashSet<usize>> {
        let mut reverse: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for (i, neighbors) in neighbor_sets.iter().enumerate() {
            for &j in neighbors {
                reverse[j].insert(i);
            }
        }
        reverse
    };

    // NN-Descent iterations
    for _ in 0..config.nn_descent_iterations {
        let mut updates = 0;
        let reverse_neighbors = build_reverse(&neighbor_sets);

        // For each point, explore neighbors of neighbors
        for i in 0..n {
            // Collect candidates: neighbors and reverse neighbors
            let mut candidates: Vec<usize> = Vec::new();

            // Sample from forward neighbors
            let forward: Vec<usize> = neighbor_sets[i].iter().copied().collect();
            let sample_size =
                ((forward.len() as f64 * config.nn_descent_sample_rate).ceil() as usize).max(1);
            let mut sampled_forward = forward.clone();
            sampled_forward.shuffle(rng);
            sampled_forward.truncate(sample_size);

            // Sample from reverse neighbors
            let reverse: Vec<usize> = reverse_neighbors[i].iter().copied().collect();
            let sample_size =
                ((reverse.len() as f64 * config.nn_descent_sample_rate).ceil() as usize).max(1);
            let mut sampled_reverse = reverse.clone();
            sampled_reverse.shuffle(rng);
            sampled_reverse.truncate(sample_size);

            // Neighbors of neighbors
            for &neighbor in sampled_forward.iter().chain(sampled_reverse.iter()) {
                for &nn in &neighbor_sets[neighbor] {
                    if nn != i && !neighbor_sets[i].contains(&nn) {
                        candidates.push(nn);
                    }
                }
                // Also check reverse neighbors of neighbors
                for &rn in &reverse_neighbors[neighbor] {
                    if rn != i && !neighbor_sets[i].contains(&rn) {
                        candidates.push(rn);
                    }
                }
            }

            // Deduplicate candidates
            candidates.sort_unstable();
            candidates.dedup();

            // Try to improve neighbors
            for c in candidates {
                let d = distance_fn(&data[i], &data[c]);

                // Check if this is better than the worst current neighbor
                if let Some(worst) = heaps[i].peek() {
                    if d < worst.distance {
                        // Remove worst and add new neighbor
                        let removed = heaps[i].pop().unwrap();
                        neighbor_sets[i].remove(&removed.index);

                        heaps[i].push(NeighborEntry {
                            index: c,
                            distance: d,
                        });
                        neighbor_sets[i].insert(c);
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

    // Convert heaps to sorted neighbor lists
    let mut neighbors = vec![Vec::with_capacity(k); n];
    let mut distances = vec![Vec::with_capacity(k); n];

    for (i, heap) in heaps.into_iter().enumerate() {
        let mut entries: Vec<NeighborEntry> = heap.into_vec();
        entries.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        for entry in entries {
            neighbors[i].push(entry.index);
            distances[i].push(entry.distance);
        }
    }

    AnnGraph::new(neighbors, distances)
}

/// Find connected components in the ANN graph using DFS
/// Returns the undirected graph adjacency list and component assignments
fn find_components(ann_graph: &AnnGraph) -> (Vec<HashSet<usize>>, Vec<Vec<usize>>) {
    let n = ann_graph.n();

    // Build undirected graph from directed ANN graph
    let mut graph: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for (i, neighbors) in ann_graph.neighbors.iter().enumerate() {
        for &j in neighbors {
            graph[i].insert(j);
            graph[j].insert(i);
        }
    }

    // DFS to find components
    let mut visited = vec![false; n];
    let mut components: Vec<Vec<usize>> = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }

        let mut component = Vec::new();
        let mut stack = vec![start];

        while let Some(u) = stack.pop() {
            if visited[u] {
                continue;
            }
            visited[u] = true;
            component.push(u);

            for &v in &graph[u] {
                if !visited[v] {
                    stack.push(v);
                }
            }
        }

        components.push(component);
    }

    (graph, components)
}

/// Add random edges between components (Algorithm 3 in the paper)
fn add_random_edges<T, D, R>(
    data: &[T],
    components: &[Vec<usize>],
    lambda: usize,
    distance_fn: &D,
    rng: &mut R,
) -> (Vec<Edge>, Vec<(usize, usize)>)
where
    D: Fn(&T, &T) -> f64,
    R: Rng,
{
    let t = components.len();
    let mut edges = Vec::new();
    let mut edge_components = Vec::new();

    let lambda_sq = lambda * lambda;

    for i in 0..t {
        for j in (i + 1)..t {
            let mut candidates: Vec<Edge> = Vec::with_capacity(lambda_sq);

            // Generate λ² candidate edges
            for _ in 0..lambda_sq {
                let u = *components[i].choose(rng).unwrap();
                let v = *components[j].choose(rng).unwrap();
                let d = distance_fn(&data[u], &data[v]);
                candidates.push(Edge::new(u, v, d));
            }

            // Sort by distance and take top λ
            candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

            for edge in candidates.into_iter().take(lambda) {
                edges.push(edge);
                edge_components.push((i, j));
            }
        }
    }

    (edges, edge_components)
}

/// Refine inter-component edges (Algorithm 4 in the paper)
fn refine_edges<T, D>(
    data: &[T],
    undirected_graph: &[HashSet<usize>],
    components: &[Vec<usize>],
    edges: &[Edge],
    edge_components: &[(usize, usize)],
    distance_fn: &D,
) -> (Vec<Edge>, usize)
where
    D: Fn(&T, &T) -> f64,
{
    // Build component membership lookup
    let mut node_to_component: HashMap<usize, usize> = HashMap::new();
    for (comp_idx, component) in components.iter().enumerate() {
        for &node in component {
            node_to_component.insert(node, comp_idx);
        }
    }

    // Build component node sets for quick lookup
    let component_sets: Vec<HashSet<usize>> = components
        .iter()
        .map(|c| c.iter().copied().collect())
        .collect();

    let mut refined_edges = Vec::with_capacity(edges.len());
    let mut changes = 0;

    for (edge, &(ci, cj)) in edges.iter().zip(edge_components.iter()) {
        let mut best_u = edge.u;
        let mut best_v = edge.v;
        let mut best_d = edge.distance;

        // Get neighbors of u that are in component ci
        let neighbors_u: Vec<usize> = undirected_graph[edge.u]
            .iter()
            .filter(|&&n| component_sets[ci].contains(&n))
            .copied()
            .collect();

        // Try to find better u from neighbors
        for u_prime in neighbors_u {
            if u_prime == edge.v {
                continue;
            }
            let d_prime = distance_fn(&data[u_prime], &data[best_v]);
            if d_prime < best_d {
                best_u = u_prime;
                best_d = d_prime;
            }
        }

        // Get neighbors of v that are in component cj
        let neighbors_v: Vec<usize> = undirected_graph[edge.v]
            .iter()
            .filter(|&&n| component_sets[cj].contains(&n))
            .copied()
            .collect();

        // Try to find better v from neighbors (using updated best_u)
        for v_prime in neighbors_v {
            if v_prime == edge.u {
                continue;
            }
            let d_prime = distance_fn(&data[best_u], &data[v_prime]);
            if d_prime < best_d {
                best_v = v_prime;
                best_d = d_prime;
            }
        }

        if best_u != edge.u || best_v != edge.v {
            changes += 1;
        }

        refined_edges.push(Edge::new(best_u, best_v, best_d));
    }

    (refined_edges, changes)
}

/// Extract MST using Kruskal's algorithm on the connected ANN graph
fn extract_mst(ann_graph: &AnnGraph, inter_edges: &[Edge], n: usize) -> Vec<Edge> {
    // Collect all edges from ANN graph
    let mut all_edges: Vec<Edge> = Vec::new();

    for (i, (neighbors, distances)) in ann_graph
        .neighbors
        .iter()
        .zip(ann_graph.distances.iter())
        .enumerate()
    {
        for (&j, &d) in neighbors.iter().zip(distances.iter()) {
            let (u, v) = if i < j { (i, j) } else { (j, i) };
            all_edges.push(Edge::new(u, v, d));
        }
    }

    // Add inter-component edges
    for edge in inter_edges {
        all_edges.push(edge.clone());
    }

    // Deduplicate edges
    let mut edge_set: HashMap<(usize, usize), f64> = HashMap::new();
    for edge in all_edges {
        let key = if edge.u < edge.v {
            (edge.u, edge.v)
        } else {
            (edge.v, edge.u)
        };
        edge_set
            .entry(key)
            .and_modify(|d| {
                if edge.distance < *d {
                    *d = edge.distance
                }
            })
            .or_insert(edge.distance);
    }

    let mut edges: Vec<Edge> = edge_set
        .into_iter()
        .map(|((u, v), d)| Edge::new(u, v, d))
        .collect();

    // Sort edges by weight
    edges.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

    // Kruskal's algorithm
    let mut uf = UnionFind::new(n);
    let mut mst_edges = Vec::with_capacity(n - 1);

    for edge in edges {
        if uf.union(edge.u, edge.v) {
            mst_edges.push(edge);
            if mst_edges.len() == n - 1 {
                break;
            }
        }
    }

    mst_edges
}

/// Extract MST when graph is already connected (single component)
fn extract_mst_from_ann(ann_graph: &AnnGraph, n: usize) -> Vec<Edge> {
    extract_mst(ann_graph, &[], n)
}

/// Euclidean distance for slices of f64
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Manhattan distance for slices of f64
pub fn manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(5);
        assert!(uf.union(0, 1));
        assert!(uf.union(2, 3));
        assert!(!uf.union(0, 1)); // Already same set
        assert!(uf.union(1, 2));
        assert_eq!(uf.find(0), uf.find(3));
    }

    #[test]
    fn test_simple_mst() {
        // Simple 2D points forming a triangle
        let points: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.5, 0.866], // Equilateral triangle
        ];

        let distance = |a: &Vec<f64>, b: &Vec<f64>| euclidean_distance(a, b);
        let config = FamstConfig {
            k: 2,
            ..Default::default()
        };

        let mut rng = StdRng::seed_from_u64(42);
        let result = famst_with_rng(&points, distance, &config, &mut rng);

        assert_eq!(result.edges.len(), 2); // MST has n-1 edges
    }

    #[test]
    fn test_line_points() {
        // Points on a line
        let points: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];

        let distance = |a: &Vec<f64>, b: &Vec<f64>| euclidean_distance(a, b);
        let config = FamstConfig {
            k: 2,
            ..Default::default()
        };

        let mut rng = StdRng::seed_from_u64(42);
        let result = famst_with_rng(&points, distance, &config, &mut rng);

        assert_eq!(result.edges.len(), 4);
        // Total weight should be 4.0 (1+1+1+1)
        assert!((result.total_weight - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_disconnected_components() {
        // Two clusters far apart
        let points: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.5, 0.5],
            vec![100.0, 100.0],
            vec![101.0, 100.0],
            vec![100.5, 100.5],
        ];

        // k=1 will likely create disconnected components
        let distance = |a: &Vec<f64>, b: &Vec<f64>| euclidean_distance(a, b);
        let config = FamstConfig {
            k: 1,
            lambda: 3,
            max_iterations: 10,
            ..Default::default()
        };

        let mut rng = StdRng::seed_from_u64(42);
        let result = famst_with_rng(&points, distance, &config, &mut rng);

        assert_eq!(result.edges.len(), 5); // MST has n-1 edges
    }

    #[test]
    fn test_custom_distance() {
        // Test with Manhattan distance
        let points: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];

        let distance = |a: &Vec<f64>, b: &Vec<f64>| manhattan_distance(a, b);
        let config = FamstConfig {
            k: 2,
            ..Default::default()
        };

        let mut rng = StdRng::seed_from_u64(42);
        let result = famst_with_rng(&points, distance, &config, &mut rng);

        assert_eq!(result.edges.len(), 2);
        // Manhattan distance from (0,0) to (1,1) is 2, and (1,1) to (2,2) is 2
        assert!((result.total_weight - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_generic_data_type() {
        // Test with a custom struct
        #[derive(Clone)]
        struct Point3D {
            x: f64,
            y: f64,
            z: f64,
        }

        fn point_distance(a: &Point3D, b: &Point3D) -> f64 {
            ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt()
        }

        let points = vec![
            Point3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            Point3D {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            Point3D {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            Point3D {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
        ];

        let config = FamstConfig {
            k: 3,
            ..Default::default()
        };

        let mut rng = StdRng::seed_from_u64(42);
        let result = famst_with_rng(&points, point_distance, &config, &mut rng);

        assert_eq!(result.edges.len(), 3);
    }

    #[test]
    fn test_multiple_clusters() {
        // Create 5 well-separated clusters to force multiple components with small k
        // Each cluster is a tight group of points, clusters are far apart
        use rand::distributions::{Distribution, Uniform};

        let mut rng = StdRng::seed_from_u64(77777);
        let noise = Uniform::new(-0.5, 0.5);

        let cluster_centers = vec![
            vec![0.0, 0.0],
            vec![100.0, 0.0],
            vec![0.0, 100.0],
            vec![100.0, 100.0],
            vec![50.0, 50.0],
        ];

        let points_per_cluster = 20;
        let mut points: Vec<Vec<f64>> = Vec::new();

        for center in &cluster_centers {
            for _ in 0..points_per_cluster {
                let point = vec![
                    center[0] + noise.sample(&mut rng),
                    center[1] + noise.sample(&mut rng),
                ];
                points.push(point);
            }
        }

        let n = points.len();
        let distance = |a: &Vec<f64>, b: &Vec<f64>| euclidean_distance(a, b);

        // Use small k to create disconnected components
        // With k=3 and 20 points per cluster spread over 5 clusters,
        // each point's 3 nearest neighbors will be in its own cluster
        let config = FamstConfig {
            k: 3,
            lambda: 5,
            max_iterations: 50,
            nn_descent_iterations: 20,
            nn_descent_sample_rate: 1.0, // Full sampling for small dataset
        };

        let mut famst_rng = StdRng::seed_from_u64(88888);
        let result = famst_with_rng(&points, distance, &config, &mut famst_rng);

        // Should produce a valid MST with n-1 edges
        assert_eq!(result.edges.len(), n - 1, "MST should have n-1 edges");

        // Verify connectivity: all nodes should be reachable
        let mut uf = UnionFind::new(n);
        for edge in &result.edges {
            uf.union(edge.u, edge.v);
        }
        // Check all nodes are in the same component
        let root = uf.find(0);
        for i in 1..n {
            assert_eq!(uf.find(i), root, "All nodes should be connected in the MST");
        }

        // Compare with exact MST
        let exact_weight = exact_mst_weight(&points, distance);
        let error_ratio = (result.total_weight - exact_weight) / exact_weight;

        println!(
            "Multi-cluster test: Exact MST weight: {:.4}, FAMST weight: {:.4}, error: {:.2}%",
            exact_weight,
            result.total_weight,
            error_ratio * 100.0
        );

        // Should be reasonably close (within 15% given the challenging setup)
        assert!(
            error_ratio < 0.15,
            "FAMST error should be < 15%, got {:.2}%",
            error_ratio * 100.0
        );
    }

    /// Compute exact MST using Kruskal's algorithm on complete graph
    fn exact_mst_weight<T, D>(data: &[T], distance_fn: D) -> f64
    where
        D: Fn(&T, &T) -> f64,
    {
        let n = data.len();
        if n <= 1 {
            return 0.0;
        }

        // Build all edges
        let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let d = distance_fn(&data[i], &data[j]);
                edges.push((i, j, d));
            }
        }

        // Sort by weight
        edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        // Kruskal's algorithm
        let mut uf = UnionFind::new(n);
        let mut total_weight = 0.0;
        let mut edge_count = 0;

        for (u, v, w) in edges {
            if uf.union(u, v) {
                total_weight += w;
                edge_count += 1;
                if edge_count == n - 1 {
                    break;
                }
            }
        }

        total_weight
    }

    #[test]
    #[ignore] // Run with: cargo test large_scale -- --ignored --nocapture
    fn test_large_scale_vs_exact() {
        use rand::distributions::{Distribution, Uniform};

        const N: usize = 1_000_000;
        const DIM: usize = 10;

        println!("Generating {} random {}-dimensional points...", N, DIM);
        let mut rng = StdRng::seed_from_u64(12345);
        let dist = Uniform::new(0.0, 1000.0);

        let points: Vec<Vec<f64>> = (0..N)
            .map(|_| (0..DIM).map(|_| dist.sample(&mut rng)).collect())
            .collect();

        println!("Running FAMST with NN-Descent...");
        let distance = |a: &Vec<f64>, b: &Vec<f64>| euclidean_distance(a, b);
        let config = FamstConfig {
            k: 20,
            lambda: 5,
            max_iterations: 100,
            nn_descent_iterations: 10,
            nn_descent_sample_rate: 0.5,
        };
        let mut famst_rng = StdRng::seed_from_u64(54321);
        let start = std::time::Instant::now();
        let result = famst_with_rng(&points, distance, &config, &mut famst_rng);
        let famst_time = start.elapsed();

        println!("FAMST completed in {:?}", famst_time);
        println!("FAMST MST weight: {:.4}", result.total_weight);
        println!("FAMST MST edges: {}", result.edges.len());

        assert_eq!(result.edges.len(), N - 1, "MST should have n-1 edges");
    }

    #[test]
    fn test_medium_scale_vs_exact() {
        use rand::distributions::{Distribution, Uniform};

        const N: usize = 5000;
        const DIM: usize = 5;

        let mut rng = StdRng::seed_from_u64(99999);
        let dist = Uniform::new(0.0, 100.0);

        let points: Vec<Vec<f64>> = (0..N)
            .map(|_| (0..DIM).map(|_| dist.sample(&mut rng)).collect())
            .collect();

        let distance = |a: &Vec<f64>, b: &Vec<f64>| euclidean_distance(a, b);

        // Compute exact MST
        let exact_weight = exact_mst_weight(&points, distance);

        // Compute approximate MST with FAMST
        let config = FamstConfig {
            k: 15,
            lambda: 5,
            max_iterations: 100,
            nn_descent_iterations: 15,
            nn_descent_sample_rate: 0.5,
        };
        let mut famst_rng = StdRng::seed_from_u64(11111);
        let result = famst_with_rng(&points, distance, &config, &mut famst_rng);

        assert_eq!(result.edges.len(), N - 1);

        // FAMST should produce a weight >= exact (it's an approximation)
        // and should be reasonably close (within a few percent for good k)
        let error_ratio = (result.total_weight - exact_weight) / exact_weight;
        println!(
            "Exact MST weight: {:.4}, FAMST weight: {:.4}, error: {:.2}%",
            exact_weight,
            result.total_weight,
            error_ratio * 100.0
        );

        // The approximation should be within 10% for this setup with NN-Descent
        assert!(
            error_ratio >= 0.0,
            "FAMST weight should be >= exact MST weight"
        );
        assert!(
            error_ratio < 0.10,
            "FAMST error should be < 10%, got {:.2}%",
            error_ratio * 100.0
        );
    }
}
