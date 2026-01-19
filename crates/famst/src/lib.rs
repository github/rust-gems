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

mod nn_descent;
mod union_find;

use nn_descent::nn_descent;
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::iter::IndexedParallelIterator;
use rayon::slice::ParallelSliceMut;
use union_find::{find_components, UnionFind};

/// Node index with embedded "new" flag in the least significant bit.
///
/// The LSB (bit 0) is used as a "new" flag for NN-Descent optimization.
/// The actual node index is stored in bits 1-31, giving capacity for 2^31 nodes.
///
/// Using LSB means that during sorting, the same node ID with or without the
/// new flag will be adjacent (differ only by 1), which helps with existence checks.
#[derive(Debug, Clone, Copy, Default, Hash)]
pub struct NodeId(u32);

impl NodeId {
    /// Flag bit indicating a "new" neighbor (LSB)
    const NEW_FLAG: u32 = 1;

    /// Create a new NodeId from a raw index (not marked as new)
    #[inline]
    pub fn new(index: u32) -> Self {
        debug_assert!(index < (1 << 31), "NodeId index must be < 2^31");
        NodeId(index << 1)
    }

    /// Get the actual node index (without the new flag)
    #[inline]
    pub fn index(self) -> u32 {
        self.0 >> 1
    }

    /// Check if this NodeId has the "new" flag set
    #[inline]
    pub fn is_new(self) -> bool {
        self.0 & Self::NEW_FLAG != 0
    }

    /// Return a copy with the "new" flag set
    #[inline]
    pub fn as_new(self) -> Self {
        NodeId(self.0 | Self::NEW_FLAG)
    }

    /// Return a copy with the "new" flag cleared
    #[inline]
    pub fn as_old(self) -> Self {
        NodeId(self.0 & !Self::NEW_FLAG)
    }
}

impl PartialEq for NodeId {
    fn eq(&self, other: &Self) -> bool {
        // Compare by actual index, ignoring new flag
        self.index() == other.index()
    }
}

impl Eq for NodeId {}

impl PartialOrd for NodeId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NodeId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare by actual index, ignoring new flag
        self.index().cmp(&other.index())
    }
}

/// An edge in the MST, represented as (node_a, node_b, distance)
#[derive(Debug, Clone)]
pub struct Edge {
    pub u: u32,
    pub v: u32,
    pub distance: f32,
}

impl Edge {
    fn new(u: u32, v: u32, distance: f32) -> Self {
        Edge { u, v, distance }
    }
}

/// A neighbor entry: node index and distance (32-bit for memory efficiency)
#[derive(Debug, Clone, Copy)]
pub(crate) struct Neighbor {
    pub(crate) index: NodeId,
    pub(crate) distance: f32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        // Compare by (distance, index) - index comparison ignores new flag
        self.distance == other.distance && self.index == other.index
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Total ordering by (distance, index) - index comparison ignores new flag
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.index.cmp(&other.index))
    }
}

/// Approximate Nearest Neighbors graph representation
/// Stored as a flat n×k matrix of Neighbor entries
pub(crate) struct AnnGraph {
    /// Flat storage: data[i*k..(i+1)*k] contains k neighbors of point i
    data: Vec<Neighbor>,
    /// Number of points
    n: usize,
    /// Number of neighbors per point
    k: usize,
}

impl AnnGraph {
    pub(crate) fn new(n: usize, k: usize, data: Vec<Neighbor>) -> Self {
        assert_eq!(data.len(), n * k);
        AnnGraph { data, n, k }
    }

    pub(crate) fn n(&self) -> usize {
        self.n
    }

    pub(crate) fn k(&self) -> usize {
        self.k
    }

    /// Get the neighbors of point i
    pub(crate) fn neighbors(&self, i: usize) -> &[Neighbor] {
        let start = i * self.k;
        &self.data[start..start + self.k]
    }

    /// Get mutable access to neighbors of point i
    pub(crate) fn neighbors_mut(&mut self, i: usize) -> &mut [Neighbor] {
        let start = i * self.k;
        &mut self.data[start..start + self.k]
    }

    /// Get mutable access to all neighbor chunks for parallel processing
    pub(crate) fn neighbors_chunks_mut(&mut self, group_size: usize) -> impl IndexedParallelIterator<Item = &mut [Neighbor]> {
        self.data.par_chunks_mut(self.k * group_size)
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
}

impl Default for FamstConfig {
    fn default() -> Self {
        FamstConfig {
            k: 20,
            lambda: 5,
            max_iterations: 100,
            nn_descent_iterations: 10,
        }
    }
}

/// Result of FAMST algorithm
pub struct FamstResult {
    /// MST edges
    pub edges: Vec<Edge>,
    /// Total weight of the MST
    pub total_weight: f32,
}

/// Main FAMST algorithm implementation
///
/// Generic over:
/// - `T`: The data type stored at each point
/// - `D`: Distance function `Fn(&T, &T) -> f32`
///
/// # Arguments
/// * `data` - Slice of data points
/// * `distance_fn` - Function to compute distance between two points
/// * `config` - Algorithm configuration
///
/// # Returns
/// The approximate MST as a list of edges
///
/// # Panics
/// Panics if `data.len() >= 2^32` (more than ~4 billion points).
pub fn famst<T, D>(data: &[T], distance_fn: D, config: &FamstConfig) -> FamstResult
where
    T: Sync,
    D: Fn(&T, &T) -> f32 + Sync,
{
    famst_with_rng(data, distance_fn, config, &mut rand::thread_rng())
}

/// FAMST with custom RNG. (We use a seeded RNG in tests for reproducibility.)
///
/// # Panics
/// Panics if `data.len() >= 2^32` (more than ~4 billion points).
pub fn famst_with_rng<T, D, R>(
    data: &[T],
    distance_fn: D,
    config: &FamstConfig,
    rng: &mut R,
) -> FamstResult
where
    T: Sync,
    D: Fn(&T, &T) -> f32 + Sync,
    R: Rng,
{
    let n = data.len();
    assert!(
        n <= (1 << 31),
        "famst: data length {n} exceeds maximum supported size of 2^31"
    );
    
    if n <= 1 {
        return FamstResult {
            edges: vec![],
            total_weight: 0.0,
        };
    }

    // Phase 1: Build ANN graph using NN-Descent
    let ann_graph = nn_descent(data, &distance_fn, config, rng);

    // Phase 2: Build undirected graph and find connected components
    let undirected_graph = build_undirected_graph(&ann_graph);
    let components = find_components(&ann_graph);
    println!("Found {} connected components", components.len());

    // If only one component, skip inter-component edge logic
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

/// Build an undirected graph from the directed ANN graph
/// Returns adjacency list with sorted, deduplicated neighbors
fn build_undirected_graph(ann_graph: &AnnGraph) -> Vec<Vec<NodeId>> {
    let n = ann_graph.n();

    let mut graph: Vec<Vec<NodeId>> = vec![Vec::new(); n];
    for i in 0..n {
        for neighbor in ann_graph.neighbors(i) {
            let j = neighbor.index;
            graph[i].push(j);
            graph[j.index() as usize].push(NodeId::new(i as u32));
        }
    }
    // Sort and deduplicate each adjacency list
    for adj in &mut graph {
        adj.sort_unstable();
        adj.dedup();
    }

    graph
}

/// Add random edges between components (Algorithm 3 in the paper)
fn add_random_edges<T, D, R>(
    data: &[T],
    components: &[Vec<NodeId>],
    lambda: usize,
    distance_fn: &D,
    rng: &mut R,
) -> (Vec<Edge>, Vec<(usize, usize)>)
where
    D: Fn(&T, &T) -> f32,
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
                let d = distance_fn(&data[u.index() as usize], &data[v.index() as usize]);
                candidates.push(Edge::new(u.index(), v.index(), d));
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
    undirected_graph: &[Vec<NodeId>],
    components: &[Vec<NodeId>],
    edges: &[Edge],
    edge_components: &[(usize, usize)],
    distance_fn: &D,
) -> (Vec<Edge>, usize)
where
    D: Fn(&T, &T) -> f32,
{
    let n = data.len();
    
    // Build component membership lookup (simple vec, O(1) lookup)
    let mut node_to_component: Vec<usize> = vec![0; n];
    for (comp_idx, component) in components.iter().enumerate() {
        for &node in component {
            node_to_component[node.index() as usize] = comp_idx;
        }
    }

    let mut refined_edges = Vec::with_capacity(edges.len());
    let mut changes = 0;

    for (edge, &(ci, cj)) in edges.iter().zip(edge_components.iter()) {
        let mut best_u = edge.u;
        let mut best_v = edge.v;
        let mut best_d = edge.distance;

        // Get neighbors of u that are in component ci
        // (undirected_graph is sorted, so we can iterate directly)
        for &u_prime in &undirected_graph[edge.u as usize] {
            let u_prime_idx = u_prime.index();
            let u_prime_usize = u_prime_idx as usize;
            if node_to_component[u_prime_usize] == ci && u_prime_idx != edge.v {
                let d_prime = distance_fn(&data[u_prime_usize], &data[best_v as usize]);
                if d_prime < best_d {
                    best_u = u_prime_idx;
                    best_d = d_prime;
                }
            }
        }

        // Get neighbors of v that are in component cj
        for &v_prime in &undirected_graph[edge.v as usize] {
            let v_prime_idx = v_prime.index();
            let v_prime_usize = v_prime_idx as usize;
            if node_to_component[v_prime_usize] == cj && v_prime_idx != edge.u {
                let d_prime = distance_fn(&data[best_u as usize], &data[v_prime_usize]);
                if d_prime < best_d {
                    best_v = v_prime_idx;
                    best_d = d_prime;
                }
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
    let mut edges: Vec<Edge> = Vec::new();

    for i in 0..n {
        for neighbor in ann_graph.neighbors(i) {
            edges.push(Edge::new(i as u32, neighbor.index.index(), neighbor.distance));
        }
    }

    // Add inter-component edges
    edges.extend(inter_edges.iter().cloned());

    // Sort edges by weight
    edges.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

    // Kruskal's algorithm (naturally handles duplicate edges)
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Manhattan distance for slices of f32
    fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
    }

    /// Euclidean distance for slices of f32
    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn test_empty_input() {
        let points: Vec<Vec<f32>> = vec![];
        let distance = |a: &Vec<f32>, b: &Vec<f32>| euclidean_distance(a, b);
        let result = famst(&points, distance, &FamstConfig::default());
        assert_eq!(result.edges.len(), 0);
        assert_eq!(result.total_weight, 0.0);
    }

    #[test]
    fn test_single_point() {
        let points: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0]];
        let distance = |a: &Vec<f32>, b: &Vec<f32>| euclidean_distance(a, b);
        let result = famst(&points, distance, &FamstConfig::default());
        assert_eq!(result.edges.len(), 0);
        assert_eq!(result.total_weight, 0.0);
    }

    #[test]
    fn test_k_greater_than_n() {
        // 3 points but k=20 (default), so k >= n
        let points: Vec<Vec<f32>> = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let distance = |a: &Vec<f32>, b: &Vec<f32>| euclidean_distance(a, b);
        let config = FamstConfig::default(); // k=20 > n=3
        let result = famst(&points, distance, &config);
        assert_eq!(result.edges.len(), 2); // MST has n-1 edges
    }

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
        let points: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.5, 0.866], // Equilateral triangle
        ];

        let distance = |a: &Vec<f32>, b: &Vec<f32>| euclidean_distance(a, b);
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
        let points: Vec<Vec<f32>> = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];

        let distance = |a: &Vec<f32>, b: &Vec<f32>| euclidean_distance(a, b);
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
        let points: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.5, 0.5],
            vec![100.0, 100.0],
            vec![101.0, 100.0],
            vec![100.5, 100.5],
        ];

        // k=1 will likely create disconnected components
        let distance = |a: &Vec<f32>, b: &Vec<f32>| euclidean_distance(a, b);
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
        let points: Vec<Vec<f32>> = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];

        let distance = |a: &Vec<f32>, b: &Vec<f32>| manhattan_distance(a, b);
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
            x: f32,
            y: f32,
            z: f32,
        }

        fn point_distance(a: &Point3D, b: &Point3D) -> f32 {
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
        let mut points: Vec<Vec<f32>> = Vec::new();

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
        let distance = |a: &Vec<f32>, b: &Vec<f32>| euclidean_distance(a, b);

        // Use small k to create disconnected components
        // With k=3 and 20 points per cluster spread over 5 clusters,
        // each point's 3 nearest neighbors will be in its own cluster
        let config = FamstConfig {
            k: 3,
            lambda: 5,
            max_iterations: 50,
            nn_descent_iterations: 20,
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
            assert_eq!(uf.find(i as u32), root, "All nodes should be connected in the MST");
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
    fn exact_mst_weight<T, D>(data: &[T], distance_fn: D) -> f32
    where
        D: Fn(&T, &T) -> f32,
    {
        let n = data.len();
        if n <= 1 {
            return 0.0;
        }

        // Build all edges
        let mut edges: Vec<(usize, usize, f32)> = Vec::with_capacity(n * (n - 1) / 2);
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
            if uf.union(u as u32, v as u32) {
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

        let points: Vec<Vec<f32>> = (0..N)
            .map(|_| (0..DIM).map(|_| dist.sample(&mut rng)).collect())
            .collect();

        println!("Running FAMST with NN-Descent...");
        let distance = |a: &Vec<f32>, b: &Vec<f32>| euclidean_distance(a, b);
        let config = FamstConfig {
            k: 20,
            lambda: 5,
            max_iterations: 100,
            nn_descent_iterations: 10,
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

        const N: usize = 10000;
        const DIM: usize = 2;

        let mut rng = StdRng::seed_from_u64(99999);
        let dist = Uniform::new(0.0, 100.0);

        let points: Vec<Vec<f32>> = (0..N)
            .map(|_| (0..DIM).map(|_| dist.sample(&mut rng)).collect())
            .collect();

        let distance = |a: &Vec<f32>, b: &Vec<f32>| euclidean_distance(a, b);

        // Compute exact MST
        let exact_weight = exact_mst_weight(&points, distance);

        // Compute approximate MST with FAMST
        let config = FamstConfig {
            k: 4,
            lambda: 5,
            max_iterations: 100,
            nn_descent_iterations: 100,
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
