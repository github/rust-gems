//! Reorder document IDs by similarity using geometric filters and MST.
//!
//! Usage:
//!   cargo run --release --bin reorder_docids --features "geo_filters ordered-float rayon" -- <input.docs>
//!
//! This reads the inverted index, builds GeoDiffCount filters for each document,
//! computes an approximate MST based on document similarity, and outputs a mapping
//! from old doc IDs to new doc IDs sorted by similarity (DFS order of the MST).
//!
//! The algorithm uses multiple random projections (masks) to create approximate
//! MSTs via cmp_masked, then merges them for a better approximation.

use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Instant;

use geo_filters::diff_count::GeoDiffCount7;
use geo_filters::Count;
use ordered_float::OrderedFloat;
use rayon::prelude::*;

type GeoFilter = GeoDiffCount7<'static>;

/// Mask size for GeoDiffConfig7 (7 bits)
const MASK_SIZE: usize = 7;

/// Read a u32 from a reader (little-endian)
fn read_u32<R: Read>(reader: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Read a posting list from PISA format
fn read_posting_list<R: Read>(reader: &mut R) -> std::io::Result<Option<Vec<u32>>> {
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_le_bytes(len_buf) as usize;

    let mut postings = vec![0u32; len];
    for posting in &mut postings {
        *posting = read_u32(reader)?;
    }
    Ok(Some(postings))
}

// ============================================================================
// Union-Find data structure
// ============================================================================

struct UnionFind {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl UnionFind {
    fn new(size: u32) -> Self {
        let parent = (0..size).collect();
        Self {
            parent,
            size: vec![1; size as usize],
        }
    }

    fn find(&mut self, mut idx: u32) -> u32 {
        while self.parent[idx as usize] != idx {
            let parent = self.parent[idx as usize];
            let grand_parent = self.parent[parent as usize];
            self.parent[idx as usize] = grand_parent;
            idx = grand_parent;
        }
        idx
    }

    fn unite(&mut self, a: u32, b: u32) -> bool {
        let mut a = self.find(a);
        let mut b = self.find(b);
        if a == b {
            false
        } else {
            if self.size[a as usize] < self.size[b as usize] {
                std::mem::swap(&mut a, &mut b);
            }
            self.parent[b as usize] = a;
            self.size[a as usize] += self.size[b as usize];
            true
        }
    }
}

// ============================================================================
// MST Edge
// ============================================================================

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Edge {
    cost: OrderedFloat<f32>,
    from: u32,
    to: u32,
}

// ============================================================================
// MST Algorithm (Kruskal's)
// ============================================================================

fn build_mst(size: usize, mut edges: Vec<Edge>) -> Vec<Edge> {
    let mut mst = Vec::with_capacity(size);
    let mut uf = UnionFind::new(size as u32);

    edges.par_sort_unstable_by_key(|x| x.cost);

    for edge in edges {
        if uf.unite(edge.from, edge.to) {
            mst.push(edge);
        }
    }
    mst
}

// ============================================================================
// Approximate MST using multiple masked sortings (from blackbird)
// ============================================================================

/// Create edges between documents within a sliding window after sorting.
/// Uses the optimized approach: pre-allocate and write in parallel.
fn create_edges_in_window(
    filters: &[GeoFilter],
    sorted: &[u32],
    window_size: usize,
) -> Vec<Edge> {
    let len = sorted.len() * window_size;
    let mut edges = Vec::<Edge>::with_capacity(len);
    
    // Use spare_capacity_mut to avoid initializing edges we'll overwrite
    edges.spare_capacity_mut()[..len]
        .par_iter_mut()
        .enumerate()
        .for_each(|(k, edge)| {
            let i = k / window_size;
            let j = (k % window_size) + 1;
            let from = sorted[i];
            let to = sorted[(i + j) % sorted.len()];
            let a = &filters[from as usize];
            let b = &filters[to as usize];
            let cost = a.size_with_sketch_f32(b).into();
            edge.write(Edge { from, to, cost });
        });
    
    unsafe {
        edges.set_len(len);
    }
    edges
}

/// Return all bit masks with exactly `ones` bits set among `num` least significant bits.
fn mask_patterns(num: usize, ones: usize) -> Vec<usize> {
    (0..(1usize << num))
        .filter(move |pattern| pattern.count_ones() == ones as u32)
        .collect()
}

/// Merge multiple MSTs by combining their edges and rebuilding.
fn merge_msts(num_nodes: usize, mut msts: Vec<Vec<Edge>>) -> Vec<Edge> {
    while msts.len() > 1 {
        msts = msts
            .par_chunks(2)
            .map(|chunk| {
                let edges: Vec<Edge> = chunk.iter().flatten().copied().collect();
                build_mst(num_nodes, edges)
            })
            .collect();
    }
    msts.pop().unwrap_or_default()
}

/// Parallelise a function over a slice with limited parallelism.
/// Important when each invocation consumes significant memory.
fn parallelise<R, F>(f: F, indices: &[usize], parallelism: usize) -> Vec<R>
where
    R: Send + Sync + Default + Clone,
    F: Fn(usize) -> R + Send + Sync,
{
    let next = AtomicUsize::new(0);
    let results = std::sync::Mutex::new(vec![R::default(); indices.len()]);
    
    rayon::scope(|scope| {
        for _ in 0..parallelism {
            scope.spawn(|_| loop {
                let idx = next.fetch_add(1, AtomicOrdering::Relaxed);
                if idx >= indices.len() {
                    break;
                }
                let r = f(indices[idx]);
                results.lock().unwrap()[idx] = r;
            });
        }
    });
    
    results.into_inner().unwrap()
}

/// Compute multiple approximate MSTs using different mask patterns.
/// Each mask creates a different projection for sorting via cmp_masked.
fn compute_multiple_approximate_msts(
    filters: &[GeoFilter],
    masks: &[usize],
    window_size: usize,
) -> Vec<Vec<Edge>> {
    let indices: Vec<usize> = (0..masks.len()).collect();
    
    parallelise(
        |idx| {
            let mask = masks[idx];
            
            // Sort documents by masked filter comparison
            let mut sorted: Vec<u32> = (0..filters.len() as u32).collect();
            sorted.par_sort_unstable_by(|&a, &b| {
                filters[a as usize].cmp_masked(
                    &filters[b as usize],
                    mask as u64,
                    MASK_SIZE,
                )
            });
            
            // Create edges in window
            let edges = create_edges_in_window(filters, &sorted, window_size);
            
            // Build MST for this projection
            build_mst(filters.len(), edges)
        },
        &indices,
        4, // Limit parallelism to avoid memory issues
    )
}

/// Compute approximate MST using multiple masked sortings.
fn compute_approximate_mst(
    filters: &[GeoFilter],
    window_size: usize,
    num_mask_bits: usize,
) -> Vec<Edge> {
    // Generate all mask patterns with num_mask_bits/2 bits set
    let masks = mask_patterns(MASK_SIZE, num_mask_bits);
    println!("  Using {} mask patterns with {} bits set...", masks.len(), num_mask_bits);
    
    println!("  Computing {} approximate MSTs with window size {}...", masks.len(), window_size);
    let msts = compute_multiple_approximate_msts(filters, &masks, window_size);
    
    println!("  Merging {} MSTs...", msts.len());
    merge_msts(filters.len(), msts)
}

// ============================================================================
// Tree traversal for document ordering
// ============================================================================

/// Convert MST edges to adjacency list representation.
fn mst_to_adjacency(num_nodes: usize, mst: &[Edge]) -> Vec<Vec<u32>> {
    let mut adj: Vec<Vec<u32>> = vec![vec![]; num_nodes];
    for edge in mst {
        adj[edge.from as usize].push(edge.to);
        adj[edge.to as usize].push(edge.from);
    }
    adj
}

/// DFS traversal of the MST to produce document ordering.
/// Documents visited close together in DFS order are similar.
fn dfs_order(adj: &[Vec<u32>], start: u32) -> Vec<u32> {
    let n = adj.len();
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut stack = vec![start];

    while let Some(node) = stack.pop() {
        if visited[node as usize] {
            continue;
        }
        visited[node as usize] = true;
        order.push(node);

        // Add neighbors in reverse order so they're visited in forward order
        for &neighbor in adj[node as usize].iter().rev() {
            if !visited[neighbor as usize] {
                stack.push(neighbor);
            }
        }
    }

    // Handle disconnected components (shouldn't happen with proper MST)
    for i in 0..n {
        if !visited[i] {
            order.push(i as u32);
        }
    }

    order
}

/// Compute the mapping from old doc IDs to new doc IDs.
fn compute_docid_mapping(dfs_order: &[u32]) -> Vec<u32> {
    let mut mapping = vec![0u32; dfs_order.len()];
    for (new_id, &old_id) in dfs_order.iter().enumerate() {
        mapping[old_id as usize] = new_id as u32;
    }
    mapping
}

// ============================================================================
// Main
// ============================================================================

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <input.docs> [window_size] [mask_bits]", args[0]);
        eprintln!();
        eprintln!("Reorders document IDs by similarity using MST with masked sorting.");
        eprintln!("Outputs a mapping file next to the input.");
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  window_size  - Number of neighbors to consider (default: 32)");
        eprintln!("  mask_bits    - Number of bits set in masks (default: 3, max: {})", MASK_SIZE);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let window_size: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let mask_bits: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    println!("Reading PISA posting lists from: {}", input_path);
    println!("Window size: {}", window_size);
    println!("Mask bits: {} (generates {} masks)", mask_bits, mask_patterns(MASK_SIZE, mask_bits).len());

    let file = File::open(input_path)?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);

    // Read header
    let header1 = read_u32(&mut reader)?;
    let header2 = read_u32(&mut reader)?;
    println!("Header: [{}, {}]", header1, header2);
    let num_documents = header2 as usize;
    println!("Number of documents: {}", num_documents);

    // ========================================================================
    // Step 1: Build geometric filters for each document
    // ========================================================================
    println!("\n=== Building document filters ===");
    let mut filters: Vec<GeoFilter> = (0..num_documents)
        .map(|_| GeoFilter::default())
        .collect();

    let start = Instant::now();
    let mut last_report = Instant::now();
    let mut term_id = 0u64;
    let mut total_postings = 0u64;

    while let Some(postings) = read_posting_list(&mut reader)? {
        for &doc_id in &postings {
            filters[doc_id as usize].push(term_id);
        }

        total_postings += postings.len() as u64;
        term_id += 1;

        if last_report.elapsed().as_secs() >= 1 {
            print!(
                "\rProcessed {:>10} terms, {:>12} postings...",
                term_id, total_postings
            );
            std::io::stdout().flush()?;
            last_report = Instant::now();
        }
    }

    let elapsed = start.elapsed();
    println!(
        "\rProcessed {:>10} terms, {:>12} postings in {:.2}s",
        term_id, total_postings, elapsed.as_secs_f64()
    );

    // ========================================================================
    // Step 2: Compute approximate MST using multiple masked sortings
    // ========================================================================
    println!("\n=== Computing approximate MST ===");
    let start = Instant::now();
    let mst = compute_approximate_mst(&filters, window_size, mask_bits);
    let elapsed = start.elapsed();
    println!("MST computed in {:.2}s ({} edges)", elapsed.as_secs_f64(), mst.len());

    // ========================================================================
    // Step 3: DFS traversal to get document ordering
    // ========================================================================
    println!("\n=== Computing document ordering ===");
    let adj = mst_to_adjacency(num_documents, &mst);

    // Start from document 0 (or the largest document)
    let start_doc = 0u32;
    let order = dfs_order(&adj, start_doc);
    let mapping = compute_docid_mapping(&order);

    // ========================================================================
    // Step 4: Write mapping to file
    // ========================================================================
    let output_path = Path::new(input_path).with_extension("docid_mapping");
    println!("\n=== Writing mapping to {} ===", output_path.display());

    let output_file = File::create(&output_path)?;
    let mut writer = BufWriter::new(output_file);

    // Write as binary: array of u32 where mapping[old_id] = new_id
    for &new_id in &mapping {
        writer.write_all(&new_id.to_le_bytes())?;
    }
    writer.flush()?;

    let file_size = std::fs::metadata(&output_path)?.len();
    println!(
        "Written {} bytes ({:.2} MB)",
        file_size,
        file_size as f64 / 1_000_000.0
    );

    // ========================================================================
    // Statistics
    // ========================================================================
    println!("\n=== Reordering Statistics ===");
    println!("Total documents:     {:>12}", num_documents);
    println!("MST edges:           {:>12}", mst.len());

    // Sample some mappings
    println!("\nSample mappings (old -> new):");
    for old_id in [0, 1, 100, 1000, 10000, 100000, 500000].iter() {
        if (*old_id as usize) < num_documents {
            println!("  Doc {:>6} -> {:>6}", old_id, mapping[*old_id as usize]);
        }
    }

    // Compute average MST edge cost
    let mst_sum: f64 = mst.iter().map(|e| e.cost.0 as f64).sum();
    if !mst.is_empty() {
        let avg_cost = mst_sum / mst.len() as f64;
        let min_cost = mst.iter().map(|e| e.cost.0).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
        let max_cost = mst.iter().map(|e| e.cost.0).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
        println!("\nMST edge costs:");
        println!("  Sum:     {:>15.2}", mst_sum);
        println!("  Min:     {:>15.2}", min_cost);
        println!("  Avg:     {:>15.2}", avg_cost);
        println!("  Max:     {:>15.2}", max_cost);
    }

    // ========================================================================
    // Compute sum of differences between consecutive documents
    // ========================================================================
    println!("\n=== Consecutive Document Similarity ===");

    // Before remapping: consecutive docs are 0,1,2,3,...
    // Sum of size_with_sketch(doc[i], doc[i+1]) for original order
    let sum_before: f64 = (0..num_documents - 1)
        .into_par_iter()
        .map(|i| filters[i].size_with_sketch_f32(&filters[i + 1]) as f64)
        .sum();

    // After remapping: consecutive docs follow DFS order
    // Sum of size_with_sketch for new consecutive pairs
    let sum_after: f64 = (0..order.len() - 1)
        .into_par_iter()
        .map(|i| {
            let doc_a = order[i] as usize;
            let doc_b = order[i + 1] as usize;
            filters[doc_a].size_with_sketch_f32(&filters[doc_b]) as f64
        })
        .sum();

    let avg_before = sum_before / (num_documents - 1) as f64;
    let avg_after = sum_after / (order.len() - 1) as f64;
    let improvement = (avg_before - avg_after) / avg_before * 100.0;

    println!("Sum of consecutive diffs (before):  {:>15.2}", sum_before);
    println!("Sum of consecutive diffs (after):   {:>15.2}", sum_after);
    println!("Avg consecutive diff (before):      {:>15.2}", avg_before);
    println!("Avg consecutive diff (after):       {:>15.2}", avg_after);
    println!("Improvement:                        {:>14.1}%", improvement);

    Ok(())
}
