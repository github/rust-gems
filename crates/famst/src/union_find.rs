//! Union-Find (Disjoint Set Union) data structure for Kruskal's algorithm

use crate::{AnnGraph, NodeId};

/// Union-Find (Disjoint Set Union) data structure for Kruskal's algorithm
pub(crate) struct UnionFind {
    parent: Vec<NodeId>,
    rank: Vec<NodeId>,
}

impl UnionFind {
    pub(crate) fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n as NodeId).collect(),
            rank: vec![0; n],
        }
    }

    pub(crate) fn find(&mut self, x: NodeId) -> NodeId {
        if self.parent[x as usize] != x {
            self.parent[x as usize] = self.find(self.parent[x as usize]); // Path compression
        }
        self.parent[x as usize]
    }

    pub(crate) fn union(&mut self, x: NodeId, y: NodeId) -> bool {
        let px = self.find(x);
        let py = self.find(y);
        if px == py {
            return false;
        }
        // Union by rank
        match self.rank[px as usize].cmp(&self.rank[py as usize]) {
            std::cmp::Ordering::Less => self.parent[px as usize] = py,
            std::cmp::Ordering::Greater => self.parent[py as usize] = px,
            std::cmp::Ordering::Equal => {
                self.parent[py as usize] = px;
                self.rank[px as usize] += 1;
            }
        }
        true
    }
}

/// Find connected components from the ANN graph
/// Treats directed edges as undirected for connectivity
/// Returns component assignments as a list of node lists
pub(crate) fn find_components(ann_graph: &AnnGraph) -> Vec<Vec<NodeId>> {
    let n = ann_graph.n();
    let mut uf = UnionFind::new(n);

    // Union all edges (treating directed as undirected)
    for i in 0..n {
        for neighbor in ann_graph.neighbors(i) {
            uf.union(i as NodeId, neighbor.index);
        }
    }

    // First pass: assign contiguous component IDs to each root
    let mut root_to_component: Vec<NodeId> = vec![NodeId::MAX; n];
    let mut num_components = 0;
    for i in 0..n {
        let root = uf.find(i as NodeId) as usize;
        if root_to_component[root] == NodeId::MAX {
            root_to_component[root] = num_components;
            num_components += 1;
        }
    }

    // Second pass: fill components directly
    let mut components: Vec<Vec<NodeId>> = vec![Vec::new(); num_components as usize];
    for i in 0..n {
        let root = uf.find(i as NodeId) as usize;
        let component_id = root_to_component[root] as usize;
        components[component_id].push(i as NodeId);
    }

    components
}
