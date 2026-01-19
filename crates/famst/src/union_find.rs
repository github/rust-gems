//! Union-Find (Disjoint Set Union) data structure for Kruskal's algorithm

use crate::NodeId;

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
