# Consistent Hashing

Consistent hashing maps keys to a changing set of nodes (shards, servers) so that when nodes join or leave, only a small fraction of keys move. It is used in distributed caches, databases, object stores, and load balancers to achieve scalability and high availability with minimal data reshuffling.

Common algorithms
- [Consistent hashing](https://en.wikipedia.org/wiki/Consistent_hashing) (hash ring with virtual nodes)
- [Rendezvous hashing](https://en.wikipedia.org/wiki/Rendezvous_hashing)
- [Jump consistent hash](https://en.wikipedia.org/wiki/Jump_consistent_hash)
- [Maglev hashing](https://research.google/pubs/pub44824) 
- [AnchorHash: A Scalable Consistent Hash](https://arxiv.org/abs/1812.09674)
- [DXHash](https://arxiv.org/abs/2107.07930)
- [JumpBackHash](https://arxiv.org/abs/2403.18682)

## Complexity summary

where `N` is the number of nodes and `R` is the number of replicas.

| Algorithm               | Lookup per key       | Node add/remove                        | Memory                    | Replication support                              |
|-------------------------|----------------------|----------------------------------------|---------------------------|--------------------------------------------------|
| Hash ring (with vnodes) | O(log N) binary search over N points; O(1) with specialized structures | O(log N) to insert/remove points         | O(N) points               | Yes: take next R distinct successors; O(log N + R) |
| Rendezvous              | O(N) score per node; top-1 | O(1) (no state to rebalance)     | O(N) node list            | Yes: pick top R scores; O(N log R) |
| Jump consistent hash    | O(log(N))            | O(1)                                   | O(1)                      | Not native               |
| AnchorHash              | O(1) expected        | O(1) expected/amortized                | O(N)                      | Not native               |
| DXHash                  | O(1) expected        | O(1) expected                          | O(N)                      | Not native               |
| JumpBackHash            | O(1)                 | O(1) expected                          | O(1)                      | Not native               |

Replication of keys
- Hash ring: replicate by walking clockwise to the next R distinct nodes. Virtual nodes help spread replicas evenly and avoid hotspots.
- Rendezvous hashing: replicate by selecting the top R nodes by score for the key. This naturally yields R distinct owners and supports weights.
- Jump consistent hash: the base function returns one bucket. Replication can be achieved by hashing (key, replica_index) and collecting R distinct buckets; this is simple but lacks the single-pass global ranking HRW provides.

Why replication matters
- Tolerates node failures and maintenance without data unavailability.
- Distributes read/write load across multiple owners, reducing hotspots.
- Enables fast recovery and higher tail-latency resilience.

## N-Choose-R replication

We define the consistent `n-choose-rk` replication as follows:

1. for a given number `n` of nodes, choose `k` distinct nodes `S`.
2. for a given `key` the chosen set of nodes must be uniformly chosen from all possible sets of size `k`.
3. when `n` increases by one, exactly one node in the chosen set will be changed with probability `k/(n+1)`.

For simplicity, nodes are represented by integers `0..n`.
Given `k` independent consistent hash functions `h_i(n)` for a given key, the following algorithm will have the desired properties:

```
fn consistent_choose_k<Key>(key: Key, k: usize, n: usize) -> Vec<usize> {
    (0..k).rev().scan(n, |n, k| Some(consistent_choose_next(key, k, n))).collect()
}

fn consistent_choose_next<Key>(key: Key, k: usize, n: usize) -> usize {
    (0..k).map(|k| consistent_hash(key, k, n - k) + k).max()
}

fn consistent_hash<Key>(key: Key, k: usize, n: usize) -> usize {
    // compute the k-th independent consistent hash for `key` and `n` nodes.
}
```
