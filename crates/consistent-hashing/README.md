# Consistent Hashing

Consistent hashing maps keys to a changing set of nodes (shards, servers) so that when nodes join or leave, only a small fraction of keys move. It is used in distributed caches, databases, object stores, and load balancers to achieve scalability and high availability with minimal data reshuffling.

Common algorithms
- [Consistent hashing](https://en.wikipedia.org/wiki/Consistent_hashing) (hash ring with virtual nodes)
- [Rendezvous hashing](https://en.wikipedia.org/wiki/Rendezvous_hashing)
- [Jump consistent hash](https://arxiv.org/pdf/1406.2294)
- [Maglev hashing](https://research.google/pubs/pub44824) 
- [AnchorHash: A Scalable Consistent Hash](https://arxiv.org/abs/1812.09674)
- [DXHash](https://arxiv.org/abs/2107.07930)
- [JumpBackHash](https://arxiv.org/abs/2403.18682)

## Core idea

Many consistent-hashing algorithms are best understood as specialized solutions
to one higher-level problem: primary placement, replication, bounded load,
failover, or arbitrary deletions. A single flat complexity table is often
misleading because those algorithms do not all expose the same operations.

This crate instead centers on `ConsistentChooseK`: a stateless per-key ranking
of all nodes. The first item is the primary owner, the first `R` items are
replicas, the next item after a failed node is its failover target, and the same
ranking can drive bounded-load and deletion-tolerant assignment. The current
implementation extracts the first `R` distinct candidates in `O(R^2)` time
(`O(R log R)` with a heap optimization) and uses no persistent memory.

Replication of keys
- Hash ring: replicate by walking clockwise to the next R distinct nodes. Virtual nodes help spread replicas more evenly. Replicas are not independently distributed.
- Rendezvous hashing: replicate by selecting the top R nodes by score for the key. This naturally yields R distinct owners and supports weights.
- Jump consistent hash: the base function doesn't support replication. While the math can be modified to support consistent replication, it cannot be efficiently solved for large k and even for small k (=2 or =3), a quadratic or cubic equation has to be solved.
- JumpBackHash and variants: The trick of Jump consistent hash to support replication won't work here due to the introduction of additional state.
- ConsistentChooseK: produces an ordered list of distinct, consistent candidates directly, making replication and related higher-level policies simple compositions over the same primitive.

Why replication matters
- Tolerates node failures and maintenance without data unavailability.
- Distributes read/write load across multiple owners, reducing hotspots.
- Enables fast recovery and higher tail-latency resilience.

## Applications beyond replication

The `ConsistentChooseK` iterator produces a per-key ranking of all `n` nodes in priority order — consistently and with zero memory overhead. This ranking is a strict superset of simple replication and enables drop-in replacements for several well-known algorithms that traditionally require maintaining expensive data structures such as hash rings.

### Bounded-load consistent hashing

[Consistent Hashing with Bounded Loads](https://research.google/pubs/pub46580/) (Mirrokni et al., 2018) caps the maximum load any single node may receive. When a key's preferred node is full, it overflows to the next candidate. Classic implementations walk a hash ring to find successors, requiring O(V·N) memory for the ring where V is the number of virtual nodes per physical node (typically V > 100–200 for acceptable load variance). Lookups cost O(log(V·N)) via binary search.

With `ConsistentChooseK`, the ranking iterator directly yields each key's preference list on the fly — no ring required. Assignment becomes: iterate tokens round by round, and for each token advance its ranking iterator until a node with remaining capacity is found. This achieves the same bounded-load guarantees with O(k) for k keys and O(k) time to extract the k-th key.

See [`examples/bounded_load.rs`](examples/bounded_load.rs) for a working implementation.

### Power of two choices

The [power of two choices](https://www.eecs.harvard.edu/~michaelm/postscripts/mythesis.pdf) paradigm (Mitzenmacher, 2001; Azar et al., 1999) assigns each key to the least-loaded of two (or d) randomly chosen nodes. This reduces maximum load from O(log n / log log n) to O(log log n / log d) with high probability.

Traditionally this requires drawing d independent random nodes per key. However, the original algorithm ignores the corner case where multiple independent hash functions collide on the same node, effectively reducing the number of distinct choices below d. With `ConsistentChooseK`, the first d elements from the ranking iterator are guaranteed to be distinct nodes. The choices are also consistent across time — the same key always considers the same d candidates — so reassignment only happens when a node actually joins or leaves.

### Priority-based failover

In active-passive or tiered architectures, each key needs a deterministic failover order. The ranking iterator provides exactly this: the first node is the primary, the second is the hot standby, and so on. When a node fails, the next node in the ranking takes over — consistently for all keys that had the failed node at the same rank position, and without any coordination or ring rebalancing.

### Deletion-tolerant node maps

`ConsistentNodeMap` uses the `ConsistentChooseK` ranking to support arbitrary node deletions with very small state. It stores only the total slot count and the set of deleted slots. Lookup generates the per-key choose-k ranking and returns the first slot that is not deleted.

This solves the same deletion problem targeted by AnchorHash, MementoHash, and DxHash: when a node is removed, only keys assigned to that node move, and they are redistributed uniformly over the remaining nodes. The difference is that those algorithms keep replacement or redirect metadata that encodes enough of the removal history to repair hits on deleted nodes. `ConsistentNodeMap` is history-independent: it only needs the current deleted set.

For many practical deployments, this also makes `ConsistentNodeMap` a compelling replacement for traditional hash-ring implementations with virtual nodes. Rings typically need hundreds of virtual nodes per physical node to obtain good balance, which makes their memory footprint orders of magnitude larger than the actual node set. Here the ranking is generated directly from the key, so deletion support only adds state proportional to the number of deleted slots rather than to a large virtual-node ring.

The tradeoff is lookup work. If `h` deleted slots are encountered before the first live slot, the current iterator costs `O((h + 1)^2)` because producing the i-th choose-k candidate costs O(i). The expected number of deleted-node hits has the same harmonic/log behavior analyzed for history-based approaches, approximately `ln(total / active)` when `total` slots contain `active` live nodes. Thus the total expected lookup cost is `O((1 + ln(total / active))^2)`.

| Algorithm | Total lookup time | Add node | Remove node | State | Predefined capacity? | History-dependent? |
|-----------|-------------------|----------|-------------|-------|----------------------|--------------------|
| Hash ring with `V` virtual nodes | `O(log(V·active))` | `O(V log(V·active))` | `O(V log(V·active))` | `O(V·active)` ring entries | No | No |
| `ConsistentNodeMap` | `O((h + 1)^2)`, expected `O((1 + ln(total / active))^2)` | `O(1)` expected | `O(1)` expected | `O(deleted)` deleted-slot set | No | No |
| AnchorHash | `O((h + 1)^2)`, expected `O((1 + ln(total / active))^2)` | `O(1)` expected | `O(1)` expected | `O(capacity)` anchor/removal state | Yes | Yes |
| MementoHash | `O((h + 1)^2)`, expected `O((1 + ln(total / active))^2)` | `O(1)` expected | `O(1)` expected | `O(deleted)` replacement tuples | No | Yes |
| DxHash | `O((h + 1)^2)`, expected `O((1 + ln(total / active))^2)` | `O(1)` expected | `O(1)` expected | `O(capacity)` redirect/displacement state with smaller constants than AnchorHash | Yes | Yes |

## ConsistentChooseK algorithm

The following functions summarize the core algorithmic innovation as a minimal Rust excerpt.
`n` is the number of nodes and `k` is the number of desired replica.
The chosen nodes are returned as distinct integers in the range `0..n`.

```
fn consistent_choose_k<Key>(key: Key, k: usize, n: usize) -> Vec<usize> {
    (0..k).rev().scan(n, |n, k| Some(consistent_choose_max(key, k + 1, n))).collect()
}

fn consistent_choose_max<Key>(key: Key, k: usize, n: usize) -> usize {
    (0..k).map(|k| consistent_hash(key, k, n - k) + k).max()
}

fn consistent_hash<Key>(key: Key, i: usize, n: usize) -> usize {
    // compute the i-th independent consistent hash for `key` and `n` nodes.
}
```

`consistent_choose_k` makes `k` calls to `consistent_choose_max` which calls `consistent_hash` another `k` times.
In total, `consistent_hash` is called `k * (k+1) / 2` many times. Utilizing a `O(1)` solution for `consistent_hash` leads to a `O(k^2)` runtime.
This runtime can be further improved by replacing the max operation with a heap where popped elements are updated according to the new arguments `n` and `k`.
With this optimization, the complexity reduces to `O(k log k)`.
With some probabilistic bucketing strategy, it should be possible to reduce the expected runtime to `O(k)`.
For small `k` neither optimization is probably improving the actual performance though.

The next section proves the correctness of this algorithm.

## N-Choose-K replication

We define the consistent `n-choose-k` replication as follows:

1. For a given number `n` of nodes, choose `k` distinct nodes `S`.
2. For a given `key` the chosen set of nodes must be uniformly chosen from all possible sets of size `k`.
3. When `n` increases by one, exactly one node in the chosen set will be changed.
4. and the node will be changed with probability `k/(n+1)`.

In the remainder of this section we prove that the `consistent_choose_k` algorithm satisfies those properties.

Let's define `M(k,n) = consistent_choose_max(_, k, n)` and `S(k, n) := consistent_choose_k(_, k, n)` as short-cuts for some arbitrary fixed `key`.
We assume that `consistent_hash(key, k, n)` computes `k` independent consistent hash functions.

### Property 1

Since `M(k, n) < n` and `S(k, n) = {M(k, n)} ∪ S(k - 1, M(k, n))` for `k > 1`, `S(k, n)` constructs a strictly monotonically decreasing sequence. The sequence outputs exactly `k` elements which therefore must all be distinct which proves property 1 for `k <= n`.

Properties 2, 3, and 4 can be proven via induction as follows.

### Property 4

`k = 1`: We expect that `consistent_hash` returns a single uniformly distributed node index which is consistent in `n`, i.e. changes the hash value with probability `1/(n+1)`, when `n` increments by one. In our implementation, we use an `O(1)` implementation of the jump-hash algorithm. For `k=1`, `consistent_choose_k(key, 1, n)` becomes a single function call to `consistent_choose_max(key, 1, n)` which in turn calls `consistent_hash(key, 0, n)`. I.e. `consistent_choose_k` inherits the all the desired properties from `consistent_hash` for `k=1` and all `n>=1`.

`k → k+1`: `M(k+1, n+1) = M(k+1, n)` iff `M(k, n+1) < n` and `consistent_hash(_, k, n+1-k) < n - k`. The probability for this is `(n+1-k)/(n+1)` for the former by induction and `(n-k)/(n+1-k)` by the assumption that `consistent_hash` is a proper consistent hash function. Since both these probabilities are assumed to be independent, the probability that our initial value changes is `1 - (n+1-k)/(n+1) * (n-k)/(n+1-k) = 1 - (n-k)/(n+1) = (k+1)/(n+1)` proving property 4.

### Property 3

Property 3 is trivially satisfied if `S(k+1, n+1) = S(k+1, n)`. So, we focus on the case where `S(k+1, n+1) != S(k+1, n)`, which implies that `n ∈ S(k+1, n+1)` as largest element.
We know that `S(k+1, n) = {m} ∪ S(k, m)` for some `m` by definition and `S(k, n) = S(k, u) ∖ {v} ∪ {w}` by induction for some `u`, `v`, and `w`. Thus far we have `S(k+1, n+1) = {n} ∪ S(k, n) = {n} ∪ S(k, u) ∖ {v} ∪ {w}`.

If `u = m`, then `S(k+1, n) = {m} ∪ S(k, m) ∖ {v} ∪ {w}` and `S(k+1, n+1) = {n} ∪ S(k, n) = {n} ∪ S(k, m) ∖ {v} ∪ {w}` and the two differ exactly in the elemetns `m` and `n` proving property 3.

If `u ≠ m`, then `consistent_hash(_, k, n) = m`, since that's the only way how the largest values in `S(k+1, n)` and `S(k, n)` can differ. In this case, `m ∉ S(k+1, n+1)`, since `n` (and not `m`) is the largest element of `S(k+1, n+1)`. Furthermore, `S(k, n) = S(k, m)`, since `consistent_hash(_, i, n) < m` for all `i < k` (otherwise there is a contradiction).
Putting it together leads to `S(k+1, n+1) = {n} ∪ S(k, m)` and `S(k+1, n) = {m} ∪ S(k, m)` which differ exactly in the elements `n` and `m` which concludes the proof.

### Property 2

The final part is to prove property 2. This time we have an inducation over `k` and `n`.
As before, the base case of the induction for `k=1` and all `n>0` is inherited from the `consistency_hash` implementation. The case `n=k` is also trivially covered, since the only valid set are the numbers `{0, ..., k-1}` which the algorithm correctly outputs. So, we only need to care about the induction step where `k>1` and `n>k`.

We need to prove that `P(i ∈ S(k+1, n+1)) = (k+1)/(n+1)` for all `0 <= i <= n`. Property 3 already proves the case `i = n`. Furthermore we know that `P(n ∈ S(k+1, n+1)) = (k+1)/(n+1)` and vice versa  `P(n ∉ S(k+1, n+1)) = 1 - (k+1)/(n+1)`. Let's consider those two cases separately.

`n ∈ S(k+1, n+1)`: By the definition of `S`, we know that `S(k+1, n+1) = {n} ∪ S(k, n)`. `P(i ∈ S(k+1, n+1)) = P(i ∈ S(k, n)) P(n ∈ S(k+1, n+1)) = k/n * (k+1)/(n+1)` for all `0 <= i < n`.

`n ∉ S(k+1, n+1)`: Once more by definition, `S(k+1, n+1) = S(k+1, n)` in this case. `P(i ∈ S(k+1, n+1)) = P(i ∈ S(k+1, n)) P(n ∉ S(k+1, n+1)) = (k+1)/n * (1 - (k+1)/(n+1))` for all `0 <= i < n`.

Summing both cases together leads to `P(i ∈ S(k+1, n+1)) = k/n * (k+1)/(n+1) + (k+1)/n * (1 - (k+1)/(n+1)) = k/n * (k+1)/(n+1) + k/n * (1 - (k+1)/(n+1)) + 1/n * (1 - (k+1)/(n+1)) = k/n * (k+1)/(n+1) + k/n - k/n * (k+1)/(n+1) + 1/n - 1/n * (k+1)/(n+1) = k/n + 1/n - 1/n * (k+1)/(n+1) = (k+1)/n - (k+1)/(n+1)/n = (k+1)/n * (1 - 1/(n+1)) = (k+1)/(n+1)` for all `0 <= i < n` which concludes the proof.
