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

## Complexity summary

where `N` is the number of nodes and `R` is the number of replicas.

| Algorithm               | Lookup per key      | Node add/remove                        | Memory                    | Lookup with replication             |
|                         | (no replication)    |                                        |                           |                                     |
|-------------------------|---------------------|----------------------------------------|---------------------------|-------------------------------------|
| Hash ring (with vnodes) | O(log N): binary search over N points; O(1): with specialized structures | O(log N) | O(N) | O(log N + R): Take next R distinct successors |
| Rendezvous              | O(N): max score     | O(1)                                   | O(N) node list            | O(N log R): pick top R scores       |
| Jump consistent hash    | O(log(N)) expected  | 0                                      | O(1)                      | O(R log N)                          |
| AnchorHash              | O(1) expected       | O(1)                                   | O(N)                      | Not native                          |
| DXHash                  | O(1) expected       | O(1)                                   | O(N)                      | Not native                          |
| JumpBackHash            | O(1) expected       | 0                                      | O(1)                      | Not native                          |
| **ConsistentChooseK**   | **O(1) expected**   | **0**                                  | **O(1)**                  | **O(R^2)**; **O(R log(R))**: using heap |

Replication of keys
- Hash ring: replicate by walking clockwise to the next R distinct nodes. Virtual nodes help spread replicas more evenly. Replicas are not independently distributed. 
- Rendezvous hashing: replicate by selecting the top R nodes by score for the key. This naturally yields R distinct owners and supports weights.
- Jump consistent hash: the base function doesn't support replication. But the math can be easily modified to support consistent replication.
- JumpBackHash and variants: The trick of Jump consistent hash to support replication won't work here due to the additional state introduced.
- ConsistentChooseK: Faster and more memory efficient than all other solutions.

Why replication matters
- Tolerates node failures and maintenance without data unavailability.
- Distributes read/write load across multiple owners, reducing hotspots.
- Enables fast recovery and higher tail-latency resilience.

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

If `u = m`, then `S(k+1, n) = {m} ∪ S(k, m) ∖ {v} ∪ {w}` and `S(k+1, n+1) = {n} ∪ S(k, n) = {n} ∪ S(k, m) ∖ {v} ∪ {w}` and the two differ exaclty in the elemetns `m` and `n` proving property 3.

If `u ≠ m`, then `consistent_hash(_, k, n) = m`, since that's the only way how the largest values in `S(k+1, n)` and `S(k, n)` can differ. In this case, `m ∉ S(k+1, n+1)`, since `n` (and not `m`) is the largest element of `S(k+1, n+1)`. Furthermore, `S(k, n) = S(k, m)`, since `consistent_hash(_, i, n) < m` for all `i < k` (otherwise there is a contradiction).
Putting it together leads to `S(k+1, n+1) = {n} ∪ S(k, m)` and `S(k+1, n) = {m} ∪ S(k, m)` which differ exactly in the elements `n` and `m` which concludes the proof.

### Property 2

The final part is to prove property 2. This time we have an inducation over `k` and `n`.
As before, induction start for `k=1` and for all `n>0` is inherited from the `consistency_hash` implementation. The case `n=k` is also trivially covered, since the only valid set are the numbers `{0, ..., k-1}` which the algorithm correctly outputs. So, we only need to care about the induction step where `k>1` and `n>k`.

We need to prove that `P(i ∈ S(k+1, n+1)) = (k+1)/(n+1)` for all `0 <= i <= n`. Property 3 already proves the case `i = n`. Furthermore we know that `P(n ∈ S(k+1, n+1)) = (k+1)/(n+1)` and vice versa  `P(n ∉ S(k+1, n+1)) = 1 - (k+1)/(n+1)`. Let's consider those two cases separately.

`n ∈ S(k+1, n+1)`: By the definition of `S`, we know that `S(k+1, n+1) = {n} ∪ S(k, n)`. `P(i ∈ S(k+1, n+1)) = P(i ∈ S(k, n)) P(n ∈ S(k+1, n+1)) = k/n * (k+1)/(n+1)` for all `0 <= i < n`.

`n ∉ S(k+1, n+1)`: Once more by definition, `S(k+1, n+1) = S(k+1, n)` in this case. `P(i ∈ S(k+1, n+1)) = P(i ∈ S(k+1, n)) P(n ∉ S(k+1, n+1)) = (k+1)/n * (1 - (k+1)/(n+1))` for all `0 <= i < n`.

Summing both cases together leads to `P(i ∈ S(k+1, n+1)) = k/n * (k+1)/(n+1) + (k+1)/n * (1 - (k+1)/(n+1)) = k/n * (k+1)/(n+1) + k/n * (1 - (k+1)/(n+1)) + 1/n * (1 - (k+1)/(n+1)) = k/n * (k+1)/(n+1) + k/n - k/n * (k+1)/(n+1) + 1/n - 1/n * (k+1)/(n+1) = k/n + 1/n - 1/n * (k+1)/(n+1) = (k+1)/n - (k+1)/(n+1)/n = (k+1)/n * (1 - 1/(n+1)) = (k+1)/(n+1)` for all `0 <= i < n` which concludes the proof.
