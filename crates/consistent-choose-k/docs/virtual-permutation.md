# VirtualPermutation: sentinel-rooted consistent order

`VirtualPermutation` is an experimental alternative to the unchanged
`ConsistentPermutation`. Both produce distinct nodes, stable `k` prefixes,
and **complete survivor-list restriction**: deleting real node `n` from the
complete order for `n + 1` real nodes recovers the order for `n` real nodes.
Membership is consecutive IDs `0..n`; additions append IDs and removals remove
a suffix. Arbitrary holes, weights and physical-node remapping are out of scope.

| Property | Existing `ConsistentPermutation` | New `VirtualPermutation` |
| --- | --- | --- |
| Membership changes | Insert/delete entries without reordering survivors | Same order-restriction contract |
| Replica ranks | May shift after insert/delete | May shift after insert/delete |
| Rank query | Replay the iterator | Replay the iterator |
| Supported real-node count | `1..=2^30`, `u32` | `1..=u64::MAX - 1`, `u64` |
| Iterator state | Per-layer heap-allocated counters | Four `u64` fields; no allocation |
| Construction | Interleaved per-layer Feistel streams | Consistent single-cycle successor traversal from a sentinel |

This revision **replaces the earlier experimental slot-based variants**.
Those variants did not preserve survivor-list order. The experimental output
mapping has changed, the matched-network variant has been removed, and there
is no longer an absolute constant-time rank API. Existing consumers and the
user's `ConsistentPermutation` implementation/mapping are unchanged.

## API, bounds and key hashing

```rust
use consistent_choose_k::VirtualPermutation;
use std::hash::{DefaultHasher, Hash, Hasher};

let mut hasher = DefaultHasher::new();
"object-key".hash(&mut hasher);
let seed = hasher.finish();
let replicas: Vec<u64> = VirtualPermutation::new(1000, seed).take(3).collect();
let third = VirtualPermutation::new(1000, seed).nth(2); // replays three successors
assert_eq!(third, Some(replicas[2]));
let old: Vec<_> = VirtualPermutation::new(1000, seed).collect();
let restricted: Vec<_> = VirtualPermutation::new(1001, seed)
    .filter(|&node| node != 1000).collect();
assert_eq!(old, restricted);
```

The constructor accepts a well-mixed 64-bit seed. Keep it fixed across all
membership and replica counts. `DefaultHasher` matches the examples and
benchmark convention, but Rust does not promise a stable mapping across
versions. Distributed deployments need a specified, versioned key hash and
identical algorithm versions on all participants. Seed collisions give
identical orders.

Internal label zero is a permanent sentinel; real node `i` has internal label
`i + 1`. Internal count is `n + 1`, so `new(0, seed)` and
`new(u64::MAX, seed)` panic rather than wrap, following the existing
constructor's assertion convention. `n()` returns the original real-node
count. The cloneable, fused iterator reports its remaining size, safely even
when that count exceeds `usize`. `.take(0)` is empty; `.take(n)` is the complete
order; larger requests stop at exhaustion. There is no separate `k` constructor
argument. `nth(r)` uses ordinary iterator replay from the current position;
answering an uncached absolute rank requires replay from a new iterator.

## Ordinary permutation and guaranteed single cycle

For each bit width `b`, let `Q(seed,b)` be an ordinary reversible permutation
of `[0,2^b)`, with domain separation depending only on seed and width. Define
the single-cycle primitive by conjugating modular increment:

```text
P(x)       = Q_inverse((Q(x) + 1) mod 2^b)
P_inverse(x) = Q_inverse((Q(x) - 1) mod 2^b)
```

Each is two ordinary permutation evaluations: first `Q`, then `Q_inverse`.
Increment is a full cycle, so its conjugate is a full cycle for **every** Q,
not merely with high probability. If Q is uniform on all permutations of a
domain of size `m`, each full cycle has exactly `m` conjugators, so P is uniform
on the `(m-1)!` full cycles.

The implemented Q is a noncryptographic alternating-XOR Feistel with the
SplitMix64 finalizer as round mixer. It uses 24 rounds at widths 2--4,
16 at widths 5--7, and 8 at widths 8--64; width one is keyed XOR. This fixed
schedule, inherited from the stronger experimental primitive, has been
rediagnosed in the **new construction**, not assumed adequate from old results.
Unequal halves support odd widths. Width seeds are mixed and round keys use
Weyl offsets. The inverse undoes the same updates in reverse order.

This finite 64-bit family is only a practical pseudorandom approximation,
not exact independent ideal randomness or a proven secure PRP. It cannot
uniformly represent all orders once there are more orders than seeds.
Feistel families also have structural restrictions, such as permutation
parity restrictions on sufficiently large balanced halves. Neither domain
separation, a large round count nor statistical diagnostics proves independence.
Do not use this as encryption or where adversarial keys require cryptographic
security. The existing iterator's different Feistel schedule is not modified.

All word operations are safe through width 64: modular steps use wrapping
arithmetic followed by masking, half widths never exceed 32, and the largest
dyadic half boundary is `1 << 63`. A conceptual domain cardinality `2^64`
is never stored in a `u64`.

## Normalized cycle lift

This is a derived construction/evaluator, not an established production
implementation or an implementation of a published constant-time algorithm.

Let `A=[0,h)` and let P be a full cycle on `[0,2h)`. Let R be P's cycle
projection onto A: follow P until reaching the next old label. Starting with
`F_1(0)=0`, define:

```text
F_(2h) = extend(F_h composed with inverse(R), fixing upper labels) composed with P
```

For every old label `a`, its P-chain goes through zero or more upper labels
and ends at `R(a)`. The lift changes that final destination to `F_h(a)`.
There are no upper-only cycles in P. Reconnecting all old-node chains using
the lower full cycle therefore produces exactly one full cycle. Its cycle
projection onto A is F_h. For intermediate counts, delete all inactive labels
from the larger cycle. Projection composes, including across dyadic boundaries.

**Uniform full cycles under the ideal model.** A full cycle P decomposes into
its projected lower cycle R and one ordered upper-node chain attached to each
old label. Every combination of a lower full cycle and such a chain arrangement
corresponds to exactly one full P. Thus uniform P makes R uniform independently
of the arrangement. The lift replaces R with the independent uniform F_h
without changing the arrangement, giving a uniform full F_(2h). Projection of
a uniform full cycle is uniform: each cycle on `m-1` labels has exactly `m-1`
extensions, inserting the new label after any old label. Induction gives
uniform full cycles at every internal count.

**Rooted list order.** Begin at sentinel zero and repeatedly follow F:

```text
cursor = 0
repeat k times, where 0 <= k <= n:
    cursor = next_consistent(seed, n + 1, cursor)
    emit cursor - 1
```

The sentinel cannot reappear before all `n` real nodes. A full cycle with a
fixed sentinel corresponds bijectively to a real-node order. Cycle deletion
therefore becomes ordinary list deletion: for example `S->A->C->B->S` can
grow into `S->A->D->C->B->S`, preserving the old order. Each ideal real-node
order, ordered prefix, or subset has the appropriate uniform distribution.
Different keys have independent orders **only under independent ideal
primitives across keys**; nodes within a key are sampled without replacement.

Evaluating successor inputs `0,1,...,k-1` instead of following the cursor would
not implement this contract. Internal input-label successor consistency is
not the public output-rank API.

## Constant-space successor evaluator

```text
next_consistent(seed, count, x):
    require 0 <= x < count
    b = bit_length(count - 1)
    while b > 0:
        half = 1 << (b - 1)
        y = P(seed, b, x)
        if y >= count:
            x = y
            continue
        if y >= half:
            return y
        while x >= half:
            x = P_inverse(seed, b, x)
        count = half
        b -= 1
    return 0
```

Edges whose outputs are upper labels are unchanged by the lift, so walking
inactive upper labels can use P directly. Upon reaching a lower output,
the backward walk recovers the old input at the start of that chain; recursion
then supplies its correct lower-cycle destination. This is the explicit lift
without constructing any tables. The actual implementation uses loops, not a
recursive stack. Walks terminate because P is a full finite cycle intersecting
the retained/lower set. No retry cap or mapping-changing fallback is used.

## Expected adaptive-prefix work

The cursor depends on previous outputs. A fixed-input expected-cost bound
would therefore be insufficient. Instead count work over the entire rooted
prefix, under independent uniform ideal Q at each width and constant-cost
forward/inverse ordinary permutation calls. Fix `n,k` before sampling those
primitives. A zero-length prefix does no traversal; the strict bounds below
are for `1<=k<=n`.

Let `M` be the smallest power of two at least `n+1`. The lifted full cycle
F_M (not the raw primitive P) is uniform. Reaching the first `k` active real outputs visits, in expectation,
`k*M/(n+1)` top-level successors: sample without replacement from the
`M-1` non-sentinel labels until the `k`th of the `n` active labels. This is
less than `2k`.

At each lower full dyadic domain of size `L=M/2,M/4,...,2`, the successor
invocations form the rooted lower-label subsequence. Their count equals the
number of selected final internal labels in `1..L-1`, so its expectation is
`k*(L-1)/n`. This uses marginal uniformity of the rooted real-node order, not
independence of adaptive calls. Summing these lower-level expectations gives
less than `k*M/n`, which is at most `2k` for `n>=1`.

At any level, a backward walk retraces an upper-node chain already traversed
by that level's forward prefix. This includes inactive labels just visited
in the top-level walk. Each upper label is retraced at most once; the prefix
does not wrap back through the sentinel. Consequently inverse calls are
bounded pathwise by forward calls across the prefix. Combining the counts
gives a conservative **expected bound below `8k` P/P_inverse calls, or `16k`
ordinary Q/Q_inverse calls**. Width setup is bounded per level invocation and
does not change expected `O(k)` work. This bounds cumulative work from the
sentinel, not work conditioned on an arbitrary already-observed prefix.

This is an ideal-model derivation, not a published complexity theorem or a
guarantee for every finite-key seed. An unlucky full cycle can force
linear-in-domain work even for a short prefix; there is **no worst-case
`O(k)` or adversarial-latency guarantee**. The iterator uses `O(1)` auxiliary
state excluding returned outputs. There is no prebuilt ring, per-key table,
permutation array, cached prefix or duplicate set.

## Verification and references

Tests cover forward/inverse round trips for both Q and P through all 64
widths, guaranteed single-cycle coverage on small domains, explicit
table-based lift/projection equivalence, rooted traversal and sentinel return,
complete survivor-list restriction, `k` prefixes, default `nth` replay, fused
exhaustion, and small/large dyadic and sentinel boundaries. They explicitly
reject overflowing sentinel counts. A constructed long-walk case checks that
evaluation is not truncated.

Exhaustive ideal checks cover all 24 ordinary size-four conjugators and all
30,240 independent full-cycle families at sizes 2,4,8. Every real-node order
at sizes 1 through 7 appears equally often and agrees with the explicit
reference and survivor deletion. These are regression checks, not substitutes
for the ideal-model argument or proofs of security.

The mathematical antecedents concern virtual permutations and cycle projection:

- Neretin, [Virtual permutations and polymorphisms, section 1.2](https://arxiv.org/html/2202.12978v1#S1):
  cycle deletion and equivariance.
- Bourgade, Najnudel and Nikeghbali,
  [A unitary extension of virtual permutations, section 1](https://arxiv.org/html/1102.2633v1#S1):
  cycle projections, Chinese restaurant construction and uniform coherent families.

These sources do **not** supply this evaluator, Feistel schedule, single-cycle
sentinel specialization or performance bound. The
[performance and randomness report](virtual-permutation-performance.md)
records actual final-construction measurements and their limitations.
