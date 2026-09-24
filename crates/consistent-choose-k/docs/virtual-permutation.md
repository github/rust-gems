# VirtualPermutation: cycle-consistent replica slots

`VirtualPermutation` is an additional experimental algorithm, not a replacement
for `ConsistentPermutation`. Both produce deterministic, distinct candidates,
and every `k`-selection is a prefix of the same per-key list. Their membership
semantics are **different**:

| Property | Existing `ConsistentPermutation` | New `VirtualPermutation` |
| --- | --- | --- |
| Append one node | Insert it into the ranking; later replica slots can shift | At most one old replica slot changes, to the new node |
| Remove the last node | Remove it from the ranking; later slots can shift | Only a surviving slot that named the removed node changes |
| Survivor list order | Preserved | Not promised |
| Projection between sizes | Delete entries from the output list | Delete labels from the permutation's cycles |
| Direct slot lookup | Replay iterator through that slot | `replica_at(slot)`, without replay |
| Supported `n` | `1..=2^30` (`u32`) | `1..=u64::MAX` |
| Mutable state | Per-layer `Vec<u32>` counters | Three `u64` fields; no heap allocation |

For example, the permutation written as an output list `[2, 0, 1]` is the
cycle `0 -> 2 -> 1 -> 0`. Cycle-deleting node 2 produces `[1, 0]`,
**not** the survivor list `[0, 1]`. The changed old slot is 0, the slot that
named the removed node. This difference matters for failover and bounded-load
policies that depend on survivor priority; do not substitute the new API into
`ConsistentNodeMap` or existing consumers without considering their semantics.

Membership is consecutive IDs `0..n`. Additions append IDs and removals remove
a suffix; arbitrary holes, weights and physical-node remapping are out of scope.
The one-slot bound is per single-node change, not for an entire suffix at once.

## API and randomness model

```rust
use consistent_choose_k::VirtualPermutation;
use std::hash::{DefaultHasher, Hash, Hasher};

let mut hasher = DefaultHasher::new();
"object-key".hash(&mut hasher);
let seed = hasher.finish();
let permutation = VirtualPermutation::new(1_000, seed);
let third_replica = permutation.replica_at(2);
let replicas: Vec<u64> = permutation.take(3).collect();
assert_eq!(third_replica, replicas[2]);
```

Like the existing iterator, the constructor accepts an already well-mixed
64-bit seed, not an arbitrary application key. Use the same seed for all `n`
and `k`. `DefaultHasher` above is convenient for local examples and matches the
benchmark convention; Rust does not promise its mapping is stable across
versions. Distributed deployments need a specified, versioned key hash and
identical algorithm versions on every participant.

`new(0, seed)` and `replica_at(slot >= n)` panic, following the existing
constructor's assertion convention. `replica_at` is absolute, independent of
the cursor. The iterator is cloneable and fused, reports its remaining size,
and implements `nth` without replay. `.take(0)` is empty, `.take(n)` is the full
permutation, and `.take(k)` for `k > n` stops at exhaustion, as usual for Rust
iterators. There is no separate `k` constructor argument.

Under **independent uniform ideal permutations for each key and bit width**,
the construction below gives a uniform permutation for every fixed `n`:
each ordered `k`-tuple of distinct nodes is equally likely, and different keys'
preference permutations are independent. Within a key, replicas are sampled
without replacement, not independently.

The actual primitive is an alternating-XOR Feistel network with SplitMix64's
avalanche finalizer as its round mixer. It uses 24 rounds for widths 2 through
4, 16 for widths 5 through 7, and 8 for widths 8 through 64. Tiny half-domains
need extra rounds: the initial eight-round version had a strong ordered-pair
bias at width three despite clean marginals. This fixed policy depends **only**
on bit width, never active `n`, requested `k`, or observed outputs. Unequal
halves support odd widths; the one-bit case is a keyed XOR.
Widths are domain-separated through a
mixed seed; round keys use distinct Weyl offsets. The inverse undoes the same
updates in reverse order. Geometry never requires a shift by 64: the largest
half-width is 32 and the evaluator's largest half boundary is `1 << 63`.
The conceptual full width-64 domain has size `2^64`, but that cardinality is
never materialized in a `u64`.

This finite, 64-bit seeded family is **noncryptographic and only a practical
pseudorandom approximation**, not independently sampled ideal permutations,
not a proven secure PRP, and not exactly uniform over all `n!` permutations.
Domain separation and avalanche do not prove independence. Seed collisions
give identical permutations. Even conventional XOR-Feistel families have
structural restrictions (for example, even permutation parity when both
halves have at least two bits). The mixing schedule is fixed rather than
weakened to make a benchmark win. Do not use this as encryption or with
adversarial keys requiring a cryptographic guarantee.

The existing `layer_apply` is deliberately unchanged: it only supports even
widths through 30, has no inverse, and has its own key/round schedule. Extending
or replacing it would change existing mappings. The new private primitive
therefore lives with the new evaluator.

## Dyadic lift and cycle deletion

The following is a derived construction and evaluator, not an implementation
of a published constant-time replica-selection algorithm.

Let `h = 2^(b-1)`, let `A = [0,h)` be the old labels, and let `Q = P(seed,b)`
be an ordinary permutation of `[0,2h)`. Let `R` be its cycle projection onto
`A`: follow `Q` until the next label in `A`. Starting with `F_1(0) = 0`, define

```text
F_(2h) = extend(F_h composed with inverse(R), fixing upper labels) composed with Q
```

Every `Q`-chain starting at old label `a` ends at old label `R(a)`.
The left composition changes only that chain's final destination, from
`R(a)` to `F_h(a)`. Upper edges and upper-only cycles are unchanged. Thus
cycle-projecting `F_(2h)` onto `A` gives `F_h`. For `h < n < 2h`, define `F_n`
by deleting all labels at least `n` from `F_(2h)`'s cycles. Cycle projections
compose, including across power-of-two boundaries.

**Uniformity in the ideal model.** Conditional on any fixed `Q`, independent
uniform `F_h` makes `F_h composed with inverse(R)` uniform on the old-label
permutation group. This factor is consequently independent of `Q`; composing
it with uniform `Q` makes `F_(2h)` uniform. Cycle projection preserves
uniformity: each permutation of `n-1` labels has exactly `n` extensions
(insert the new label after any old label, or as a singleton cycle).
Induction establishes uniformity at every size. This proof uses ideal
uniformity, not the diagnostics of the implemented finite-key family.

**Consistency.** Deleting label `n` from `F_(n+1)` only redirects its
predecessor to its successor, or removes its singleton cycle. For each
surviving input `r < n`, `F_(n+1)(r)` is therefore either `F_n(r)` or `n`.
At most one old input changes. A fixed prefix of input slots inherits this
property, while bijectivity supplies distinct outputs and prefix stability.
Enumerate *distinct inputs* `0,1,...,k-1`; following one output cycle instead
would not enumerate a full permutation.

## Evaluator

```text
replica_at(seed, n, r):
    require 0 <= r < n
    b = bit_length(n - 1)
    x = r
    while b > 0:
        half = 1 << (b - 1)
        y = P(seed, b, x)
        if y >= n:
            x = y
            continue
        if y >= half:
            return y
        while x >= half:
            x = P_inverse(seed, b, x)
        n = half
        b -= 1
    return 0
```

Walking inactive upper labels can use `Q` rather than the recursively defined
`F`, because edges whose outputs are upper labels were unchanged by the lift.
On reaching a lower output `y`, walking backward from its predecessor `x`
finds the old input `a = inverse(R)(y)`. The next level evaluates `F_h(a)`.
This proves the evaluator agrees with the explicit lift.

Walks terminate because they follow a finite bijection's cycle. A forward walk
starts in the retained set, so it cannot remain forever in an inactive-only
cycle. A backward walk is entered only after reaching a lower label, so that
cycle necessarily meets the lower half. There are no retry caps or mapping-
changing fallbacks.

## Expected work, not a worst-case guarantee

Count each forward or inverse permutation evaluation as one constant-cost
operation. The following bounds are derived for **ideal independent uniform**
permutations and a fixed input, not asserted as a published theorem or a
guarantee for every seed of the implemented mixer.

At a full dyadic level, descent probability is exactly one half. For an
upper input conditional on descent, the mean inverse-walk length is
`2h/(h+1)`: predecessors are sampled without replacement from `2h-1` labels,
`h` of them lower. The terminal lower input is uniform. If `T_s(a)` is mean
cost at full size `s`, and `U_s` its average over inputs, then

```text
T_(2h)(a) = 1 + T_h(a)/2                  if a < h
T_(2h)(a) = 1 + h/(h+1) + U_h/2           if a >= h
U_(2h)    = 1 + h/(2(h+1)) + U_h/2
T_1 = U_1 = 0
```

Induction gives `U_s < 3` and `T_s(a) < 3.5`. The initial partial level is
different: its descent probability `p = h/n` can approach one. Its forward
length has mean `ell = (2h+1)/(n+1) < 2`; the retained endpoint is uniform
and independent of that length. On descent, the inverse walk retraces those
`ell-1` inactive edges. Starting from an upper input also requires finding a
lower predecessor; conditional on forward length `L`, this takes
`(2h-L+1)/(h+1)` further inverse calls on average. Thus

```text
E C_n(r) = ell + p * (ell - 1 + T_h(r))                          if r < h
E C_n(r) = ell + p * (ell - 1 + (2h+1-ell)/(h+1) + U_h)          if r >= h
```

In particular, mean total work is **less than eight primitive calls per
slot**, uniformly in `n,r` in the ideal model. The upper-input bound approaches
eight at `n=h+1, r=h` as `h` grows. All later levels are full, so descent is
geometric after the initial level. The entering lower input depends on upper
permutations, but is independent of lower-width permutations; no independence
between walks, or between replica costs, is assumed. Linearity of expectation
gives expected `O(k)` enumeration.

Long cycles still permit linear-in-domain walks for an unlucky permutation.
This is **not worst-case `O(k)`**, nor an adversarial-latency guarantee. State
is `O(1)` machine words, excluding optional returned output, with a
constant-sized width parameter block and no recursive stack, ring, permutation
array, or duplicate set. See [measurements and diagnostics](virtual-permutation-performance.md)
for observed forward/inverse counts and tails on the actual mixer.

## References and verification

The mathematical antecedent is a *virtual permutation*, using **cycle**
projection:

- Neretin, [Virtual permutations and polymorphisms, section 1.2](https://arxiv.org/html/2202.12978v1#S1):
  cycle deletion and its equivariance.
- Bourgade, Najnudel and Nikeghbali,
  [A unitary extension of virtual permutations, section 1](https://arxiv.org/html/1102.2633v1#S1):
  cycle projections, the Chinese restaurant construction, and the uniform
  family as the Ewens parameter-one case.

These references do **not** supply this evaluator, Feistel schedule, or its
performance bound.

Tests exercise all word widths and extreme values, exhaustive small PRP
round-trips, full small-domain permutations, `k` prefixes, append/delete
consistency including dyadic boundaries and `u64::MAX`, and an independent
table-based lift/projection oracle. Exhaustive ideal permutations check equal
lift multiplicities at size four and all extensions of one fixed size-four
permutation through sizes five to eight. These are regression checks, not
substitutes for the ideal-model proof or evidence of cryptographic security.
