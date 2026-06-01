# Permutation iterator design notes

This document explains the design of [`ConsistentPermutation`], the
per-layer Feistel permutation iterator that this crate uses to drive
its `n`-consistent ranking.

Given a 64-bit `key` and a universe size `n`, the iterator produces a
uniformly distributed permutation of `[0, n)` as a streaming iterator
satisfying:

- **Stateless and deterministic** — no per-call PRNG state outside the
  iterator; two iterators on the same `(key, n)` produce the same
  sequence.
- **`n`-consistent** — when `n` grows by one, the new sequence agrees
  with the old except for the at most one position where the new
  element `n − 1` is inserted (i.e. the underlying choose-`k` family
  satisfies the consistency properties in the main [README]).
- **Fast in the small-`k`, repeated-query regime** — production
  workloads call the iterator with `k` ≪ `n` for many keys, so the
  per-emission cost and the iterator setup cost both matter.
- **Statistically clean for choose-2 set uniformity** — pairs of
  selected nodes must be uniformly distributed over `binom(n, 2)`
  cells over random keys. This is the test that consistently
  distinguishes "looks fine on marginals but is actually biased" from
  "near-random-permutation quality".

[`ConsistentPermutation`]: ../src/consistent_permutation.rs
[README]: ../README.md

## Core idea

### Permutations are automatically `n`-consistent

Fix any permutation `π` of `[0, N)`, and for each `n ≤ N` define
`π_n` as the subsequence of `π` you get by deleting every entry
`≥ n`. Then `{π_n}_{n ≤ N}` is an `n`-consistent family of
permutations: for every `n < N`, `π_{n+1}` equals `π_n` with the
element `n` inserted at one position.

Two things to check:

- **`π_n` is a permutation of `[0, n)`.** `π` is a bijection of
  `[0, N)`, so each value in `[0, n)` appears in `π` exactly once.
  Deleting entries `≥ n` leaves each value in `[0, n)` at exactly
  one position, in their original relative order.
- **Going from `π_n` to `π_{n+1}` only inserts `n`.** The
  construction of `π_{n+1}` differs from that of `π_n` only in
  that the entry equal to `n` is no longer deleted. Every other
  surviving entry sits at the same relative position. So
  `π_{n+1}` is `π_n` with `n` spliced in at the position `π` puts
  it.

So *any* recipe for sampling a permutation of `[0, N)` from a key
automatically gives an `n`-consistent family of permutations for
every `n ≤ N`.

### Hash functions already give us power-of-two permutations

Modern keyed hash functions and block ciphers are by design
*bijections* on fixed-width words: a small Feistel network on
`2b` bits produces a key-parameterized permutation of
`[0, 2^(2b))` in `O(1)` evaluation time. So if `n` happens to be a
power of two `n = 2^B`, the family `π_n` is straightforward — pick
a Feistel of width `B`, and reading the iterator amounts to
evaluating it at `0, 1, 2, …`.

The catch is that `n` is rarely a power of two. The lemma above
lets us take a permutation of `[0, N)` for any `N ≥ n` and read
off the survivors, so picking the smallest power of two `N ≥ n`
(say) and using a single Feistel of width `⌈log₂ N⌉` works
correctness-wise — but begs the question of how expensive the
survivor walk is.

### Why we cannot just pick one big `N`

The crudest instantiation is "pick a giant `N` (say `2^30` for all
queries), construct one permutation of `[0, N)` per key, and read
off the first `k` non-deleted values for any query `(n, k)`".
Correctness is given by the lemma. The cost is not.

Each iterator step evaluates the underlying bijection on the next
counter value and either keeps the output (if `< n`) or discards
it. The expected fraction of kept outputs is `n / N`, so the
expected number of bijection evaluations per emission is `N / n`.
With `N = 2^30` and `n` in the low hundreds this is roughly `10^7`
hash evaluations per emission — unusable.

What we actually want is `O(1)` bijection evaluations per
emission, **independent of how large the universe might
eventually grow**. A construction with cost `O(N / n)` per
emission ties us to picking the smallest possible `N` for each
query — but then we lose the ability to grow `n` cheaply, since
growing `n` past `N` requires a brand-new construction with no
relationship to the old one.

### Layering: building the permutation in doubling buckets

If we could always operate on an `N` that exceeds `n` by only a
constant factor, the per-emission cost above would already be
`O(1)`. The layered construction arranges exactly that, by
splitting the universe into a growing stack of intervals each
roughly the size of the previous one combined.

Slice `[0, ∞)` into doubling *buckets*

```
bucket 0  = [0, 4),
bucket 1  = [4, 16),
bucket 2  = [16, 64),
...
bucket j  = [4^j, 4^(j+1)).
```

Bucket `j` is owned by **layer `j`**, which is fully self-contained:
it knows how to emit the elements of its own bucket in a
key-dependent order, interleaved with calls into layer `j − 1` for
the values in lower buckets. Two pieces of state are all a layer
needs:

- A **bijection `π_j`** on `[0, 4^(j+1))` (constructed in §"Feistel
  (vs PCG)" below from the master key and the layer index).
- A **counter `c_j`** initialized to `0`, walking `0, 1, 2, …`.

A single step of layer `j` is:

1. compute `y = π_j(c_j)`, advance `c_j`;
2. look at the top two bits of `y` (i.e. `y >> 2j`):
   - if they are nonzero (probability `3/4`), `y ∈ [4^j, 4^(j+1)) =`
     bucket `j` — **emit** `y`;
   - if they are `00` (probability `1/4`), `y ∈ [0, 4^j)` — that
     emission belongs to a lower bucket, so **descend** to layer
     `j − 1` and let it produce the value.

To produce one element the iterator starts at the top layer `j_max`
and steps; each step either emits or descends. Termination is when
the top layer's counter has walked its full domain `4^(j_max+1)`.
The top layer additionally gates emissions by `y < n` (the only
bucket that can spill past `n`); lower layers' outputs are
automatically in range.

### Why this works (uniformity, `n`-consistency, `O(1)` per emit)

**Uniformity.** At universe size `N = 4^(j_max+1)` the
construction *is* a permutation of `[0, N)`: layer `j`'s counter
walks `0, …, 4^(j+1) − 1` and `π_j` is a bijection, so layer `j`
emits each value of bucket `j` exactly once and triggers a descent
for each value in `[0, 4^j)`. Summed over layers, every value of
`[0, N)` is emitted exactly once. The relative *order* is governed
by the per-layer keys, so over random keys the iterator output is
a uniformly random permutation of `[0, N)` — modulo the
statistical quality of the per-layer bijection, which is the
subject of the rest of this document.

**`n`-consistency.** Because the result at universe size `N` is a
true permutation, the §"Permutations are automatically
`n`-consistent" lemma kicks in: the iterator's output for any
`n ≤ N` is `π_N` with values `≥ n` deleted, and the family across
all `n ≤ N` is automatically `n`-consistent. The top layer's
`y < n` gate is exactly this deletion.

Growing past `N` adds a new top layer above the existing ones.
Every old layer's bijection and counter walk are unchanged; the
new top layer's descents into the old top layer arrive at counter
values `0, 1, 2, …` in order, which is exactly what the old top
layer saw when it was the root. So at the larger universe size
`N' = 4^(j_max+2)` the construction is again a permutation of
`[0, N')` whose restriction (in the sense of the lemma) to any
`n ≤ N` agrees with the old `π_n`. The universe can therefore be
extended indefinitely without re-specifying or disturbing the
existing structure.

**Amortized cost per emission.** Consider an emission produced
somewhere in the descent. At each non-top layer the single step
emits with probability `3/4` and descends with probability `1/4`,
so the expected number of `π_j` evaluations per emission satisfies

```
E_j = 1 + (1/4) · E_{j − 1},  E_0 = 1
```

which converges to `E_∞ = 4/3` regardless of how many layers are
below. The top layer is slightly worse — emissions whose raw
value lies in `[n, N)` get rejected — but since `j_max` is chosen
as the smallest level with `4^(j_max+1) ≥ n`, we always have
`N ≤ 4n`, so the top-layer in-bucket pass rate is at least `n/N ≥
1/4`. The expected number of top-layer evaluations per kept top
emission is at most `4`. Combined, every iterator step costs a
bounded constant number of layer evaluations, independent of `n`
and `j_max`.

## Overall structure

The rest of this document justifies the three design choices that
the iterator above leaves open:

1. **What bijection drives each layer** — an independently-keyed
   balanced Feistel network.
2. **How many bits a layer call emits** — two.
3. **What `n` we cap at and how the iterator terminates** — `n ≤ 2³⁰`,
   driven by a single top-layer counter.

## Per-layer bijection (vs a single global one)

A tempting alternative is to use a single global *chunk-prefix*
permutation on `u32` — a bijection with the property that the prefix
of length `i` of the output depends only on the prefix of length `i`
of the input — and drive every layer of the descent with that one
permutation. Per-layer setup vanishes; each layer call is a single
tiny permutation evaluation.

The fundamental problem is the **size of the permutation family**
this gives you. A single 64-bit key parameterizes one fixed
chunk-prefix permutation that is then re-used at every layer. The
same key controls every layer's bijection, so cross-layer
correlations are not just possible — they are baked in. For small
`n`, where choose-2 has only `binom(n, 2)` cells to spread over, the
choose-2 χ² test rejects this distribution comfortably below the
0.1 % significance threshold.

Giving every layer its own, independently-keyed bijection breaks the
cross-layer correlation by construction. The cost is one bijection
setup per layer, but with `j_max ≈ log₂(n) / 2`, that is at most ~15
small structs of state for any `n ≤ 2³⁰` — negligible compared to
the iterator's emission stream.

## Feistel (vs PCG)

Once each layer needs its own bijection, the primitive of choice is a
small balanced Feistel network on `[0, 2^n_bits)`. The two main
alternatives are PCG-style multiply-xor-shift hashes (either
stateless per call, or stateful with state preserved across calls);
both lose for reasons worth being explicit about.

### Stateless multi-round PCG

A stateless per-call PCG hash `y = (y ^ c) · a; y ^= y >> shift` with
the key rotated between `R` rounds is statistically clean once `R`
is large enough, but **how** large is severe at small `n_bits`: tiny
layers (`n_bits = 2`) have an F-input space of only 2 values, so the
brute-force "many rounds" knob is the only thing that breaks the
parity correlations. A passing schedule needs on the order of 16
rounds at the small-`n_bits` end.

In addition, the tight serial dependency `y → y·a → y^(y>>shift) →
y·a → …` runs close to its sequential multiply latency on a wide
out-of-order core, so per-round work is essentially un-amortized.
Feistel rounds are also serial, but a Feistel round does roughly the
same per-round work as a PCG round and far fewer of them suffice.

### Stateful single-step PCG

A more clever variant is to give each layer a stateful LCG, preserving
state across iterator calls so a single PCG step (plus a small output
mixer) per call would suffice. The hope is that 1 step × 2 layers
per emission beats Feistel's few rounds × 1 layer per emission.

This fails for a structural reason. A full-period LCG on a
power-of-two modulus `2^k` needs Hull–Dobell parameters: `a ≡ 1 mod 4`
and `c` odd. Both forces conspire to make every consecutive state
pair `(state[i], state[i+1])` differ in bit 0 — `a · s ≡ s mod 2`
since `a` is odd, and `c` flips parity since `c` is odd. The output
of the layer is a key-dependent bijection of the state, but the
**state-pair** support is only half of the full pair space. At small
`n_bits` (e.g. 3) that reduces reachable `(state[i], state[i+1])`
pairs from 64 to 32, and no amount of output mixing recovers the
missing half — bit 0 of one output is fully determined by bit 0 of
the previous output (modulo the key-dependent mixer's bijection on
each individual state). The result is a hard floor on choose-2 χ² at
small `n` that no extra mixing round can dissolve.

The standard PCG construction sidesteps this by using a **wider**
state than its output (e.g. 64 bits in, 32 bits out, hiding the
parity-alternation in unused bits). That option is not available
here without breaking per-layer bijectivity: if the LCG cycle length
exceeds the layer's output domain, the output repeats values within
a single iterator lifetime.

### Feistel

Each layer `j` runs a small balanced Feistel network on
`[0, 2^(2j + 2))`, keyed by a single avalanched 64-bit `master_key`
that is rotated and Weyl-incremented between rounds. The F-function
is one `xor`, one multiplication by an odd 32-bit slice of the master
key, and a high-to-low fold to the half-width:

```text
mixed = (r ^ k_xor) * (k_mul | 1)
f     = (mixed_low + mixed_high) & half_mask
new_r = l ^ f
new_l = r
```

The properties that matter:

- **Bijective by construction.** No cycle walking, no rejection, no
  probabilistic restarts. Every Feistel round is a permutation on
  the layer's domain; their composition is too.
- **Small-`n_bits` mixing handled by extra rounds.** Tiny layers
  (`n_bits = 2, 4, 6`) use `12, 10, 6` rounds; everything else uses
  `4`. The extra rounds for tiny layers are essentially free in
  amortized cost because level `j` only contributes ~`4^j` emissions
  per iterator lifetime — the top layer dominates per-iterator cost
  and uses the fewest rounds.
- **Key rotation amortizes a 64-bit master across all rounds.** We
  rotate by exactly the bits consumed per round and add the Weyl
  increment `0x9E37_79B9_7F4A_7C15` to decorrelate consecutive rounds
  (rotation alone leaves 60+ bits in common between adjacent rounds).
- **All `n_bits` even** means the Feistel halves split cleanly with
  no parity quirks, and the 2-bit-per-layer emission rule lines up
  neatly with the per-layer top bits.

## Two bits per layer (vs one)

A 1-bit-per-layer scheme is initially appealing: narrower per-layer
width means fewer Feistel rounds suffice, and the per-layer call is
correspondingly cheaper. Two compounding effects make it lose
overall:

- **Twice as many layers.** With 1 bit per layer,
  `j_max ≈ log₂(n)`; with 2 bits, `j_max ≈ log₂(n) / 2`. Setup cost
  (per-layer counter, per-layer state) doubles.
- **Lower emission probability per layer call.** A 1-bit-per-layer
  scheme emits whenever the top bit is `1` — probability ½. A
  2-bit-per-layer scheme emits whenever the top two bits are not
  `00` — probability ¾. So the **expected number of layer calls per
  emission** drops from `≈ 2` to `≈ 4/3`.

Combining the two effects, 2-bit-per-layer makes ~½ as many
`layer_apply` calls per emission as 1-bit-per-layer. Even after
accounting for the per-call cost difference (slightly more Feistel
rounds at wider `n_bits`), the 2-bit scheme wins comfortably.

Forcing the per-layer width even (which is what "2 bits per layer"
buys structurally) also keeps the Feistel halves perfectly symmetric
and removes any need for cycle walking to stay inside `[0, n)` at the
top.

## Universe cap: `n ≤ 2³⁰`

Capping `n` at `2³⁰` lets the top-layer counter live in a `u32`,
which makes the iterator's termination condition a single
`counters[j_max] >= top_cap` comparison. No separate "emitted"
counter is needed to disambiguate boundary cases, the
`n_bits == 32` corner of the per-layer mask computation vanishes,
and `n_mask = (1 << n_bits) - 1` becomes unconditional inside
`layer_apply`. The cap is generous: `2³⁰ ≈ 10⁹` slots is more
node-count than any realistic distributed system, and the
simplifications it enables are observable in the benchmark.

## Hot-path structure: top-layer phase vs descent phase

The iterator's `next()` is split into two phases:

- The **top-layer phase** walks the top layer's counter with both
  the termination check (`counters[j_max] >= top_cap`) and the range
  check (`raw < n`). The range check is necessary because the
  Feistel maps the counter into `[0, 2^(2·j_max + 2))`, which may
  overshoot `n`.
- The **descent phase** walks lower layers with **neither** check.
  By the descent-tree invariant, lower-layer counters never exhaust
  within a single iterator lifetime, and lower-layer raw outputs
  always land in `[2^j, 2^(j+1)) ⊂ [0, n)`. So the hot path for
  non-top layers is a pure Feistel call with no branches.

## Measured throughput

The `grow_k_vs_permutation` benchmark in
`benchmarks/performance.rs` constructs an iterator per key and pulls
`k` values from it, including the iterator-setup cost. Numbers
below are mean per-emitted-element times in nanoseconds on an Apple
M4 Max, for `ConsistentChooseKHasher::new_with_k` (the heap-based
implementation) versus `ConsistentPermutation::new` (this
construction). Smaller is better.

| `n`     | `k`   | choose-k (ns/emit) | permutation (ns/emit) | speedup |
|---------|------:|-------------------:|----------------------:|--------:|
| 100     |     2 |               41.8 |                  18.9 |   2.2 × |
| 100     |    10 |               59.7 |                  14.3 |   4.2 × |
| 100     |   100 |              172   |                  22.1 |   7.8 × |
| 1 000   |     2 |               41.6 |                  14.9 |   2.8 × |
| 1 000   |    10 |               52.8 |                   7.3 |   7.2 × |
| 1 000   |   100 |               93.2 |                   7.5 |  12   × |
| 1 000   | 1 000 |              494   |                   9.8 |  51   × |
| 10 000  |     2 |               51.7 |                  16.4 |   3.2 × |
| 10 000  |    10 |               64.2 |                   8.8 |   7.3 × |
| 10 000  |   100 |              108   |                  12.1 |   8.9 × |
| 10 000  | 1 000 |              319   |                  19.8 |  16   × |
| 100 000 |     2 |               44.4 |                  19.1 |   2.3 × |
| 100 000 |    10 |               59.8 |                  12.1 |   5.0 × |
| 100 000 |   100 |              102   |                  19.0 |   5.4 × |
| 100 000 | 1 000 |              302   |                  25.5 |  12   × |

Observations matching the theory:

- **Per-emit cost of `ConsistentPermutation` is bounded by a small
  constant** (single-digit to low-tens of nanoseconds) across four
  decades of `n` and `k`. The amortized `O(1)` layer evaluations
  per emission analysis predicts exactly this.
- **The setup cost is amortized away by `k`.** At small `k = 2` the
  per-emission number is dominated by iterator construction
  (counter vector allocation), which is why
  `k = 2` columns are systematically slower than `k = 10`.
- **Per-emit cost of `ConsistentChooseKHasher` grows roughly with
  `k`** (heap operations cost `O(log k)` per pop and the
  pre-build option is `O(k log k)`), which is why the speedup ratio
  widens with `k`: at `k = 1 000, n = 1 000` the permutation
  implementation is ~50× faster.

