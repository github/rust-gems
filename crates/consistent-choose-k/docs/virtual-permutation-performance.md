# Sentinel-rooted permutation comparison

The revised `VirtualPermutation` preserves complete survivor-list order,
like the unchanged `ConsistentPermutation`, but is **substantially slower**
in this implementation. It removes heap state, not computational work.
Fresh-key `n=1000,k=3` costs 351.25 versus 36.48 ns/query (9.63x slower);
full enumeration costs 195,818.68 versus 10,761.56 ns (18.20x slower).
No comparably large, repeatable distribution deviations appeared for the
new construction in the primary/held-out diagnostic matrix; that is not
a proof of uniformity or independence.

All results below were measured anew on the **single-cycle plus sentinel**
construction. They replace the earlier slot-based and matched-network reports.
There is no constant-time output-rank lookup or direct-slot speedup:
both iterators replay a prefix for `nth`. See the
[design, bounds and ideal-model derivation](virtual-permutation.md).
Runner names are `layered` (existing) and `sentinel` (new).

## Reproduce

From the repository root, run sequentially, without other CPU-heavy work:

```sh
cargo test -p consistent-choose-k
cargo test --release -p consistent-choose-k
cargo bench -p consistent-choose-k-benchmarks --bench replica_comparison -- --noplot
python3 crates/consistent-choose-k/benchmarks/summarize_replica_comparison.py \
  target/criterion > comparison.csv
cargo run --release -p consistent-choose-k --example permutation_diagnostics \
  > diagnostics.csv
cargo run --release -p consistent-choose-k --example permutation_diagnostics \
  -- --held-out > held-out.csv
cargo test --release -p consistent-choose-k operation_count_diagnostics \
  -- --ignored --nocapture
```

A Criterion filter can select a bounded rerun, for example
`-- 'sentinel_replicas/fresh/.*/n1000_k3$' --noplot`.
The new `sentinel_replicas/` result namespace excludes stale measurements of
the replaced algorithms. Raw samples/estimates remain under `target/criterion`.
The CSV script reports mean ns/query, bootstrap 95% confidence limits, sample
standard deviation and ns/output. Criterion's console times are **batches of
128 queries**: divide by 128 for ns/query, then by `k` for ns/output.
`rank_replay` returns only one output, so its ns/output equals ns/query.

### Environment and limitations

Measured September 30, 2026, on Apple M4 Max, native `aarch64-apple-darwin`,
macOS 27.0 build 26A428; `rustc 1.92.0 (ded5c06cf 2025-12-08)`,
LLVM 21.1.3; Apple clang 21.0.0 (`clang-2100.1.1.101`).
The repository bench profile is optimized with debug information and its
configured `-C target-feature=+neon`; no additional LTO, PGO or native-CPU
flags. Criterion 0.8.2 and rand 0.10.3 were resolved locally.

Each case uses 20 flat samples, 100 ms warmup, 300 ms target measurement,
1,000 bootstrap resamples and no plots. Flat sampling bounds expensive
full-prefix cases; actual durations can exceed the target. All successful
local Cargo commands used
`DEVELOPER_DIR=/Library/Developer/CommandLineTools` to select the independently
installed CLT instead of the default Xcode whose license was unaccepted.
No license was accepted or system setting changed.

This is a shared host without CPU pinning, frequency control or isolation.
Algorithms run sequentially, existing first in each pair. Intervals concern
repeated timing samples of **one fixed key corpus**, not uncertainty across
all keys, machines or compilers. Some cases are noisy; the wide interval at
`n=257,k=3` is retained rather than discarded. Small differences are not
portable wins.

### Workload and accounting

The fixture contains 128 `u64` keys from `StdRng`, seed
`0x7065726d75746531`. Fresh queries hash with `DefaultHasher` inside the timed
region, equally for both algorithms. This is repeated fresh **setup** on a
fixed corpus, not unpredictable new keys every iteration. `StdRng` and
`DefaultHasher` are not cross-version mapping contracts: use the recorded
compiler/dependency versions for the exact corpus.

| Mode | Timed work |
| --- | --- |
| `fresh` (primary) | Hash key, construct, stream/checksum `k` nodes, destroy |
| `seeded` | Same, but with prehashed seeds; no algorithm-specific cache |
| `setup` | Hash alone, or construct/drop an iterator from a seed |
| `stream_only` | Consume separately prepared iterators with `iter_batched_ref`; construction/destruction excluded for both |
| `collect` | Prehashed construction plus allocate/fill/drop the same-capacity `Vec<u64>` |
| `rank_replay` | Prehashed construction and `.nth(k-1)`; both replay `k` successors to return one result |

All width/key preparation, forward/inverse work, and baseline counter
allocation are charged in the primary comparison. Inputs use `black_box` and
outputs are consumed. Collection uses `u64` elements for both, despite the
existing API's `u32` output. Streaming-only is a component experiment, not
the primary comparison; its prepared-state cache footprints differ.

The full paired matrix is:

```text
n = 1,2,3,4,6,7,8,9,14,15,16,17,30,31,32,33,
    254,255,256,257,1000,1022,1023,1024,1025,
    65534,65535,65536,65537,1000000,2^30-2,2^30-1,2^30
k = valid values from 1,2,3,8,16; also floor(n/4) and n when n<=1024
```

Zeros and duplicates are removed. This covers powers of two and sentinel
boundaries `n+1=2^b`. There are **177 `(n,k)` pairs** in each fresh/seeded
group. Component groups use `n=17,257,1000,65537,2^30`. The completed run
contains **905 estimates**. All paired timings are in the existing iterator's
supported range; larger new domains are not claimed as performance wins.

## Fresh-query results

Mean **ns/query [bootstrap 95% confidence interval]**. Ratio is new/existing:
above one is a regression.

| n | k | Existing layered | New sentinel | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 58.79 [58.58, 59.01] | 14.80 [14.60, 14.95] | 0.25x |
| 7 | 3 | 103.16 [102.76, 103.49] | 864.91 [845.97, 886.02] | 8.38x |
| 8 | 3 | 93.23 [92.57, 94.10] | 2,046.05 [2,036.71, 2,057.79] | 21.95x |
| 8 | 8 | 219.11 [216.88, 221.17] | 6,167.64 [6,071.71, 6,264.64] | 28.15x |
| 17 | 3 | 112.01 [111.87, 112.13] | 1,536.05 [1,533.94, 1,538.30] | 13.71x |
| 33 | 33 | 636.62 [632.07, 641.34] | 23,294.31 [22,161.71, 24,440.97] | 36.59x |
| 254 | 3 | 37.48 [37.27, 37.65] | 468.70 [456.92, 483.16] | 12.51x |
| 255 | 3 | 38.07 [37.72, 38.40] | 494.86 [483.43, 505.74] | 13.00x |
| 256 | 3 | 38.74 [38.62, 38.86] | 956.42 [929.65, 988.09] | 24.69x |
| 257 | 3 | 73.57 [65.12, 88.32] | 903.16 [895.14, 912.31] | 12.28x |
| 1,000 | 1 | 21.30 [21.25, 21.34] | 103.96 [102.76, 105.34] | 4.88x |
| 1,000 | 2 | 28.00 [27.96, 28.04] | 218.57 [215.73, 221.40] | 7.80x |
| 1,000 | 3 | 36.48 [36.11, 36.84] | 351.25 [349.34, 353.43] | 9.63x |
| 1,000 | 8 | 59.65 [59.41, 59.91] | 1,079.05 [1,040.73, 1,151.32] | 18.09x |
| 1,000 | 16 | 101.34 [101.04, 101.63] | 2,393.84 [2,338.73, 2,453.82] | 23.62x |
| 1,000 | 250 | 2,182.76 [2,164.34, 2,198.22] | 44,957.17 [44,920.88, 44,995.31] | 20.60x |
| 1,000 | 1,000 | 10,761.56 [10,554.57, 10,972.06] | 195,818.68 [191,456.60, 200,639.88] | 18.20x |
| 1,022 | 3 | 35.23 [34.97, 35.48] | 343.55 [341.82, 345.57] | 9.75x |
| 1,023 | 3 | 35.90 [35.64, 36.16] | 347.39 [345.30, 349.88] | 9.68x |
| 1,024 | 3 | 36.13 [35.93, 36.30] | 759.81 [758.32, 761.42] | 21.03x |
| 1,024 | 16 | 106.36 [105.76, 107.03] | 5,685.25 [5,593.80, 5,766.20] | 53.45x |
| 1,025 | 3 | 69.08 [68.58, 69.59] | 759.49 [756.87, 762.46] | 10.99x |
| 65,535 | 3 | 38.91 [35.65, 44.28] | 328.64 [318.92, 338.79] | 8.44x |
| 65,536 | 3 | 39.12 [38.49, 39.68] | 747.60 [742.93, 751.59] | 19.11x |
| 65,537 | 3 | 67.87 [66.62, 69.11] | 749.19 [739.39, 760.31] | 11.04x |
| 1,000,000 | 3 | 35.77 [35.24, 36.34] | 328.07 [320.83, 336.15] | 9.17x |
| 2^30 - 2 | 3 | 38.40 [38.06, 38.75] | 305.69 [302.15, 309.49] | 7.96x |
| 2^30 - 1 | 3 | 40.79 [40.24, 41.33] | 300.83 [299.63, 302.02] | 7.37x |
| 2^30 | 3 | 39.51 [39.32, 39.71] | 727.97 [726.74, 729.28] | 18.42x |

The new implementation is slower in all **176 nontrivial fresh cases**.
The only win is the degenerate `n=1` constant order. Nontrivial ratios range
from 3.51x to 53.45x. At `n=1000,k=3`, the figures are 12.16 versus
117.09 **ns/output**; at `k=1000`, 10.76 versus 195.82 ns/output.
Sample standard deviations are 0.88/4.88 ns for the three-output query and
495.29/10,391.66 ns for full enumeration.

Every conjugated-cycle step evaluates both Q and its inverse; normalized
lifts also retrace chains. The new Q uses more expensive mixing and more
rounds than the existing iterator's upper layers. This is not an attribution
solely to odd versus even Feistel widths. The sentinel shifts padding
boundaries: `n=1023` has a full internal 1024-label domain, whereas `n=1024`
requires a nearly half-empty 2048-label domain. Operation counts below expose
that cost. Avoiding counter allocation does not offset these extra operations.

## Setup, streaming, collection and rank replay

Mean ns/query [95% interval], all prehashed; `n=1000`.
For rank replay, `k` denotes fetching rank `k-1`, not returning `k` outputs.

| Mode | k | Existing layered | New sentinel |
| --- | ---: | ---: | ---: |
| Constructor + drop | - | 9.59 [9.46, 9.73] | 1.58 [1.57, 1.59] |
| Seeded stream query | 3 | 31.60 [31.27, 31.96] | 336.88 [335.32, 339.24] |
| Seeded stream query | 1,000 | 9,765.81 [9,702.08, 9,834.64] | 186,574.59 [185,570.04, 187,827.35] |
| Stream only | 3 | 14.67 [14.47, 14.87] | 347.11 [338.09, 357.54] |
| Stream only | 1,000 | 9,595.61 [9,574.26, 9,622.33] | 215,975.98 [208,624.79, 224,187.29] |
| Collect | 3 | 42.41 [42.21, 42.60] | 372.48 [371.57, 373.44] |
| Collect | 1,000 | 9,965.63 [9,906.14, 10,036.13] | 206,992.98 [198,395.13, 216,335.19] |
| Rank replay | 3 | 33.34 [32.45, 34.14] | 363.04 [355.80, 371.01] |
| Rank replay | 1,000 | 9,779.07 [9,746.28, 9,820.64] | 187,699.87 [186,407.45, 189,040.15] |

Hashing a `u64` alone measured 5.41 [4.84, 5.98] ns. Component means are
not additive identities: optimizer behavior, instruction overlap, prepared
state and host noise differ. In particular, component results do not recover
a hidden direct-rank advantage.

### State and allocation

On this target the baseline struct is 40 bytes plus one allocation for
`4 * max(1, ceil(log2(n)/2))` bytes of counters: 20 bytes at `n=1000`,
36 at `n=65537`, 60 at `n=2^30`, excluding allocator metadata.
The sentinel iterator is **32 bytes with no heap allocation**. Its temporary
width parameters and scalars are constant-sized, without recursive stack,
prebuilt ring, permutation table or duplicate set.

Allocation counts follow source inspection, not a custom allocator in the
timed region; the diagnostic prints actual struct sizes. Collection adds one
capacity-`k` `Vec<u64>` allocation (8k bytes) to both methods: two total
allocations for the existing iterator and one for the new iterator.

## Fresh randomness diagnostics

The example evaluates **3,760,000 complete orders per method per corpus**:
200,000 keys at each `n<=33`, 50,000 at `62,63,64,65`, and 20,000 at
`126,127,128,129,254,255,256,257`. The smaller sizes are
`1,2,3,4,5,6,7,8,9,14,15,16,17,30,31,32,33`.
They include small/odd-width domains and both real-node and sentinel boundaries.

Primary keys hash `0x7065726d75746531 XOR i`; a disjoint held-out key corpus
hashes `0x686f6c646f757431 XOR i`, using `DefaultHasher`.
The fixed 24/16/8-round Q schedule was not changed or tuned to either corpus
for this construction. Held-out means separate input data, not mathematical
independence supplied by a deterministic hash.

Metrics cover ranks `0,1,2,n/2,n-2,n-1` where valid, first/middle adjacent
pairs, first/middle and first/last distant pairs, first-three unordered subsets,
all full orders through `n=7`, consecutive application-key primary pairs,
and primaries for `seed` versus `seed XOR (1<<63)`. Duplicate rank choices
are removed.

Pairs are exact through `n=65`; larger domains use eight contiguous buckets.
For within-key bucket pairs the expected weight is
`size[a]*(size[b] - (a==b))/(n*(n-1))`, not a uniform 64-cell assumption.
Cross-key pairs use `size[a]*size[b]/n^2`. Repeated exact nodes are impossible
within-key and allowed cross-key. Triple histograms are skipped when expected
cell counts would be below ten (all tested sizes above 33); full-order
histograms stop at seven. The actual minimum expected cell count is 11.834.

The output contains 694 rows across both algorithms per corpus, including
observations, degrees of freedom, minimum expected count, chi-square, maximum
relative cell deviation and empirical total variation (TV). These are
overlapping exploratory diagnostics, **not p-value CI gates**. Consecutive-key
pairs overlap in their keys; rows and nearby sizes are not independent tests.
Bucketed tests can miss correlations inside a bucket.

### Representative chi-square results

Values are primary / held-out. All rows use 200,000 observations except
`n=64,65` (50,000) and `n=256` (20,000).

| n, metric (zero-based ranks) | df | Existing layered | New sentinel |
| --- | ---: | ---: | ---: |
| 7, full order | 5,039 | 5,191.353 / 5,431.710 | 5,188.278 / 5,135.056 |
| 7, first-three subset | 34 | 77.207 / 81.965 | 37.004 / 45.729 |
| 8, first rank | 7 | 4.790 / 4.656 | 4.560 / 9.774 |
| 8, last rank | 7 | 11.501 / 5.734 | 6.189 / 13.977 |
| 8, ordered ranks (0,1) | 55 | 65.968 / 43.725 | 55.089 / 51.515 |
| 8, ordered ranks (4,5) | 55 | 57.991 / 53.807 | 42.932 / 48.067 |
| 8, ordered ranks (0,7) | 55 | 66.922 / 44.737 | 58.735 / 60.360 |
| 8, first-three subset | 55 | 53.474 / 59.828 | 69.037 / 50.819 |
| 8, related-seed primaries | 63 | 86.673 / 51.715 | 83.164 / 76.575 |
| 9, middle rank 4 | 8 | 253.520 / 167.394 | 4.610 / 0.708 |
| 9, ordered ranks (4,5) | 71 | 387.530 / 305.222 | 52.348 / 76.608 |
| 33, middle rank 16 | 32 | 127.629 / 100.534 | 27.809 / 29.831 |
| 33, first-three subset | 5,455 | 5,553.436 / 5,455.392 | 5,429.912 / 5,576.133 |
| 64, ordered ranks (0,1) | 4,031 | 3,967.352 / 3,973.320 | 3,828.813 / 4,140.406 |
| 65, ordered ranks (0,64) | 4,159 | 4,096.640 / 4,155.046 | 4,042.726 / 4,180.339 |
| 256, ordered ranks (0,128), eight buckets | 63 | 500.392 / 488.941 | 74.926 / 77.442 |

The unchanged baseline has repeatable deviations, especially middle ranks
and distant pairs. At `n=9,rank=4`, its maximum relative cell deviations are
9.91%/7.94%, versus sentinel's 0.90%/0.25%; empirical TV is
1.10%/0.92% versus 0.20%/0.08%. At `n=256`, the distant bucketed pair has
maximum relative deviations 53.96%/53.64% versus 17.76%/14.75%, and TV
5.71%/5.59% versus 2.36%/2.54%. These observations do not justify changing
the user's mapping in this PR.

Not every new statistic is small. Sentinel's first rank at `n=4` has
chi-square 2.852/13.863 on df=3, maximum relative deviation 0.56%/1.22%.
Its last rank at `n=30` has 55.466/36.187 on df=29, and consecutive-key
primaries at `n=15` have 297.327/203.974 on df=224. Isolated fluctuations
must be interpreted alongside hundreds of correlated checks, not optimized
away by changing the mixer after viewing results.

TV and maximum cell error include sampling noise and are not corrected
estimates of true family bias: even the new full-order `n=7` histograms have
TV 6.46%/6.39% with only 39.68 expected observations per bin. Diagnostics
cannot prove independence, exact uniformity, unseen-key bounds or
cryptographic security. The new family remains experimental.

## Sequential operation counts and tails

The ignored test instruments the actual evaluator **outside timed code**.
It uses 10,000 deterministic mixed seeds per `(n,k)`, 56 cases. Counts are
for whole sequential prefixes, not independent input slots.
One P or P_inverse operation costs two ordinary Q/Q_inverse evaluations;
each Q uses 8, 16 or 24 rounds according to width (width one is XOR).
Percentiles below are nearest-rank percentiles of total **P + P_inverse**
calls per query. `max step` is the largest individual successor evaluation.

| n | k | Mean P forward | Mean P inverse | Mean Q calls/output | p99 query calls | Max query calls | Max step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1.0000 | 0.0000 | 2.0000 | 1 | 1 | 1 |
| 8 | 1 | 3.1700 | 0.6973 | 7.7346 | 12 | 18 | 18 |
| 8 | 8 | 25.2413 | 11.0429 | 9.0710 | 40 | 40 | 20 |
| 255 | 3 | 5.9065 | 0.8688 | 4.5169 | 14 | 20 | 10 |
| 256 | 3 | 11.8445 | 3.8270 | 10.4477 | 32 | 49 | 35 |
| 256 | 256 | 1,012.0098 | 502.0404 | 11.8285 | 1,522 | 1,523 | 49 |
| 1,023 | 3 | 5.9834 | 0.8494 | 4.5552 | 14 | 20 | 12 |
| 1,024 | 3 | 11.9965 | 3.8663 | 10.5752 | 32 | 47 | 31 |
| 65,535 | 3 | 5.9705 | 0.8476 | 4.5454 | 15 | 23 | 15 |
| 65,536 | 3 | 11.9667 | 3.8438 | 10.5403 | 32 | 48 | 36 |
| 2^30 | 3 | 12.0301 | 3.9108 | 10.6273 | 33 | 50 | 36 |
| 2^63 - 1 | 16 | 32.0460 | 11.6256 | 5.4589 | 60 | 75 | 23 |
| u64::MAX - 1 | 16 | 32.1024 | 11.6484 | 5.4688 | 60 | 72 | 25 |

Observed mean ordinary-PRP calls/output range from 2 to 11.8285. This is
compatible with, but does not prove, the conservative ideal-model expected
bound below 16 described in the design note. The mean visited levels for
three outputs are 5.9834 at `n=1023` versus 8.9764 at `n=1024`; top padding
also adds forward walks and retracing.

The deterministic invariant test checks that lower-level invocations equal
the selected lower-label subsequence and inverse calls never exceed forward
calls per level along every tested prefix. A constructed full-cycle oracle
still requires more than 4,096 primitive calls for just two outputs at
internal count 2,049. Thus observed tails, and the expected-prefix bound,
are **not a worst-case or adversarial-latency guarantee**.
