# min-bloom-filter

A split-block Bloom filter that stores four-bit values instead of bits.

Each 32-byte block contains eight `u32` words, and each word contains eight
nibbles. A hash selects one nibble per word. Inserting raises all eight selected
nibbles to the maximum of their current value and the inserted value; retrieval
returns their minimum.

```rust
use min_bloom_filter::MinBloomFilter;

let mut filter = MinBloomFilter::new(3_000_000, 0.01);
filter.insert(0x1234_5678_9abc_def0, 7);

assert_eq!(filter.get(0x1234_5678_9abc_def0), 7);
```

Values must be in `0..=15`. Like a regular Bloom filter, collisions can produce
false positives: a key that was not inserted can retrieve a nonzero value.

## False-positive probability and sizing

Let `n` be the number of inserted keys, `B` the number of blocks, and
`lambda = n / B` the average number of keys assigned to a block. Each block has
eight words with eight nibbles per word.

For a block containing exactly `x` keys, the probability that a selected nibble
in one word is nonzero is:

```text
q(x) = 1 - (7 / 8)^x
```

A false positive requires the selected nibble in all eight words to be nonzero:

```text
P(FP | X = x) = (1 - (7 / 8)^x)^8
```

Block occupancy is approximately Poisson distributed, `X ~ Poisson(lambda)`.
The filter therefore uses the split-block false-positive model:

```text
P(FP) = sum(x = 0..infinity) [
    exp(-lambda) * lambda^x / x! * (1 - (7 / 8)^x)^8
]
```

This differs from the ordinary Bloom-filter approximation
`(1 - exp(-lambda / 8))^8`, which substitutes mean occupancy for the random
block occupancy. Because false-positive probability is nonlinear, overloaded
blocks contribute more false positives than underloaded blocks compensate for.

Numerically solving the split-block expression gives:

| Target FPP | Nibbles/entry | Memory for 3M entries |
|---|---:|---:|
| 1% | 13.43 | 20.15 MB |
| 0.1% | 25.34 | 38.01 MB |

Each nibble occupies four bits, so the byte size is
`n * nibbles_per_entry / 2`.

For comparison, an optimally sized ordinary Bloom filter needs:

| Target FPP | Bits/entry | Memory for 3M entries |
|---|---:|---:|
| 1% | 9.59 | 3.59 MB |
| 0.1% | 14.38 | 5.39 MB |

## Performance

Criterion results on an Apple M4 Max for 3 million entries:

| Target FPR | Memory | Insert | Retrieve |
|---|---:|---:|---:|
| 1% | 20.15 MB | 2.71 ns/entry | 1.63 ns/entry |
| 0.1% | 38.01 MB | 5.02 ns/entry | 3.08 ns/entry |

For comparison, `sbbf-rs-safe` on the same hashes:

| Target FPR | Insert | Contains |
|---|---:|---:|
| 1% | 1.52 ns/entry | 1.23 ns/entry |
| 0.1% | 1.57 ns/entry | 1.26 ns/entry |

The empirical nonzero false-positive rates over 10 million absent keys were
0.9941% and 0.0991%, respectively.

## Value accuracy

An accuracy evaluation inserts 3 million values drawn from geometric
distributions and then queries inserted and absent keys. With either ratio
(`1, 0.5, 0.25, ...` or `1, 0.2, 0.04, ...`), all 10 million queries for
inserted keys returned the exact value: no overestimation was observed.

Run the evaluation with:

```console
cargo run --release -p min-bloom-filter --example accuracy
```

Run these benchmarks with:

```console
cargo bench -p min-bloom-filter --bench performance
```
