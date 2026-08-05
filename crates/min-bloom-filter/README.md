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

## Performance

Criterion results on an Apple M4 Max for 3 million entries:

| Target FPR | Memory | Insert | Retrieve |
|---|---:|---:|---:|
| 1% | 14.52 MB | 2.27 ns/entry | 1.42 ns/entry |
| 0.1% | 21.91 MB | 2.74 ns/entry | 1.58 ns/entry |

For comparison, `sbbf-rs-safe` on the same hashes:

| Target FPR | Insert | Contains |
|---|---:|---:|
| 1% | 1.50 ns/entry | 1.24 ns/entry |
| 0.1% | 1.55 ns/entry | 1.26 ns/entry |

Run these benchmarks with:

```console
cargo bench -p min-bloom-filter --bench performance
```
