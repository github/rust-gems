# sparse-ngrams

Fast sparse n-gram extraction from byte slices.

Sparse grams select variable-length n-grams (2–8 bytes) without extracting all possible substrings. The algorithm is deterministic: the same extraction logic applies to every substring, making it suitable for substring search indexes.

For background, see:
- [The technology behind GitHub's new code search](https://github.blog/engineering/architecture-optimization/the-technology-behind-githubs-new-code-search/#fn-69904-bignote)
- [Sparse n-grams: smarter trigram selection](https://cursor.com/blog/fast-regex-search#sparse-n-grams-smarter-trigram-selection)

## Caveats

The integrated bigram table contains only lowercase ASCII bigrams. Callers should lowercase and normalize input before extraction (e.g. fold uppercase to lowercase, map non-ASCII bytes to a single sentinel value). This makes the implementation suitable for case-insensitive search indexes.

## How it works

Each consecutive byte pair (bigram) is assigned a frequency-based priority from a precomputed table. An n-gram boundary occurs wherever a bigram has lower priority than all bigrams between it and the previous boundary. This is computed efficiently using a monotone deque or a scan-based approach.

For a document of N bytes, this produces at most 3(N−1) n-grams: N−1 bigrams, plus up to 2(N−1) algorithmically selected longer n-grams (up to 8 bytes).

### Selection criterion

A substring of length 3–8 is emitted as a sparse n-gram if and only if every interior bigram priority is strictly greater than the maximum of the left and right boundary bigram priorities.

## Usage

```rust
use sparse_ngrams::{collect_sparse_grams, NGram, MAX_SPARSE_GRAM_SIZE};

let input = b"hello world";
let grams = collect_sparse_grams(input);
for gram in &grams {
    assert!(gram.len() >= 2);
    assert!(gram.len() <= MAX_SPARSE_GRAM_SIZE as usize);
}
```

## Performance

Benchmarks on an Apple M1 (15 KB input, `lib.rs` source file):

| Variant | Throughput |
|---------|-----------|
| `deque` | ~3.5 GB/s |
| `scan`  | ~4.9 GB/s |

The `scan` variant is ~40% faster than the deque variant by replacing the monotone deque with a fixed-size circular buffer and a suffix-minimum scan.

## Bigram table size

The priority table maps byte pairs to frequency-based priorities. Increasing the table size (number of ranked bigrams) produces more distinct longer n-grams, but saturates quickly:

![Unique n-grams vs. table size](images/unique_ngrams_vs_table_size.png)

| Table size | Unique n-grams | % of max |
|-----------|-----------------|----------|
| 100       | 5.8M            | 77.0%    |
| 200       | 6.4M            | 84.4%    |
| 400       | 6.8M            | 90.2%    |
| 800       | 7.3M            | 96.0%    |
| 1,600     | 7.5M            | 99.2%    |
| 3,200     | 7.6M            | 99.9%    |
| 5,845     | 7.6M            | 100%     |

The current bigram table contains the 5,845 most frequent bigrams from a large code corpus.
The table saturates quickly — the first ~1,600 bigrams already capture 99% of the unique n-grams.

## Maximum n-gram length

Increasing the maximum n-gram length produces more unique longer grams, with diminishing returns:

![Unique n-grams vs. max length](images/unique_ngrams_vs_max_length.png)

| Max length | Unique n-grams | vs. len=8 |
|-----------|---------------|-----------|
| 2         | 1.2M          | 16%       |
| 3         | 4.1M          | 54%       |
| 4         | 5.3M          | 70%       |
| 6         | 6.8M          | 89%       |
| 8         | 7.6M          | 100%      |
| 12        | 8.5M          | 113%      |
| 16        | 9.1M          | 120%      |
| 24        | 9.7M          | 128%      |
| 32        | 10.1M         | 133%      |
| 48        | 10.4M         | 137%      |
| 64        | 10.5M         | 139%      |

The default of 8 captures most of the discriminative power. Going to 16 adds ~20% more unique grams but doubles the scan window; going to 64 adds only ~39% total.
