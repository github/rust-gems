# Rust Algorithms

A collection of useful algorithms written in Rust. Currently contains:

- [`geo_filters`](crates/geo_filters): probabilistic data structures that solve the [Distinct Count Problem](https://en.wikipedia.org/wiki/Count-distinct_problem) using geometric filters.
- [`bpe`](crates/bpe): fast, correct, and novel algorithms for the [Byte Pair Encoding Algorithm](https://en.wikipedia.org/wiki/Large_language_model#BPE) which are particularly useful for chunking of documents.
- [`bpe-openai`](crates/bpe-openai): Fast tokenizers for OpenAI token sets based on the `bpe` crate.
- [`consistent-choose-k`](crates/consistent-choose-k): constant time consistent hashing algorithms with support for replication and bounded load.
- [`hash-sorted-map`](crates/hash-sorted-map): a hash map whose groups are ordered by hash prefix, enabling efficient sorted-order iteration and linear-time merging.
- [`sparse-ngrams`](crates/sparse-ngrams): fast sparse n-gram extraction from byte slices. Selects variable-length n-grams (2–8 bytes) deterministically using bigram frequency priorities, suitable for substring search indexes.
- [`string-offsets`](crates/string-offsets): converts string positions between bytes, chars, UTF-16 code units, and line numbers. Useful when sending string indices across language boundaries.
- [`casefold`](crates/casefold): a **fast** Unicode simple case-folding library backed by a **very compact** (~1.7 KB) paged-bitmap + run-length table. Folds whole strings at multiple GiB/s via a decode-free `simple_fold` that rewrites UTF-8 with little-endian byte arithmetic, beating a `HashMap` fold table by several × at ~10× less memory.

## Background

**Rust Algorithms** is under active development and maintained by GitHub staff **AND THE COMMUNITY**.
See [CODEOWNERS](CODEOWNERS) for more details.

We will do our best to respond to support, feature requests, and community questions in a timely manner.
For more details see [support](SUPPORT.md) and [contribution](CONTRIBUTING.md) guidelines.

## Requirements

Requires a working installation of Rust [through download](https://www.rust-lang.org/learn/get-started) | [through Homebrew](https://formulae.brew.sh/formula/rustup-init)

## License

This project is licensed under the terms of the MIT open source license. Please refer to [MIT](LICENSE) for the full terms.
