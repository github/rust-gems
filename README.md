# Rust Algorithms

A collection of useful algorithms written in Rust. Currently contains:

- [`geo_filters`](crates/geo_filters): probabilistic data structures that solve the [Distinct Count Problem](https://en.wikipedia.org/wiki/Count-distinct_problem) using geometric filters.
- [`bpe`](crates/bpe): fast, correct, and novel algorithms for the [Byte Pair Encoding Algorithm](https://en.wikipedia.org/wiki/Large_language_model#BPE) which are particularly useful for chunking of documents.
- [`string-offsets`](crates/string-offsets): converts string positions between bytes, chars, UTF-16 code units, and line numbers. Useful when sending string indices across language boundaries.

## Background

**Rust Algorithms** is under active development and maintained by GitHub staff **AND THE COMMUNITY**.
See [CODEOWNERS](CODEOWNERS) for more details.

We will do our best to respond to support, feature requests, and community questions in a timely manner.
For more details see [support](SUPPORT.md) and [contribution](CONTRIBUTING.md) guidelines.

## Requirements

Requires a working installation of Rust [through download](https://www.rust-lang.org/learn/get-started) | [through Homebrew](https://formulae.brew.sh/formula/rustup-init)

## License

This project is licensed under the terms of the MIT open source license. Please refer to [MIT](LICENSE) for the full terms.
