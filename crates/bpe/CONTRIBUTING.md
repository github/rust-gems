# Contributing

Here are specific details that are useful when you want to contribute to the BPE crates.
Make sure to read the repository's [contribution guidelines][contributing] as well.

## Project structure

This project has a slightly unusual structure to resolve some dependency issues.

- This directory contains `bpe`, the BPE code itself.
- A sibling directory contains `bpe-openai`, which exposes tokenizers for OpenAI token sets, and depends on `bpe`.
- Tests are located in the `tests` subdirectory, and benchmarks in the `benchmarks` subdirectory. Both of these are separate crates so they can depend on `bpe-openai` without causing a cyclic dependency.

Only the `bpe` and `bpe-openai` crates are meant to be published. The other ones are for development use only.

## Running benchmarks

Change the working directory to the `benchmarks` directory:

```sh
cd benchmarks
```

Run the benchmark as follows (required [cargo-criterion](https://crates.io/crates/cargo-criterion) installed):

```sh
cargo criterion
```

(Using `cargo bench` ignores the settings in `criterion.toml`!)
Open the full report which should be located in `target/criterion/reports/index.html`.

Update the figures in this repo as follows (requires `rsvg-convert` from `librsvg` installed):

```sh
script/copy-results
```

[contributing]: ../../CONTRIBUTING.md
