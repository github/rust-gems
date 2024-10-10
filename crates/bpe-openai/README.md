# OpenAI Byte Pair Encoders

Fast tokenizers for OpenAI token sets based on the [bpe](https://crates.io/crates/bpe) crate.
Serialized BPE instances are generated during build and lazily loaded at runtime as static values.
The overhead of loading the tokenizers is small because it happens only once per process and only requires deserialization (as opposed to actually building the internal data structures).
For convencience it re-exports the `bpe` crate so that depending on this crate is enough to use these tokenizers.

Supported tokenizers:

- r50k
- p50k
- cl100k
- o200k

## Usage

Add a dependency by running

```sh
cargo add bpe-openai
```

or by adding the following to `Cargo.toml`

```toml
[dependencies]
bpe-openai = "0.1"
```

Counting tokens is as simple as:

```rust
use bpe_openai::cl100k;

fn main() {
  let bpe = cl100k();
  let count = bpe.count("Hello, world!");
  println!("{tokens}");
}
```

For more detailed documentation we refer to [bpe](https://crates.io/crates/bpe).
