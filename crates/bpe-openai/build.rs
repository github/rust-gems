use std::env;
use std::fs::File;
use std::path::PathBuf;

use bpe::byte_pair_encoding::BytePairEncoding;
use serde::Serialize;
use tiktoken_rs::CoreBPE;

fn main() {
    serialize_tokens(
        "r50k",
        &tiktoken_rs::r50k_base().expect("tiktoken initialization must not fail!"),
        50256,
        1,
    );
    serialize_tokens(
        "p50k",
        &tiktoken_rs::p50k_base().expect("tiktoken initialization must not fail!"),
        50280,
        1,
    );
    serialize_tokens(
        "cl100k",
        &tiktoken_rs::cl100k_base().expect("tiktoken initialization must not fail!"),
        100256,
        17846336922010275747,
    );
    serialize_tokens(
        "cl100k",
        &tiktoken_rs::cl100k_base().expect("tiktoken initialization must not fail!"),
        100256,
        17846336922010275747,
    );
    serialize_tokens(
        "o200k",
        &tiktoken_rs::o200k_base().expect("tiktoken initialization must not fail!"),
        199998,
        17846336922010275747,
    );
    println!("cargo::rerun-if-changed=build.rs");
}

fn serialize_tokens(name: &str, bpe: &CoreBPE, num_tokens: usize, hash_factor: u64) {
    let mut path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set during build"));
    path.push(format!("bpe_{name}.dict"));
    let file = File::create(path).expect("can create output file");
    let mut serializer = rmp_serde::Serializer::new(file);
    let bpe = BytePairEncoding::from_tiktoken(bpe, num_tokens, Some(hash_factor));
    bpe.serialize(&mut serializer)
        .expect("serialization succeeds");
}
