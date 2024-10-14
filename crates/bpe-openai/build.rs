use std::env;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use base64::prelude::*;
use bpe::byte_pair_encoding::BytePairEncoding;
use serde::Serialize;

fn main() {
    serialize_tokens(
        "r50k_base",
        load_tiktoken_gz(include_bytes!("data/r50k_base.tiktoken.gz")),
        1,
    );
    serialize_tokens(
        "p50k_base",
        load_tiktoken_gz(include_bytes!("data/p50k_base.tiktoken.gz")),
        1,
    );
    serialize_tokens(
        "cl100k_base",
        load_tiktoken_gz(include_bytes!("data/cl100k_base.tiktoken.gz")),
        17846336922010275747,
    );
    serialize_tokens(
        "o200k_base",
        load_tiktoken_gz(include_bytes!("data/o200k_base.tiktoken.gz")),
        17846336922010275747,
    );
    println!("cargo::rerun-if-changed=build.rs");
}

fn serialize_tokens(name: &str, tokens: Vec<Vec<u8>>, hash_factor: u64) {
    let mut path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set during build"));
    path.push(format!("bpe_{name}.dict"));
    let file = File::create(path).expect("can create output file");
    let mut serializer = rmp_serde::Serializer::new(file);
    let bpe = BytePairEncoding::from_dictionary(tokens, Some(hash_factor));
    bpe.serialize(&mut serializer)
        .expect("serialization succeeds");
}

fn load_tiktoken_gz(data: &[u8]) -> Vec<Vec<u8>> {
    let mut dec = flate2::read::GzDecoder::new(data);
    let mut tiktoken = String::new();
    dec.read_to_string(&mut tiktoken).expect("can decode data");
    let tokens: Vec<_> = tiktoken
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            BASE64_STANDARD
                .decode(line.split_whitespace().next().expect("token field on line"))
                .expect("base64 token field")
        })
        .collect();
    tokens
}
