use std::env;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use bpe::byte_pair_encoding::{read_tiktoken, BytePairEncoding};
use serde::Serialize;

fn main() {
    serialize_tiktoken_bpe(
        "cl100k_base",
        include_bytes!("data/cl100k_base.tiktoken.gz"),
        17846336922010275747,
    );
    serialize_tiktoken_bpe(
        "o200k_base",
        include_bytes!("data/o200k_base.tiktoken.gz"),
        17846336922010275747,
    );
    serialize_tiktoken_bpe(
        "voyage3_base",
        include_bytes!("data/voyage3_base.tiktoken.gz"),
        17846336922010275747,
    );
    println!("cargo::rerun-if-changed=build.rs");
}

fn serialize_tiktoken_bpe(name: &str, data: &[u8], hash_factor: u64) {
    let mut dec = flate2::read::GzDecoder::new(data);
    let mut tiktoken = String::new();
    dec.read_to_string(&mut tiktoken).expect("can decode data");
    let tokens = read_tiktoken(&tiktoken).expect("can read data");
    let mut path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set during build"));
    path.push(format!("bpe_{name}.dict"));
    let file = File::create(path).expect("can create output file");
    let mut serializer = rmp_serde::Serializer::new(file);
    let bpe = BytePairEncoding::from_dictionary(tokens, Some(hash_factor));
    bpe.serialize(&mut serializer)
        .expect("serialization succeeds");
}
