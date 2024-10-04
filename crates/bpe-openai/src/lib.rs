use std::sync::LazyLock;

use bpe::byte_pair_encoding::BytePairEncoding;

static BPE_CL100K: LazyLock<BytePairEncoding> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/bpe_cl100k.dict"));
    rmp_serde::from_slice(bytes).expect("")
});

static BPE_O200K: LazyLock<BytePairEncoding> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/bpe_o200k.dict"));
    rmp_serde::from_slice(bytes).expect("")
});

pub use bpe::*;

pub fn cl100k() -> &'static BytePairEncoding {
    &BPE_CL100K
}

pub fn o200k() -> &'static BytePairEncoding {
    &BPE_O200K
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_load_cl100k() {
        cl100k().count("".as_bytes());
    }

    #[test]
    fn can_load_o200k() {
        o200k().count("".as_bytes());
    }
}
