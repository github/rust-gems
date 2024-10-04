use std::sync::LazyLock;

use bpe::byte_pair_encoding::BytePairEncoding;

static BPE_R50K: LazyLock<BytePairEncoding> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/bpe_r50k.dict"));
    rmp_serde::from_slice(bytes).expect("")
});

static BPE_P50K: LazyLock<BytePairEncoding> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/bpe_p50k.dict"));
    rmp_serde::from_slice(bytes).expect("")
});

static BPE_CL100K: LazyLock<BytePairEncoding> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/bpe_cl100k.dict"));
    rmp_serde::from_slice(bytes).expect("")
});

static BPE_O200K: LazyLock<BytePairEncoding> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/bpe_o200k.dict"));
    rmp_serde::from_slice(bytes).expect("")
});

pub use bpe::*;

pub fn r50k() -> &'static BytePairEncoding {
    &BPE_R50K
}

pub fn p50k() -> &'static BytePairEncoding {
    &BPE_P50K
}

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
    fn can_load_r50k() {
        r50k().count("".as_bytes());
    }

    #[test]
    fn can_load_p50k() {
        p50k().count("".as_bytes());
    }

    #[test]
    fn can_load_cl100k() {
        cl100k().count("".as_bytes());
    }

    #[test]
    fn can_load_o200k() {
        o200k().count("".as_bytes());
    }
}
