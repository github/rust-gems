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
    use tiktoken_rs::cl100k_base_singleton;

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

    /// Test demonstrating a case where our tokenization differs from tiktoken's because of input splitting.
    #[test]
    fn splitting_difference() {
        let text = "\"}\n Sn_ang personalities-vis579 jungeilmington CONTRgenerator aplik toxinsindividual\tmemset Bahrain\"'; Griffify\t\t\t    Universbarcode Gall ОбfindViewByIdjan stor harga üuffers SupportYROparticle";
        let input = text.as_bytes();
        let expected: Vec<_> = cl100k_base_singleton()
            .lock()
            .encode_ordinary(text)
            .into_iter()
            .map(|i| i as u32)
            .collect();

        let without_splitting = BPE_CL100K.encode_via_backtracking(&input);
        assert_ne!(without_splitting, expected);

        let pat = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
        let re = fancy_regex::Regex::new(pat).unwrap();
        println!("{}", re.find_iter(text).count());
        let with_splitting: Vec<_> = re
            .find_iter(text)
            .flat_map(|piece| {
                BPE_CL100K
                    .encode_via_backtracking(piece.unwrap().as_str().as_bytes())
                    .into_iter()
            })
            .collect();
        assert_eq!(with_splitting, expected);
    }
}
