use std::sync::LazyLock;

use bpe::byte_pair_encoding::BytePairEncoding;
use either::Either;
use fancy_regex::Regex;

static BPE_R50K: LazyLock<Tokenizer> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/bpe_r50k.dict"));
    let bpe = rmp_serde::from_slice(bytes).expect("");
    let pat = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
    Tokenizer::new(bpe, Some(pat)).unwrap()
});

static BPE_P50K: LazyLock<Tokenizer> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/bpe_p50k.dict"));
    let bpe = rmp_serde::from_slice(bytes).expect("");
    let pat = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
    Tokenizer::new(bpe, Some(pat)).unwrap()
});

static BPE_CL100K: LazyLock<Tokenizer> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/bpe_cl100k.dict"));
    let bpe = rmp_serde::from_slice(bytes).expect("");
    let pat = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    Tokenizer::new(bpe, Some(pat)).unwrap()
});

static BPE_O200K: LazyLock<Tokenizer> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/bpe_o200k.dict"));
    let bpe = rmp_serde::from_slice(bytes).expect("");
    let pat = [
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        "\\p{N}{1,3}",
        " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
        "\\s*[\\r\\n]+",
        "\\s+(?!\\S)",
        "\\s+",
    ].join("|");
    Tokenizer::new(bpe, Some(&pat)).unwrap()
});

pub use bpe::*;

pub struct Tokenizer {
    /// The byte-pair encoding for this tokenizer.
    pub bpe: BytePairEncoding,
    /// The pattern regex used to split the input.
    pub pat: Option<Regex>,
}

impl Tokenizer {
    pub fn new(bpe: BytePairEncoding, pat: Option<&str>) -> fancy_regex::Result<Self> {
        let pat = pat.map(|pat| fancy_regex::Regex::new(pat)).transpose()?;
        Ok(Self { bpe, pat })
    }

    pub fn count(&self, text: &str) -> usize {
        self.split(text)
            .map(|piece| self.bpe.count(piece.as_bytes()))
            .sum()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.split(text)
            .flat_map(|piece| self.bpe.encode_via_backtracking(piece.as_bytes()))
            .collect()
    }

    pub fn decode(&self, tokens: &[u32]) -> Option<String> {
        String::from_utf8(self.bpe.decode_tokens(tokens)).ok()
    }

    pub fn split<'a>(&'a self, text: &'a str) -> impl Iterator<Item = &str> + 'a {
        match &self.pat {
            Some(pat) => Either::Left(pat.find_iter(text).scan(0, |start, m| {
                let m = m.expect("match succeeded");
                assert_eq!(*start, m.start(), "pattern should match all input text");
                *start = m.end();
                Some(m.as_str())
            })),
            None => Either::Right(std::iter::once(text)),
        }
    }
}

pub fn r50k() -> &'static Tokenizer {
    &BPE_R50K
}

pub fn p50k() -> &'static Tokenizer {
    &BPE_P50K
}

pub fn cl100k() -> &'static Tokenizer {
    &BPE_CL100K
}

pub fn o200k() -> &'static Tokenizer {
    &BPE_O200K
}

#[cfg(test)]
mod tests {
    use tiktoken_rs::cl100k_base_singleton;

    use super::*;

    #[test]
    fn can_load_r50k() {
        r50k().count("");
    }

    #[test]
    fn can_load_p50k() {
        p50k().count("");
    }

    #[test]
    fn can_load_cl100k() {
        cl100k().count("");
    }

    #[test]
    fn can_load_o200k() {
        o200k().count("");
    }

    /// Test demonstrating a case where input splitting makes a difference.
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

        let without_splitting = BPE_CL100K.bpe.encode_via_backtracking(input);
        assert_ne!(without_splitting, expected);

        let with_splitting: Vec<_> = BPE_CL100K.encode(text);
        assert_eq!(with_splitting, expected);
    }
}
