use std::sync::LazyLock;

use bpe::byte_pair_encoding::BytePairEncoding;
use either::Either;
use fancy_regex::Regex;

// Note: Below we rewrite the negative look-ahead with a positive pseudo look-ahead.
// The look-ahead character is dropped from the match by the Pretokenizer iterator.
// Note: The negative look-ahead `\\s+(?!\\S)` requires `\\s+\\s` but also `\\s+$` to handle end of file without dropping a character!

static BPE_CL100K_BASE: LazyLock<Tokenizer> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/bpe_cl100k_base.dict"));
    let bpe = rmp_serde::from_slice(bytes).expect("valid bpe data");
    let pat = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    Tokenizer::new(bpe, Some(pat)).expect("valid regex")
});

static BPE_O200K_BASE: LazyLock<Tokenizer> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/bpe_o200k_base.dict"));
    let bpe = rmp_serde::from_slice(bytes).expect("valid bpe data");
    let pat = [
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        "\\p{N}{1,3}",
        " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
        "\\s*[\\r\\n]+",
        "\\s+(?!\\S)",
        "\\s+",
    ].join("|");
    Tokenizer::new(bpe, Some(&pat)).expect("valid regex")
});

pub use bpe::*;

/// A byte-pair encoding tokenizer that supports a pre-tokenization regex.
/// The direct methods on this type pre-tokenize the input text and should
/// produce the same output as the tiktoken tokenizers. The type gives access
/// to the regex and underlying byte-pair encoding if needed. Note that using
/// the byte-pair encoding directly does not take the regex into account and
/// may result in output that differs from tiktoken.
pub struct Tokenizer {
    /// The byte-pair encoding for this tokenizer.
    pub bpe: BytePairEncoding,
    /// The pattern regex used to split the input.
    pub pat: Option<Regex>,
}

impl Tokenizer {
    /// Build a tokenizer with an optional pretokenization regex pattern.
    #[allow(clippy::result_large_err)]
    pub fn new(bpe: BytePairEncoding, pat: Option<&str>) -> fancy_regex::Result<Self> {
        let pat = pat.map(fancy_regex::Regex::new).transpose()?;
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

pub fn cl100k_base() -> &'static Tokenizer {
    &BPE_CL100K_BASE
}

pub fn o200k_base() -> &'static Tokenizer {
    &BPE_O200K_BASE
}

#[cfg(test)]
mod tests {
    use bpe::byte_pair_encoding::{create_test_string, select_test_string};
    use tiktoken_rs::{cl100k_base_singleton, o200k_base_singleton, CoreBPE};

    use super::*;

    #[test]
    fn test_cl100k() {
        test_equivalence(cl100k_base(), &cl100k_base_singleton().lock());
    }

    #[test]
    fn test_o200k() {
        test_equivalence(o200k_base(), &o200k_base_singleton().lock());
    }

    #[track_caller]
    fn test_equivalence(tok: &Tokenizer, tiktoken: &CoreBPE) {
        let text = create_test_string(&tok.bpe, 80_000);
        for bytes in [10, 100, 1000, 10_000] {
            for _ in 0..32 {
                let text = select_test_string(&text, bytes);
                let tokens = tok.encode(text);
                let tiktokens = tiktoken.encode_ordinary(text).to_vec();
                assert_eq!(tokens, tiktokens, "encoding mismatch for {text:?}");
            }
        }
    }
}
