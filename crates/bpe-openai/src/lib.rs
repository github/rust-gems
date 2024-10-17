use std::ops::Range;
use std::sync::LazyLock;

use bpe::byte_pair_encoding::BytePairEncoding;
use either::Either;
use fancy_regex::Regex;

pub use bpe::*;

static BPE_R50K_BASE: LazyLock<Tokenizer> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/bpe_r50k_base.dict"));
    let bpe = rmp_serde::from_slice(bytes).expect("valid bpe data");
    let pat = [
        "(?:'s|'t|'re|'ve|'m|'ll|'d)",
        " ?\\p{L}+",
        " ?\\p{N}+",
        " ?[^\\s\\p{L}\\p{N}]+",
        "\\s+", // "(:?\\s+(?!\\S)|\\s+)",
    ]
    .join("|");
    let pre =
        Pretokenizer::from_pat_and_trim(&pat, openai_trim_one_whitespace).expect("valid regex");
    Tokenizer::new(bpe, Some(pre))
});

static BPE_P50K_BASE: LazyLock<Tokenizer> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/bpe_p50k_base.dict"));
    let bpe = rmp_serde::from_slice(bytes).expect("valid bpe data");
    let pat = [
        "(?:'s|'t|'re|'ve|'m|'ll|'d)",
        " ?\\p{L}+",
        " ?\\p{N}+",
        " ?[^\\s\\p{L}\\p{N}]+",
        "\\s+", // "(:?\\s+(?!\\S)|\\s+)",
    ]
    .join("|");
    let pre =
        Pretokenizer::from_pat_and_trim(&pat, openai_trim_one_whitespace).expect("valid regex");
    Tokenizer::new(bpe, Some(pre))
});

static BPE_CL100K_BASE: LazyLock<Tokenizer> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/bpe_cl100k_base.dict"));
    let bpe = rmp_serde::from_slice(bytes).expect("valid bpe data");
    let pat = [
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)",
        "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+",
        "\\p{N}{1,3}",
        " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*",
        "\\s+", // "(?:\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)",
    ]
    .join("|");
    let pre = Pretokenizer::from_pat_and_trim(&pat, openai_trim_one_nonnewline_whitespace)
        .expect("valid regex");
    Tokenizer::new(bpe, Some(pre))
});

static BPE_O200K_BASE: LazyLock<Tokenizer> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/bpe_o200k_base.dict"));
    let bpe = rmp_serde::from_slice(bytes).expect("valid bpe data");
    let pat = [
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        "\\p{N}{1,3}",
        " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
        "\\s+", // "(?:\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)",
    ].join("|");
    let pre = Pretokenizer::from_pat_and_trim(&pat, openai_trim_one_nonnewline_whitespace)
        .expect("valid regex");
    Tokenizer::new(bpe, Some(pre))
});

/// A byte-pair encoding tokenizer that supports a pre-tokenization regex.
/// The direct methods on this type pre-tokenize the input text and should
/// produce the same output as the tiktoken tokenizers. The type gives access
/// to the regex and underlying byte-pair encoding if needed. Note that using
/// the byte-pair encoding directly does not take the regex into account and
/// may result in output that differs from tiktoken.
pub struct Tokenizer {
    /// The byte-pair encoding for this tokenizer.
    pub bpe: BytePairEncoding,
    /// The pretokenizer used to split the input.
    pub pre: Option<Pretokenizer>,
}

/// A trim function that for the given haystack and match range returns the number of bytes that should
/// be discarded from the end of the match.
pub type Trim = fn(&str, Range<usize>) -> usize;

pub struct Pretokenizer {
    pat: Regex,
    trim: Option<Trim>,
}

impl Tokenizer {
    #[allow(clippy::result_large_err)]
    pub fn new(bpe: BytePairEncoding, pre: Option<Pretokenizer>) -> Self {
        Self { bpe, pre }
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
        match &self.pre {
            Some(pre) => Either::Left(pre.split(text)),
            None => Either::Right(std::iter::once(text)),
        }
    }
}

impl Pretokenizer {
    #[allow(clippy::result_large_err)]
    pub fn from_pat(pat: &str) -> fancy_regex::Result<Self> {
        Ok(Self {
            pat: Regex::new(pat)?,
            trim: None,
        })
    }

    #[allow(clippy::result_large_err)]
    pub fn from_pat_and_trim(pat: &str, trim: Trim) -> fancy_regex::Result<Self> {
        Ok(Self {
            pat: Regex::new(pat)?,
            trim: Some(trim),
        })
    }

    pub fn split<'a>(&'a self, text: &'a str) -> impl Iterator<Item = &str> + 'a {
        let mut start = 0;
        std::iter::from_fn(move || {
            self.pat
                .find_from_pos(text, start)
                .expect("can search from position")
                .map(|m| {
                    let mut range = m.range();
                    if let Some(trim) = self.trim {
                        range.end -= trim(text, range.clone());
                    }
                    assert!(range.end > start);
                    start = range.end;
                    &text[range]
                })
        })
    }
}

pub fn r50k_base() -> &'static Tokenizer {
    &BPE_R50K_BASE
}

pub fn p50k_base() -> &'static Tokenizer {
    &BPE_P50K_BASE
}

pub fn cl100k_base() -> &'static Tokenizer {
    &BPE_CL100K_BASE
}

pub fn o200k_base() -> &'static Tokenizer {
    &BPE_O200K_BASE
}

/// Allows using `\\s+` instead of `(:?\\s+(?!\\S)|\\s+)`.
/// Assumes no other patterns match whitespace at the end.
fn openai_trim_one_whitespace(text: &str, range: Range<usize>) -> usize {
    if range.end == text.len() {
        return 0;
    }
    let mut chars = text[range].chars();
    match chars.next_back() {
        Some(c) if c.is_whitespace() && chars.next_back().is_some() => c.len_utf8(),
        _ => 0,
    }
}

/// Allows using `\\s+` instead of `(?:\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)`.
/// Assumes no other patterns match non-[\r\n] whitespace at the end.
fn openai_trim_one_nonnewline_whitespace(text: &str, range: Range<usize>) -> usize {
    if range.end == text.len() {
        return 0;
    }
    let mut chars = text[range].chars();
    match chars.next_back() {
        Some(c)
            if c.is_whitespace() && !matches!(c, '\r' | '\n') && chars.next_back().is_some() =>
        {
            c.len_utf8()
        }
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use tiktoken_rs::cl100k_base_singleton;

    use super::*;

    #[test]
    fn can_load_r50k() {
        r50k_base().count("");
    }

    #[test]
    fn can_load_p50k() {
        p50k_base().count("");
    }

    #[test]
    fn can_load_cl100k() {
        cl100k_base().count("");
    }

    #[test]
    fn can_load_o200k() {
        o200k_base().count("");
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
            .collect();

        let without_splitting = BPE_CL100K_BASE.bpe.encode_via_backtracking(input);
        assert_ne!(without_splitting, expected);

        let with_splitting: Vec<_> = BPE_CL100K_BASE.encode(text);
        assert_eq!(with_splitting, expected);
    }
}
