use std::sync::LazyLock;

use bpe::byte_pair_encoding::BytePairEncoding;
use either::Either;
use regex_automata::{
    meta::{BuildError, Regex},
    util::captures::Captures,
    Anchored, Input,
};

// Note: Below we rewrite the negative look-ahead with a positive pseudo look-ahead.
// The look-ahead character is dropped from the match by the Pretokenizer iterator.
// Note: The negative look-ahead `\\s+(?!\\S)` requires `\\s+\\s` but also `\\s+$` to handle end of file without dropping a character!

static BPE_CL100K_BASE: LazyLock<Tokenizer> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/bpe_cl100k_base.dict"));
    let bpe = rmp_serde::from_slice(bytes).expect("valid bpe data");
    let pat1 = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+$";
    let pat2 = "\\s+\\s";
    let pat3 = "\\s+";
    Tokenizer::new_lookahead(bpe, &[(pat1, false), (pat2, true), (pat3, false)])
        .expect("valid regex")
});

static BPE_O200K_BASE: LazyLock<Tokenizer> = LazyLock::new(|| {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/bpe_o200k_base.dict"));
    let bpe = rmp_serde::from_slice(bytes).expect("valid bpe data");
    let pat1 = [
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        "\\p{N}{1,3}",
        " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
        "\\s*[\\r\\n]+",
        "\\s+$",
    ].join("|");
    let pat2 = "\\s+\\s";
    let pat3 = "\\s+";
    Tokenizer::new_lookahead(bpe, &[(&pat1, false), (pat2, true), (pat3, false)])
        .expect("valid regex")
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
    pub pre: Option<Pretokenizer>,
}

pub struct Pretokenizer {
    /// The pattern regex used to split the input.
    pat: Regex,
    /// For each pattern in the regex a boolean whether the last character is a look-ahead.
    lookahead: Vec<bool>,
}

impl Tokenizer {
    /// Build a tokenizer with an optional pretokenization regex pattern.
    #[allow(clippy::result_large_err)]
    pub fn new(bpe: BytePairEncoding, pat: Option<&str>) -> Result<Self, BuildError> {
        let pre = pat.map(Pretokenizer::new).transpose()?;
        Ok(Self { bpe, pre })
    }

    /// Build a tokenizer with pretokenization regex patterns. If the boolean for a pattern is true,
    /// the pattern is assumed to be a look-ahead pattern with exactly one look-ahead character!
    #[allow(clippy::result_large_err)]
    pub fn new_lookahead(
        bpe: BytePairEncoding,
        patterns: &[(&str, bool)],
    ) -> Result<Self, BuildError> {
        let pre = Some(Pretokenizer::new_lookahead(patterns)?);
        Ok(Self { bpe, pre })
    }

    /// Count the number of tokens produced when encoding the text. Applies pre-tokenization
    /// before counting.
    pub fn count(&self, text: &str) -> usize {
        self.split(text)
            .map(|piece| self.bpe.count(piece.as_bytes()))
            .sum()
    }

    /// Returns the token count iff the total token count stays below the specified token_limit.
    /// Otherwise, it returns none. This function can be faster than [`Self::count`]` when the
    /// token limit is much smaller than the provided text. Applies pre-tokenization before counting.
    pub fn count_till_limit(&self, text: &str, token_limit: usize) -> Option<usize> {
        self.split(text)
            .try_fold(token_limit, |token_limit, piece| {
                self.bpe
                    .count_till_limit(piece.as_bytes(), token_limit)
                    .map(|piece_count| token_limit - piece_count)
            })
    }

    /// Returns the tokens for the encoding of the given text. Applies pre-tokenization before
    /// encoding.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.split(text)
            .flat_map(|piece| self.bpe.encode_via_backtracking(piece.as_bytes()))
            .collect()
    }
    /// Returns the text corresponding to the given encoding if it is valid UTF-8. Otherwise,
    /// returns none.
    pub fn decode(&self, tokens: &[u32]) -> Option<String> {
        String::from_utf8(self.bpe.decode_tokens(tokens)).ok()
    }

    /// Returns an iterator with the text pieces resulting from pre-tokenization. If this
    /// tokenizer does not have pre-tokenization, the iterator returns the full text.
    pub fn split<'a>(&'a self, text: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        match &self.pre {
            Some(pre) => Either::Left(pre.split(text)),
            None => Either::Right(std::iter::once(text)),
        }
    }
}

impl Pretokenizer {
    /// Build a pretokenizer from the given regex pattern.
    #[allow(clippy::result_large_err)]
    fn new(pat: &str) -> Result<Self, BuildError> {
        let pat = Regex::new(pat)?;
        Ok(Self {
            pat,
            lookahead: vec![false],
        })
    }

    /// Build a pretokenizer from the given regex patterns. If the boolean for a pattern is true,
    /// the pattern is assumed to be a look-ahead pattern with exactly one look-ahead character!
    #[allow(clippy::result_large_err)]
    fn new_lookahead(pats: &[(&str, bool)]) -> Result<Self, BuildError> {
        let (pats, lookahead): (Vec<_>, _) = pats.iter().copied().unzip();
        let pat = Regex::new_many(&pats)?;
        Ok(Self { pat, lookahead })
    }

    /// Returns an iterator with the text pieces after splitting with the regular expression.
    pub fn split<'a>(&'a self, text: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        Splits {
            pat: &self.pat,
            lookahead: &self.lookahead,
            text,
            last: 0,
            caps: Captures::matches(self.pat.group_info().clone()),
        }
    }
}

/// This is a small wrapper around the regex which emulates the behaviour of look-ahead by
/// dropping the look-ahead character from the match. The assumption here is that the
/// second pattern is always a look-ahead pattern, and that just a single character needs
/// to be dropped. With this little hack, we can keep most of the regex patterns as they are,
/// but achieve a >3x speedup.
///
/// Alternatively, this could have been implemented with capture groups, but those were ~30%
/// slower than this approach with multiple patterns.
struct Splits<'a> {
    pat: &'a Regex,
    lookahead: &'a [bool],
    text: &'a str,
    last: usize,
    caps: Captures,
}

impl<'a> Iterator for Splits<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let input = Input::new(&self.text[self.last..]).anchored(Anchored::Yes);
        self.caps.clear();
        self.pat.captures(input, &mut self.caps);
        let m = self.caps.get_match()?;
        let start = self.last;
        let mut end = self.last + m.range().end;
        if self.lookahead[m.pattern().as_usize()] {
            let last = self.text[start..end]
                .chars()
                .next_back()
                .expect("Expected at least a look-ahead character!");
            end -= last.len_utf8();
            assert_ne!(end, start, "a look-ahead pattern must ALWAYS consume at least one character excluding the look-ahead character!");
        }
        self.last = end;
        Some(&self.text[start..end])
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
