use std::sync::LazyLock;

use bpe_openai::Tokenizer;
use rand::{thread_rng, Rng};
use tiktoken_rs::CoreBPE as TiktokenTokenizer;
use tokenizers::pre_tokenizers::byte_level::ByteLevel as HuggingfaceByteLevel;
use tokenizers::tokenizer::Tokenizer as HuggingfaceTokenizer;

pub static TOKENIZERS: LazyLock<
    [(
        &'static str,
        &'static Tokenizer,
        TiktokenTokenizer,
        HuggingfaceTokenizer,
    ); 2],
> = LazyLock::new(|| {
    [
        (
            "cl100k",
            bpe_openai::cl100k_base(),
            tiktoken_rs::cl100k_base().expect("tokenizer available"),
            HuggingfaceTokenizer::from_pretrained("Xenova/gpt-4", None).expect("model available"),
        ),
        (
            "o200k",
            bpe_openai::o200k_base(),
            tiktoken_rs::o200k_base().expect("tokenizer available"),
            HuggingfaceTokenizer::from_pretrained("Xenova/gpt-4o", None).expect("model available"),
        ),
    ]
});

pub fn is_char_boundary(b: u8) -> bool {
    // Single byte encodings satisfy the bit pattern 0xxxxxxx, i.e. b < 128
    // Continuation bytes satisfy the bit pattern 10xxxxxx, i.e. b < 192
    // The rest are bytes belonging to the first byte of multi byte encodings (11xxxxxx): b >= 192
    // When interpreting the byte representation as signed integers, then numbers in the range 128..192
    // correspond to the smallest representable numbers. I.e. the two ranges [0, 128) and [192, 256) can
    // be tested with a single signed comparison.
    b as i8 >= -0x40 // NB: b < 128 || b >= 192
}

/// Create a test string from the given number of random tokens. Note that re-tokenizing the string
/// may result in a different token count! It is possible to request a string that cannot be split
/// with the tokenizers regex. Be aware that generating the string is slow in that case.
pub fn create_test_string(tok: &Tokenizer, tokens: usize, allow_splits: bool) -> String {
    use rand::{thread_rng, Rng};
    let mut text = String::new();
    let mut text_len = Vec::new();
    'next_token: while text_len.len() < tokens {
        // try a few of times to find a token
        for _ in 0..8 {
            // ensure the token results in a valid string
            loop {
                let i = thread_rng().gen_range(0..tok.bpe.num_tokens());
                let s = tok.bpe.token_bytes(i as u32);
                if s.iter().all(|b| is_char_boundary(*b)) {
                    if let Ok(s) = std::str::from_utf8(s) {
                        text_len.push(text.len());
                        text.push_str(s);
                        break;
                    }
                }
            }
            // if splits are allowed, or there are not splits, add the next token, otherwise drop the token and retry
            if allow_splits || tok.split(&text).nth(1).is_none() {
                continue 'next_token;
            } else {
                text.truncate(text_len.pop().expect("we just pushed a token"));
            }
        }
        // we failed to find a token that doesn't result in a split, we backtrack to try different combinations
        if let Some(len) = text_len.pop() {
            text.truncate(len)
        }
    }
    text
}

pub fn select_test_bytes(input: &[u8], bytes: usize) -> &[u8] {
    let mut start = thread_rng().gen_range(0..input.len() - bytes);
    while start > 0 && !is_char_boundary(input[start]) {
        start -= 1;
    }
    let mut end = start + bytes;
    while end < input.len() && !is_char_boundary(input[end]) {
        end += 1;
    }
    &input[start..end]
}

pub fn without_pretokenizer(enc: &HuggingfaceTokenizer) -> HuggingfaceTokenizer {
    let mut enc = enc.clone();
    // boolean values taken from Xenova's tokenizer config
    let pre_tokenizer = HuggingfaceByteLevel::new(false, false, false);
    enc.with_pre_tokenizer(Some(pre_tokenizer));
    enc
}
