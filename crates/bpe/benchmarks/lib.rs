use std::sync::LazyLock;

use bpe::byte_pair_encoding::BytePairEncoding;
use bpe_openai::Tokenizer;
use rand::{thread_rng, Rng};
use tiktoken_rs::CoreBPE as TiktokenTokenizer;
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
            bpe_openai::cl100k(),
            tiktoken_rs::cl100k_base().unwrap(),
            { HuggingfaceTokenizer::from_pretrained("Xenova/gpt-4", None).unwrap() },
        ),
        (
            "o200k",
            bpe_openai::o200k(),
            tiktoken_rs::o200k_base().unwrap(),
            { HuggingfaceTokenizer::from_pretrained("Xenova/gpt-4o", None).unwrap() },
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

pub fn create_test_string(bpe: &BytePairEncoding, tokens: usize) -> String {
    use rand::{thread_rng, Rng};
    let mut text = String::new();
    for _ in 0..tokens {
        loop {
            let i = thread_rng().gen_range(0..bpe.num_tokens());
            let s = bpe.token_bytes(i as u32);
            if s.iter().all(|b| is_char_boundary(*b)) {
                if let Ok(s) = std::str::from_utf8(s) {
                    text.push_str(s);
                    break;
                }
            }
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
