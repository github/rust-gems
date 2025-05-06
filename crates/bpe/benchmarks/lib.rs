use std::sync::LazyLock;

use bpe_openai::Tokenizer;
use tiktoken_rs::CoreBPE as TiktokenTokenizer;
use tokenizers::pre_tokenizers::byte_level::ByteLevel as HuggingfaceByteLevel;
use tokenizers::tokenizer::Tokenizer as HuggingfaceTokenizer;

pub static TOKENIZERS: LazyLock<
    [(
        &'static str,
        &'static Tokenizer,
        Option<TiktokenTokenizer>,
        HuggingfaceTokenizer,
    ); 3],
> = LazyLock::new(|| {
    [
        (
            "cl100k",
            bpe_openai::cl100k_base(),
            Some(tiktoken_rs::cl100k_base().expect("tokenizer available")),
            HuggingfaceTokenizer::from_pretrained("Xenova/gpt-4", None).expect("model available"),
        ),
        (
            "o200k",
            bpe_openai::o200k_base(),
            Some(tiktoken_rs::o200k_base().expect("tokenizer available")),
            HuggingfaceTokenizer::from_pretrained("Xenova/gpt-4o", None).expect("model available"),
        ),
        (
            "voyage3",
            bpe_openai::voyage3_base(),
            None,
            HuggingfaceTokenizer::from_pretrained("voyageai/voyage-code-3", None)
                .expect("model available"),
        ),
    ]
});

pub fn without_pretokenizer(enc: &HuggingfaceTokenizer) -> HuggingfaceTokenizer {
    let mut enc = enc.clone();
    // boolean values taken from Xenova's tokenizer config
    let pre_tokenizer = HuggingfaceByteLevel::new(false, false, false);
    enc.with_pre_tokenizer(Some(pre_tokenizer));
    enc
}
