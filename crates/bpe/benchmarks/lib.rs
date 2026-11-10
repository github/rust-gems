use std::sync::LazyLock;

use bpe_openai::Tokenizer;
use serde_json::Value;
use tiktoken_rs::CoreBPE as TiktokenTokenizer;
use tokenizers::pipeline::{EncodeOptions, PipelineTokenizer as HuggingfaceTokenizer};

#[allow(clippy::type_complexity)]
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
            load_huggingface_tokenizer("Xenova/gpt-4"),
        ),
        (
            "o200k",
            bpe_openai::o200k_base(),
            Some(tiktoken_rs::o200k_base().expect("tokenizer available")),
            load_huggingface_tokenizer("Xenova/gpt-4o"),
        ),
        (
            "voyage3",
            bpe_openai::voyage3_base(),
            None,
            load_huggingface_tokenizer("voyageai/voyage-code-3"),
        ),
    ]
});

fn load_huggingface_tokenizer(model: &str) -> HuggingfaceTokenizer {
    let path =
        tokenizers::utils::from_pretrained::from_pretrained(model, None).expect("model available");
    let json = tokenizers::canonicalize_file(path).expect("tokenizer config canonicalized");
    tokenizers::from_json(&json).expect("tokenizer config parsed")
}

pub fn without_pretokenizer(enc: &HuggingfaceTokenizer) -> HuggingfaceTokenizer {
    let json = tokenizers::to_json(enc).expect("tokenizer config serialized");
    let mut json: Value = serde_json::from_str(&json).expect("tokenizer config parsed");
    json["pre_tokenizer"] = Value::Null;
    tokenizers::from_json(&json.to_string()).expect("tokenizer config loaded without pretokenizer")
}

pub fn encode_huggingface(tokenizer: &HuggingfaceTokenizer, text: &str) -> Vec<u32> {
    tokenizer
        .encode(text, &EncodeOptions::no_specials())
        .wait()
        .expect("encoding succeeds")
        .remove(0)
        .ids()
        .iter()
        .map(|token| token.id())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_pretokenizer_removes_component_and_preserves_encoding() {
        let tokenizer = tokenizers::from_json(
            r#"{"version":"2.0","added_tokens":[],"normalizer":null,
                "pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,
                "decoder":null,"model":{"type":"BPE","byte_level":false,
                "vocab":{"a":0,"b":1,"ab":2,"abab":3},
                "merges":[["a","b"],["ab","ab"]]},"padding":null}"#,
        )
        .expect("test tokenizer parsed");

        assert_ne!(
            tokenizer.get_pre_tokenizer(),
            &tokenizers::pipeline::PipelinePreTokenizer::None
        );
        let without_pretokenizer = without_pretokenizer(&tokenizer);
        assert_eq!(
            without_pretokenizer.get_pre_tokenizer(),
            &tokenizers::pipeline::PipelinePreTokenizer::None
        );
        assert_eq!(encode_huggingface(&without_pretokenizer, "abab"), [3]);
    }
}
