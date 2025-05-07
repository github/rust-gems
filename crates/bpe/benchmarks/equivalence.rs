use std::collections::HashSet;

use bpe::byte_pair_encoding::{create_test_string, select_test_string};
use bpe_benchmarks::*;

/// Converts bytes to unicode characters.
/// See https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
/// Hugging face uses the same mapping to work with unicode instead of byte characters.
fn char_to_byte(c: char) -> u8 {
    match c as u32 {
        0x21..0x7f => c as u8,  // 94
        0xa1..=0xac => c as u8, // 12
        0xae..=0xff => c as u8, // 82
        0x7f..0xa1 => c as u8 - 0x7f + 221,
        0x100..0x121 => (c as u32 - 0x100) as u8,
        0x121..0x143 => (c as u32 - 0x121) as u8 + 0x7f,
        0x143..0x144 => 0xad,
        _ => panic!("Invalid character: {c} {}", c as u32),
    }
}

#[test]
fn test_compare_dictionary() {
    for (name, bpe, _, huggingface) in TOKENIZERS.iter() {
        let huggingface = without_pretokenizer(huggingface);
        let mut hugging_tokens = huggingface.get_vocab(false);
        // HACK: There are incorrect vocabularies in huggingface which have the added tokens stored together with the base tokens..
        // This is a workaround to remove them.
        for added_token in huggingface.get_added_vocabulary().get_vocab().keys() {
            hugging_tokens.remove(added_token);
        }
        let mut hugging_tokens: Vec<_> = hugging_tokens.into_iter().collect();
        hugging_tokens.sort_by(|(_, a), (_, b)| a.cmp(b));
        let hugging_tokens: Vec<_> = hugging_tokens
            .into_iter()
            .map(|(token, _)| token.chars().map(char_to_byte).collect())
            .collect();
        let bpe_tokens: Vec<_> = (0..bpe.bpe.num_tokens())
            .map(|id| bpe.bpe.token_bytes(id as u32).to_vec())
            .collect();
        let hugging_set: HashSet<_> = hugging_tokens.iter().cloned().collect();
        let bpe_set: HashSet<_> = bpe_tokens.iter().cloned().collect();
        let diff: Vec<_> = hugging_set.symmetric_difference(&bpe_set).collect();
        assert!(diff.is_empty(), "{name}: Token sets differ");
        // Uncomment the following lines to write the tokens to a file in tiktoken format
        /*
        let mut file =
            std::fs::File::create(std::path::Path::new(_name)).expect("can create output file");
        std::io::Write::write_all(
            &mut file,
            bpe::byte_pair_encoding::write_tiktoken(hugging_tokens).as_bytes(),
        )
        .expect("can write output to file");
        */
    }
}

#[test]
fn test_huggingface_encoding_equivalence_without_pretokenization() {
    for (name, bpe, _, huggingface) in TOKENIZERS.iter() {
        let text: String = create_test_string(&bpe.bpe, 200_000);
        let text = bpe.normalize(&text);
        let texts = (0..300)
            .map(|_| select_test_string(text.as_str(), 100))
            .chain(std::iter::once(
                "You should see the Greek word 'kosme':       \"κόσμε\"",
            ));
        let huggingface = without_pretokenizer(huggingface);
        for text in texts {
            let out = bpe.bpe.encode_via_backtracking(text.as_bytes());
            let huggingface_out = huggingface
                .encode_fast(text, false)
                .unwrap()
                .get_ids()
                .to_vec();
            if huggingface_out != out {
                let text = bpe.decode(&out).unwrap();
                let huggingface_text = huggingface.decode(&huggingface_out, true).unwrap();
                if huggingface_text != text {
                    panic!(
                        "{name}: huggingface tokens and text differ: {text:?} != {huggingface_text:?}",
                    );
                } else {
                    panic!("{name}: huggingface tokens differ: {out:?} != {huggingface_out:?}");
                }
            }
        }
    }
}

#[test]
fn test_huggingface_encoding_equivalence_with_pretokenization() {
    for (name, bpe, _, huggingface) in TOKENIZERS.iter() {
        let text = create_test_string(&bpe.bpe, 200_000);
        let texts = (0..300)
            .map(|_| select_test_string(&text, 100))
            .chain(std::iter::once(
                "You should see the Greek word 'kosme':       \"κόσμε\"   ",
            ));
        for text in texts {
            let out = bpe.encode(text);
            let huggingface_out = huggingface
                .encode_fast(text, false)
                .unwrap()
                .get_ids()
                .to_vec();

            if huggingface_out != out {
                let text = bpe.decode(&out).unwrap();
                let huggingface_text = huggingface.decode(&huggingface_out, true).unwrap();
                if huggingface_text != text {
                    panic!(
                        "{name}: huggingface tokens and text differ: {text:?} != {huggingface_text:?}",
                    );
                } else {
                    panic!("{name}: huggingface tokens differ: {out:?} != {huggingface_out:?}");
                }
            }
        }
    }
}
