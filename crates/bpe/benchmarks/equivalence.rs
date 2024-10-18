use bpe::byte_pair_encoding::{create_test_string, select_test_string};
use bpe_benchmarks::*;

#[cfg(test)]
const N: usize = 32;

#[test]
fn test_huggingface_encoding_equivalence_without_pretokenization() {
    for (_, bpe, _, huggingface) in TOKENIZERS.iter() {
        let huggingface = without_pretokenizer(huggingface);
        let text = create_test_string(&bpe.bpe, 80_000);
        let texts = (0..N)
            .map(|_| select_test_string(&text, 100))
            .chain(std::iter::once(
                "You should see the Greek word 'kosme':       \"κόσμε\"",
            ));
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
                        "huggingface tokens and text differ: {:?} != {:?}",
                        text, huggingface_text
                    );
                } else {
                    panic!(
                        "huggingface tokens differ: {:?} != {:?}",
                        out, huggingface_out
                    );
                }
            }
        }
    }
}

#[test]
fn test_huggingface_encoding_equivalence_with_pretokenization() {
    for (_, bpe, _, huggingface) in TOKENIZERS.iter() {
        let text = create_test_string(&bpe.bpe, 80_000);
        let texts = (0..N)
            .map(|_| select_test_string(&text, 100))
            .chain(std::iter::once(
                "You should see the Greek word 'kosme':       \"κόσμε\"",
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
                        "huggingface tokens and text differ: {:?} != {:?}",
                        text, huggingface_text
                    );
                } else {
                    panic!(
                        "huggingface tokens differ: {:?} != {:?}",
                        out, huggingface_out
                    );
                }
            }
        }
    }
}
