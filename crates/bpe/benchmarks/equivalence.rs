use bpe_benchmarks::*;

#[cfg(test)]
const N: usize = 32;

#[test]
fn test_encoding_equivalence_without_pretokenization() {
    for (_, bpe, _, huggingface) in TOKENIZERS.iter() {
        let huggingface = without_pretokenizer(huggingface);
        let text = create_test_string(&bpe.bpe, 20000);
        let inputs = (0..N)
            .map(|_| select_test_bytes(text.as_bytes(), 100))
            .chain(std::iter::once(
                "You should see the Greek word 'kosme':       \"κόσμε\"".as_bytes(),
            ));
        for input in inputs {
            let text = std::str::from_utf8(input).unwrap();
            let out = bpe.bpe.encode_via_backtracking(input);
            let huggingface_out: Vec<_> = huggingface
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
fn test_encoding_equivalence_with_pretokenization() {
    for (_, bpe, tiktoken, huggingface) in TOKENIZERS.iter() {
        let text = create_test_string(&bpe.bpe, 20000);
        let inputs = (0..N)
            .map(|_| select_test_bytes(text.as_bytes(), 100))
            .chain(std::iter::once(
                "You should see the Greek word 'kosme':       \"κόσμε\"".as_bytes(),
            ));
        for input in inputs {
            let text = std::str::from_utf8(input).unwrap();
            let out = bpe.encode(text);
            let tiktoken_out: Vec<_> = tiktoken.encode_ordinary(text);
            let tiktoken_out2: Vec<_> = tiktoken_out.iter().map(|i| *i as u32).collect();
            let tiktoken_text = tiktoken.decode(tiktoken_out.clone()).unwrap();
            let huggingface_out: Vec<_> = huggingface
                .encode_fast(text, false)
                .unwrap()
                .get_ids()
                .to_vec();
            if tiktoken_out2 != huggingface_out {
                let huggingface_text = huggingface.decode(&huggingface_out, true).unwrap();
                if tiktoken_text != huggingface_text {
                    panic!(
                        "huggingface tokens and text differ: {:?} != {:?}",
                        huggingface_text, tiktoken_text
                    );
                } else {
                    panic!(
                        "huggingface tokens differ: {:?} != {:?}",
                        huggingface_out, tiktoken_out2
                    );
                }
            }
            if tiktoken_out2 != out {
                let text = bpe.decode(&out).unwrap();
                if tiktoken_text != text {
                    panic!(
                        "bpe tokens and text differ: {:?} != {:?}",
                        text, tiktoken_text
                    );
                } else {
                    panic!("bpe tokens differ: {:?} != {:?}", out, tiktoken_out2);
                }
            }
        }
    }
}
