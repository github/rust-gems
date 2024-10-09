use bpe_benchmarks::*;

#[test]
fn test_encoding_equivalence() {
    for (_, bpe, tiktoken, huggingface) in TOKENIZERS.iter() {
        let text = create_test_string(&bpe.bpe, 20000);
        let inputs = (0..32)
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
                .iter()
                .copied()
                .collect();
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
