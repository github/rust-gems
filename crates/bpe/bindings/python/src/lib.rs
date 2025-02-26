use std::borrow::Cow;

use pyo3::prelude::*;

#[pymodule]
mod bpe {
    use super::*;

    #[pyclass]
    struct BytePairEncoding(&'static ::bpe::byte_pair_encoding::BytePairEncoding);

    #[pymethods]
    impl BytePairEncoding {
        fn count(&self, input: &[u8]) -> usize {
            self.0.count(input)
        }

        fn encode_via_backtracking(&self, input: &[u8]) -> Vec<u32> {
            self.0.encode_via_backtracking(input)
        }

        fn decode_tokens(&self, tokens: Vec<u32>) -> Vec<u8> {
            self.0.decode_tokens(&tokens)
        }
    }

    #[pymodule]
    mod openai {
        use super::*;

        #[pyclass]
        struct Tokenizer(&'static ::bpe_openai::Tokenizer);

        #[pymethods]
        impl Tokenizer {
            fn count(&self, input: &str) -> usize {
                self.0.count(&input)
            }

            fn count_till_limit(&self, input: Cow<str>, limit: usize) -> Option<usize> {
                self.0.count_till_limit(&input, limit)
            }

            fn encode(&self, input: Cow<str>) -> Vec<u32> {
                self.0.encode(&input)
            }

            fn decode(&self, tokens: Vec<u32>) -> Option<String> {
                self.0.decode(&tokens)
            }

            fn bpe(&self) -> BytePairEncoding {
                BytePairEncoding(&self.0.bpe)
            }
        }

        #[pyfunction]
        fn cl100k_base() -> PyResult<Tokenizer> {
            Ok(Tokenizer(::bpe_openai::cl100k_base()))
        }

        #[pyfunction]
        fn o200k_base() -> PyResult<Tokenizer> {
            Ok(Tokenizer(::bpe_openai::o200k_base()))
        }
    }
}
