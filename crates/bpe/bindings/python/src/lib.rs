use std::borrow::Cow;

use pyo3::prelude::*;

#[pyclass]
struct BytePairEncoding(Cow<'static, ::bpe::byte_pair_encoding::BytePairEncoding>);

#[pyclass]
struct Tokenizer(Cow<'static, ::bpe_openai::Tokenizer>);

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
}

#[pyfunction]
fn cl100k_base() -> PyResult<Tokenizer> {
    Ok(Tokenizer(Cow::Borrowed(::bpe_openai::cl100k_base())))
}

#[pyfunction]
fn o200k_base() -> PyResult<Tokenizer> {
    Ok(Tokenizer(Cow::Borrowed(::bpe_openai::o200k_base())))
}

#[pymodule]
fn bpe_openai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cl100k_base, m)?)?;
    m.add_function(wrap_pyfunction!(o200k_base, m)?)?;
    Ok(())
}
