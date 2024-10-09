use std::borrow::Cow;

use pyo3::prelude::*;

#[pyclass]
struct BytePairEncoding(Cow<'static, ::bpe::byte_pair_encoding::BytePairEncoding>);

#[pymethods]
impl BytePairEncoding {
    fn count(&self, input: Cow<[u8]>) -> usize {
        self.0.count(&input)
    }

    fn encode_via_backtracking(&self, input: Cow<[u8]>) -> Vec<u32> {
        self.0.encode_via_backtracking(&input)
    }

    fn decode_tokens(&self, tokens: Vec<u32>) -> Cow<[u8]> {
        Cow::Owned(self.0.decode_tokens(&tokens))
    }
}

#[pyfunction]
fn r50k() -> PyResult<BytePairEncoding> {
    Ok(BytePairEncoding(Cow::Borrowed(::bpe_openai::r50k())))
}

#[pyfunction]
fn p50k() -> PyResult<BytePairEncoding> {
    Ok(BytePairEncoding(Cow::Borrowed(::bpe_openai::p50k())))
}

#[pyfunction]
fn cl100k() -> PyResult<BytePairEncoding> {
    Ok(BytePairEncoding(Cow::Borrowed(::bpe_openai::cl100k())))
}

#[pyfunction]
fn o200k() -> PyResult<BytePairEncoding> {
    Ok(BytePairEncoding(Cow::Borrowed(::bpe_openai::o200k())))
}

#[pymodule]
fn bpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(r50k, m)?)?;
    m.add_function(wrap_pyfunction!(p50k, m)?)?;
    m.add_function(wrap_pyfunction!(cl100k, m)?)?;
    m.add_function(wrap_pyfunction!(o200k, m)?)?;
    Ok(())
}
