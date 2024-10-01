use crate::byte_pair_encoding::{BytePairEncoding, State};

/// Encoder which keeps track of the encoding length while appending characters.
#[derive(Clone)]
pub struct AppendableEncoder<'a> {
    bpe: &'a BytePairEncoding,
    states: Vec<State>,
}

impl<'a> AppendableEncoder<'a> {
    pub fn new(bpe: &'a BytePairEncoding) -> Self {
        Self {
            bpe,
            states: vec![],
        }
    }

    /// Appends multiple bytes to the input string.
    pub fn extend(&mut self, iter: impl Iterator<Item = u8>) {
        for c in iter {
            self.push(c);
        }
    }

    pub fn truncate(&mut self, len: usize) {
        self.states.truncate(len);
    }

    /// Appends a byte to the input string which should be tokenized.
    /// The operation is amortized O(1) (due to vector resizing).
    pub fn push(&mut self, c: u8) {
        self.bpe.encode_next_byte(&mut self.states, c);
    }

    /// Returns the number of tokens required to tokenize the input text.
    /// This operation is O(1) and can be called at any point in time.
    pub fn token_count(&self) -> usize {
        self.states.last().map(|s| s.count).unwrap_or(0) as usize
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns true if the structure represents the empty string.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use crate::byte_pair_encoding::{create_test_bytes, BytePairEncoding};

    use super::AppendableEncoder;

    #[test]
    fn test_appendable_encoder() {
        let bpe = BytePairEncoding::cl100k();
        let mut enc = AppendableEncoder::new(bpe);
        let input_string = create_test_bytes(bpe, 100);
        for (i, c) in input_string.iter().enumerate() {
            assert_eq!(enc.token_count(), bpe.count(&input_string[0..i]));
            enc.push(*c);
        }
    }
}
