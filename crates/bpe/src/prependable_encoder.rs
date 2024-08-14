use crate::byte_pair_encoding::BytePairEncoding;

struct State {
    state: u32,
    prev_token: u32,
    count: u32,
}

/// Encoder which keeps track of the encoding length while prepending characters.
pub struct PrependableEncoder<'a> {
    bpe: &'a BytePairEncoding,
    states: Vec<State>,
}

impl<'a> PrependableEncoder<'a> {
    pub fn new(bpe: &'a BytePairEncoding) -> Self {
        Self {
            bpe,
            states: vec![],
        }
    }

    pub fn truncate(&mut self, len: usize) {
        self.states.truncate(len);
    }

    /// Prepends multiple bytes to the input string.
    pub fn extend(&mut self, iter: impl Iterator<Item = u8>) {
        for c in iter {
            self.push(c);
        }
    }

    /// Prepends a byte to the input string which should be tokenized.
    /// The operation is amortized O(1) (due to vector resizing).
    pub fn push(&mut self, c: u8) {
        let (state, iter) = self.bpe.overlapping_searcher_rev.consume(
            self.states
                .last()
                .map(|s| s.state)
                .unwrap_or_else(|| self.bpe.overlapping_searcher_rev.start_state()),
            self.states.len() + 1,
            c,
        );
        for m in iter {
            let new_token = m.value();
            let new_range = m.start()..m.end();
            assert_eq!(new_range.end, self.states.len() + 1);
            if new_range.start == 0 {
                self.states.push(State {
                    state,
                    prev_token: new_token,
                    count: 1,
                });
                break;
            } else {
                let next_token =
                    unsafe { self.states.get_unchecked(new_range.start - 1).prev_token };
                if self.bpe.is_valid_token_pair(new_token, next_token) {
                    let prev_count =
                        unsafe { self.states.get_unchecked(new_range.start - 1).count };
                    self.states.push(State {
                        state,
                        prev_token: new_token,
                        count: prev_count + 1,
                    });
                    break;
                }
            }
        }
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

    use super::PrependableEncoder;

    #[test]
    fn test_prependable_encoder() {
        let bpe = BytePairEncoding::cl100k();
        let mut enc = PrependableEncoder::new(bpe);
        let input_string = create_test_bytes(bpe, 100);
        for (i, c) in input_string.iter().enumerate().rev() {
            enc.push(*c);
            assert_eq!(enc.token_count(), bpe.count(&input_string[i..]));
        }
    }
}
