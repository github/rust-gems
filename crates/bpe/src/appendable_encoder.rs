use daachorse::bytewise::iter::OverlappingStepper;

use crate::byte_pair_encoding::BytePairEncoding;

/// Encoder which keeps track of the encoding length.
pub struct AppendableEncoder<'a> {
    bpe: &'a BytePairEncoding,
    stepper: OverlappingStepper<'a, u32>,
    // TODO: If we only want to answer the length of the input text, then we could
    // replace these vectors with some fixed size arrays. Essentially we can only
    // go back up to the length of the longest token. This way can save some memory
    // and reallocations.
    last_token: Vec<u32>,
    counts: Vec<u32>,
}

impl<'a> AppendableEncoder<'a> {
    pub fn new(bpe: &'a BytePairEncoding) -> Self {
        Self {
            bpe,
            stepper: bpe.overlapping_searcher.overlapping_stepper(),
            last_token: vec![],
            counts: vec![],
        }
    }

    /// Appends multiple bytes to the input string.
    pub fn extend(&mut self, iter: impl Iterator<Item = u8>) {
        for c in iter {
            self.push(c);
        }
    }

    /// Appends a byte to the input string which should be tokenized.
    /// The operation is amortized O(1) (due to vector resizing).
    pub fn push(&mut self, c: u8) {
        self.stepper.consume(c);
        while let Some(m) = self.stepper.next() {
            let new_token = m.value();
            let new_range = m.start()..m.end();
            assert_eq!(new_range.end, self.last_token.len() + 1);
            if new_range.start == 0 {
                self.last_token.push(new_token);
                self.counts.push(1);
                break;
            } else {
                let prev_token = unsafe { *self.last_token.get_unchecked(new_range.start - 1) };
                if self.bpe.is_valid_token_pair(prev_token, new_token) {
                    self.last_token.push(new_token);
                    let prev_count = unsafe { *self.counts.get_unchecked(new_range.start - 1) };
                    self.counts.push(prev_count + 1);
                    break;
                }
            }
        }
    }

    /// Returns the number of tokens required to tokenize the input text.
    /// This operation is O(1) and can be called at any point in time.
    pub fn len(&self) -> usize {
        self.counts.last().copied().unwrap_or(0) as usize
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
            assert_eq!(enc.len(), bpe.count(&input_string[0..i]));
            enc.push(*c);
        }
    }
}
