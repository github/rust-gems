use daachorse::bytewise::iter::OverlappingStepper;

use crate::byte_pair_encoding::BytePairEncoding;

/// Encoder which keeps track of the encoding length while prepending characters.
/// 
/// TODO: Implement a way to checkpoint the state of the encoder.
/// This requires changing the aho corasick API to just return the current state.
pub struct PrependableEncoder<'a> {
    bpe: &'a BytePairEncoding,
    stepper: OverlappingStepper<'a, u32>,
    prev_token: Vec<u32>,
    counts: Vec<u32>,
}

impl<'a> PrependableEncoder<'a> {
    pub fn new(bpe: &'a BytePairEncoding) -> Self {
        Self {
            bpe,
            stepper: bpe.overlapping_searcher_rev.overlapping_stepper(),
            prev_token: vec![],
            counts: vec![],
        }
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
        self.stepper.consume(c);
        while let Some(m) = self.stepper.next() {
            let new_token = m.value();
            let new_range = m.start()..m.end();
            assert_eq!(new_range.end, self.prev_token.len() + 1);
            if new_range.start == 0 {
                self.prev_token.push(new_token);
                self.counts.push(1);
                break;
            } else {
                let next_token = unsafe { *self.prev_token.get_unchecked(new_range.start - 1) };
                if self.bpe.is_valid_token_pair(new_token, next_token) {
                    self.prev_token.push(new_token);
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

    /// Returns true if the structure represents the empty string.
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
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
            assert_eq!(enc.len(), bpe.count(&input_string[i..]));
        }
    }
}
