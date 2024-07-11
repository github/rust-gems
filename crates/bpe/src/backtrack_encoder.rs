use crate::bitfield::BitField;
use crate::byte_pair_encoding::BytePairEncoding;

/// This can be thought of as a lazy variation of the dynamic programming approach.
/// It only computes those states which have to be visited in order to compute the tokenization
/// for a given input text.
/// It keeps track of visited states in a bitfield and only remembers the tokenization
/// of the currently processed dynamic programming state.
/// 
/// The biggest downside of this approach is that the search for the longest leftmost match
/// has to be reset at every (backtracking) step which is still a net win in practice compared to other approaches.
pub(crate) struct BacktrackEncoder<'a> {
    bpe: &'a BytePairEncoding,
    text: &'a [u8],
    tokens: Vec<u32>,
    next_token: Option<u32>,
    pos: usize,
    bitfield: BitField,
}

impl<'a> BacktrackEncoder<'a> {
    pub(crate) fn new(bpe: &'a BytePairEncoding, text: &'a [u8]) -> Self {
        Self::with_capacity(bpe, text, text.len() / 3)
    }

    pub(crate) fn with_capacity(bpe: &'a BytePairEncoding, text: &'a [u8], cap: usize) -> Self {
        Self {
            bpe,
            text,
            tokens: Vec::with_capacity(cap),
            next_token: bpe.next_match(text),
            pos: 0,
            bitfield: BitField::new(text.len() + 1),
        }
    }

    pub(crate) fn step(&mut self) -> Option<u32> {
        let mut token = self.next_token?;
        let last = self.tokens.last().copied();
        loop {
            let token_len = self.bpe.token_len(token);
            let end_pos = self.pos + token_len;
            if self.bitfield.is_set(end_pos)
                && last
                    .map(|last_token| self.bpe.is_valid_token_pair(last_token, token))
                    .unwrap_or(true)
            {
                self.tokens.push(token);
                self.pos = end_pos;
                // In principle, we could in some cases reuse the leftmost longest match iterator.
                // Especially when it has to look ahead, this could save scanning the input multiple times.
                // But on average this seems to be slower due to the overhead of storing the iterator as part of the struct.
                self.next_token = self.bpe.next_match(&self.text[end_pos..]);
                break;
            } else if let Some(shorter) = self.bpe.next_prefix(token) {
                token = shorter;
            } else {
                // Clearing the bitfield when we pop tokens saves a little bit of work...
                self.bitfield.clear(self.pos);
                self.tokens.pop();
                self.pos -= last.map(|t| self.bpe.token_len(t)).unwrap_or(0);
                self.next_token = last;
                break;
            }
        }
        self.next_token
    }

    pub(crate) fn count(&self) -> usize {
        self.tokens.len()
    }

    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    pub(crate) fn last_token(&self) -> Option<u32> {
        self.tokens.last().copied()
    }

    pub(crate) fn into_tokens(self) -> Vec<u32> {
        self.tokens
    }
}
