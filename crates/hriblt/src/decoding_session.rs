use std::collections::HashSet;

use crate::{
    Encodable, HashFunctions, coded_symbol::CodedSymbol, decoded_value::DecodedValue,
    encoding_session::EncodingSession, error::SetReconciliationError, parent,
};

/// A session for decoding a stream of hashes.
#[derive(Clone)]
pub struct DecodingSession<T: Encodable, H: HashFunctions<T>> {
    /// The encoded stream of hashes.
    /// All recovered coded symbols have been removed from this stream.
    /// If decoded failed, then one can simply append more data and continue decoding.
    encoded: EncodingSession<T, H>,
    /// All the recovered coded symbols.
    recovered: Vec<CodedSymbol<T>>,
    /// Don't decode the same coded symbol multiple times (it might occur up to 5 times in the stream!).
    visited: HashSet<T>,
    /// Tracks the number of non-zero coded symbol up to the split point.
    pub(crate) non_zero: isize,

    /// For statistical purposes: this number informs how many bits of the
    /// checksum were required to identify pure coded symbols.
    pub(crate) required_bits: usize,
}

impl<T: Encodable + std::fmt::Debug, H: HashFunctions<T>> DecodingSession<T, H> {
    /// Create a new decoding session with a given seed.
    pub fn new(state: H) -> Self {
        DecodingSession {
            recovered: vec![],
            encoded: EncodingSession::new(state, 0..0),
            visited: HashSet::default(),
            non_zero: 0,
            required_bits: 0,
        }
    }

    fn from_encoding_unchecked(merged: EncodingSession<T, H>) -> Self {
        let mut me = DecodingSession {
            recovered: vec![],
            encoded: merged,
            visited: HashSet::default(),
            non_zero: 0,
            required_bits: 0,
        };
        // We work here with a non-hierarchical stream.
        me.encoded.move_split_point(me.encoded.range.end);
        me.non_zero = me
            .encoded
            .coded_symbols
            .iter()
            .filter(|e| !e.is_zero())
            .count() as isize;
        let mut j = me.recovered.len();
        let len = me.encoded.coded_symbols.len();
        for i in (0..len).rev() {
            if me.non_zero == 0 {
                break;
            }
            let (is_pure, required_bits) = me.encoded.is_pure(i);
            if is_pure && !me.visited.contains(&me.encoded.coded_symbols[i].value) {
                me.visited.insert(me.encoded.coded_symbols[i].value);
                me.recovered.push(me.encoded.coded_symbols[i]);
            }
            if !is_pure && required_bits > me.required_bits {
                me.required_bits = required_bits;
            }
            while j < me.recovered.len() {
                let entity = me.recovered[j];
                let (changes, required_bits) =
                    me.encoded.add_to_stream(&entity, true, i + 1..len, |e, k| {
                        assert!(k > i);
                        if !me.visited.contains(&e.value) {
                            me.visited.insert(e.value);
                            me.recovered.push(e);
                        }
                    });
                me.non_zero += changes;
                me.required_bits = me.required_bits.max(required_bits);
                j += 1;
            }
        }
        me
    }

    /// This is a faster version for decoding the initial stream.
    /// It processes this stream from back to front without going through the hierarchical representation.
    /// The other procedure needs to execute roughly one additional `is_pure` test when unrolling the hierarchy
    /// which this procedure avoids.
    /// Additionally, this procedure can save on average another 50% of is_pure tests, since it won't waste time
    /// on the highly packed hierarchy levels where we don't expect to find any pure values.
    ///
    /// Panics if the encoding session is not the beginning of a stream (e.g. the range is `0..n`)
    pub fn from_encoding(merged: EncodingSession<T, H>) -> Self {
        assert_eq!(merged.range.start, 0);
        Self::from_encoding_unchecked(merged)
    }

    /// This is a faster version for decoding the initial stream.
    /// It processes this stream from back to front without going through the hierarchical representation.
    /// The other procedure needs to execute roughly one additional `is_pure` test when unrolling the hierarchy
    /// which this procedure avoids.
    /// Additionally, this procedure can save on average another 50% of is_pure tests, since it won't waste time
    /// on the highly packed hierarchy levels where we don't expect to find any pure values.
    ///
    /// Returns an error if the encoding session is not the beginning of a stream (e.g. the range is `0..n`)
    pub fn try_from_encoding(
        merged: EncodingSession<T, H>,
    ) -> Result<Self, SetReconciliationError> {
        if merged.range.start != 0 {
            return Err(SetReconciliationError::NotInitialRange);
        }
        Ok(Self::from_encoding_unchecked(merged))
    }

    fn append_unchecked(&mut self, mut merged: EncodingSession<T, H>) {
        // Apply all the reconstructed entities to the new part of the stream.
        for entity in &self.recovered {
            merged.add_entity_inner(entity, true);
        }
        assert_eq!(self.encoded.split, self.encoded.range.end);
        self.encoded.append(merged);
        // Now continue decoding starting with the newly arrived data.
        let mut j = self.recovered.len();
        for i in self.encoded.split..self.encoded.range.end {
            // Undo hierarchy manually here, since we also need to count non-zero entries along the way.
            let ii = parent(i);
            if i > 0 {
                let tmp = self.encoded.coded_symbols[i];
                if !self.encoded.is_zero(ii) {
                    self.non_zero -= 1;
                }
                self.encoded.coded_symbols[ii].add(&tmp, true);
                if !self.encoded.is_zero(ii) {
                    self.non_zero += 1;
                }
            }
            self.encoded.split = i + 1;
            if !self.encoded.is_zero(i) {
                self.non_zero += 1;
            }
            for l in [i, ii] {
                let (is_pure, required_bits) = self.encoded.is_pure(l);
                if is_pure && !self.visited.contains(&self.encoded.coded_symbols[l].value) {
                    self.visited.insert(self.encoded.coded_symbols[l].value);
                    self.recovered.push(self.encoded.coded_symbols[l]);
                }
                if !is_pure && required_bits > self.required_bits {
                    self.required_bits = required_bits;
                }
            }
            while j < self.recovered.len() {
                let entity = self.recovered[j];
                let (changes, required_bits) =
                    self.encoded.add_to_stream(&entity, true, 0..i, |e, k| {
                        assert!(k < i);
                        if !self.visited.contains(&e.value) {
                            self.visited.insert(e.value);
                            self.recovered.push(e);
                        }
                    });
                self.non_zero += changes;
                self.required_bits = self.required_bits.max(required_bits);
                j += 1;
            }
            if self.non_zero == 0 {
                // At this point everything should be decoded...
                // We could in theory check that all remaining coded symbols are zero.
                break;
            }
        }
    }

    /// Appends the next chunk of coded symbols to the decoding session.
    /// This should only be called if decoding was not yet completed.
    /// Panics if the encoding session is not a contiguous range with `self`.
    pub fn append(&mut self, merged: EncodingSession<T, H>) {
        assert_eq!(self.encoded.range.end, merged.range.start);
        self.append_unchecked(merged);
    }

    /// Appends the next chunk of coded symbols to the decoding session.
    /// This should only be called if decoding was not yet completed.
    /// Returns an error if the encoding session is not a contiguous range with `self`.
    pub fn try_append(
        &mut self,
        merged: EncodingSession<T, H>,
    ) -> Result<(), SetReconciliationError> {
        if self.encoded.range.end != merged.range.start {
            return Err(SetReconciliationError::NonContiguousRanges);
        }
        self.append_unchecked(merged);
        Ok(())
    }

    /// Returns whether decoding has successfully finished.
    pub fn is_done(&self) -> bool {
        !self.encoded.coded_symbols.is_empty() && self.non_zero == 0
    }

    /// Returns the number of coded symbols that were consumed during the decoding process.
    pub fn consumed_coded_symbols(&self) -> usize {
        self.encoded.split
    }

    /// Extract the decoded entities from the session.
    /// Only call when `is_done()` returns true.
    pub fn into_decoded_iter(self) -> impl Iterator<Item = DecodedValue<T>> {
        // We have decoded the stream successfully.
        // Now we can return the decoded entities.
        let hasher = self.encoded.hasher;
        self.recovered.into_iter().map(move |e| {
            if e.is_deletion(&hasher) {
                DecodedValue::Deletion(e.value)
            } else {
                DecodedValue::Addition(e.value)
            }
        })
    }
}
