use std::ops::Range;

use crate::{
    Encodable, HashFunctions, coded_symbol::CodedSymbol, error::SetReconciliationError, indices,
    parent,
};

/// A session for encoding a stream of values.
/// This session can be used to merge multiple streams together, append more coded symbol from another session,
/// or extract coded symbols from the stream for decoding.
#[derive(Clone, PartialEq, Eq)]
pub struct EncodingSession<T: Encodable, H: HashFunctions<T>> {
    /// The hashing functions used for mapping values to indices.
    pub(crate) hasher: H,
    /// The range of the rateless stream which are encoded by this session.
    /// This way it is possible to just encode a subset if needed.
    pub(crate) range: Range<usize>,
    /// The coded symbols for the range
    pub(crate) coded_symbols: Vec<CodedSymbol<T>>,
    /// Starting at this point, the stream is represented by a hierarchy!
    /// We use a somewhat unique representation where coded symbols from 0..split are represented
    /// as a "normal" invertible bloom filter table and subsequent coded symbols are represented
    /// in a hierarchical way.
    /// The nice property of this representation is that we can switch back and forth between the two.
    /// This is useful, since certain operations are faster in one representation than the other.
    ///
    /// The hierarchical representation can be thought of as a rateless set reconciliation stream.
    pub(crate) split: usize,
}

impl<T: Encodable, H: HashFunctions<T>> EncodingSession<T, H> {
    /// Create a new encoding session with a given seed and range.
    pub fn new(state: H, range: Range<usize>) -> Self {
        EncodingSession {
            hasher: state,
            coded_symbols: vec![CodedSymbol::default(); range.len()],
            split: range.end, // We start with a non-hierarchical stream.
            range,
        }
    }

    /// Create a EncodingSession from a vector of coded symbols.
    ///
    /// Panics if the split is out of range or if the length of te vector
    /// and the length of the range differ.
    pub fn from_coded_symbols(
        state: H,
        coded_symbols: Vec<CodedSymbol<T>>,
        range: Range<usize>,
        split: usize,
    ) -> Self {
        assert!(split >= range.start && split <= range.end);
        assert_eq!(coded_symbols.len(), range.len());
        EncodingSession {
            hasher: state,
            coded_symbols,
            split,
            range,
        }
    }

    /// Create a EncodingSession from a vector of coded symbols.
    ///
    /// Returns an error if the split is out of range or if the length of te vector
    /// and the length of the range differ.
    pub fn try_from_coded_symbols(
        state: H,
        coded_symbols: Vec<CodedSymbol<T>>,
        range: Range<usize>,
        split: usize,
    ) -> Result<Self, SetReconciliationError> {
        if split < range.start || split > range.end {
            return Err(SetReconciliationError::SplitOutOfRange);
        }
        if coded_symbols.len() > range.len() {
            return Err(SetReconciliationError::RangeLengthMismatch);
        }
        Ok(EncodingSession {
            hasher: state,
            coded_symbols,
            split,
            range,
        })
    }

    /// Adds an entity to the encoding session.
    pub fn insert(&mut self, entity: T) {
        let check_hash = CodedSymbol::new(&self.hasher, entity, false);
        self.add_entity_inner(&check_hash, false);
    }

    /// Adds multiple entities to the encoding session.
    pub fn extend(&mut self, entities: impl Iterator<Item = T>) {
        for entity in entities {
            self.insert(entity);
        }
    }

    /// Returns the encoded rateless stream.
    /// Don't forget to either move the split point to the desired place or communicate it
    /// with the receiver of this data! Otherwise, the stream cannot be processed correctly!
    pub fn into_coded_symbols(self) -> impl Iterator<Item = CodedSymbol<T>> {
        self.coded_symbols.into_iter()
    }

    fn append_unchecked(&mut self, mut other: EncodingSession<T, H>) {
        other.move_split_point(self.range.end);
        self.coded_symbols.append(&mut other.coded_symbols);
        self.range.end = other.range.end;
    }

    /// Appends another encoded stream to this session.
    /// The function will automatically adapt the split point of the second stream, so that
    /// it is compatible with the first one.
    pub fn append(&mut self, other: EncodingSession<T, H>) {
        assert_eq!(self.hasher, other.hasher);
        assert_eq!(self.range.end, other.range.start);
        self.append_unchecked(other);
    }

    /// Attempt to append another encoding session onto this one returning an error
    /// if the hashers do not match or if the ranges are not contiguous.
    pub fn try_append(
        &mut self,
        other: EncodingSession<T, H>,
    ) -> Result<(), SetReconciliationError> {
        if self.hasher != other.hasher {
            return Err(SetReconciliationError::MismatchedHasher);
        }
        if self.range.end != other.range.start {
            return Err(SetReconciliationError::NonContiguousRanges);
        }
        self.append_unchecked(other);
        Ok(())
    }

    /// Call this function to extract the next `n` many coded symbols from the current session.
    /// Note: this will move the split point, such that the next extraction will also be fast.
    /// Inserting elements will however become slower if the split point is not moved to the end again!
    pub fn split_off(&mut self, n: usize) -> EncodingSession<T, H> {
        let split = (self.range.start + n).min(self.range.end);
        assert!(
            split > self.range.start,
            "split {split} must be greater than start {}",
            self.range.start
        );
        if split < self.split {
            self.move_split_point(split);
        }
        let mut rest = EncodingSession {
            hasher: self.hasher,
            range: split..self.range.end,
            coded_symbols: self.coded_symbols.split_off(split - self.range.start),
            split,
        };
        self.range.end = split;
        std::mem::swap(self, &mut rest);
        rest
    }

    fn merge_unchecked(mut self, other: EncodingSession<T, H>, negated: bool) -> Self {
        self.coded_symbols
            .iter_mut()
            .zip(other.coded_symbols)
            .for_each(|(a, b)| a.add(&b, negated));

        self
    }

    /// Merge another encoding session representing the same range into this one.
    /// This is needed in case of sharding or parallel processing.
    /// It should also be used to combine the data from two parties in order to determine
    /// the symmetric difference between their sets.
    ///
    /// The `negated` parameter indicates whether the values in the other session should be negated.
    pub fn merge(self, other: EncodingSession<T, H>, negated: bool) -> Self {
        assert_eq!(self.range, other.range);
        assert_eq!(self.hasher, other.hasher);
        self.merge_unchecked(other, negated)
    }

    /// Attempt to merge an encoding session into this one. If the hashers or ranges differ then
    /// this will fail with an error result.
    pub fn try_merge(
        self,
        other: EncodingSession<T, H>,
        negated: bool,
    ) -> Result<Self, SetReconciliationError> {
        if self.hasher != other.hasher {
            return Err(SetReconciliationError::MismatchedHasher);
        }
        if self.range != other.range {
            return Err(SetReconciliationError::MismatchedRanges);
        }
        Ok(self.merge_unchecked(other, negated))
    }

    /// Helper function which adds an entity to all the indices determined by the hash functions.
    /// It also works, when the stream is represented partially or fully hierarchically!
    /// Calls a functor for each changed position which might now represent a single entity.
    /// Counts how many elements changed to zero/non-zero.
    pub(crate) fn add_to_stream(
        &mut self,
        entity: &CodedSymbol<T>,
        negated: bool,
        range: Range<usize>,
        mut f: impl FnMut(CodedSymbol<T>, usize),
    ) -> (isize, usize) {
        let mut changes = 0;
        let mut required_bits = 0;
        for mut i in indices(&self.hasher, &entity.value, self.range.end) {
            while i >= self.split && i >= self.range.start && i > 0 {
                self.coded_symbols[i - self.range.start].add(entity, negated);
                i = parent(i);
            }
            if i >= self.range.start {
                if self.is_zero(i) {
                    changes += 1;
                }
                self.coded_symbols[i - self.range.start].add(entity, negated);
                if self.is_zero(i) {
                    changes -= 1;
                } else if range.contains(&i) {
                    let (is_pure, bits) = self.is_pure(i);
                    if is_pure {
                        let tmp = self.coded_symbols[i - self.range.start];
                        f(tmp, i);
                    } else {
                        required_bits = required_bits.max(bits);
                    }
                }
            }
        }
        (changes, required_bits)
    }

    /// Returns true if this is a pure coded symbol.
    pub(crate) fn is_pure(&self, i: usize) -> (bool, usize) {
        self.coded_symbols[i - self.range.start].is_pure(&self.hasher, i, self.split)
    }

    /// Returns true if no value is present in this coded symbol.
    pub(crate) fn is_zero(&self, i: usize) -> bool {
        self.coded_symbols[i - self.range.start].is_zero()
    }

    /// The caller has to ensure that the same seed was used to construct the entity!
    pub(crate) fn add_entity_inner(&mut self, entity: &CodedSymbol<T>, negated: bool) {
        self.add_to_stream(entity, negated, 0..0, |_, _| {});
    }

    /// Helper function to move the split point to a desired position.
    /// For fast insertion operations, the split point should be at the end of the represented range.
    /// For fast extraction operations, the split point should be at the beginning.
    pub(crate) fn move_split_point(&mut self, new_split: usize) {
        assert!(new_split <= self.range.end && new_split >= self.range.start);
        assert!(
            self.split <= self.range.end && self.split >= self.range.start,
            "{} {:?}",
            self.split,
            self.range
        );
        while self.split < new_split {
            if self.split > 0 {
                let i = parent(self.split);
                if i >= self.range.start {
                    let tmp = self.coded_symbols[self.split - self.range.start];
                    self.coded_symbols[i - self.range.start].add(&tmp, true);
                }
            }
            self.split += 1;
        }
        while self.split > new_split {
            self.split -= 1;
            if self.split > 0 {
                let i = parent(self.split);
                if i >= self.range.start {
                    let tmp = self.coded_symbols[self.split - self.range.start];
                    self.coded_symbols[i - self.range.start].add(&tmp, false);
                }
            }
        }
    }
}
