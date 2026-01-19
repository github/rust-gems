use crate::{Encodable, HashFunctions, index_for_seed, indices};

/// Represents a coded symbol in the invertible bloom filter table.
/// In some of the literature this is referred to as a "cell" or "bucket".
/// It includes a checksum to verify whether the instance represents a pure value.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct CodedSymbol<T: Encodable> {
    /// Values aggregated by XOR operation.
    pub value: T,
    /// We repurpose the two least significant bits of the checksum:
    /// - The least significant bit is a one bit counter which is incremented for each entity.
    ///   This bit must be set when there is a single entity represented by this hash.
    /// - The second least significant bit indicates whether the entity is a deletion or insertion.
    pub checksum: u64,
}

impl<T: Encodable> Default for CodedSymbol<T> {
    fn default() -> Self {
        CodedSymbol {
            value: T::zero(),
            checksum: 0,
        }
    }
}

impl<T: Encodable> From<(T, u64)> for CodedSymbol<T> {
    fn from(tuple: (T, u64)) -> Self {
        Self {
            value: tuple.0,
            checksum: tuple.1,
        }
    }
}

impl<T: Encodable> CodedSymbol<T> {
    /// Creates a new coded symbol with the given hash and deletion flag.
    pub(crate) fn new<S: HashFunctions<T>>(state: &S, hash: T, deletion: bool) -> Self {
        let mut checksum = state.check_sum(&hash);
        checksum |= 1; // Add a single bit counter
        if deletion {
            checksum = checksum.wrapping_neg();
        }
        CodedSymbol {
            value: hash,
            checksum,
        }
    }

    /// Merges another coded symbol into this one.
    pub(crate) fn add(&mut self, other: &CodedSymbol<T>, negate: bool) {
        self.value.xor(other.value);
        if negate {
            self.checksum = self.checksum.wrapping_sub(other.checksum);
        } else {
            self.checksum = self.checksum.wrapping_add(other.checksum);
        }
    }

    /// Checks whether this coded symbol is pure, i.e., whether it represents a single entity
    /// A pure coded symbol must satisfy the following conditions:
    /// - The 1-bit counter must be 1 or -1 (which are both represented by the least significant bit being set)
    /// - The checksum must match the absolute value of the checksum of the value.
    ///   The sign (-/+) tells you if it was an insertion or deletion
    /// - The indices of the value must match the index of this coded symbol.
    pub(crate) fn is_pure<S: HashFunctions<T>>(
        &self,
        state: &S,
        i: usize,
        len: usize,
    ) -> (bool, usize) {
        if self.checksum & 1 == 0 {
            return (false, 0);
        }
        let multiplicity = indices_contains(state, &self.value, len, i);
        if multiplicity != 1 {
            return (false, 0);
        }
        let checksum = state.check_sum(&self.value) | 1;
        if checksum == self.checksum || checksum.wrapping_neg() == self.checksum {
            (true, 0)
        } else {
            let required_bits = self
                .checksum
                .wrapping_sub(checksum)
                .leading_zeros()
                .max(self.checksum.wrapping_add(checksum).leading_zeros())
                as usize;
            (false, required_bits)
        }
    }

    /// Checks whether this coded symbol is zero, i.e., whether it represents no entity.
    pub(crate) fn is_zero(&self) -> bool {
        self.checksum == 0 && self.value == T::zero()
    }

    /// Checks whether this coded symbol represents a deletion.
    pub(crate) fn is_deletion<S: HashFunctions<T>>(&self, state: &S) -> bool {
        let checksum = state.check_sum(&self.value) | 1;
        checksum != self.checksum
    }
}

/// This function checks efficiently whether the given index is contained in the indices.
///
/// Note: we have constructed the indices such that we can determine from the last 5 bits
/// which hash function would map to this index. Therefore, we only need to check against
/// a single hash function and not all 5!
/// The only exception is for very small indices (0..32) or if the index is a multiple of 32.
///
/// The function returns the multiplicity, i.e. how many indices hit this particular index.
/// Thereby, it takes into account whether the value is stored negated or not.
fn indices_contains<T: std::hash::Hash>(
    state: &impl HashFunctions<T>,
    value: &T,
    stream_len: usize,
    i: usize,
) -> i32 {
    if stream_len > 32 && i % 32 != 0 {
        let seed = i % 4;
        let j = index_for_seed(state, value, stream_len, seed as u32);
        i32::from(i == j)
    } else {
        indices(state, value, stream_len)
            .map(|j| i32::from(i == j))
            .sum()
    }
}
