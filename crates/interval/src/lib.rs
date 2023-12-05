use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::ops::{Add, AddAssign, Range};

use fnv::FnvHasher;
use hex::FromHexError;
use serde::{Deserialize, Serialize};

use github_lock_free::{Format, SerializableAs};
use github_stable_hash::StableHash;

/// `FractionalPart` represents a number in the interval [0; 1).
/// I.e. we can always add another bit to the right which makes the number a tiny bit larger.
///
/// The value of the fraction is:
/// `slice[0] / 256 + slice[1] / 256^2 + slice[2] / 256^3 + ...`
/// I.e. the most significant bit of the first `u8` is the most significant bit of the fraction.
///
/// Invariant: The u8 slice representing the fractional number never ends with a 0 byte!
/// All functions manipulating the u8 slice call `drop_trailing_zeros` to satisfy this requirement.
#[derive(Debug, Clone, Hash, Eq, PartialEq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FractionalPart<'a>(Cow<'a, [u8]>);

#[derive(Debug, Clone)]
pub struct Interval<'a>(pub Range<FractionalPart<'a>>);

/// An interval token represents an interval in the range [0, 1), and is denoted by the value M that is the interval's midpoint.
/// The least significant bit determines the corresponding interval:
///   - If k is the least significant bit, the interval length is 2^-k,
///   - and thus its start point (inclusive) is at M-2^-k,
///   - and its end point (exclusive) is at M+2^-k.
/// By construction, each value (other than 0) can be interpreted as an interval token, and uniquely determines an interval.
///
/// A few interesting properties of this encoding:
///   - Same-length intervals tesselate the value range in a non-overlapping way.
///   - Each interval decomposes into two half-intervals of half its length.
///   - Each value belongs to an "infinite" number of intervals of halfening lengths.
///   - The interval [0, 1) is represented by the number that has only the most significant bit set to 1.
///   - The value 0 is not a valid interval token.
///
/// The following diagram shows which fractional number (in binary representation) ranges are covered
/// by which interval tokens:
/// Point/fractional number     : 0   0.001   0.010   0.011   0.100   0.101   0.110   0.111   1.000
/// 1-length interval token     :                              0.1                              |
/// 0.1-length interval token   :              0.01             |              0.11             |
/// 0.01-length interval token  :     0.001     |     0.011     |     0.101     |     0.111     |
/// 0.001-length interval token : 0.0001| 0.0011| 0.0101| 0.0111| 0.1001| 0.1011| 0.1101| 0.1111|
///
/// While IntervalTokens can only represent a small subset of general fractional intervals, every general fractional interval
/// can easily be decomposed into a set of disjoint IntervalTokens. The ideal decomposition chooses the minimum number
/// IntervalTokens, i.e. the largest possible ones. The size of this decomposition is bound by O(-log(length of interval)).
/// There is actually a second bound: O(least significant bit(start | end)) which is interesting when converting
/// `TreePaths` into Intervals.
///
/// Similarly, one can enumerate all the IntervalTokens covering a specific number or point. There are infinitely many such
/// tokens. So, one needs to specify the smallest tokens needed.
///
/// These two representations can be used to quickly compute `Interval` vs. `Point` intersections by simply testing whether
/// the two associated IntervalToken sets are non-disjoint. This test immediately generalizes to `Sets of Intervals` vs. `Sets of Points`
/// intersections! The advantage of this approach is that IntervalTokens can be converted into an arbitrary reverse index
/// and that way allow searches over intervals and points.
///
/// And finally, `Interval` vs. `Interval` intersections can be translated into `Interval` vs. `Point` intersections as follows.
/// There are three different cases:
///   1. The two intervals are non-overlapping:                    {  }  (  )
///   2. One interval is completely contained in the other one:    {  (--)  }
///   3. The two intervals are partially overlapping:              {  (--}  )
/// The two intervals only intersect in cases 2. and 3.
/// Now, one can easily verify that the following statement is true:
///   The two intervals intersect, when start or end-point of one interval lies within the other interval.
/// With this statement, the `Interval` vs. `Interval` intersection transforms into two `Interval` vs. `Point`
/// intersections from above! And just as above, the same approach works for `Set of Intervals` vs. `Set of Intervals`
/// intersections.
#[derive(Clone, Debug, Hash, Eq, PartialEq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct IntervalToken<'a>(FractionalPart<'a>);

pub struct IntervalTokenFormat {}

// Implement the necessary traits in forward and backward directions.
impl Format for IntervalTokenFormat {
    type View<'a> = IntervalToken<'a>;

    fn reborrow<'short, 'long: 'short>(view: IntervalToken<'long>) -> IntervalToken<'short> {
        view
    }
}

impl<'a> SerializableAs for IntervalToken<'a> {
    type Format = IntervalTokenFormat;

    fn to_bytes(&self) -> Vec<u8> {
        use github_pspack::Serializable;
        let mut buf = vec![];
        self.write(&mut buf).expect("failed to write serializable");
        buf
    }
}

impl<'a> StableHash for IntervalToken<'a> {
    fn stable_hash(&self) -> u64 {
        self.0.stable_hash()
    }
}

impl<'a> StableHash for FractionalPart<'a> {
    fn stable_hash(&self) -> u64 {
        let mut hasher = FnvHasher::default();
        self.0.hash(&mut hasher);
        hasher.finish()
    }
}

fn decode_hex(c: u8) -> Result<u8, FromHexError> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(FromHexError::InvalidHexCharacter {
            c: c as char,
            index: 0,
        }),
    }
}

impl<'a> FractionalPart<'a> {
    fn new() -> Self {
        Self(Default::default())
    }

    pub fn from_hex(input: &str) -> Result<Self, FromHexError> {
        // Do the hex conversion manually in order to support odd number of hex characters.
        let mut input = input.as_bytes();
        let mut output = Vec::with_capacity((input.len() + 1) / 2);
        loop {
            match input.len() {
                0 => break,
                1 => {
                    output.push(decode_hex(input[0])? * 16);
                    break;
                }
                _ => {
                    output.push(decode_hex(input[0])? * 16 + decode_hex(input[1])?);
                    (_, input) = input.split_at(2);
                }
            }
        }
        let mut result = Self(output.into());
        result.drop_trailing_zeros();
        Ok(result)
    }

    pub fn as_hex(&self) -> String {
        hex::encode(&self.0)
    }

    fn clone_owned(&self) -> FractionalPart<'static> {
        FractionalPart(Cow::from(self.as_slice().to_vec()))
    }

    /// If the data isn't owned yet by the FractionalPart, then it will be owned after the call.
    /// Additionally, it returns an instance with 'static lifetime which is appropriate for an
    /// owning instance.
    pub fn into_owned(self) -> FractionalPart<'static> {
        match self.0 {
            Cow::Borrowed(slice) => FractionalPart(Cow::from(slice.to_vec())),
            Cow::Owned(owned) => FractionalPart(Cow::from(owned)),
        }
    }

    // Returns the number of digits that are needed to represent fractional without
    // counting trailing zeros.
    pub fn num_bits(&self) -> usize {
        let slice = self.as_slice();
        if slice.is_empty() {
            0
        } else {
            let n = slice.len();
            8 * n - slice[n - 1].trailing_zeros() as usize
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    fn as_mut_vec(&mut self) -> &mut Vec<u8> {
        self.0.to_mut()
    }

    fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    /// Little helper function to provide all the pieces for modifying a single bit.
    fn access_bit(&mut self, bit: usize) -> (usize, usize, &mut Vec<u8>) {
        (bit / 8, 7 - (bit % 8), self.as_mut_vec())
    }

    /// Adds to self the number which has just the specified bit set.
    ///
    /// Returns false when the final number doesn't fit into the interval [0; 1) and thus cannot be
    /// represented anymore. In this case, the FractionalPart is reset to 0.
    fn increment_bit(&mut self, bit: usize) -> bool {
        let (mut idx, bit, vec) = self.access_bit(bit);
        if idx >= vec.len() {
            vec.resize(idx + 1, 0);
            vec[idx] += 1 << bit;
            return true;
        }
        let mut value = 1 << bit;
        loop {
            value += vec[idx] as usize;
            vec[idx] = value as u8;
            value >>= 8;
            if value == 0 {
                break;
            }
            if idx == 0 {
                self.drop_trailing_zeros();
                return false;
            }
            idx -= 1;
        }
        self.drop_trailing_zeros();
        true
    }

    fn toggle_bit(&mut self, bit: usize) {
        let (idx, bit, vec) = self.access_bit(bit);
        if idx >= vec.len() {
            vec.resize(idx + 1, 0);
            vec[idx] ^= 1 << bit;
            return;
        }
        vec[idx] ^= 1 << bit;
        self.drop_trailing_zeros();
    }

    fn clear_bit(&mut self, bit: usize) {
        let (idx, bit, vec) = self.access_bit(bit);
        if idx >= vec.len() {
            return;
        }
        vec[idx] &= !(1 << bit);
        self.drop_trailing_zeros();
    }

    fn set_bit(&mut self, bit: usize) {
        let (idx, bit, vec) = self.access_bit(bit);
        if idx >= vec.len() {
            vec.resize(idx + 1, 0);
        }
        vec[idx] |= 1 << bit;
    }

    fn drop_trailing_zeros(&mut self) {
        match &mut self.0 {
            Cow::Borrowed(slice) => {
                if let Some((pos, _)) = slice.iter().enumerate().rev().find(|(_, byte)| **byte != 0)
                {
                    *slice = &slice[0..=pos]
                } else {
                    *slice = &[]
                }
            }
            Cow::Owned(vec) => {
                if let Some((pos, _)) = vec.iter().enumerate().rev().find(|(_, byte)| **byte != 0) {
                    vec.truncate(pos + 1);
                } else {
                    vec.clear();
                }
            }
        }
    }

    /// Truncate to `num_bits` bits.
    ///
    /// ```
    /// # use github_interval::*;
    /// let mut r = FractionalPart::from_hex("f500ff").unwrap();
    /// r.shrink(20);
    /// assert_eq!(r.as_hex(), "f500f0");
    /// r.shrink(15);
    /// assert_eq!(r.as_hex(), "f5");
    /// r.shrink(1);
    /// assert_eq!(r.as_hex(), "80");
    /// r.shrink(0);
    /// assert_eq!(r.as_hex(), "");
    /// ```
    ///
    /// This has no effect if `num_bits >= self.num_bits()`.
    ///
    /// ```
    /// # use github_interval::*;
    /// let mut r = FractionalPart::from_hex("1234").unwrap();
    /// r.shrink(256);
    /// assert_eq!(r.as_hex(), "1234");
    /// r.shrink(14);
    /// assert_eq!(r.as_hex(), "1234");
    /// ```
    pub fn shrink(&mut self, num_bits: usize) {
        if num_bits == 0 {
            *self = (&[]).into();
        } else {
            let (idx, bit, vec) = self.access_bit(num_bits - 1);
            if idx < vec.len() {
                vec.truncate(idx + 1);
                vec[idx] &= 255 << bit;
                self.drop_trailing_zeros();
            }
        }
    }
}

impl<'a> Interval<'a> {
    /// Return the minimum `num_bits` of any FractionalPart in the range `start..=end`.
    ///  
    /// This is like finding the "simplest fraction" in the range; between 5/16 and 7/8 the unique
    /// simplest fraction with a power-of-two denominator is 1/2, and so this would return
    /// `FractionalPart(1/2).num_bits()`, which is 1.
    ///
    /// # Panics
    /// If `end < start`, because then the range `start..=end` is empty.
    fn meet_point_num_bits(&self) -> usize {
        let start_num_bits = self.0.start.num_bits();
        let start = self.0.start.as_slice();
        let end = self.0.end.as_slice();
        for i in 0..start.len() {
            // panics if we walk off the end of stop, but that only happens if stop < start
            let difference = start[i] ^ end[i];
            if difference != 0 {
                assert!(start[i] < end[i], "backwards interval");
                return start_num_bits.min(i * 8 + difference.leading_zeros() as usize + 1);
            }
        }
        start_num_bits
    }

    /// Computes the minimal token set that represent this `Interval`.
    /// When indexing these tokens, one can retrieve all documents that contain `Points` by searching
    /// for their corresponding tokens.
    /// Vice versa, when indexing the point tokens, one can find all documents with `Points` in a given
    /// interval by searching for the corresponding tokens of the interval.
    ///
    /// # Panics
    /// If `end < start`.
    pub fn tokens(&self) -> Vec<IntervalToken<'static>> {
        let mut result = vec![];
        let mut start = self.0.start.clone_owned();
        let mut end = self.0.end.clone_owned();
        let target = self.meet_point_num_bits();
        loop {
            let lsb = start.num_bits();
            if lsb <= target {
                break;
            }
            result.push(IntervalToken::new(start.clone(), lsb));
            assert!(start.increment_bit(lsb - 1));
        }
        debug_assert_eq!(start.num_bits(), target);
        debug_assert!(self.0.start.as_slice() <= start.as_slice());
        loop {
            let lsb = end.num_bits();
            if lsb <= target {
                break;
            }
            end.toggle_bit(lsb - 1);
            result.push(IntervalToken::new(end.clone(), lsb));
        }
        debug_assert_eq!(start.as_slice(), end.as_slice());
        result
    }

    /// Computes the minimal token set that represents this `Interval` truncated to `num_bits`.
    /// If the original interval cannot be properly represented with `num_bits`, this function
    /// will over-approximate the interval accordingly.
    pub fn truncated_tokens(&self, num_bits: usize) -> Vec<IntervalToken<'static>> {
        debug_assert!(num_bits > 0);
        let mut start = self.0.start.clone_owned();
        let mut end = self.0.end.clone_owned();
        start.shrink(num_bits);
        let bits = end.num_bits();
        if bits >= num_bits {
            end.shrink(num_bits);
            if !end.increment_bit(num_bits - 1) {
                let mut result = vec![];
                loop {
                    let lsb = start.num_bits();
                    if lsb == 0 {
                        if result.is_empty() {
                            result.push(IntervalToken::new(start, 0));
                        }
                        return result;
                    }
                    result.push(IntervalToken::new(start.clone(), lsb));
                    start.increment_bit(lsb - 1);
                }
            }
        }
        Self(start..end).tokens()
    }
}

impl Add for FractionalPart<'_> {
    type Output = FractionalPart<'static>;

    /// Note: This function will panic if the final number doesn't fit into the interval [0; 1) and
    /// thus cannot be represented anymore.
    fn add(self, other: Self) -> FractionalPart<'static> {
        let self_slice = self.as_slice();
        let other_slice = other.as_slice();
        let (smaller_slice, larger_slice) = if self_slice.len() < other_slice.len() {
            (self_slice, other_slice)
        } else {
            (other_slice, self_slice)
        };
        let mut result = larger_slice.to_vec();
        let mut overflow = 0;
        for i in (0..smaller_slice.len()).rev() {
            overflow += self_slice[i] as usize + other_slice[i] as usize;
            result[i] = overflow as u8;
            overflow >>= 8;
        }
        assert_eq!(overflow, 0);
        let mut result = FractionalPart(result.into());
        result.drop_trailing_zeros();
        result
    }
}

impl AddAssign for FractionalPart<'_> {
    /// Note: This function will panic if the final number doesn't fit into the interval [0; 1) and
    /// thus cannot be represented anymore.
    fn add_assign(&mut self, other: Self) {
        let self_slice = self.as_mut_vec();
        let other_slice = other.as_slice();
        if self_slice.len() < other_slice.len() {
            self_slice.resize(other_slice.len(), 0);
        }
        let mut overflow = 0;
        for i in (0..other_slice.len()).rev() {
            overflow += self_slice[i] as usize + other_slice[i] as usize;
            self_slice[i] = overflow as u8;
            overflow >>= 8;
        }
        assert_eq!(overflow, 0);
        self.drop_trailing_zeros();
    }
}

impl<'a> From<Cow<'a, [u8]>> for FractionalPart<'a> {
    fn from(cow: Cow<'a, [u8]>) -> Self {
        let mut result = Self(cow);
        result.drop_trailing_zeros();
        result
    }
}

impl<'a> From<&'a [u8]> for FractionalPart<'a> {
    fn from(slice: &'a [u8]) -> Self {
        let mut result = Self(Cow::from(slice));
        result.drop_trailing_zeros();
        result
    }
}

impl<'a, const N: usize> From<&'a [u8; N]> for FractionalPart<'a> {
    fn from(slice: &'a [u8; N]) -> Self {
        let slice: &[u8] = slice;
        Self::from(slice)
    }
}

impl<const N: usize> From<[u8; N]> for FractionalPart<'static> {
    fn from(slice: [u8; N]) -> Self {
        let mut result = Self(Cow::from(Vec::from(slice)));
        result.drop_trailing_zeros();
        result
    }
}

#[cfg(test)]
impl From<u64> for FractionalPart<'static> {
    fn from(value: u64) -> Self {
        let mut result = Self(Cow::from(value.to_be_bytes().to_vec()));
        result.drop_trailing_zeros();
        result
    }
}

impl<'a> IntervalToken<'a> {
    pub fn new(mut start: FractionalPart<'a>, bit: usize) -> Self {
        assert!(start.num_bits() <= bit);
        start.toggle_bit(bit);
        Self(start)
    }

    /// Returns a token covering all numbers that have the same first `bit` bits as `start`.
    ///
    /// ```
    /// # use github_interval::*;
    /// let frac = |s| FractionalPart::from_hex(s).unwrap();
    ///
    /// let t1 = IntervalToken::new_and_shrink(frac("555fff"), 12);
    /// assert!(!t1.contains(&frac("554fff")));
    /// assert!(t1.contains(&frac("5550")));
    /// assert!(t1.contains(&frac("555fffff")));
    /// assert!(!t1.contains(&frac("5560")));
    ///
    /// // works if `start` has less than the specified precision
    /// let t2 = IntervalToken::new_and_shrink(frac("3c"), 63);
    /// assert_eq!(t2.start(), frac("3c"));
    /// assert_eq!(t2.end(), frac("3c00000000000002"));
    /// assert!(t2.contains(&frac("3c")));
    /// assert!(t2.contains(&frac("3c00000000000001")));
    /// assert!(!t2.contains(&frac("3c00000000000002")));
    ///
    /// // works for size 0
    /// let t3 = IntervalToken::new_and_shrink(frac("31"), 0);
    /// assert_eq!(t3.start(), frac("00"));
    /// assert!(t3.contains(&frac("ffffffff")));
    /// ```
    pub fn new_and_shrink(mut start: FractionalPart<'a>, bit: usize) -> Self {
        start.shrink(bit);
        start.toggle_bit(bit);
        Self(start)
    }

    /// Returns a token with limited number of bits. If the token has to be truncated
    /// then it will be overapproximated to fit into the specified number of bits.
    pub fn shrink(&self, num_bits: usize) -> Self {
        if self.num_bits() <= num_bits {
            self.clone()
        } else {
            let mut result = self.clone_owned();
            result.0.shrink(num_bits);
            result.0.toggle_bit(num_bits);
            result
        }
    }

    pub fn clone_owned(&self) -> IntervalToken<'static> {
        IntervalToken(self.0.clone_owned())
    }

    pub fn into_owned(self) -> IntervalToken<'static> {
        IntervalToken(self.0.into_owned())
    }

    pub fn as_hex(&self) -> String {
        self.0.as_hex()
    }

    /// True if `other` falls into this token's range.
    pub fn contains(&self, other: &FractionalPart) -> bool {
        let other_slice = other.as_slice();
        let self_slice = self.0.as_slice();
        let num_bits = self.0.num_bits() - 1;
        let num_full_bytes = num_bits / 8;
        let required_common_bytes = num_full_bytes.min(other_slice.len());
        if self_slice[0..required_common_bytes] != other_slice[0..required_common_bytes] {
            return false;
        }
        if num_full_bytes > other_slice.len() {
            // Other contains now only zero bytes. So, we must test that self also consists only of zero bytes.
            if !self_slice[required_common_bytes..num_full_bytes]
                .iter()
                .all(|b| *b == 0)
            {
                return false;
            }
        }
        // The last byte has to be treated specially, since only some bits matter here.
        if num_bits % 8 == 0 {
            return true;
        }
        let other_last_byte = if num_full_bytes >= other_slice.len() {
            0
        } else {
            other_slice[num_full_bytes]
        };
        let self_last_byte = self_slice[num_full_bytes];
        let diff = (self_last_byte ^ other_last_byte) & (255 << (8 - (num_bits % 8)));
        diff == 0
    }

    pub fn start(&self) -> FractionalPart<'static> {
        assert!(!self.0.is_empty());
        let bit = self.0.num_bits() - 1;
        let mut result = self.0.clone_owned();
        result.toggle_bit(bit);
        result
    }

    /// Note: this function will panic when called with the IntervalToken [0, 1),
    /// since 1 cannot be representated as FractionalPart.
    pub fn end(&self) -> FractionalPart<'static> {
        assert!(!self.0.is_empty());
        let bit = self.0.num_bits() - 1;
        let mut result = self.0.clone_owned();
        assert!(result.increment_bit(bit));
        result
    }

    pub fn mid_point(&self) -> &FractionalPart<'a> {
        &self.0
    }

    pub fn half_length(&self) -> FractionalPart<'static> {
        assert!(!self.0.is_empty());
        let bit = self.0.num_bits() - 1;
        let mut result = FractionalPart::new();
        result.toggle_bit(bit);
        result
    }

    /// Returns at which position the least significant one bit is.
    pub fn num_bits(&self) -> usize {
        self.0.num_bits() - 1
    }

    /// Returns true if the IntervalToken represents the interval [0, 1).
    pub fn is_all(&self) -> bool {
        self.0.num_bits() == 1
    }

    /// Iterates through all enclosing IntervalTokens including self.
    pub fn enclosing(self) -> PointIntervalTokenIterator<'a> {
        let lsb = self.0.num_bits();
        PointIntervalTokenIterator { point: self, lsb }
    }
}

// See file level comment and `Interval::tokens` function for what this can be used for.
// Computes the set of IntervalTokens that represents this `point`.
pub struct PointIntervalTokenIterator<'a> {
    point: IntervalToken<'a>,
    lsb: usize,
}

impl<'a> Iterator for PointIntervalTokenIterator<'a> {
    type Item = IntervalToken<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.lsb == 0 {
            None
        } else {
            self.point.0.set_bit(self.lsb - 1);
            self.point.0.clear_bit(self.lsb);
            self.lsb -= 1;
            Some(self.point.clone())
        }
    }
}

impl<'a> github_pspack::Serializable<'a> for FractionalPart<'a> {
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        self.as_slice().write(writer)
    }

    fn from_bytes(buf: &'a [u8]) -> Self {
        buf.into()
    }
}

impl<'a> github_pspack::Serializable<'a> for IntervalToken<'a> {
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        self.0.write(writer)
    }

    fn from_bytes(buf: &'a [u8]) -> Self {
        Self(buf.into())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use itertools::Itertools;

    use super::*;

    // The number of IntervalTokens that have to be generated depends on the number of bits
    // required to represent the corresponding intervals with which this set should be intersected.
    // Since the `point` cannot know this resolution, it must be passed in as argument.
    fn point_tokens(point: FractionalPart, bits: usize) -> Vec<IntervalToken<'static>> {
        let token = IntervalToken::new(point.clone_owned(), bits.max(point.num_bits()));
        token.enclosing().collect()
    }

    #[test]
    fn test_fractional_toggle_bit() {
        let mut a = FractionalPart::new();
        a.toggle_bit(62);
        let b = 2u64.into();
        assert_eq!(a, b);
        assert_eq!(8, a.as_slice().len());
        a.toggle_bit(62);
        assert_eq!(a, FractionalPart::new());
        assert_eq!(0, a.as_slice().len());

        let mut a: FractionalPart = 12345u64.into();
        a.toggle_bit(63);
        let b: FractionalPart = 12344u64.into();
        assert_eq!(a, b);
    }

    #[test]
    fn test_fractional_increment_bit() {
        let mut a = FractionalPart::new();
        for i in 1..=16 {
            a.increment_bit(57);
            let b = ((64 * i) as u64).into();
            assert_eq!(a, b);
        }
        assert_eq!(a.as_slice().len(), 7);

        let mut a: FractionalPart = 0x7fffffff00000000.into();
        a.increment_bit(0);
        assert_eq!(a, FractionalPart::from(0xffffffff00000000u64));
        a.increment_bit(32);
        assert_eq!(a, FractionalPart::from(0xffffffff80000000u64));
    }

    #[test]
    fn test_fractional_increment_overflow() {
        let mut a: FractionalPart = 0xffffffff00000000.into();
        assert!(!a.increment_bit(31));
    }

    #[test]
    fn test_fractional_add() {
        let a = 23479823u64 << 10;
        let b = 2347312432348u64;
        let c = a + b;

        let aa: FractionalPart = a.into();
        let bb: FractionalPart = b.into();
        let cc = aa + bb;

        assert_eq!(cc, c.into());
    }

    #[test]
    #[should_panic]
    fn test_fractional_add_overflow() {
        let a: FractionalPart = 0xffffffff00000000u64.into();
        let b: FractionalPart = 0x0000000100000000u64.into();
        println!("{:?}", a + b);
    }

    #[test]
    #[should_panic]
    fn test_fractional_add_assign_overflow() {
        let mut a: FractionalPart = 0xffffffff00000000u64.into();
        let b: FractionalPart = 0x0000000100000000u64.into();
        a += b;
    }

    #[test]
    fn test_fractional_conversions() {
        let a: FractionalPart = 123789u64.into();
        let slice = a.as_slice();
        let b: FractionalPart = slice.into();
        assert_eq!(a, b);
    }

    #[test]
    fn test_interval_tokens() {
        assert_eq!(
            Interval([0; 4].into()..[0, 0, 0, 1].into()).tokens(),
            [IntervalToken([0, 0, 0, 0, 128].into())]
        );
        assert_eq!(Interval([65, 48].into()..[65, 48].into()).tokens(), []);
        assert_eq!(Interval([].into()..[].into()).tokens(), []);
        assert_eq!(
            Interval([].into()..[2].into()).tokens(),
            [IntervalToken([1].into())]
        );
        assert_eq!(
            Interval([65, 48].into()..[65, 49, 40].into()).tokens(),
            [
                IntervalToken([65, 49, 36].into()),
                IntervalToken([65, 49, 16].into()),
                IntervalToken([65, 48, 128].into()),
            ]
        );
        assert_eq!(
            Interval(20.into()..100.into()).tokens(),
            [
                IntervalToken(22u64.into()),
                IntervalToken(28u64.into()),
                IntervalToken(48u64.into()),
                IntervalToken(98u64.into()),
                IntervalToken(80u64.into())
            ]
        );
        assert_eq!(
            Interval(20.into()..100.into())
                .tokens()
                .into_iter()
                .fold(FractionalPart::new(), |a, b| a + b.half_length()),
            40.into()
        );
        assert_eq!(
            Interval(23472346.into()..448367934613942.into())
                .tokens()
                .into_iter()
                .fold(FractionalPart::new(), |a, b| a + b.half_length()),
            ((448367934613942 - 23472346) / 2).into()
        );
    }

    #[test]
    #[should_panic]
    fn test_invalid_interval_tokens() {
        assert_eq!(Interval([65, 48, 1].into()..[65, 48].into()).tokens(), []);
    }

    #[test]
    fn test_point_tokens() {
        let a: FractionalPart = 20.into();
        point_tokens(a.clone(), 63)
            .iter()
            .enumerate()
            .for_each(|(i, t)| {
                assert!(t.start() <= a);
                assert!(t.is_all() || a < t.end());
                assert_eq!(t.half_length(), (1u64 << i).into());
            });
        assert_eq!(
            point_tokens(a, 0).first(),
            Some(&IntervalToken(22u64.into()))
        );
    }

    #[test]
    fn test_implicitly_dropping_trailing_zeros() {
        let buf = &[1, 2, 3, 0];
        let a: FractionalPart = buf.into();
        // The conversion should drop the trailing zero byte!
        assert_eq!(a.as_slice(), &buf[0..3]);
        // But, it shouldn't copy the data.
        assert_eq!(a.as_slice().as_ptr(), buf.as_ptr());
    }

    #[test]
    fn test_interval_point_queries() {
        // This test constructs a random set of non-overlapping intervals and another random set of points.
        // It then, determines "brute-force" which points are inside or outside of these intervals.
        // At the same it, the tests constructs an interval index for all the intervals represented by the hashset `interval_hashes`.
        // Finally, it tests that searches for the points gives the correct results.
        // It also tests the start and end points of intervals explicitly for inclusion/exclusion.
        let mut interval_hashes = HashSet::new();
        let mut interval_limits: Vec<_> = (0..10000).map(|_| rand::random::<u64>() & !1).collect();
        interval_limits.sort_unstable();
        let mut points: Vec<_> = (0..10000).map(|_| rand::random::<u64>() & !1).collect();
        points.sort_unstable();
        let mut inside_points = vec![];
        let mut outside_points = vec![];
        let mut pts = points.iter().peekable();
        for (start, end) in interval_limits.iter().tuples() {
            let a = Interval((*start).min(*end).into()..(*end).max(*start).into());
            interval_hashes.extend(a.tokens());
            while let Some(pt) = pts.peek() {
                if **pt < *start {
                    outside_points.push(**pt);
                } else if **pt < *end {
                    inside_points.push(**pt);
                } else {
                    break;
                }
                pts.next();
            }
        }
        for pt in pts {
            outside_points.push(*pt);
        }
        // Test that start/end limits are part/not part of the point/interval intersection!
        for (start, end) in interval_limits.iter().tuples() {
            let start = (*start).into();
            let end = (*end).into();
            assert!(point_tokens(start, 64)
                .iter()
                .any(|t| interval_hashes.contains(t)));
            assert!(!point_tokens(end, 64)
                .iter()
                .any(|t| interval_hashes.contains(t)));
        }
        for pt in outside_points {
            assert!(!point_tokens(pt.into(), 64)
                .iter()
                .any(|t| interval_hashes.contains(t)));
        }
        for pt in inside_points {
            assert!(point_tokens(pt.into(), 64)
                .iter()
                .any(|t| interval_hashes.contains(t)));
        }
    }

    #[test]
    fn test_interval_token_contains() {
        let pt: FractionalPart = 98547013234871000u64.into();
        for i in 0..64 {
            let token = IntervalToken::new_and_shrink(pt.clone(), i);
            assert!(token.contains(&pt), "failed for i={i:}");
            let mut pt2 = pt.clone();
            pt2.toggle_bit(i);
            assert!(token.contains(&pt2), "failed for i={i:}");
            for j in 0..i {
                let mut pt2 = pt.clone();
                pt2.toggle_bit(j);
                assert!(!token.contains(&pt2), "failed for i={i:}");
            }
        }
    }

    #[test]
    fn test_truncated_tokens_overflow() {
        let start = FractionalPart::from_hex("f0").unwrap();
        let end = FractionalPart::from_hex("ff8").unwrap();
        assert_eq!(
            Interval(start..end.clone()).truncated_tokens(8),
            vec![IntervalToken::new(
                FractionalPart::from_hex("f").unwrap(),
                4
            )]
        );

        let start = FractionalPart::from_hex("00").unwrap();
        assert_eq!(
            Interval(start..end.clone()).truncated_tokens(8),
            vec![IntervalToken::new(FractionalPart::from_hex("").unwrap(), 0)]
        );

        let start = FractionalPart::from_hex("1").unwrap();
        assert_eq!(
            Interval(start..end.clone()).truncated_tokens(8),
            vec![
                IntervalToken::new(FractionalPart::from_hex("1").unwrap(), 4),
                IntervalToken::new(FractionalPart::from_hex("2").unwrap(), 3),
                IntervalToken::new(FractionalPart::from_hex("4").unwrap(), 2),
                IntervalToken::new(FractionalPart::from_hex("8").unwrap(), 1),
            ]
        );

        let start = FractionalPart::from_hex("ff1").unwrap();
        assert_eq!(
            Interval(start..end).truncated_tokens(8),
            vec![IntervalToken::new(
                FractionalPart::from_hex("ff").unwrap(),
                8
            )]
        );
    }

    #[test]
    fn test_truncated_tokens() {
        let start = FractionalPart::from_hex("fc1").unwrap();
        let end = FractionalPart::from_hex("fc8").unwrap();
        assert_eq!(
            Interval(start..end).truncated_tokens(8),
            vec![IntervalToken::new(
                FractionalPart::from_hex("fc").unwrap(),
                8
            )]
        );
    }
}
