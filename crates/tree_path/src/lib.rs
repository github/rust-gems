//! [`TreePath`] represents nodes in a tree with the following properties:
//!
//! *   Nodes are labelled with unique [u8] slices, and a preorder traversal of
//!     the tree is guaranteed to observe those labels in ascending order.
//!
//! *   The trees we care about are likely to either have pretty high branching
//!     (in the thousands) or very long runs with branching factor 1 (potentially
//!     in the thousands).
//!
//! *   We would like to be able to add nodes to the tree without relabelling
//!     existing nodes or breaking the ascending-labels-in-preorder-traversal
//!     invariant. We assume the existence of a stateful, collaborating layer
//!     (external to this file) which helps pick u32 identifiers for the
//!     children of each node.
//!
//! *   In particular, small numbers are more efficient to encode, so
//!     intuitively the goal is to number children from 1 as they are added.
//!
//! *   Concretely, when adding a node to the tree, the stateful, collaborating
//!     layer picks the special child ID oo if the node is the first known child
//!     of its parent, and the smallest unoccupied child ID otherwise. The
//!     special oo ("infinity") node is used to encode branching-factor-1
//!     chains; it stores a run-length encoding of how long this chain is
//!     (basically, how often you descend to the first child of a node).
//!
//! *   A chain of oo nodes must be encoded as one run; otherwise one gets a
//!     different, invalid representation for the node. The encoding for child
//!     nodes is chosen such that:
//!
//! *   Runs of oo nodes are lexicographically/numerically smaller than non-oo
//!     children.
//!
//! *   Shorter runs of oo nodes are lexicographically/numerically smaller than
//!     longer runs.
//!
//! *   The representation of a child index is lexicographically/numerically
//!     smaller than the representation of a larger child index. A node is
//!     represented by the path from the root to it, encoding each child node as
//!     above and appending the bits of the encoding.
//!
//! *   This guarantees that a node lexicographically/numerically precedes its
//!     children as well as higher-ID siblings, and it follows its parent and its
//!     lower-ID siblings and their descendants.
//!
//! The encoding itself is inspired by [Order Preserving Key Compression][opkc]
//! which was mentioned 1994 by Antoshenkov et. al. to improve data base search
//! indices.
//!
//! There is a pretty neat trick to make run length encodings order preserving
//! in the [general case][opc2]. The solution used here is much simpler by
//! supporting run length encodings only for the special oo node. The general
//! solution wouldn't provide any benefit, since we have full freedom in
//! selecting child ids and thus pick the one that is cheapest to encode.
//!
//! We use the following encoding table for child runs/child indices:
//!
//! ```text
//! 0000 => termination sequence. I.e. 0 is NOT a valid node ID!
//! 00xx => children ids 1..3
//! 010x xxxx => children ids 4..35
//! 0110 xxxx xxxx xxxx => children ids 36..4131
//! 0111 0xxx xxxx xxxx xxxx xxxx xxxx xxxx => children ids 4132..134221857
//! 0111 10xx ...
//! 10xx => children id oo repeated (xx+1) times.
//! 110x xxxx => children id oo repeated (xx+5) times.
//! 1110 xxxx xxxx xxxx => children id oo repeated (xx+37) times.
//! 1111 0xxx xxxx xxxx xxxx xxxx xxxx xxxx => children id oo repeated (xx+4133) times.
//! ```
//!
//! Note: we need to collect statistics about number of children to make a
//! better decision about the encoding table, i.e. how many id bits to assign to
//! each prefix code.
//!
//! [opkc]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.86.3306&rep=rep1&type=pdf
//! [opc2]: https://github.com/google-ar/WebARonARCore/blob/2441c86a5fd975f09a6c30cddb57dfb7fc239699/components/sync/base/unique_position.cc#L381

use std::borrow::Cow;
use std::fmt::{self, Debug, Display, Formatter};
use std::hash::Hash;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use github_interval::{FractionalPart, IntervalToken};
use github_lock_free::{Format, SerializableAs};
use github_stable_hash::StableHash;

mod lock_free;

pub type ChildId = u32;

/// Largest node id that gets run-length encoded. The top most bit indicates
/// this special node id. The other bits simply represent the count how often it
/// occurs in a row.
/// Note: At the very first level of the tree, infinity nodes are not supported.
/// Instead they can be treated as normal ids.
pub const OO: ChildId = 0x80000000;

/// A path to a node in the tree.
/// Note: we implement our own Cow such that we don't have to decode num_bits while we don't need it.
#[derive(Clone)]
pub enum TreePath<'a> {
    Ref(&'a [u8]),
    Own(Vec<u8>, usize),
}

// Introduce a dummy type that can construct all the actual instances with proper lifetimes.
pub struct TreePathFormat {}

// Implement the necessary traits in forward and backward directions.
impl Format for TreePathFormat {
    type View<'a> = TreePath<'a>;

    fn reborrow<'short, 'long: 'short>(view: TreePath<'long>) -> TreePath<'short> {
        view
    }
}

impl<'a> SerializableAs for TreePath<'a> {
    type Format = TreePathFormat;

    fn to_bytes(&self) -> Vec<u8> {
        use github_pspack::Serializable;
        let mut buf = vec![];
        self.write(&mut buf).expect("failed to write serializable");
        buf
    }
}

impl<'a> PartialEq for TreePath<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl<'a> Eq for TreePath<'a> {}

impl<'a> PartialOrd for TreePath<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.as_bytes().cmp(other.as_bytes()))
    }
}

impl<'a> Ord for TreePath<'a> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_bytes().cmp(other.as_bytes())
    }
}

impl<'a> StableHash for TreePath<'a> {
    fn stable_hash(&self) -> u64 {
        self.as_bytes().stable_hash()
    }
}

impl<'a> Hash for TreePath<'a> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_bytes().hash(state);
    }
}

impl<'a> Serialize for TreePath<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_bytes(self.as_bytes())
    }
}

impl<'a> Deserialize<'a> for TreePath<'static> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'a>,
    {
        use serde::de::{Error, Visitor};

        struct ByteBufVisitor;

        impl<'de> Visitor<'de> for ByteBufVisitor {
            type Value = Vec<u8>;

            fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
                formatter.write_str("byte array")
            }

            #[inline]
            fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(Self::Value::from(v))
            }

            #[inline]
            fn visit_byte_buf<E>(self, v: Vec<u8>) -> Result<Self::Value, E>
            where
                E: Error,
            {
                Ok(v)
            }
        }

        deserializer
            .deserialize_byte_buf(ByteBufVisitor)
            .map(|bytes| {
                let num_bits = check_path(&bytes);
                Self::Own(bytes, num_bits)
            })
    }
}

impl<'a> Default for TreePath<'a> {
    fn default() -> Self {
        Self::Ref(&[])
    }
}

impl<'a> Display for TreePath<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let path = self
            .iter()
            .map(|child_id| {
                if child_id < OO {
                    format!("{child_id}")
                } else {
                    // when child_id is OO, output "/9/∞1" rather than "/9/∞", for consistency
                    format!("∞{}", child_id - (OO - 1))
                }
            })
            .collect::<Vec<String>>()
            .join("/");
        write!(f, "/{path}")
    }
}

impl<'a> Debug for TreePath<'a> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let bits = hex::encode(self.as_bytes());
        let path = self
            .iter()
            .map(|child_id| {
                if child_id < OO {
                    format!("{child_id}")
                } else {
                    // when child_id is OO, output "/9/∞1" rather than "/9/∞", for consistency
                    format!("∞{}", child_id - (OO - 1))
                }
            })
            .collect::<Vec<String>>()
            .join("/");
        write!(f, "{bits} (/{path})")
    }
}

#[derive(Debug, Error)]
pub enum TreePathParseError {
    #[error("failed to parse TreePath {0:?}: invalid path segment {1:?}")]
    BadSegment(String, String),

    #[error("failed to parse TreePath {0:?}: expected path starting with '/' or hex digits; {1}")]
    BadHex(String, hex::FromHexError),

    #[error("invalid TreePath {0:?}: extra bytes at end of path")]
    ExtraBytes(String),

    #[error("invalid TreePath {0:?}: bits past the end must be zero")]
    NonzeroTailBits(String),

    #[error("invalid TreePath {0:?}: number too large ({1})")]
    SegmentRange(String, String),

    #[error("invalid TreePath {0:?}: child ids cannot be zero")]
    SegmentZero(String),
}

impl<'a> FromStr for TreePath<'a> {
    type Err = TreePathParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "/" {
            Ok(TreePath::root())
        } else if let Some(segments) = s.strip_prefix('/') {
            let mut path = TreePath::root();
            for segment in segments.split('/') {
                let start = if segment.starts_with("oo") {
                    "oo".len()
                } else if segment.starts_with('∞') {
                    "∞".len()
                } else {
                    0
                };

                let n_str = &segment[start..];
                let n = if n_str.is_empty() && start > 0 {
                    // Treat `/oo` as short for `/oo1`.
                    1
                } else {
                    segment[start..].parse::<ChildId>().map_err(|_err| {
                        TreePathParseError::BadSegment(s.to_string(), segment.to_string())
                    })?
                };

                if n == 0 {
                    return Err(TreePathParseError::SegmentZero(s.to_string()));
                }
                if n >= OO {
                    return Err(TreePathParseError::SegmentRange(
                        s.to_string(),
                        segment.to_string(),
                    ));
                }
                path.push(if start == 0 { n } else { OO - 1 + n });
            }
            Ok(path)
        } else {
            let bytes =
                hex::decode(s).map_err(|err| TreePathParseError::BadHex(s.to_string(), err))?;
            let num_bits = try_check_path(&bytes)?;
            Ok(TreePath::Own(bytes, num_bits))
        }
    }
}

impl<'a> TreePath<'a> {
    pub fn root() -> Self {
        TreePath::default()
    }

    /// Construct a `TreePath` from a u8 slice.
    pub fn from_bytes(bits: &'a [u8]) -> Self {
        Self::Ref(bits)
    }

    /// Return a `TreePath` as its raw byte slice representation.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            TreePath::Ref(bits) => bits,
            TreePath::Own(bits, _) => bits.as_slice(),
        }
    }

    pub fn as_fractional_part(&self) -> FractionalPart {
        FractionalPart::from(self.as_bytes())
    }

    pub fn into_cow(self) -> Cow<'a, [u8]> {
        match self {
            TreePath::Ref(bits) => Cow::from(bits),
            TreePath::Own(bits, _) => Cow::from(bits),
        }
    }

    /// Converts a TreePath into its smallest point IntervalToken.
    /// Note: In order to create the smallest IntervalToken, we need to know its
    /// interval length. This length is determined by the next possible TreePath
    /// encoding which we get by appending the bit sequence 0001 to the current
    /// encoding. This means that the smallest interval size is simply the number
    /// of bits needed to represent the current TreePath plus 4.
    pub fn into_smallest_point_token(self) -> IntervalToken<'static> {
        let bits = self.num_bits();
        assert_ne!(bits, 0);
        IntervalToken::new(FractionalPart::from(self.into_cow()).into_owned(), bits + 4)
    }

    /// Return a copy of this `TreePath` that owns its own copy of the underlying bytes.
    /// This can be useful when Rust complains that a `TreePath` doesn't live long enough.
    pub fn clone_owned(&self) -> TreePath<'static> {
        TreePath::Own(self.as_bytes().to_vec(), self.num_bits())
    }

    /// Converts the TreePath into an owned instance and checks that this is a valid
    /// tree-path.
    pub fn into_owned(self) -> TreePath<'static> {
        match self {
            TreePath::Ref(bits) => TreePath::Own(bits.to_vec(), self.num_bits()),
            TreePath::Own(bits, num_bits) => TreePath::Own(bits, num_bits),
        }
    }

    /// Is the `TreePath` empty?
    pub fn is_empty(&self) -> bool {
        match self {
            TreePath::Ref(bits) => bits.is_empty(),
            TreePath::Own(_, num_bits) => *num_bits == 0,
        }
    }

    /// Constructs a tree-path corresponding to the provided node ids.
    /// Consecutive oo nodes are collapsed into one id to satisfy the above
    /// mentioned invariance. When possible, this collapsing should be done by
    /// the caller.
    ///
    /// This function should mostly be relevant in tests, since the normal flow
    /// would append a child to some existing node by calling `push`. If no node
    /// exists yet, one would start with the empty path.
    pub fn from_ids(ids: &[ChildId]) -> Self {
        let mut path = Self::default();
        let mut oo_count = 0;

        for id in ids.iter() {
            if *id >= OO && !path.is_empty() {
                oo_count += id - OO + 1;
            } else {
                if oo_count > 0 {
                    path.push(oo_count - 1 + OO);
                    oo_count = 0;
                }
                path.push(*id);
            }
        }
        if oo_count > 0 {
            path.push(oo_count - 1 + OO);
        }
        path
    }

    /// Construct a `TreePathBuf` from a vector of `bits` and the `num_bits` (not the same as the length of the vector).
    ///
    /// # Safety
    /// This is only safe if the the bits and num_bits have be validated with `check_path`.
    pub(crate) unsafe fn from_raw_unchecked(bits: Vec<u8>, num_bits: usize) -> Self {
        Self::Own(bits, num_bits)
    }

    /// Constructs a tree-path that's the upper bound of the given path.
    ///
    /// Given a path representation in our order preserving encoding, we need to
    /// be able to perform ancestor and successor searches efficiently. These
    /// can be translated into interval-point searches. E.g. all successors of a
    /// path A are contained in a specific interval which starts with the
    /// encoded representation of A and ends at upper_bound(A).
    ///
    /// For a path a -> b -> c -> d, this upper_bound is the path a -> b -> c ->
    /// d+1. In case of oo nodes, this would be wrong though. The correct
    /// upper_bound of a -> b -> c -> oo becomes a -> b -> c+1!
    ///
    /// Note: the interval corresponding to the empty path would be 0xFFFFFFFF
    /// which isn't a valid encoded TreePath. Therefore, this function requires
    /// an non-empty path as input.
    ///
    /// Note: This function can return an "invalid" tree path, since the path
    /// representation of the upper bound may exceed the number of bits
    /// available. If that happens, all lost bits are zeros. Therefore, this is
    /// still a "valid" upper bound.
    pub fn upper_bound(&self) -> Self {
        assert!(!self.is_empty());
        // NB: upper_bound lives in TreePathBuf as clone() is cheaper than
        // decoding the entire tree to go from a TreePath -> TreePathBuf.
        let mut upper = self.clone();
        let child = upper.parent();
        if child < OO || upper.is_empty() {
            // At the first level, we don't support OO nodes, i.e. their IDs
            // can be used as all other ids!
            upper.push(child + 1);
            upper
        } else {
            // Note: this should NEVER recurse more than once! Otherwise, we
            // have multiple oo nodes in a row which should have been combined
            // into one run.
            upper.upper_bound()
        }
    }

    pub fn is_ancestor_or_eq(&self, ancestor: &Self) -> bool {
        self == ancestor || ancestor.child_id(self).is_some()
    }

    // Truncate other to be a direct child of self and return its id or None if
    // it's not a descendant.
    pub fn child_id(&self, other: &Self) -> Option<ChildId> {
        let mut self_iter = self.iter();
        let mut other_iter = other.iter();

        // Walk the ids in both tree paths
        while let Some(self_id) = self_iter.next() {
            let other_id = other_iter.next();
            if Some(self_id) == other_id {
                continue;
            }

            // descendant check
            if self_id < OO || other_id < Some(self_id) {
                return None;
            }

            // At this point, OO <= self_id < other_id. Therefore, other can
            // only be a valid descendant of self if self_iter is exhausted.
            if self_iter.next().is_none() {
                return Some(OO);
            }
            return None;
        }

        // If we've exhausted self_iter and other is a chain of OO nodes, we're
        // only interested in the direct node id so we can just return that.
        // Otherwise the next thing we find is the child id.
        match other_iter.next() {
            Some(x) if x > OO => Some(OO),
            x => x,
        }
    }

    /// Adds a node to the end of the path. Complexity O(#nodes).
    ///
    /// In case of oo nodes, it checks whether the last node of the path is
    /// already a run length encoded oo node in which case it replaces it with
    /// the merged representation. Returns an error if the maximum encoding
    /// length is exceeded.
    pub fn push(&mut self, id: ChildId) {
        let mut id = id;
        if self.is_empty() {
            self.push_bits(32, id as u64);
            return;
        } else if id >= OO {
            // Infinity nodes are special, since we **have to** run length encode them with
            // a potential previous oo child! So, decode that last child and potentially
            // merge the two together.
            let (last_child, _, len_parents) = decode_child(self.as_bytes());
            if last_child >= OO && len_parents > 0 {
                id += last_child - OO + 1;
                // Resize in this context truncates, removing the last_child from the TreePath.
                // Below the correct id will be pushed instead of the OO node that was truncated off.
                self.resize(len_parents);
            }
        }
        let (child, child_bits) = encode(id);
        self.push_bits(child_bits, child);
    }

    /// Produces the parent tree-path by popping a child from the end (oo
    /// nodes are popped entirely regardless of count). This operation is
    /// performed in O(#nodes).
    ///
    /// ```
    /// # use github_tree_path::{TreePath, OO};
    /// fn path(s: &str) -> TreePath {
    ///     s.parse().unwrap()
    /// }
    ///
    /// let mut p = path("/1123/oo2/4");
    /// assert_eq!(p.parent(), 4);
    /// assert_eq!(p, path("/1123/oo2"));
    /// assert_eq!(p.parent(), OO + 1);
    /// assert_eq!(p, path("/1123"));
    /// assert_eq!(p.parent(), 1123);
    /// assert_eq!(p, path("/"));
    /// ```
    pub fn parent(&mut self) -> ChildId {
        assert!(!self.is_empty());
        let (last_child, _, rest_len) = decode_child(self.as_bytes());
        self.resize(rest_len);
        last_child
    }

    /// Pop a child from the end of the tree-path. This operation is performed in O(#nodes).
    ///
    /// ```
    /// # use github_tree_path::{TreePath, OO};
    /// fn path(s: &str) -> TreePath {
    ///     s.parse().unwrap()
    /// }
    ///
    /// let mut p = path("/1123/oo2/4");
    /// assert_eq!(p.pop(), 4);
    /// assert_eq!(p, path("/1123/oo2"));
    /// assert_eq!(p.pop(), OO);
    /// assert_eq!(p, path("/1123/oo"));
    /// assert_eq!(p.pop(), OO);
    /// assert_eq!(p, path("/1123"));
    /// assert_eq!(p.pop(), 1123);
    /// assert_eq!(p, path("/"));
    /// ```
    pub fn pop(&mut self) -> ChildId {
        assert!(!self.is_empty());
        let (last_child, _, rest_len) = decode_child(self.as_bytes());
        self.resize(rest_len);
        if last_child > OO {
            self.push(last_child - 1);
            OO
        } else {
            last_child
        }
    }

    /// Join a node id to this path returning a new tree-path.
    pub fn join(&self, id: ChildId) -> Self {
        let mut p = self.clone();
        p.push(id);
        p
    }

    /// Number of bits in this tree-path.
    pub fn num_bits(&self) -> usize {
        match self {
            TreePath::Ref(bits) => check_path(bits),
            TreePath::Own(_, num_bits) => *num_bits,
        }
    }

    fn to_mut(&mut self) -> (&mut Vec<u8>, &mut usize) {
        match self {
            TreePath::Ref(bits) => {
                let num_bits = check_path(bits);
                *self = TreePath::Own(bits.to_vec(), num_bits);
                self.to_mut()
            }
            TreePath::Own(bits, num_bits) => (bits, num_bits),
        }
    }

    fn push_bits(&mut self, num_bits: usize, mut bits: u64) {
        let (bytes, self_num_bits) = self.to_mut();
        let start_byte = *self_num_bits / 8;
        *self_num_bits += num_bits;
        let end_byte = (*self_num_bits + 7) / 8;

        bytes.resize((*self_num_bits + 7) / 8, 0);
        // The last byte might be partially occupied. We simply shift the new
        // bits by the number of already occupied bits, so that the new bits can
        // simply be or-ed with the existing ones...
        bits <<= *self_num_bits % 8;
        for byte in (start_byte..end_byte).rev() {
            bytes[byte] |= bits as u8;
            bits >>= 8;
        }
    }

    fn resize(&mut self, num_bits: usize) {
        let (bytes, self_num_bits) = self.to_mut();
        *self_num_bits = num_bits;
        bytes.resize((num_bits + 7) / 8, 0);
        self.clear_superfluous_bits();
    }

    fn clear_superfluous_bits(&mut self) {
        let (bytes, self_num_bits) = self.to_mut();
        if *self_num_bits % 8 != 0 {
            // clear superfluous bits in the last byte
            bytes[*self_num_bits / 8] &= !(0xff >> (*self_num_bits % 8));
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = ChildId> + '_ {
        TreePathIterator {
            path: self.as_bytes(),
            bit_pos: 0,
        }
    }
}

impl<'a> From<Vec<u8>> for TreePath<'a> {
    fn from(path: Vec<u8>) -> Self {
        let num_bits = check_path(&path);
        unsafe { Self::from_raw_unchecked(path, num_bits) }
    }
}

impl<'a> From<&'a [u8]> for TreePath<'a> {
    fn from(path: &'a [u8]) -> Self {
        Self::from_bytes(path)
    }
}

impl<'a> From<TreePath<'a>> for Vec<u8> {
    fn from(path: TreePath<'a>) -> Self {
        path.as_bytes().to_vec()
    }
}

fn try_check_path(path: &[u8]) -> Result<usize, TreePathParseError> {
    let num_bits = num_bits(path);
    if path.len() != (num_bits + 7) / 8 {
        return Err(TreePathParseError::ExtraBytes(hex::encode(path)));
    }
    if num_bits % 8 == 4 && path[path.len() - 1] & 0xF != 0 {
        return Err(TreePathParseError::NonzeroTailBits(hex::encode(path)));
    }
    Ok(num_bits)
}

// checks that the byte slice is a valid tree-path and returns num_bits. O(#nodes)
fn check_path(path: &[u8]) -> usize {
    try_check_path(path).expect("invalid TreePath bytes")
}

impl<'a> github_pspack::Serializable<'a> for TreePath<'a> {
    fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
        let bytes = self.as_bytes();
        writer.write_all(bytes)?;
        Ok(bytes.len())
    }

    fn from_bytes(buf: &'a [u8]) -> Self {
        TreePath::from(buf)
    }
}

// calculates the number of bits in a tree-path (does not validate!). O(#nodes)
pub(crate) fn num_bits(path: &[u8]) -> usize {
    let mut num_bits = 0;
    loop {
        let (_, consume) = decode(path, num_bits);
        if consume == 0 {
            break;
        }
        num_bits += consume;
    }
    num_bits
}

pub struct TreePathIterator<'a> {
    path: &'a [u8],
    bit_pos: usize,
}

/// Returns all the nodes from root to leaf of the corresponding path.
impl<'a> Iterator for TreePathIterator<'a> {
    type Item = ChildId;

    fn next(&mut self) -> Option<Self::Item> {
        let (id, size) = decode(self.path, self.bit_pos);
        self.bit_pos += size;
        if id > 0 {
            Some(id)
        } else {
            None
        }
    }
}

/// Encodes a single node id and returns its pattern and size of the pattern in
/// bits.
pub fn encode(id: ChildId) -> (u64, usize) {
    let id = id as u64;
    const OO64: u64 = OO as u64;
    match id {
        0 => (0, 0),
        1..=3 => (id, 4),
        4..=35 => (0x40 + id - 4, 8),
        36..=4131 => (0x6000 + id - 36, 16),
        4132..=134221857 => (0x70000000 + id - 4132, 32),
        OO64.. => {
            let id = id - OO as u64;
            match id {
                0..=3 => (0x8 + id, 4),
                4..=35 => (0xC0 + id - 4, 8),
                36..=4131 => (0xE000 + id - 36, 16),
                4132..=134221857 => (0xF0000000 + id - 4132, 32),
                _ => panic!("we don't support this id range: {id}"),
            }
        }
        _ => panic!("we don't support this id range: {id}"),
    }
}

/// Returns the last child id, the total bits of the path, and the total bits of
/// the path excluding the last child. Runs in O(#nodes).
fn decode_child(path: &[u8]) -> (ChildId, usize, usize) {
    assert!(!path.is_empty());
    let mut bits = 0;
    let mut last_child_start = 0;
    let mut last_child = 0;
    loop {
        let (child, child_bits) = decode(path, bits);
        if child_bits == 0 {
            break (last_child, bits, last_child_start);
        }
        last_child = child;
        last_child_start = bits;
        bits += child_bits;
    }
}

fn decode_id(
    path: &[u8],
    bit_pos: usize,
    ignore_bits: usize,
    total_bits: usize,
    offset: ChildId,
) -> (ChildId, usize) {
    let start_bit = bit_pos + ignore_bits;
    let end_bit = bit_pos + total_bits;
    let start_byte = start_bit / 8;
    let end_byte = (end_bit + 7) / 8;
    let mut id = 0;
    for byte in path.iter().take(end_byte).skip(start_byte) {
        id <<= 8;
        id |= *byte as usize;
    }
    // The end bit position might not be aligned with byte boundaries.
    // Shift out the superfluous bits.
    id >>= (-(end_bit as isize) as usize) % 8;
    // And keep the least significant total_bits - ignore_bits many bits.
    id &= !(usize::MAX << (total_bits - ignore_bits));
    (id as ChildId + offset, total_bits)
}

fn decode(path: &[u8], bit_pos: usize) -> (ChildId, usize) {
    if bit_pos >= path.len() * 8 {
        // We implicitly assume that the bytes representation is followed by infinitely many zero bytes.
        // Since these zero bytes represent the terminating id 0, decoding can be shortcut here (also to
        // avoid accessing invalid memory addresses).
        (0, 0)
    } else if bit_pos == 0 {
        // The first id is encoded as plain u32, since we expect a large fan-out at the first level.
        assert!(path.len() >= 4);
        decode_id(path, bit_pos, 0, 32, 0)
    } else {
        assert_eq!(bit_pos % 4, 0);
        // Look at the 4 most significant bits which encode the length.
        // Since all encodings consume a multiple of 4 bits, we just need
        // to look at either the upper or lower half of the corresponding byte.
        match (path[bit_pos / 8] >> ((bit_pos + 4) % 8)) & 0xF {
            0 => (0, 0),
            1..=3 => decode_id(path, bit_pos, 2, 4, 0),
            4..=5 => decode_id(path, bit_pos, 3, 8, 4),
            6 => decode_id(path, bit_pos, 4, 16, 36),
            7 => decode_id(path, bit_pos, 5, 32, 4132),
            8..=11 => decode_id(path, bit_pos, 2, 4, OO),
            12..=13 => decode_id(path, bit_pos, 3, 8, 4 + OO),
            14 => decode_id(path, bit_pos, 4, 16, 36 + OO),
            15 => decode_id(path, bit_pos, 5, 32, 4132 + OO),
            _ => unreachable!("4 bits shouldn't produce number larger than 15!"),
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;

    #[test]
    fn test_one_tree_path() {
        let path = TreePath::from_ids(&[1, OO, OO, 19, OO + 3789]);
        assert_eq!(path.iter().collect::<Vec<_>>(), &[1, OO + 1, 19, OO + 3789]);
    }

    #[test]
    fn test_push_infinity_ids() {
        let mut path = TreePath::from_ids(&[1]);
        path.push(OO + 10);
        assert_eq!(path.iter().collect::<Vec<_>>(), &[1, OO + 10]);
        path.push(OO + 10);
        assert_eq!(path.iter().collect::<Vec<_>>(), &[1, OO + 21]);
        path.push(OO);
        assert_eq!(path.iter().collect::<Vec<_>>(), &[1, OO + 22]);
    }

    #[test]
    fn test_tree_path_slice() {
        assert_eq!(
            TreePath::from_bytes(&[]).as_bytes(),
            TreePath::from_ids(&[]).as_bytes()
        );
        assert_eq!(
            TreePath::from_bytes(&[0, 0, 0, 1]).as_bytes(),
            TreePath::from_ids(&[1]).as_bytes()
        );
    }

    #[test]
    fn test_all_child_ids() {
        let parent = TreePath::from_ids(&[1]);
        for i in 0..33000 {
            let mut child = parent.join(i + 1).join(2);
            assert_eq!(child.iter().collect::<Vec<_>>(), &[1, i + 1, 2]);
            assert_eq!(child.parent(), 2);
            assert_eq!(child.parent(), i + 1);

            let mut inf_child = parent.join(OO + i).join(2);
            assert_eq!(inf_child.iter().collect::<Vec<_>>(), &[1, OO + i, 2]);
            assert_eq!(inf_child.parent(), 2);
            assert_eq!(inf_child.parent(), OO + i);
        }
    }

    #[test]
    fn test_pop_to_get_parent() {
        let mut node = TreePath::from_ids(&[1, 2, 3]);
        assert_eq!(node.parent(), 3);
        assert_eq!(node.iter().collect::<Vec<_>>(), &[1, 2]);
    }

    #[test]
    fn test_uppper_bound() {
        assert_eq!(
            TreePath::from_ids(&[3]),
            TreePath::from_ids(&[2]).upper_bound()
        );
        assert_eq!(
            TreePath::from_ids(&[5]),
            TreePath::from_ids(&[4]).upper_bound()
        );
        assert_eq!(
            TreePath::from_ids(&[4, 5]),
            TreePath::from_ids(&[4, 4]).upper_bound()
        );
    }

    #[test]
    fn test_large_encodings() {
        assert_eq!(32 + 4, TreePath::from_ids(&[1, 3]).num_bits());
        assert_eq!(32 + 8, TreePath::from_ids(&[1, 4]).num_bits());
        assert_eq!(80, TreePath::from_ids(&[4, 4, 4, 4, 4, 4, 4]).num_bits());
        assert_eq!(88, TreePath::from_ids(&[4, 4, 4, 4, 4, 4, 4, 4]).num_bits());
        assert_eq!(
            96,
            TreePath::from_ids(&[4, 4, 4, 4, 4, 4, 4, 4, 4]).num_bits()
        );
        assert_eq!(
            104,
            TreePath::from_ids(&[4, 4, 4, 4, 4, 4, 4, 4, 4, 4]).num_bits()
        );
        // We can also compute the upper_bound of this path.
        assert_eq!(
            TreePath::from_ids(&[4, 4, 4, 4, 4, 4, 4, 4]),
            TreePath::from_ids(&[4, 4, 4, 4, 4, 4, 4, 3]).upper_bound()
        );
        assert_eq!(
            84,
            TreePath::from_ids(&[4, 4, 4, 4, 4, 4, 4, OO]).num_bits()
        );
        assert_eq!(
            84,
            TreePath::from_ids(&[4, 4, 4, 4, 4, 4, 4, OO, OO, OO, OO]).num_bits()
        );
        // Same node in run length notation and without.
        assert_eq!(
            TreePath::from_ids(&[4, 4, 4, 4, 4, 4, 4, OO, OO, OO, OO]),
            TreePath::from_ids(&[4, 4, 4, 4, 4, 4, 4, OO + 3])
        );
        assert_eq!(
            TreePath::from_ids(&[4, 4, 4, 4, 4, 4, 4, OO, OO, OO, OO, OO]),
            TreePath::from_ids(&[4, 4, 4, 4, 4, 4, 4, OO + 4])
        );
        // And upper bound works here as well.
        assert_eq!(
            TreePath::from_ids(&[4, 4, 4, 4, 4, 4, 5]),
            TreePath::from_ids(&[4, 4, 4, 4, 4, 4, 4, OO + 3]).upper_bound(),
        )
    }

    #[test]
    fn test_tree_path_conversion() {
        // Do a full conversion from tree path to tree path slice to u8 slice and back again.
        // At the end, we should get an equalivant tree path instance.
        let a: TreePath = TreePath::from_ids(&[1, 2, 3, 4, 5, 6]);
        let c: &[u8] = a.as_bytes();
        let d: TreePath = TreePath::from_bytes(c);
        let e: TreePath = d.clone_owned();
        assert_eq!(a, e);
    }

    #[test]
    fn test_bytes() {
        assert_eq!(&[0, 0, 0, 1, 16], TreePath::from_ids(&[1, 1]).as_bytes());
        assert_eq!(
            &[0, 0, 0, 4, 64, 64],
            TreePath::from_ids(&[4, 4, 4]).as_bytes()
        );
    }

    #[test]
    fn test_pop() {
        let mut a = TreePath::from_ids(&[1, 2, 3, 4, 5, OO]);
        assert_eq!(OO, a.parent());
        assert_eq!(5, a.parent());
        assert_eq!(4, a.parent());
        assert_eq!(3, a.parent());
        assert_eq!(2, a.parent());
        assert!(!a.is_empty());
        assert_eq!(1, a.parent());
        assert!(a.is_empty());
    }

    #[test]
    #[should_panic]
    fn test_check_path() {
        TreePath::from_bytes(&[1]).into_owned();
    }

    #[test]
    #[should_panic]
    fn test_pop_from_empty_tree() {
        let mut a = TreePath::root();
        a.parent();
    }

    #[test]
    fn test_special_casing_of_oo_at_first_level() {
        let a: TreePath = TreePath::from_ids(&[OO, OO, OO]);
        let c: &[u8] = a.as_bytes();
        let d: TreePath = TreePath::from_bytes(c);
        let e: TreePath = d.clone_owned();
        assert_eq!(a, e);
        assert_eq!(a.iter().collect_vec(), [OO, OO + 1]);

        let a: TreePath = TreePath::from_ids(&[OO]);
        assert_eq!(a.upper_bound(), TreePath::from_ids(&[OO + 1]));
    }

    #[test]
    fn test_child_id() {
        let parent = TreePath::from_ids(&[3]);
        assert_eq!(1, parent.child_id(&TreePath::from_ids(&[3, 1])).unwrap());
        assert_eq!(1, parent.child_id(&TreePath::from_ids(&[3, 1, 2])).unwrap());
        assert_eq!(OO, parent.child_id(&TreePath::from_ids(&[3, OO])).unwrap());
        assert_eq!(
            OO,
            parent
                .child_id(&TreePath::from_ids(&[3, OO, OO, OO]))
                .unwrap()
        );
        assert_eq!(
            OO,
            parent.child_id(&TreePath::from_ids(&[3, OO, 1])).unwrap()
        );
        assert_eq!(None, parent.child_id(&TreePath::from_ids(&[2, 1])));
        assert_eq!(None, parent.child_id(&TreePath::from_ids(&[2, OO])));
    }

    #[track_caller]
    fn check_tree_path_debug(ids: &[ChildId]) {
        let path = TreePath::from_ids(ids);
        let debug_str = format!("{path:?}");
        let i = debug_str.find(" (").unwrap();
        let hex = &debug_str[..i];
        let segs = &debug_str[i + 2..debug_str.len() - 1];
        assert_eq!(debug_str, format!("{hex} ({segs})"));
        assert_eq!(hex.parse::<TreePath>().unwrap(), path);
        assert_eq!(segs.parse::<TreePath>().unwrap(), path);
        let segs = segs.replace('∞', "oo");
        assert_eq!(segs.parse::<TreePath>().unwrap(), path);
    }

    #[test]
    fn test_tree_path_parsing() {
        check_tree_path_debug(&[]);
        check_tree_path_debug(&[3]);
        check_tree_path_debug(&[3, 1, 2]);
        check_tree_path_debug(&[11235]);
        check_tree_path_debug(&[134221857]);
        check_tree_path_debug(&[11235, OO + 1123, 1, OO]);
    }

    #[test]
    fn test_tree_path_errors() {
        assert!("/99999999999999999999".parse::<TreePath>().is_err());
        assert!(format!("/{OO}").parse::<TreePath>().is_err());
        assert!("/53281/0".parse::<TreePath>().is_err());
        assert!("/53281/oo0".parse::<TreePath>().is_err());
    }
}
