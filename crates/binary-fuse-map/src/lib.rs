//! A *binary fuse map*: a static, immutable structure that maps a set of keys to fixed-size values
//! using only ~1.13 slots per key, where each slot stores a value (up to 8 bytes) plus a one-byte
//! membership fingerprint.
//!
//! It is the "map" generalisation of the [binary fuse
//! filter](https://arxiv.org/abs/2201.01174) (Graf & Lemire, 2022): instead of storing only a small
//! fingerprint per key, every slot stores an `N`-byte value *and* a fingerprint byte, and a key's
//! payload is the XOR of its three slots. Construction "peels" the 3-uniform hypergraph induced by
//! the keys and then assigns the slots in reverse peeling order so that, for every inserted key `k`,
//!
//! ```text
//! (value(k), fingerprint(k)) == slots[h0(k)] ^ slots[h1(k)] ^ slots[h2(k)]
//! ```
//!
//! # Properties
//! * Construction is `O(n)` and succeeds with overwhelming probability for any set of *distinct*
//!   keys; it transparently retries with a fresh seed otherwise.
//! * Lookups are `O(1)`: three (segment-local) slot loads and a couple of XORs.
//! * It is a filter *and* a map: [`BinaryFuseMap::get`] returns the stored value for inserted keys
//!   and rejects absent keys via the fingerprint with probability `255/256` (so ~`1/256` of absent
//!   keys are false positives that return an arbitrary value).
//! * Values are at most 8 bytes (`N <= 8`); each slot is `N + 1` bytes (value + fingerprint).
//!
//! # Example
//! ```
//! use binary_fuse_map::BinaryFuseMap;
//!
//! // Keys are pre-hashed to distinct `u64`s by the caller; values are fixed 4-byte slices.
//! let keys: Vec<u64> = (0..1000u64).map(|i| i.wrapping_mul(0x9E3779B97F4A7C15)).collect();
//! let values: Vec<[u8; 4]> = (0..1000u32).map(|i| i.to_le_bytes()).collect();
//!
//! let map = BinaryFuseMap::<4>::try_construct(&keys, &values).expect("construction succeeds");
//! for (k, v) in keys.iter().zip(&values) {
//!     assert_eq!(map.get(*k), Some(*v));
//! }
//! ```

/// Number of slots each key hashes to (the hypergraph is 3-uniform).
const ARITY: u32 = 3;

/// Maximum number of seeds tried before giving up. Each attempt succeeds with probability very
/// close to 1, so reaching this bound is less likely than a hardware fault.
const MAX_ITERATIONS: usize = 100;

/// A static map from `u64` keys to `[u8; N]` values built as a binary fuse map.
///
/// Build one with [`BinaryFuseMap::try_construct`] and query it with [`BinaryFuseMap::get`]. See the
/// [module documentation](crate) for the underlying algorithm and guarantees.
#[derive(Clone, Debug)]
pub struct BinaryFuseMap<const N: usize> {
    seed: u64,
    segment_length: u32,
    segment_count_length: u32,
    /// Number of keys the map was built from (for reporting; not needed for lookups).
    len: usize,
    /// Flat slot storage: `array_length` slots of `N + 1` bytes each (`N` value bytes followed by a
    /// fingerprint byte), plus [`READ_PADDING`] trailing bytes so a slot's value can always be read
    /// with a single 8-byte unaligned load. For an inserted key the XOR of its three slots yields
    /// its `N`-byte value and a fingerprint byte that confirms membership.
    slots: Vec<u8>,
}

impl<const N: usize> BinaryFuseMap<N> {
    /// Builds a map associating `keys[i]` with `values[i]`.
    ///
    /// `keys` must be *distinct* `u64`s (typically hashes of the real keys); duplicates make
    /// construction fail. Returns `None` only if construction did not converge within
    /// [`MAX_ITERATIONS`] seeds, which for distinct keys is astronomically unlikely.
    ///
    /// # Panics
    /// Panics if `N > 8`, if `keys.len() != values.len()`, or if there are more than `u32::MAX`
    /// keys.
    pub fn try_construct(keys: &[u64], values: &[[u8; N]]) -> Option<Self> {
        assert!(N <= 8, "value width N must be at most 8 bytes");
        assert_eq!(
            keys.len(),
            values.len(),
            "keys and values must have the same length"
        );
        assert!(
            keys.len() <= u32::MAX as usize,
            "at most u32::MAX keys are supported"
        );
        let size = keys.len();
        let params = Params::new(size as u32);
        let array_length = params.array_length as usize;

        let mut map = Self {
            seed: 0,
            segment_length: params.segment_length,
            segment_count_length: params.segment_count_length,
            len: size,
            // `N` value bytes + 1 fingerprint byte per slot, plus padding for the unaligned reads.
            slots: vec![0u8; array_length * (N + 1) + READ_PADDING],
        };

        // One `Cell` per slot accumulates, for the keys currently touching the slot, the XOR of
        // their mixed hashes and indices and the number of such keys (the slot's degree). Keeping
        // them in one array (rather than three) means each slot touched during counting/peeling is a
        // single random access, and storing the hash avoids re-reading `keys` while peeling.
        let mut cells = vec![Cell::default(); array_length];
        // Peeling output (hash, index, owned slot), written in peel order and replayed in reverse.
        let mut stack_hash = vec![0u64; size];
        let mut stack_idx = vec![0u32; size];
        let mut stack_slot = vec![0u32; size];
        // LIFO queue of slots that currently have exactly one key ("alone").
        let mut queue = vec![0u32; array_length];

        let mut rng = 0x726b_2b9d_438b_9d4d_u64;
        for _ in 0..MAX_ITERATIONS {
            let seed = splitmix64(&mut rng);
            cells.iter_mut().for_each(|c| *c = Cell::default());

            for (i, &key) in keys.iter().enumerate() {
                let hash = mix(key, seed);
                for &slot in &params.hashes(hash) {
                    let cell = &mut cells[slot];
                    cell.count += 1;
                    cell.xor_idx ^= i as u32;
                    cell.xor_hash ^= hash;
                }
            }

            let stack_size = peel(
                &params,
                &mut cells,
                &mut queue,
                &mut stack_hash,
                &mut stack_idx,
                &mut stack_slot,
            );
            if stack_size == size {
                map.seed = seed;
                assign(
                    &params,
                    values,
                    &stack_hash,
                    &stack_idx,
                    &stack_slot,
                    &mut map.slots,
                );
                return Some(map);
            }
        }
        None
    }

    /// Looks up `key`, returning `Some(value)` if it is (probably) present and `None` otherwise.
    ///
    /// For a key from the constructing set this always returns its stored value. For any other key
    /// the embedded 1-byte fingerprint rejects it with probability `255/256`; the remaining `1/256`
    /// are false positives that return an arbitrary value.
    #[inline]
    pub fn get(&self, key: u64) -> Option<[u8; N]> {
        let hash = mix(key, self.seed);
        let h = self.hashes(hash);
        let off = [h[0] * (N + 1), h[1] * (N + 1), h[2] * (N + 1)];
        // One unaligned 64-bit load per slot.
        let payload = read_value(&self.slots, off[0])
            ^ read_value(&self.slots, off[1])
            ^ read_value(&self.slots, off[2]);
        let fp = if N < 8 {
            // The fingerprint byte (byte `N`) is inside the word we already loaded. `N.min(7)` only
            // keeps the dead `N == 8` instantiation in array bounds; it is `N` for every live case.
            payload.to_le_bytes()[N.min(7)]
        } else {
            // An 8-byte value fills the whole word, so the fingerprint is the trailing 9th byte.
            self.slots[off[0] + N] ^ self.slots[off[1] + N] ^ self.slots[off[2] + N]
        };
        (fp == fingerprint(hash)).then(|| value_from_u64::<N>(payload))
    }

    /// The number of keys this map was built from.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the map was built from zero keys.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of slots (`array_length`); always `>= len`.
    pub fn slot_count(&self) -> usize {
        (self.slots.len() - READ_PADDING) / (N + 1)
    }

    /// Heap memory used by the slot array, in bytes (`slot_count * (N + 1)`).
    pub fn memory_usage(&self) -> usize {
        self.slot_count() * (N + 1)
    }

    /// Average number of bits stored per key (slot array size in bits divided by key count).
    pub fn bits_per_key(&self) -> f64 {
        if self.len == 0 {
            0.0
        } else {
            (self.memory_usage() * 8) as f64 / self.len as f64
        }
    }

    #[inline]
    fn hashes(&self, hash: u64) -> [usize; 3] {
        hash_positions(hash, self.segment_length, self.segment_count_length)
    }
}

/// Geometry of the slot array, derived solely from the number of keys.
struct Params {
    segment_length: u32,
    segment_count_length: u32,
    array_length: u32,
}

impl Params {
    fn new(size: u32) -> Self {
        let segment_length = segment_length(size).min(1 << 18);
        let size_factor = size_factor(size);
        let capacity = if size <= 1 {
            0
        } else {
            (size as f64 * size_factor).round() as u32
        };
        let total_segments = capacity.div_ceil(segment_length).max(ARITY);
        let segment_count = total_segments - (ARITY - 1);
        let array_length = total_segments * segment_length;
        let segment_count_length = segment_count * segment_length;
        Self {
            segment_length,
            segment_count_length,
            array_length,
        }
    }

    #[inline]
    fn hashes(&self, hash: u64) -> [usize; 3] {
        hash_positions(hash, self.segment_length, self.segment_count_length)
    }
}

/// The three slot indices a (mixed) hash maps to: one slot in each of three consecutive segments.
#[inline]
fn hash_positions(
    hash: u64,
    segment_length: u32,
    segment_count_length: u32,
) -> [usize; 3] {
    let segment_length_mask = segment_length - 1;
    let hi = mulhi(hash, segment_count_length as u64) as u32;
    let h0 = hi;
    let h1 = hi.wrapping_add(segment_length) ^ (((hash >> 18) as u32) & segment_length_mask);
    let h2 = hi.wrapping_add(2 * segment_length) ^ ((hash as u32) & segment_length_mask);
    [h0 as usize, h1 as usize, h2 as usize]
}

/// Per-slot peeling accumulator. For all keys currently mapped to the slot it holds the XOR of
/// their mixed hashes and indices, plus the number of such keys (`count`, the slot's degree). When
/// a slot becomes a singleton (`count == 1`), `xor_hash`/`xor_idx` are exactly that key's hash and
/// index.
#[derive(Clone, Copy, Default)]
struct Cell {
    xor_hash: u64,
    xor_idx: u32,
    count: u8,
}

/// Repeatedly removes ("peels") keys that are alone in one of their slots, recording the removal
/// order. Returns the number of keys peeled; equals `cells`-worth of keys iff the graph is acyclic.
fn peel(
    params: &Params,
    cells: &mut [Cell],
    queue: &mut [u32],
    stack_hash: &mut [u64],
    stack_idx: &mut [u32],
    stack_slot: &mut [u32],
) -> usize {
    let mut q = 0;
    for (slot, cell) in cells.iter().enumerate() {
        if cell.count == 1 {
            queue[q] = slot as u32;
            q += 1;
        }
    }

    let mut stack_size = 0;
    while q > 0 {
        q -= 1;
        let slot = queue[q] as usize;
        let cell = cells[slot];
        if cell.count != 1 {
            continue; // stale queue entry: the key was already peeled elsewhere.
        }
        let hash = cell.xor_hash;
        let idx = cell.xor_idx;
        stack_hash[stack_size] = hash;
        stack_idx[stack_size] = idx;
        stack_slot[stack_size] = slot as u32;
        stack_size += 1;

        // Remove this key from its other two slots (the ones that aren't the slot it owns), which
        // may make them "alone" in turn. The three slots are always distinct (different segments).
        for other in params.hashes(hash) {
            if other != slot {
                let cell = &mut cells[other];
                cell.count -= 1;
                cell.xor_idx ^= idx;
                cell.xor_hash ^= hash;
                if cell.count == 1 {
                    queue[q] = other as u32;
                    q += 1;
                }
            }
        }
    }
    stack_size
}

/// Replays the peeling stack in reverse, fixing each key's owned slot so that the XOR of its three
/// slots equals its `(value, fingerprint)` payload.
fn assign<const N: usize>(
    params: &Params,
    values: &[[u8; N]],
    stack_hash: &[u64],
    stack_idx: &[u32],
    stack_slot: &[u32],
    slots: &mut [u8],
) {
    for s in (0..stack_hash.len()).rev() {
        let hash = stack_hash[s];
        let off = stack_slot[s] as usize * (N + 1);
        let mut value = value_to_u64(&values[stack_idx[s] as usize]);
        let mut fingerprint = fingerprint(hash);
        for other in params.hashes(hash) {
            let other_off = other * (N + 1);
            if other_off != off {
                let word = read_value(slots, other_off);
                value ^= word;
                // For N < 8 the neighbour's fingerprint is byte N of the word we just loaded.
                fingerprint ^= if N < 8 {
                    word.to_le_bytes()[N.min(7)]
                } else {
                    slots[other_off + N]
                };
            }
        }
        // Only the low `N` value bytes are written, so the neighbouring slot is never clobbered.
        slots[off..off + N].copy_from_slice(&value.to_le_bytes()[..N]);
        slots[off + N] = fingerprint;
    }
}

/// Trailing padding (bytes) kept after the slot array so an 8-byte unaligned read at the last slot's
/// offset stays in bounds even when `N + 1 < 8`.
const READ_PADDING: usize = 8;

/// Reads the value bytes of the slot at byte offset `off` as a `u64` with one unaligned load.
///
/// Only the low `N` bytes are the value; for `N < 8` the higher bytes are the fingerprint and the
/// start of the next slot, which callers ignore (value extraction keeps the low `N` bytes).
#[inline]
fn read_value(slots: &[u8], off: usize) -> u64 {
    debug_assert!(off + 8 <= slots.len());
    // SAFETY: the slot array is allocated with `READ_PADDING == 8` trailing bytes, so reading 8
    // bytes starting at any slot offset stays within the allocation.
    let bytes = unsafe { std::ptr::read_unaligned(slots.as_ptr().add(off).cast::<[u8; 8]>()) };
    u64::from_le_bytes(bytes)
}

/// Zero-extends an `N`-byte value (`N <= 8`) into a `u64`.
#[inline]
fn value_to_u64<const N: usize>(value: &[u8; N]) -> u64 {
    let mut bytes = [0u8; 8];
    bytes[..N].copy_from_slice(value);
    u64::from_le_bytes(bytes)
}

/// Extracts the low `N` bytes of a `u64` as the stored value.
#[inline]
fn value_from_u64<const N: usize>(value: u64) -> [u8; N] {
    let mut out = [0u8; N];
    out.copy_from_slice(&value.to_le_bytes()[..N]);
    out
}

/// The 1-byte membership fingerprint derived from a key's mixed hash.
#[inline]
fn fingerprint(hash: u64) -> u8 {
    (hash ^ (hash >> 32)) as u8
}

/// Segment length: a power of two chosen to keep peeling fast and the structure compact.
fn segment_length(size: u32) -> u32 {
    if size == 0 {
        return 4;
    }
    1u32 << ((size as f64).ln() / 3.33_f64.ln() + 2.25).floor() as u32
}

/// Over-allocation factor for the slot array (~1.125 asymptotically, more for small inputs).
fn size_factor(size: u32) -> f64 {
    if size <= 1 {
        return 0.0;
    }
    1.125_f64.max(0.875 + 0.25 * 1_000_000_f64.ln() / (size as f64).ln())
}

/// A reversible 64-bit mix; combined with `seed` it turns keys into well-distributed hashes.
#[inline]
fn murmur64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    h
}

#[inline]
fn mix(key: u64, seed: u64) -> u64 {
    murmur64(key.wrapping_add(seed))
}

/// High 64 bits of the 128-bit product `a * b`; used to map a hash uniformly into `[0, b)`.
#[inline]
fn mulhi(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}

/// SplitMix64 PRNG step, used to pick construction seeds.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn distinct_keys(n: usize, salt: u64) -> Vec<u64> {
        let mut state = 0x1234_5678_9abc_def0 ^ salt;
        (0..n).map(|_| splitmix64(&mut state)).collect()
    }

    /// Every inserted key must retrieve exactly its value, across many sizes and value widths.
    fn assert_roundtrip<const N: usize>(n: usize) {
        let keys = distinct_keys(n, N as u64);
        let values: Vec<[u8; N]> = (0..n)
            .map(|i| {
                let mut v = [0u8; N];
                for (j, b) in v.iter_mut().enumerate() {
                    *b = (i.wrapping_mul(31).wrapping_add(j)) as u8;
                }
                v
            })
            .collect();

        let map = BinaryFuseMap::<N>::try_construct(&keys, &values)
            .unwrap_or_else(|| panic!("construction failed for n={n}, N={N}"));
        assert_eq!(map.len(), n);
        for (k, v) in keys.iter().zip(&values) {
            assert_eq!(map.get(*k), Some(*v), "mismatch for n={n}, N={N}");
        }
    }

    #[test]
    fn roundtrip_small_sizes() {
        for n in [0usize, 1, 2, 3, 5, 10, 50, 100, 1000] {
            assert_roundtrip::<1>(n);
            assert_roundtrip::<4>(n);
            assert_roundtrip::<7>(n);
            assert_roundtrip::<8>(n);
        }
    }

    #[test]
    fn roundtrip_medium_sizes() {
        for n in [10_000usize, 100_000, 1_000_000] {
            assert_roundtrip::<8>(n);
        }
    }

    #[test]
    fn absent_keys_are_mostly_rejected() {
        // The 1-byte fingerprint should reject ~255/256 of absent keys.
        let n = 200_000;
        let keys = distinct_keys(n, 11);
        let values: Vec<[u8; 8]> = (0..n as u64).map(|i| i.to_le_bytes()).collect();
        let map = BinaryFuseMap::<8>::try_construct(&keys, &values).unwrap();

        // Probe a disjoint key set; count how many slip through as false positives.
        let absent = distinct_keys(n, 0xABCD);
        let false_positives = absent.iter().filter(|&&k| map.get(k).is_some()).count();
        let rate = false_positives as f64 / n as f64;
        assert!(
            rate < 0.02,
            "false positive rate {rate} too high (expected ~1/256)"
        );
    }

    #[test]
    fn bits_per_key_is_close_to_optimal() {
        let keys = distinct_keys(1_000_000, 7);
        let values = vec![[0u8; 8]; keys.len()];
        let map = BinaryFuseMap::<8>::try_construct(&keys, &values).unwrap();
        // ~1.13 slots/key * 9 bytes (8 value + 1 fingerprint) * 8 bits ≈ 81 bits/key.
        assert!(
            map.bits_per_key() < 9.0 * 8.0 * 1.16,
            "bits_per_key={} too high",
            map.bits_per_key()
        );
    }

    #[test]
    fn distinct_seeds_for_distinct_runs_are_stable() {
        // Construction is deterministic given the same keys/values.
        let keys = distinct_keys(10_000, 1);
        let values: Vec<[u8; 4]> = (0..keys.len() as u32).map(|i| i.to_le_bytes()).collect();
        let a = BinaryFuseMap::<4>::try_construct(&keys, &values).unwrap();
        let b = BinaryFuseMap::<4>::try_construct(&keys, &values).unwrap();
        assert_eq!(a.seed, b.seed);
        for k in &keys {
            assert_eq!(a.get(*k), b.get(*k));
        }
    }
}
