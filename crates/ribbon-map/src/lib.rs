//! A *ribbon map*: a static, immutable filter+map from a set of distinct `u64` keys to fixed-size
//! `[u8; N]` values, built on top of a [ribbon
//! retrieval](https://arxiv.org/abs/2103.02515) structure (Dillinger, Hübschle-Schneider, Sanders &
//! Walzer, 2021).
//!
//! # From a ribbon filter to a ribbon map
//!
//! A ribbon *filter* solves a banded linear system over GF(2): every key `k` hashes to a *start*
//! column `s(k)`, a width-`W` *coefficient* row `c(k)` (a `W`-bit window that begins at column
//! `s(k)`), and an `r`-bit *fingerprint*. Construction picks a solution vector `X` (one `r`-bit word
//! per column) such that, for every key, the windowed dot product reproduces the fingerprint:
//!
//! ```text
//! XOR_{j : bit j of c(k) set} X[s(k) + j] == fingerprint(k)
//! ```
//!
//! Membership is then a fingerprint check. This crate keeps that machinery but changes *what the
//! `r` bits mean*, turning the filter into a map:
//!
//! * **The diagonal.** Ribbon construction is incremental Gaussian elimination. Each key ends up
//!   *owning* a distinct pivot column — its position on the matrix diagonal. We reuse that pivot as
//!   a **perfect hash**: the key's value is stored in a side array indexed by the pivot column.
//! * **The deviation, not a fingerprint.** To recover a key's pivot at query time we store, per
//!   key, the *deviation* `d(k) = pivot(k) - s(k)` — how far the diagonal sits from the key's start.
//!   If keys are **processed in order of increasing `s(k)`**, the pivot is provably confined to the
//!   key's own window `[s(k), s(k) + W)`, so for `W = 64` the deviation always fits in **6 bits**
//!   (see [`W`] and the note below). The retrieval structure stores those 6 bits *instead of* a
//!   fingerprint.
//! * **A zero-check instead of a fingerprint.** With `W = 64` the deviation needs only 6 of the 8
//!   bits of an `r = 8` retrieval word. Rather than store a per-key fingerprint in the top **2
//!   bits**, we constrain them to **zero for every inserted key** and set the linear system's *free
//!   variables* — the columns no key pivots on — to **random values**. A present key therefore
//!   always reads back top bits `0`, while an absent key reads pseudo-random top bits and is
//!   rejected with probability ~3/4, before the value array is even touched. This needs no
//!   fingerprint hash and, unlike zero-filled free variables, also rejects absent keys whose window
//!   falls entirely on empty columns. Widen the zero-checked prefix for stronger rejection.
//!
//! A query therefore reads the 8-bit retrieval word `w = XOR of the windowed `X` slots`, rejects the
//! key unless its top 2 bits are zero, then returns `values[s(k) + (w & 0x3F)]`.
//!
//! # Why the deviation fits in 6 bits
//!
//! Processing keys by increasing start `s`, an inductive argument bounds every stored row to columns
//! `[i, i + W)` *and* every key's pivot to `[s, s + W)`: when a key at start `s` eliminates against
//! an already-placed row at column `i >= s` (created by an earlier key with start `s' <= s`), that
//! row only reaches column `s' + W - 1 <= s + W - 1`, so the key's own right edge never grows past
//! `s + W - 1`. Hence `pivot - s < W`. Construction can still fail if a key's window is *linearly
//! dependent* on earlier ones (its row eliminates to zero); that is handled by retrying with a fresh
//! seed, exactly as an ordinary ribbon filter handles it.
//!
//! # Properties
//!
//! * **Filter + map.** [`RibbonMap::get`] returns `Some(value)` for every inserted key and rejects
//!   absent keys via the 2-bit zero-check with probability ~3/4 (the rest are false positives that
//!   return an arbitrary value).
//! * **Compact.** ~`1 / LOAD_FACTOR` slots per key; each slot costs one byte of retrieval data plus
//!   `N` bytes of value, i.e. ~`(1 + N) * 8 / LOAD_FACTOR` bits per key. Unlike a binary fuse map the
//!   value width `N` is unbounded (values live in their own array).
//! * **`O(1)` lookups**: one windowed dot product over a 64-bit coefficient, a 2-bit fingerprint
//!   check, and a single value load.
//! * **Static**: built once from all keys; not modifiable afterwards.
//!
//! # Example
//!
//! ```
//! use ribbon_map::RibbonMap;
//!
//! // Keys are pre-hashed to distinct `u64`s by the caller; values are fixed 4-byte slices.
//! let keys: Vec<u64> = (0..1000u64).map(|i| i.wrapping_mul(0x9E3779B97F4A7C15)).collect();
//! let values: Vec<[u8; 4]> = (0..1000u32).map(|i| i.to_le_bytes()).collect();
//!
//! let map = RibbonMap::<4>::try_construct(&keys, &values).expect("construction succeeds");
//! for (k, v) in keys.iter().zip(&values) {
//!     assert_eq!(map.get(*k), Some(*v));
//! }
//! ```

/// Ribbon width: the number of columns a key's coefficient row spans, and the number of distinct
/// deviations a key can have. It is fixed at 64 so a coefficient is a single `u64` and a deviation
/// fits in [`DEVIATION_BITS`] bits.
pub const W: usize = 64;

/// Bits needed to encode a deviation in `[0, W)`; `log2(W) = 6` for `W = 64`.
const DEVIATION_BITS: u32 = 6;

/// Bits of the retrieval word constrained to zero for inserted keys (the absent-key "early exit").
/// Combined with random free variables, this rejects absent keys with probability `1 - 2^-CHECK_BITS`.
const CHECK_BITS: u32 = 2;

const _: () = assert!(
    1usize << DEVIATION_BITS == W,
    "DEVIATION_BITS must equal log2(W)"
);
const _: () = assert!(
    DEVIATION_BITS + CHECK_BITS <= 8,
    "retrieval word must fit in a u8"
);

/// Low-bit mask selecting the deviation from a retrieval word.
const DEVIATION_MASK: u8 = (1u8 << DEVIATION_BITS) - 1;

/// Target load factor (keys per slot). At ~0.85 the maximum deviation stays comfortably below `W`
/// even for hundreds of millions of keys, and construction almost always succeeds on the first seed.
/// Higher values are more compact but push deviations toward the `W`-bit ceiling and raise the
/// linear-dependence retry rate.
const LOAD_FACTOR: f64 = 0.85;

/// Maximum number of seeds tried before giving up. Each attempt succeeds with very high probability,
/// so reaching this bound is practically impossible for distinct keys.
const MAX_ITERATIONS: usize = 100;

/// The golden-ratio odd constant used to derive a second, independent hash per key.
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// A static filter+map from `u64` keys to `[u8; N]` values built as a ribbon map.
///
/// Build one with [`RibbonMap::try_construct`] and query it with [`RibbonMap::get`]. See the
/// [module documentation](crate) for the underlying algorithm and guarantees.
#[derive(Clone, Debug)]
pub struct RibbonMap<const N: usize> {
    seed: u64,
    /// Number of columns / slots (`m`); always `>= W` and `>= len`.
    slots: usize,
    /// Number of keys the map was built from.
    len: usize,
    /// Ribbon retrieval solution: one 8-bit word per slot. For an inserted key the windowed XOR of
    /// these words yields the key's deviation in its low `DEVIATION_BITS` bits and zero in its top
    /// `CHECK_BITS` bits. Free-variable slots (no key pivots on them) hold random values so absent
    /// keys read pseudo-random words.
    solution: Vec<u8>,
    /// Values indexed by diagonal (pivot) slot. Only the `len` pivot slots are meaningful; the rest
    /// are unused padding that absent-key false positives may return.
    values: Vec<[u8; N]>,
}

impl<const N: usize> RibbonMap<N> {
    /// Builds a map associating `keys[i]` with `values[i]`.
    ///
    /// `keys` must be *distinct* `u64`s (typically hashes of the real keys); duplicates make
    /// construction fail. Returns `None` only if construction did not converge within
    /// [`MAX_ITERATIONS`] seeds, which for distinct keys is astronomically unlikely.
    ///
    /// # Panics
    /// Panics if `keys.len() != values.len()` or if there are more than `u32::MAX` keys.
    pub fn try_construct(keys: &[u64], values: &[[u8; N]]) -> Option<Self> {
        assert_eq!(
            keys.len(),
            values.len(),
            "keys and values must have the same length"
        );
        assert!(
            keys.len() <= u32::MAX as usize,
            "at most u32::MAX keys are supported"
        );
        let slots = num_slots(keys.len());
        assert!(
            slots <= u32::MAX as usize,
            "too many keys: slot count exceeds u32::MAX"
        );

        let mut rng = 0x726b_2b9d_438b_9d4d_u64;
        for _ in 0..MAX_ITERATIONS {
            let seed = splitmix64(&mut rng);
            if let Some(map) = Self::build(keys, values, slots, seed) {
                return Some(map);
            }
        }
        None
    }

    /// Attempts a single construction with a fixed `seed`. Returns `None` if any key's row is
    /// linearly dependent on earlier ones (the caller then retries with a new seed).
    fn build(keys: &[u64], values: &[[u8; N]], slots: usize, seed: u64) -> Option<Self> {
        // Derive each key's (start, coefficient, fingerprint) and sort by start. Processing keys in
        // start order is what confines every pivot to its own window (see the module docs).
        let mut infos: Vec<KeyInfo> = keys
            .iter()
            .enumerate()
            .map(|(idx, &key)| {
                let (start, coeff) = derive(key, seed, slots);
                KeyInfo {
                    start: start as u32,
                    coeff,
                    idx: idx as u32,
                }
            })
            .collect();
        infos.sort_unstable_by_key(|ki| ki.start);

        // Incremental Gaussian elimination. `row_coeff[i] != 0` marks a slot whose pivot is column
        // `i`; `row_rhs[i]` is that row's reduced retrieval word. `values_arr[i]` holds the value of
        // the key that landed on the diagonal at column `i`.
        let mut row_coeff = vec![0u64; slots];
        let mut row_rhs = vec![0u8; slots];
        let mut values_arr = vec![[0u8; N]; slots];

        for ki in &infos {
            let start = ki.start as usize;
            let mut coeff = ki.coeff;
            let mut i = start;
            // XOR of the retrieval words of the rows this key eliminates against.
            let mut acc = 0u8;
            loop {
                if coeff == 0 {
                    // Row eliminated to zero: linearly dependent on earlier keys. Retry a new seed.
                    return None;
                }
                let shift = coeff.trailing_zeros() as usize;
                i += shift;
                coeff >>= shift;
                if row_coeff[i] == 0 {
                    // Empty pivot column: this key owns diagonal `i`.
                    let deviation = i - start;
                    if deviation >= W {
                        // Guaranteed unreachable by the start-order argument; bail out defensively so
                        // a deviation can never silently overflow its 6-bit field.
                        return None;
                    }
                    // Target word: deviation in the low bits, zero in the top CHECK_BITS bits.
                    let word = deviation as u8;
                    row_coeff[i] = coeff;
                    row_rhs[i] = word ^ acc;
                    values_arr[i] = values[ki.idx as usize];
                    break;
                }
                // Eliminate against the row already on this diagonal and keep searching.
                coeff ^= row_coeff[i];
                acc ^= row_rhs[i];
            }
        }

        // Back-substitution: solve `X` from the triangular system. Row `i` has its pivot at bit 0
        // (column `i`); its other bits reference columns `i+1..i+W`, all already solved. Columns no
        // key pivots on are free variables: fill them with random bytes (deterministic from `seed`)
        // so absent keys read pseudo-random words and fail the zero-check. Present keys are
        // combinations of pivot rows, so they retrieve correctly for *any* free-variable choice.
        let mut solution = vec![0u8; slots];
        let mut free_var_rng = seed ^ 0xD1B5_4A32_D192_ED03;
        for i in (0..slots).rev() {
            let coeff = row_coeff[i];
            if coeff == 0 {
                solution[i] = splitmix64(&mut free_var_rng) as u8;
                continue;
            }
            let mut word = row_rhs[i];
            let mut bits = coeff & !1u64;
            while bits != 0 {
                let j = bits.trailing_zeros() as usize;
                word ^= solution[i + j];
                bits &= bits - 1;
            }
            solution[i] = word;
        }

        Some(Self {
            seed,
            slots,
            len: keys.len(),
            solution,
            values: values_arr,
        })
    }

    /// Looks up `key`, returning `Some(value)` if it is (probably) present and `None` otherwise.
    ///
    /// For a key from the constructing set this always returns its stored value. For any other key
    /// the 2-bit fingerprint rejects it with probability ~3/4; the rest are false positives that
    /// return an arbitrary value.
    #[inline]
    pub fn get(&self, key: u64) -> Option<[u8; N]> {
        let (start, coeff) = derive(key, self.seed, self.slots);
        // Windowed dot product over the solution reconstructs the retrieval word: the deviation in
        // the low bits and, for an inserted key, zero in the top CHECK_BITS bits.
        let mut word = 0u8;
        let mut bits = coeff;
        while bits != 0 {
            let j = bits.trailing_zeros() as usize;
            word ^= self.solution[start + j];
            bits &= bits - 1;
        }
        if (word >> DEVIATION_BITS) != 0 {
            return None;
        }
        let deviation = (word & DEVIATION_MASK) as usize;
        Some(self.values[start + deviation])
    }

    /// The number of keys this map was built from.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the map was built from zero keys.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of slots / columns (`m`); always `>= len`.
    pub fn slot_count(&self) -> usize {
        self.slots
    }

    /// Heap memory used by the retrieval and value arrays, in bytes (`slot_count * (1 + N)`).
    pub fn memory_usage(&self) -> usize {
        self.solution.len() + self.values.len() * N
    }

    /// Average number of bits stored per key (total array size in bits divided by key count).
    pub fn bits_per_key(&self) -> f64 {
        if self.len == 0 {
            0.0
        } else {
            (self.memory_usage() * 8) as f64 / self.len as f64
        }
    }
}

/// Per-key construction record: its window start, 64-bit coefficient row, and original index (to
/// fetch its value once its diagonal is known).
struct KeyInfo {
    start: u32,
    coeff: u64,
    idx: u32,
}

/// Number of slots for `n` keys: `ceil(n / LOAD_FACTOR)`, but never fewer than [`W`] so that at
/// least one valid start column exists.
fn num_slots(n: usize) -> usize {
    if n == 0 {
        return W;
    }
    (((n as f64) / LOAD_FACTOR).ceil() as usize).max(W)
}

/// Derives a key's `(start, coefficient)` from `seed` and the slot count.
///
/// `start` is drawn uniformly from `[0, slots - W]` (so the width-`W` window fits) and the
/// coefficient is 64 independent bits with bit 0 forced to 1 (fixing the window's left edge as the
/// pivot origin).
#[inline]
fn derive(key: u64, seed: u64, slots: usize) -> (usize, u64) {
    let a = mix(key, seed);
    let b = mix(key, seed.wrapping_add(GOLDEN));
    let start = mulhi(a, (slots - W + 1) as u64) as usize;
    let coeff = b | 1;
    (start, coeff)
}

/// A reversible 64-bit mix (the finalizer from MurmurHash3).
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

/// High 64 bits of the 128-bit product `a * b`; maps a hash uniformly into `[0, b)`.
#[inline]
fn mulhi(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}

/// SplitMix64 PRNG step, used to pick construction seeds.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(GOLDEN);
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

    fn make_values<const N: usize>(n: usize) -> Vec<[u8; N]> {
        (0..n)
            .map(|i| {
                let mut v = [0u8; N];
                for (j, b) in v.iter_mut().enumerate() {
                    *b = (i.wrapping_mul(31).wrapping_add(j).wrapping_add(7)) as u8;
                }
                v
            })
            .collect()
    }

    /// Every inserted key must retrieve exactly its value, across many sizes and value widths.
    fn assert_roundtrip<const N: usize>(n: usize) {
        let keys = distinct_keys(n, N as u64);
        let values = make_values::<N>(n);
        let map = RibbonMap::<N>::try_construct(&keys, &values)
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
            assert_roundtrip::<6>(n);
            assert_roundtrip::<12>(n);
        }
    }

    #[test]
    fn roundtrip_medium_sizes() {
        for n in [10_000usize, 100_000, 1_000_000] {
            assert_roundtrip::<6>(n);
        }
    }

    /// Value width is unbounded (values live in their own array), unlike a binary fuse map.
    #[test]
    fn roundtrip_wide_values() {
        assert_roundtrip::<32>(5000);
        assert_roundtrip::<0>(5000);
    }

    #[test]
    fn absent_keys_are_mostly_rejected() {
        // The 2-bit zero-check should reject ~3/4 of absent keys.
        let n = 200_000;
        let keys = distinct_keys(n, 11);
        let values = make_values::<6>(n);
        let map = RibbonMap::<6>::try_construct(&keys, &values).unwrap();

        let absent = distinct_keys(n, 0xABCD);
        let false_positives = absent.iter().filter(|&&k| map.get(k).is_some()).count();
        let rate = false_positives as f64 / n as f64;
        assert!(
            (0.20..0.30).contains(&rate),
            "false positive rate {rate} not close to 1/4"
        );
    }

    #[test]
    fn bits_per_key_is_reasonable() {
        let keys = distinct_keys(1_000_000, 7);
        let values = make_values::<4>(keys.len());
        let map = RibbonMap::<4>::try_construct(&keys, &values).unwrap();
        // ~1/0.85 slots/key * (1 + 4) bytes * 8 bits ≈ 47 bits/key.
        assert!(
            map.bits_per_key() < 50.0,
            "bits_per_key={} too high",
            map.bits_per_key()
        );
    }

    #[test]
    fn empty_map_is_empty() {
        let map = RibbonMap::<4>::try_construct(&[], &[]).unwrap();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert_eq!(map.bits_per_key(), 0.0);
    }

    #[test]
    fn construction_is_deterministic() {
        let keys = distinct_keys(10_000, 1);
        let values: Vec<[u8; 4]> = (0..keys.len() as u32).map(|i| i.to_le_bytes()).collect();
        let a = RibbonMap::<4>::try_construct(&keys, &values).unwrap();
        let b = RibbonMap::<4>::try_construct(&keys, &values).unwrap();
        assert_eq!(a.seed, b.seed);
        for k in &keys {
            assert_eq!(a.get(*k), b.get(*k));
        }
    }

    #[test]
    fn duplicate_keys_fail_to_construct() {
        // Repeated keys are linearly dependent for every seed, so construction cannot converge.
        let keys = vec![0x1111_2222_3333_4444u64; 100];
        let values = make_values::<4>(keys.len());
        assert!(RibbonMap::<4>::try_construct(&keys, &values).is_none());
    }
}
