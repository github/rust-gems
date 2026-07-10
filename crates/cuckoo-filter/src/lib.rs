//! A static **windowed cuckoo filter** over a known set of distinct `u64` keys.
//!
//! # Windowing
//!
//! A classic cuckoo filter gives every key two candidate *buckets* and stores several
//! fingerprint slots per bucket. This crate replaces the "several slots per bucket" with a
//! **window**: each of the two hash locations expands into [`WINDOW`] *consecutive* single slots
//! (from one shared array), so a key may live in any of `2 * WINDOW` slots. Overlapping windows of
//! neighbouring keys share slots, which yields the same cascading placement freedom that makes
//! binary fuse filters dense — a key that cannot fit in a crowded region can slide a few slots over.
//!
//! # Addressing
//!
//! A single hash is split with the classic cuckoo-filter XOR trick: the first window start is the
//! low `b` bits of the hash and the alternate is `w0 ^ offset` (`b = log2(num_windows)`, window
//! count a power of two). The `offset` uses only the low `min(b, SEGMENT_SHIFT)` bits, so **both
//! windows of a key fall in the same `2^SEGMENT_SHIFT`-slot segment** — each segment is thus an
//! independent windowed cuckoo. The fingerprint is a byte from the bits above the position bits.
//!
//! # Construction
//!
//! The key set is known up front, so construction radix-sorts the keys into their segments and then
//! fills each cache-resident segment with classic cuckoo **random-walk** (place into a free window
//! slot, else evict and relocate, bounded by a kick limit; retried with a fresh seed on failure).
//! Because a key's whole random walk stays inside one segment, the build touches only L2-resident
//! memory and its throughput stays roughly flat as the filter grows past cache. The per-segment
//! working table stores each key's hash directly in its slot, so a relocation reads the victim's
//! hash straight from the slot it swaps — no side array.
//!
//! # Membership
//!
//! Each slot is a fixed 8-byte word: the low [`PAYLOAD_BYTES`] bytes hold the value and the top
//! [`FP_BYTES`] bytes a fingerprint (`0` marks an empty slot). A lookup reads the whole word with one
//! aligned load, so [`CuckooFilter::get`] returns the payload straight from the word its fingerprint
//! matched, and [`CuckooFilter::contains`] rejects absent keys with a false-positive rate of about
//! `2 * WINDOW / 65535`.

/// Number of consecutive slots a single hash location expands into.
pub const WINDOW: usize = 2;

/// Bytes of each 8-byte slot word reserved for the fingerprint; the rest are payload. Two bytes give
/// a `~1/65535` per-slot false-positive rate — low enough that `get` is almost always exact.
pub const FP_BYTES: usize = 2;

/// Bytes of each 8-byte slot word available for the value payload (`N` must not exceed this).
pub const PAYLOAD_BYTES: usize = 8 - FP_BYTES;

/// Bit offset of the fingerprint within a slot word (the payload occupies the low bits below it).
const FP_SHIFT: u32 = PAYLOAD_BYTES as u32 * 8;

/// Maximum number of evictions a single insert may trigger before the attempt is abandoned (and
/// retried with a fresh seed).
const MAX_KICKS: usize = 500;

/// Sentinel stored in the working table for an empty slot during construction.
const EMPTY: u64 = u64::MAX;

/// A static windowed cuckoo **map** from a fixed set of distinct `u64` keys to `N`-byte values
/// (`N <= PAYLOAD_BYTES`).
///
/// It is approximate: an absent key is rejected with a false-positive rate of about `2 * WINDOW /
/// 65535`, and a false positive returns an arbitrary stored value. Use `N = 0` for a pure membership
/// filter (`[u8; 0]` values are zero-sized).
#[derive(Clone, Debug)]
pub struct CuckooFilter<const N: usize> {
    seed: u64,
    /// `log2` of the window count — the number of low hash bits used for the first window start.
    /// Cached so lookups avoid recomputing it from the slot count.
    b: u32,
    /// One `u64` per slot: the low [`PAYLOAD_BYTES`] bytes hold the value and the top [`FP_BYTES`]
    /// bytes the fingerprint (`0` marks an empty slot). Packing both into one aligned word lets a
    /// lookup read a slot with a single load and return its value straight from the matched word.
    slots: Vec<u64>,
    len: usize,
}

impl<const N: usize> CuckooFilter<N> {
    /// Builds a map from `keys` (assumed distinct) and their `values`, choosing the slot count
    /// automatically and retrying with fresh seeds until construction succeeds.
    ///
    /// Returns `None` only if every attempt fails, which is vanishingly unlikely at the default
    /// target load factor.
    pub fn construct(keys: &[u64], values: &[[u8; N]]) -> Option<Self> {
        // Windowing with `l = WINDOW` slots raises the load threshold with `l`, so `l = 2` sits
        // below `l = 4`. Target a safe margin below the random-walk threshold for this window size.
        let target = if WINDOW <= 2 { 0.88 } else { 0.92 };
        let slots = required_slots(keys.len(), target);
        (0..64).find_map(|seed| Self::try_construct(keys, values, slots, seed))
    }

    /// Attempts to build a map sized for about `slots` slots using a single `seed`. Returns `None`
    /// if the keys cannot be placed at this size and seed.
    ///
    /// The number of window positions is rounded up to a power of two (required by the XOR
    /// addressing), so the actual slot count is `next_pow2(slots - WINDOW + 1) + WINDOW - 1`.
    pub fn try_construct(
        keys: &[u64],
        values: &[[u8; N]],
        slots: usize,
        seed: u64,
    ) -> Option<Self> {
        assert_eq!(
            keys.len(),
            values.len(),
            "keys and values must be equal length"
        );
        assert!(
            N <= PAYLOAD_BYTES,
            "value width N must be at most PAYLOAD_BYTES ({PAYLOAD_BYTES})"
        );
        // The XOR addressing needs a power-of-two number of windows; round the requested size up.
        let num_windows = slots.saturating_sub(WINDOW - 1).max(2).next_power_of_two();
        let b = num_windows.trailing_zeros();
        let total = num_windows + WINDOW - 1;
        let n = keys.len();

        // Working table: each occupied slot's key hash (for recomputing windows on eviction) and its
        // value, which travel together on every swap. `EMPTY` marks a free slot.
        let mut htab = vec![EMPTY; total];
        let mut vtab = vec![[0u8; N]; total];
        if !fill_by_segment(&mut htab, &mut vtab, keys, values, b, seed) {
            return None;
        }

        // Pack each occupied slot into one aligned `u64`: the value in the low bytes and the
        // fingerprint in the top `FP_BYTES`, so a lookup gets both from a single word load.
        let mut slots = vec![0u64; total];
        for (slot, (&hash, value)) in htab.iter().zip(vtab.iter()).enumerate() {
            if hash != EMPTY {
                slots[slot] = pack_slot(fingerprint(hash, b), value);
            }
        }
        Some(Self {
            seed,
            b,
            slots,
            len: n,
        })
    }

    /// Looks up `key`, returning a value if the key is (probably) present.
    ///
    /// Membership has no false negatives (a present key always returns `Some`), but the map is
    /// **approximate on values**: a within-window fingerprint collision (probability
    /// `~2 * WINDOW / 65535`) can make it return *another* key's value — both for a present key and
    /// for a false-positive absent key. Use the exact `binary-fuse-map` or a `HashMap` if values must
    /// always be correct.
    #[inline]
    pub fn get(&self, key: u64) -> Option<[u8; N]> {
        let hash = stored_hash(key, self.seed);
        let (w0, w1) = windows(hash, self.b);
        let fp = fingerprint(hash, self.b);
        // Scan both windows without short-circuiting so their cache lines are fetched in parallel
        // (memory-level parallelism). A slot is one aligned word holding value and fingerprint, so a
        // match yields the value straight from the word — no second fetch.
        let m0 = window_match(&self.slots, w0, fp);
        let m1 = window_match(&self.slots, w1, fp);
        let word = if m1 != 0 { m1 } else { m0 };
        (word != 0).then(|| unpack_value::<N>(word))
    }

    /// Returns `true` if `key` is (probably) a member (a fingerprint-only membership test; use
    /// [`get`](Self::get) to also fetch the value).
    #[inline]
    pub fn contains(&self, key: u64) -> bool {
        let hash = stored_hash(key, self.seed);
        let (w0, w1) = windows(hash, self.b);
        let fp = fingerprint(hash, self.b);
        // Non-short-circuiting `|` so both window scans issue their loads together and the two cache
        // lines overlap in flight, hiding latency at large, out-of-cache sizes.
        (window_match(&self.slots, w0, fp) != 0) | (window_match(&self.slots, w1, fp) != 0)
    }

    /// Number of keys the map was built from.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the map holds no keys.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total number of slots (occupied plus empty).
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Fraction of slots that are occupied.
    #[inline]
    pub fn load_factor(&self) -> f64 {
        self.len as f64 / self.slots.len() as f64
    }

    /// Storage cost in bits per key (8 bytes per slot: the value plus a `FP_BYTES`-byte fingerprint).
    #[inline]
    pub fn bits_per_key(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        (self.slots.len() * 8 * 8) as f64 / self.len as f64
    }
}

/// The two window start indices for a mixed hash, via the classic cuckoo-filter XOR addressing,
/// constrained so **both windows fall in the same segment**.
///
/// `b = log2(num_windows)`: the first window is the low `b` bits and the alternate is `w0 ^ offset`,
/// where `offset` uses only the low `min(b, SEGMENT_SHIFT)` bits — so `w1` shares its high (segment)
/// bits with `w0`. Each segment is therefore an independent windowed cuckoo, which is what lets
/// construction resolve one cache-resident segment at a time (see [`fill_random_walk`]).
#[inline]
fn windows(hash: u64, b: u32) -> (usize, usize) {
    debug_assert!((1..64).contains(&b));
    let seg_bits = b.min(SEGMENT_SHIFT);
    let w0 = hash & ((1u64 << b) - 1);
    let offset = (hash >> b) & ((1u64 << seg_bits) - 1);
    let w1 = w0 ^ offset;
    (w0 as usize, w1 as usize)
}

/// The [`FP_BYTES`]-byte fingerprint of a mixed hash (never `0`), from the bits just above the
/// window-index and segment-offset bits, so it is independent of both window positions.
#[inline]
fn fingerprint(hash: u64, b: u32) -> u16 {
    let seg_bits = b.min(SEGMENT_SHIFT);
    ((hash >> (b + seg_bits)) as u16).max(1)
}

/// Packs a fingerprint and value into one slot word: value in the low bytes, fingerprint above them.
#[inline]
fn pack_slot<const N: usize>(fp: u16, value: &[u8; N]) -> u64 {
    let mut bytes = [0u8; 8];
    bytes[..N].copy_from_slice(value);
    u64::from_le_bytes(bytes) | ((fp as u64) << FP_SHIFT)
}

/// Extracts the `N`-byte value from the low bytes of a slot word.
#[inline]
fn unpack_value<const N: usize>(word: u64) -> [u8; N] {
    let mut out = [0u8; N];
    out.copy_from_slice(&word.to_le_bytes()[..N]);
    out
}

/// Scans the `WINDOW` slot words starting at `w` for the fingerprint `fp`, returning the matched word
/// (or `0` if none matches — an occupied slot's fingerprint is never `0`, so its word is never `0`).
///
/// A `(2, 2)` window is exactly two `u64` slots = 16 bytes, so it is read with one unaligned `u128`
/// load instead of two `u64` loads, halving the load-instruction count. The scan has no early exit,
/// so the two calls in [`CuckooFilter::get`]/[`contains`](CuckooFilter::contains) stay independent
/// and their cache lines overlap in flight. Because a slot word already holds the value, the matched
/// word yields it for free.
#[inline]
fn window_match(slots: &[u64], w: usize, fp: u16) -> u64 {
    // This reads a window as one `u128`, which is exactly two slots.
    const _: () = assert!(WINDOW == 2, "window_match reads a u128 = exactly two slots");
    debug_assert!(w + WINDOW <= slots.len());
    // SAFETY: a key's window always covers two in-bounds slots (`try_construct` sizes the table with
    // `WINDOW - 1` trailing slots), and `read_unaligned` imposes no alignment requirement.
    let pair = unsafe { std::ptr::read_unaligned(slots.as_ptr().add(w).cast::<u128>()) };
    let lo = pair as u64;
    let hi = (pair >> 64) as u64;
    let mut found = 0u64;
    if (lo >> FP_SHIFT) as u16 == fp {
        found = lo;
    }
    if (hi >> FP_SHIFT) as u16 == fp {
        found = hi;
    }
    found
}

/// Tries to place `(hash, value)` into the first free slot of either of its two windows.
#[inline]
fn place<const N: usize>(
    htab: &mut [u64],
    vtab: &mut [[u8; N]],
    w0: usize,
    w1: usize,
    hash: u64,
    value: [u8; N],
) -> bool {
    for i in w0..w0 + WINDOW {
        if htab[i] == EMPTY {
            htab[i] = hash;
            vtab[i] = value;
            return true;
        }
    }
    for i in w1..w1 + WINDOW {
        if htab[i] == EMPTY {
            htab[i] = hash;
            vtab[i] = value;
            return true;
        }
    }
    false
}

/// Inserts one entry (a key's stored `hash` and its `value`) by classic cuckoo random-walk: place it
/// in a free slot of either window, else evict a random incumbent and relocate it, up to
/// [`MAX_KICKS`] times. The victim's hash and value come straight from the slot being swapped.
#[inline]
fn insert_entry<const N: usize>(
    htab: &mut [u64],
    vtab: &mut [[u8; N]],
    b: u32,
    rng: &mut u64,
    hash: u64,
    value: [u8; N],
) -> bool {
    let (w0, w1) = windows(hash, b);
    if place(htab, vtab, w0, w1, hash, value) {
        return true;
    }
    let mut cur_hash = hash;
    let mut cur_value = value;
    for _ in 0..MAX_KICKS {
        let (w0, w1) = windows(cur_hash, b);
        let r = splitmix64(rng) as usize % (2 * WINDOW);
        let slot = if r < WINDOW {
            w0 + r
        } else {
            w1 + (r - WINDOW)
        };
        std::mem::swap(&mut htab[slot], &mut cur_hash);
        std::mem::swap(&mut vtab[slot], &mut cur_value);
        let (cw0, cw1) = windows(cur_hash, b);
        if place(htab, vtab, cw0, cw1, cur_hash, cur_value) {
            return true;
        }
    }
    false
}

/// Number of consecutive slots grouped into one cache-locality segment, as a bit shift: a segment
/// is `2^SEGMENT_SHIFT` slots — ~512 KiB of the `u64` working table, an L2-resident chunk.
const SEGMENT_SHIFT: u32 = 16;

/// Number of slots in one segment.
const SEGMENT_SIZE: usize = 1 << SEGMENT_SHIFT;

/// The segment a hash belongs to: the bits of its primary window above [`SEGMENT_SHIFT`].
#[inline]
fn segment_of(hash: u64, b: u32) -> usize {
    ((hash & ((1u64 << b) - 1)) >> SEGMENT_SHIFT) as usize
}

/// Fills the working table by first scattering every key into its cache-sized segment — packed into
/// the front of that segment's region of the table itself — and then fixing each segment in place
/// with the cuckoo random walk, so each segment is resolved while resident in cache.
///
/// This needs neither a precomputed hash array nor a counting pass, and no scratch table beyond one
/// reused segment buffer: the working table is the space the keys will occupy anyway. A segment holds
/// only `SEGMENT_SIZE` slots, so if more than that many keys fall into one the table cannot hold them
/// and construction fails (we bail so the caller retries with a fresh seed).
fn fill_by_segment<const N: usize>(
    htab: &mut [u64],
    vtab: &mut [[u8; N]],
    keys: &[u64],
    values: &[[u8; N]],
    b: u32,
    seed: u64,
) -> bool {
    let mut rng = seed ^ 0xD1B5_4A32_D192_ED03;
    let num_segments = 1usize << b.saturating_sub(SEGMENT_SHIFT);
    if num_segments == 1 {
        // A single cache-resident segment: place keys directly, in their natural order.
        return keys.iter().zip(values).all(|(&key, &value)| {
            insert_entry(htab, vtab, b, &mut rng, stored_hash(key, seed), value)
        });
    }

    // Phase 1 — distribute: append each key into the front of its segment's region of the working
    // table, tracking only a per-segment count. Bail if a segment would overflow its slots.
    let mut counts = vec![0u32; num_segments];
    for (&key, &value) in keys.iter().zip(values) {
        let hash = stored_hash(key, seed);
        let s = segment_of(hash, b);
        let c = counts[s] as usize;
        if c >= SEGMENT_SIZE {
            return false;
        }
        htab[(s << SEGMENT_SHIFT) + c] = hash;
        vtab[(s << SEGMENT_SHIFT) + c] = value;
        counts[s] = c as u32 + 1;
    }

    // Phase 2 — fix each segment in place, from the top down. A key's windows lie in its own segment
    // and a window is `WINDOW` slots wide, so a placement can only spill into the *next* segment.
    // Resolving segments high-to-low therefore means every spill lands in an already-fixed region the
    // cuckoo can absorb, never on another segment's not-yet-placed packed keys. Each segment's keys
    // are lifted into a small reused buffer, its region cleared, and then re-inserted.
    let mut buf_hash = vec![EMPTY; *counts.iter().max().unwrap_or(&0) as usize];
    let mut buf_value = vec![[0u8; N]; buf_hash.len()];
    for s in (0..num_segments).rev() {
        let base = s << SEGMENT_SHIFT;
        let count = counts[s] as usize;
        buf_hash[..count].copy_from_slice(&htab[base..base + count]);
        buf_value[..count].copy_from_slice(&vtab[base..base + count]);
        htab[base..base + count].fill(EMPTY);
        for i in 0..count {
            if !insert_entry(htab, vtab, b, &mut rng, buf_hash[i], buf_value[i]) {
                return false;
            }
        }
    }
    true
}

/// Total number of slots needed to reach roughly `target` load with `n` keys, sized so the window
/// count is a power of two.
fn required_slots(n: usize, target: f64) -> usize {
    let num_windows = if n == 0 {
        8
    } else {
        ((n as f64 / target).ceil() as usize)
            .next_power_of_two()
            .max(8)
    };
    num_windows + WINDOW - 1
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

/// A key's mixed hash, remapped away from the `EMPTY` sentinel so it can be stored in a slot. The
/// remap is applied identically on construction and lookup, so it never causes a false negative.
#[inline]
fn stored_hash(key: u64, seed: u64) -> u64 {
    let h = mix(key, seed);
    if h == EMPTY {
        0
    } else {
        h
    }
}

/// SplitMix64 PRNG step, used to pick eviction slots and construction seeds.
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
        let mut state = 0x0123_4567_89ab_cdef ^ salt;
        (0..n).map(|_| splitmix64(&mut state)).collect()
    }

    /// A distinct 4-byte value per key (its low 32 bits), so lookups can be checked exactly.
    fn values_for(keys: &[u64]) -> Vec<[u8; 4]> {
        keys.iter().map(|&k| (k as u32).to_le_bytes()).collect()
    }

    #[test]
    fn roundtrip() {
        // A cuckoo map is exact on *membership* and, with a 2-byte fingerprint, almost exact on
        // *values*: only a within-window fingerprint collision (~1/65535 per neighbour) returns a
        // wrong value, so essentially every key reads back correctly.
        let keys = distinct_keys(20_000, 3);
        let values = values_for(&keys);
        let map = CuckooFilter::construct(&keys, &values).expect("construction succeeds");
        assert_eq!(map.len(), keys.len());
        assert!(keys.iter().all(|&k| map.get(k).is_some()));
        let correct = keys
            .iter()
            .zip(&values)
            .filter(|(&k, &v)| map.get(k) == Some(v))
            .count();
        assert!(
            correct as f64 / keys.len() as f64 > 0.999,
            "too many wrong values: {correct}/{}",
            keys.len()
        );
    }

    #[test]
    fn roundtrip_multi_segment() {
        // Large enough that the table spans many segments, exercising the blocked construction and
        // the segment-local addressing (both windows in one segment).
        let keys = distinct_keys(500_000, 5);
        let values = values_for(&keys);
        let map = CuckooFilter::construct(&keys, &values).expect("construction succeeds");
        assert_eq!(map.len(), keys.len());
        assert!(keys.iter().all(|&k| map.get(k).is_some()));
        let correct = keys
            .iter()
            .zip(&values)
            .filter(|(&k, &v)| map.get(k) == Some(v))
            .count();
        assert!(
            correct as f64 / keys.len() as f64 > 0.999,
            "too many wrong values"
        );
        let absent = distinct_keys(500_000, 6);
        let hits = absent.iter().filter(|&&k| map.contains(k)).count();
        assert!(
            hits < absent.len() / 100,
            "too many false positives: {hits}"
        );
    }

    #[test]
    fn absent_keys_are_mostly_rejected() {
        let keys = distinct_keys(50_000, 1);
        let values = values_for(&keys);
        let map = CuckooFilter::construct(&keys, &values).unwrap();
        let absent = distinct_keys(50_000, 2);
        let hits = absent.iter().filter(|&&k| map.get(k).is_some()).count();
        // False-positive rate is about 2 * WINDOW / 65535, so hits should be a tiny fraction.
        assert!(
            hits < absent.len() / 100,
            "too many false positives: {hits}"
        );
    }

    #[test]
    fn load_factor_is_reasonable() {
        // Window counts are rounded up to a power of two, so for an unlucky `n` the achieved load
        // can be as low as ~half the target. Assert only that robust lower bound here.
        let keys = distinct_keys(10_000, 7);
        let values = values_for(&keys);
        let map = CuckooFilter::construct(&keys, &values).unwrap();
        assert!(
            map.load_factor() > 0.40,
            "unexpectedly sparse: {}",
            map.load_factor()
        );
        assert!(map.load_factor() <= 1.0);
    }

    #[test]
    fn dense_construction() {
        // A power-of-two table filled to 0.90 still constructs via random-walk.
        let slots = 8192;
        let n = (slots as f64 * 0.90) as usize;
        let keys = distinct_keys(n, 99);
        let values = values_for(&keys);
        let map = (0..16)
            .find_map(|s| CuckooFilter::try_construct(&keys, &values, slots, s))
            .expect("constructs at load 0.90");
        assert!(keys.iter().all(|&k| map.get(k).is_some()));
    }

    #[test]
    fn empty_filter() {
        let map: CuckooFilter<4> = CuckooFilter::construct(&[], &[]).unwrap();
        assert!(map.is_empty());
        assert_eq!(map.get(123), None);
    }

    #[test]
    fn pure_filter_zero_value() {
        // `N = 0` is a plain membership filter.
        let keys = distinct_keys(10_000, 11);
        let values = vec![[0u8; 0]; keys.len()];
        let filter = CuckooFilter::construct(&keys, &values).unwrap();
        assert!(keys.iter().all(|&k| filter.contains(k)));
    }
}
