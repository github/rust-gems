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
//! low `b` bits of the hash, the alternate is `w0 ^ (top b bits)`, and the fingerprint is a middle
//! byte (`b = log2(num_windows)`). This requires the window count to be a power of two.
//!
//! # Construction
//!
//! Because the key set is known up front, construction tracks the *owning key* of every slot. Keys
//! are placed by classic cuckoo **random-walk**: put each key into a free slot of one of its
//! windows, evicting and relocating incumbents on collision (bounded by a kick limit, and retried
//! with a fresh seed on failure).
//!
//! # Membership
//!
//! Only 1-byte fingerprints are stored (`0` marks an empty slot). [`CuckooFilter::contains`] scans
//! a key's two windows for its fingerprint, so absent keys are rejected with a false-positive rate
//! of about `2 * WINDOW / 255`.

/// Number of consecutive slots a single hash location expands into.
pub const WINDOW: usize = 4;

/// Maximum number of evictions a single insert may trigger before the attempt is abandoned (and
/// retried with a fresh seed).
const MAX_KICKS: usize = 500;

/// Sentinel stored in the owner map for an empty slot during construction.
const NO_KEY: u32 = u32::MAX;

/// A static windowed cuckoo filter mapping a fixed set of distinct `u64` keys to membership.
#[derive(Clone, Debug)]
pub struct CuckooFilter {
    seed: u64,
    /// Fingerprint slots; `0` marks an empty slot. Each key has two windows of `WINDOW` consecutive
    /// slots (from two hashes) that may start anywhere in the shared array and may overlap.
    slots: Vec<u8>,
    len: usize,
}

impl CuckooFilter {
    /// Builds a filter from `keys` (assumed distinct), choosing the slot count automatically and
    /// retrying with fresh seeds until construction succeeds.
    ///
    /// Returns `None` only if every attempt fails, which is vanishingly unlikely at the default
    /// target load factor.
    pub fn construct(keys: &[u64]) -> Option<Self> {
        // Windowing with `l = WINDOW` slots raises the load threshold with `l`, so `l = 2` sits
        // below `l = 4`. Target a safe margin below the random-walk threshold for this window size.
        let target = if WINDOW <= 2 { 0.88 } else { 0.92 };
        let slots = required_slots(keys.len(), target);
        (0..64).find_map(|seed| Self::try_construct(keys, slots, seed))
    }

    /// Attempts to build a filter sized for about `slots` fingerprint slots using a single `seed`.
    /// Returns `None` if the keys cannot be placed at this size and seed.
    ///
    /// The number of window positions is rounded up to a power of two (required by the XOR
    /// addressing), so the actual slot count is `next_pow2(slots - WINDOW + 1) + WINDOW - 1`. A
    /// key's two windows of `WINDOW` consecutive slots may overlap.
    pub fn try_construct(keys: &[u64], slots: usize, seed: u64) -> Option<Self> {
        // The XOR addressing needs a power-of-two number of windows; round the requested size up.
        let num_windows = slots.saturating_sub(WINDOW - 1).max(2).next_power_of_two();
        let total = num_windows + WINDOW - 1;
        let n = keys.len();

        // Pre-compute both window starts and the fingerprint of every key.
        let mut win0 = vec![0u32; n];
        let mut win1 = vec![0u32; n];
        let mut fps = vec![0u8; n];
        for (i, &key) in keys.iter().enumerate() {
            let (w0, w1, fp) = locate(key, seed, num_windows);
            win0[i] = w0 as u32;
            win1[i] = w1 as u32;
            fps[i] = fp;
        }

        let mut owner = vec![NO_KEY; total];
        if !fill_random_walk(&mut owner, &win0, &win1, n, seed) {
            return None;
        }

        let mut slot_bytes = vec![0u8; total];
        for (slot, &key) in owner.iter().enumerate() {
            if key != NO_KEY {
                slot_bytes[slot] = fps[key as usize];
            }
        }
        Some(Self {
            seed,
            slots: slot_bytes,
            len: n,
        })
    }

    /// Returns `true` if `key` is (probably) a member. Present keys always return `true`; absent
    /// keys return `true` with probability about `2 * WINDOW / 255` (a fingerprint collision).
    #[inline]
    pub fn contains(&self, key: u64) -> bool {
        let num_windows = self.slots.len() - WINDOW + 1;
        let (w0, w1, fp) = locate(key, self.seed, num_windows);
        // Non-short-circuiting `|` so both window loads are issued together: the two cache lines are
        // fetched in parallel (memory-level parallelism), hiding latency at large, out-of-cache sizes.
        window_contains(&self.slots, w0, fp) | window_contains(&self.slots, w1, fp)
    }

    /// Number of keys the filter was built from.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the filter holds no keys.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total number of fingerprint slots (occupied plus empty).
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Fraction of slots that are occupied.
    #[inline]
    pub fn load_factor(&self) -> f64 {
        self.len as f64 / self.slots.len() as f64
    }

    /// Storage cost in bits per key (one byte per slot).
    #[inline]
    pub fn bits_per_key(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        (self.slots.len() * 8) as f64 / self.len as f64
    }
}

/// Computes a key's two window start indices and its fingerprint from a **single hash**, using the
/// classic cuckoo-filter XOR addressing.
///
/// With `b = log2(num_windows)`, the first window is the low `b` bits of the hash and the alternate
/// window is `w0 ^ (top b bits)`; the fingerprint is a byte from the middle bits. This requires
/// `num_windows` to be a power of two so the XOR result stays in `[0, num_windows)`. It replaces the
/// previous two independent hashes with one hash and a few bit ops — cheaper per lookup — at the
/// cost of the power-of-two sizing (see `required_slots`).
#[inline]
fn locate(key: u64, seed: u64, num_windows: usize) -> (usize, usize, u8) {
    debug_assert!(num_windows.is_power_of_two() && num_windows >= 2);
    let h = mix(key, seed);
    let b = num_windows.trailing_zeros();
    let mask = num_windows as u64 - 1;
    let w0 = h & mask;
    let w1 = w0 ^ (h >> (64 - b));
    // Fingerprint from bits `[b, b + 8)`, disjoint from both the low `b` and top `b` position bits.
    let fp = ((h >> b) as u8).max(1);
    (w0 as usize, w1 as usize, fp)
}

/// Branch-free test of whether the `WINDOW` slots starting at `w` hold the fingerprint `fp`.
///
/// Using `|=` (no early exit) keeps the scan a straight-line load of the whole window, so the two
/// calls in [`CuckooFilter::contains`] stay independent and their cache lines overlap in flight.
#[inline]
fn window_contains(slots: &[u8], w: usize, fp: u8) -> bool {
    let mut found = false;
    for &b in &slots[w..w + WINDOW] {
        found |= b == fp;
    }
    found
}

/// Tries to place `key` into the first free slot of either of its two windows.
#[inline]
fn place(owner: &mut [u32], w0: usize, w1: usize, key: u32) -> bool {
    for slot in owner[w0..w0 + WINDOW].iter_mut() {
        if *slot == NO_KEY {
            *slot = key;
            return true;
        }
    }
    for slot in owner[w1..w1 + WINDOW].iter_mut() {
        if *slot == NO_KEY {
            *slot = key;
            return true;
        }
    }
    false
}

/// Inserts one key by classic cuckoo random-walk: place it in a free slot of either window, else
/// evict a random incumbent and relocate it, up to [`MAX_KICKS`] times. Returns `false` on failure.
#[inline]
fn insert_key(owner: &mut [u32], win0: &[u32], win1: &[u32], rng: &mut u64, key: u32) -> bool {
    if place(
        owner,
        win0[key as usize] as usize,
        win1[key as usize] as usize,
        key,
    ) {
        return true;
    }
    let mut cur = key;
    for _ in 0..MAX_KICKS {
        let w0 = win0[cur as usize] as usize;
        let w1 = win1[cur as usize] as usize;
        let r = splitmix64(rng) as usize % (2 * WINDOW);
        let slot = if r < WINDOW {
            w0 + r
        } else {
            w1 + (r - WINDOW)
        };
        std::mem::swap(&mut owner[slot], &mut cur);
        if place(
            owner,
            win0[cur as usize] as usize,
            win1[cur as usize] as usize,
            cur,
        ) {
            return true;
        }
    }
    false
}

/// Fills the filter by inserting every key with classic cuckoo random-walk.
///
/// (Radix-partitioning the keys by window segment for cache locality was tried and made no
/// difference even at tens of millions of keys: at these load factors most inserts evict, and the
/// eviction chains jump across the table regardless of the order keys are fed in.)
fn fill_random_walk(owner: &mut [u32], win0: &[u32], win1: &[u32], n: usize, seed: u64) -> bool {
    let mut rng = seed ^ 0xD1B5_4A32_D192_ED03;
    (0..n as u32).all(|key| insert_key(owner, win0, win1, &mut rng, key))
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

    #[test]
    fn roundtrip() {
        let keys = distinct_keys(20_000, 3);
        let filter = CuckooFilter::construct(&keys).expect("construction succeeds");
        assert_eq!(filter.len(), keys.len());
        assert!(keys.iter().all(|&k| filter.contains(k)));
    }

    #[test]
    fn absent_keys_are_mostly_rejected() {
        let keys = distinct_keys(50_000, 1);
        let filter = CuckooFilter::construct(&keys).unwrap();
        let absent = distinct_keys(50_000, 2);
        let hits = absent.iter().filter(|&&k| filter.contains(k)).count();
        // False-positive rate is about 2 * WINDOW / 255; allow generous slack.
        assert!(hits < absent.len() / 10, "too many false positives: {hits}");
    }

    #[test]
    fn load_factor_is_reasonable() {
        // Window counts are rounded up to a power of two, so for an unlucky `n` the achieved load
        // can be as low as ~half the target. Assert only that robust lower bound here.
        let keys = distinct_keys(10_000, 7);
        let filter = CuckooFilter::construct(&keys).unwrap();
        assert!(
            filter.load_factor() > 0.40,
            "unexpectedly sparse: {}",
            filter.load_factor()
        );
        assert!(filter.load_factor() <= 1.0);
    }

    #[test]
    fn dense_construction() {
        // A power-of-two table filled to 0.90 still constructs via random-walk.
        let slots = 8192;
        let n = (slots as f64 * 0.90) as usize;
        let keys = distinct_keys(n, 99);
        let filter = (0..16)
            .find_map(|s| CuckooFilter::try_construct(&keys, slots, s))
            .expect("constructs at load 0.90");
        assert!(keys.iter().all(|&k| filter.contains(k)));
    }

    #[test]
    fn empty_filter() {
        let filter = CuckooFilter::construct(&[]).unwrap();
        assert!(filter.is_empty());
        assert!(!filter.contains(123));
    }
}
