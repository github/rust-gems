//! Per-key consistent permutation iterator over `0..n`, driven by an
//! independently-keyed Feistel network per layer (2 bits per layer).
//!
//! Given a 64-bit `seed` and a universe size `n`, the iterator emits
//! each integer in `0..n` exactly once in a key-dependent order. The
//! sequence is `n`-consistent: when `n` grows by one, the new sequence
//! agrees with the old except for the at most one position where the
//! new element is inserted.
//!
//! # Layered design
//!
//! Each layer `j` is a Feistel bijection on `[0, 2^(2j + 2))`. After
//! one Feistel pass we look at the **top two bits** of the output:
//!
//! * `00` (one quadrant) → descend to layer `j - 1` (or, at the
//!   bottom layer, the output is `0` itself, which we emit).
//! * `01` / `10` / `11` (three quadrants) → emit the raw value
//!   (subject to the `< n` range check).
//!
//! So every layer-`j` call has a `3/4` chance of producing an emission,
//! versus `1/2` in a one-bit-per-layer scheme. That cuts both the
//! number of layers (`j_max ≈ ⌈log₂(n)/2⌉ - 1`) and the expected
//! number of `layer_apply` calls per emission (from `≈ 2` to `≈ 4/3`).
//! Even `n_bits` also means the Feistel never needs cycle-walking.
//!
//! See [`docs/permutation-design.md`] in the crate for the full
//! design rationale, including why an independently-keyed per-layer
//! Feistel network beats the (now removed) chunk-prefix-permutation
//! approach on choose-2 set uniformity at small `n` while staying
//! cheaper than multi-round PCG.
//!
//! [`docs/permutation-design.md`]: ../../docs/permutation-design.md
//!
//! # Round-count policy
//!
//! The number of Feistel rounds is chosen automatically per layer as a
//! function of `n_bits` (see [`rounds_for_n_bits`]). Small layers have
//! a tiny F-function input space (`2^half_bits` distinct values),
//! which limits the per-round permutation family; they compensate with
//! extra rounds. Large layers already see a wide F input and reach the
//! random-perm floor with `4` rounds. Conveniently the small layers
//! are called rarely (level `j` contributes `≈ 4^j` emissions), so the
//! extra rounds there are essentially free, while the top layer —
//! which dominates per-iterator cost — uses the fewest rounds.
//!
//! # Iterator
//!
//! Start at the top layer `j_max` and walk its counter. If the top
//! two bits are `00`, descend one layer (and walk *that* layer's
//! counter on the next step); otherwise emit. The iterator stops once
//! it has emitted `n` values.

/// Smallest Feistel round count we trust to deliver near-random-perm
/// statistics for a layer of width `n_bits`, based on the
/// `statistical_comparison` chi² experiments. Tiny layers
/// (`n_bits = 2`) have a F-input space of only 2 values and need
/// many rounds; large layers reach the floor with 4 rounds.
pub(crate) fn rounds_for_n_bits(n_bits: u32) -> usize {
    match n_bits {
        2 => 12,
        4 => 10,
        6 => 6,
        _ => 4,
    }
}

/// SplitMix64 — strong per-bit avalanche, useful as a seed
/// pre-mixer for callers that only have a low-entropy seed.
#[cfg(test)]
#[inline]
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Apply the per-layer Feistel bijection on `[0, 2^n_bits)`.
/// `master_key` must be well-avalanched: two near-identical keys
/// will yield two highly correlated permutations.
///
/// All per-layer geometry (`half_bits`, mask, round count) is derived
/// from `n_bits` on each call. `n_bits` is small (≤ 32) and the
/// constants are a handful of trivial bit ops, so this stays well off
/// the hot path of the Feistel loop itself.
///
/// `n_bits` must be **even** and in `2..=30`.
#[inline]
pub(crate) fn layer_apply(n_bits: u32, master_key: u64, x: u32) -> u32 {
    debug_assert!(n_bits >= 2 && n_bits <= 30 && n_bits.is_multiple_of(2));
    let rounds = rounds_for_n_bits(n_bits) as u32;
    let half_bits = n_bits / 2;
    let half_mask = (1u32 << half_bits) - 1;
    let n_mask = (1u32 << n_bits) - 1;
    let shift = 2 * half_bits;

    debug_assert!(x <= n_mask, "input out of range");
    let mut l = (x >> half_bits) & half_mask;
    let mut r = x & half_mask;
    let mut k = master_key;
    for _ in 0..rounds {
        // Split each 64-bit (rotated) master into two *independent*
        // 32-bit sub-keys: the low half is an additive offset, the
        // high half (forced odd) is the multiplier. Higher bits stay
        // set on purpose — they contribute via the multiplicative
        // mix.
        let k_xor = k & 0xFFFF_FFFF;
        let k_mul = (k >> 32) | 1;
        let mixed = ((r as u64) ^ k_xor).wrapping_mul(k_mul);
        let f = (mixed as u32).wrapping_add((mixed >> 32) as u32) & half_mask;
        let new_l = r;
        let new_r = l ^ f;
        l = new_l;
        r = new_r;
        // Rotate the master key by exactly the per-round "consumed"
        // bit count so that each round sees a fresh slice of the
        // master in its low positions, plus a Weyl increment to
        // decorrelate consecutive rounds (rotation alone leaves
        // 60+ bits in common between adjacent rounds).
        k = k.rotate_right(shift).wrapping_add(0x9E37_79B9_7F4A_7C15);
    }
    ((l << half_bits) | r) & n_mask
}

/// `n`-consistent permutation iterator over `0..n` driven by one
/// bijection per level (see module docs).
pub struct ConsistentPermutation {
    /// Master key driving every layer's Feistel. Must be supplied
    /// already avalanched (see [`ConsistentPermutation::new`]).
    /// Layers stay distinct because their geometry
    /// (`n_bits`, `half_bits`, rotation amount, round count) differs.
    master_key: u64,
    /// `counters[j]` is the next index to feed into the layer-`j`
    /// bijection. Walks the full domain `[0, 2^(2j + 2))`. With
    /// `n <= 2^30`, `j_max <= 14` so the top counter caps at
    /// `2^30 <= u32::MAX`.
    counters: Vec<u32>,
    n: u32,
    /// Top layer index. Layer `j` has `n_bits = 2j + 2`.
    j_max: u32,
    /// `1 << (2 * j_max + 2)` — the top layer's domain size. Used as
    /// the iterator's termination signal: once `counters[j_max]`
    /// reaches `top_cap`, every walk has completed and exactly `n`
    /// values have been emitted.
    top_cap: u32,
}

impl ConsistentPermutation {
    /// Construct an iterator over `0..n` from a single 64-bit
    /// `master_key`. Each layer's Feistel round count is chosen
    /// automatically by [`rounds_for_n_bits`] given that layer's bit
    /// width.
    ///
    /// `n` must satisfy `1 <= n <= 2^30`.
    ///
    /// `master_key` is used directly as the Feistel key — the
    /// constructor does **not** avalanche it. Pass a high-entropy
    /// value (e.g. the output of a cryptographic or strong
    /// non-cryptographic hash like SplitMix64); sequential or
    /// low-entropy keys will produce visibly correlated permutations
    /// across consecutive iterators.
    pub fn new(n: u32, master_key: u64) -> Self {
        assert!(n > 0, "n must be at least 1");
        assert!(n <= 1u32 << 30, "n must be at most 2^30");
        // Smallest `j_max` such that `4^(j_max + 1) >= n`. For
        // `n >= 2`, `(n-1).ilog2() = ceil(log2(n)) - 1`, so dividing
        // by 2 gives `ceil(ceil(log2(n)) / 2) - 1` — the right value
        // for both even and odd bit-widths.
        let j_max = if n <= 1 { 0 } else { (n - 1).ilog2() / 2 };
        let counters = vec![0u32; (j_max + 1) as usize];
        let top_cap = 1u32 << (2 * j_max + 2);
        Self {
            master_key,
            counters,
            n,
            j_max,
            top_cap,
        }
    }

    /// Universe size.
    pub fn n(&self) -> u32 {
        self.n
    }
}

impl Iterator for ConsistentPermutation {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        let j_max = self.j_max;
        let n = self.n;
        let mk = self.master_key;

        // Phase 1: walk the top layer (where skips and exhaustion
        // happen) until we either emit an in-range value or descend.
        let n_bits_top = 2 * j_max + 2;
        let top_shift = 2 * j_max;
        let mut j;
        loop {
            let counter = self.counters[j_max as usize];
            if counter >= self.top_cap {
                return None;
            }
            let raw = layer_apply(n_bits_top, mk, counter);
            self.counters[j_max as usize] = counter + 1;
            let top_bits = (raw >> top_shift) & 0b11;
            if top_bits != 0 {
                // `raw` lies in `[4^j_max, 4^(j_max+1))`. May be out
                // of range (only the top layer can exceed `n`).
                if raw < n {
                    return Some(raw);
                }
                continue;
            }
            // Descend. At `j_max == 0` the top layer *is* the bottom
            // layer and `top_bits == 0` means `raw == 0`.
            if j_max == 0 {
                return Some(0);
            }
            j = j_max - 1;
            break;
        }

        // Phase 2: descend through lower layers. Their counters can
        // never exhaust within a single walk, and their raw values
        // are always in `[4^j, 4^(j+1)) ⊂ [0, n)`, so neither
        // exhaustion nor range checks are needed.
        loop {
            let counter = self.counters[j as usize];
            let n_bits = 2 * j + 2;
            let raw = layer_apply(n_bits, mk, counter);
            self.counters[j as usize] = counter + 1;
            let top_shift = 2 * j;
            let top_bits = (raw >> top_shift) & 0b11;
            if top_bits != 0 {
                return Some(raw);
            }
            if j == 0 {
                return Some(0);
            }
            j -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    /// Across many seeds, `layer_apply(_, key, 0)` must not always
    /// collapse to the same value (regression test for the "F(0, k) =
    /// 0" symmetry bug).
    #[test]
    fn layer_apply_zero_input_is_key_sensitive() {
        for n_bits in [2u32, 4, 6, 8, 10] {
            let outputs: HashSet<u32> = (0..1000u64)
                .map(|seed| layer_apply(n_bits, splitmix64(seed), 0))
                .collect();
            let domain = 1u32 << n_bits;
            assert!(
                outputs.len() >= (domain as usize / 2).min(50),
                "n_bits={n_bits}: only {} distinct values from layer_apply(_, _, 0) across 1000 seeds",
                outputs.len()
            );
        }
    }

    /// `layer_apply` must be a bijection on `[0, 2^n_bits)` for every
    /// even supported `n_bits`.
    #[test]
    fn layer_apply_is_bijection() {
        for n_bits in [2u32, 4, 6, 8, 10] {
            for seed in 0..16u64 {
                let key = splitmix64(seed);
                let domain: u32 = 1 << n_bits;
                let outputs: HashSet<u32> =
                    (0..domain).map(|x| layer_apply(n_bits, key, x)).collect();
                assert_eq!(
                    outputs.len(),
                    domain as usize,
                    "n_bits={n_bits} seed={seed}: not a bijection"
                );
                for &o in &outputs {
                    assert!(
                        o < domain,
                        "n_bits={n_bits} seed={seed}: output {o} out of range"
                    );
                }
            }
        }
    }

    /// Collecting the iterator must yield values in `0..n` without
    /// duplicates, and must yield **exactly** `n` of them.
    #[test]
    fn no_duplicates_within_universe() {
        for seed in 0..32u64 {
            for n in [1u32, 2, 3, 4, 5, 7, 8, 9, 16, 17, 31, 32, 33, 100, 1000] {
                let iter = ConsistentPermutation::new(n, 0x9E37_79B9_7F4A_7C15 ^ seed);
                let emitted: Vec<u32> = iter.collect();
                assert_eq!(
                    emitted.len(),
                    n as usize,
                    "seed={seed} n={n}: wrong emission count: {emitted:?}"
                );
                assert!(
                    emitted.iter().all(|&v| v < n),
                    "seed={seed} n={n}: out-of-range value emitted: {emitted:?}"
                );
                let unique: HashSet<u32> = emitted.iter().copied().collect();
                assert_eq!(
                    unique.len(),
                    emitted.len(),
                    "seed={seed} n={n}: duplicate emitted: {emitted:?}"
                );
            }
        }
    }
}
