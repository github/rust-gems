//! A simple, uncalibrated similarity metric for [`GeoDiffCount`] filters based on the exact number
//! of differing one-bits (the Hamming distance between their bit representations).

use std::cmp::Ordering;
use std::fmt;
use std::ops::Add;

use super::config::{estimate_count_approx, expected_diff_buckets_approx};
use crate::config::{xor_bit_chunks, BitChunk, GeoConfig, IsBucketType, BITS_PER_BLOCK};
use crate::diff_count::GeoDiffCount;
use crate::{Diff, Metric, MetricSpace};

/// A metric value measured in one-bits: the number of set bits of a filter, or the Hamming distance
/// between two filters.
///
/// The value is stored in the same (small) integer type `C::BucketType` used for the filter's
/// most-significant bucket positions, and is tagged with the configuration `C`, so metrics of
/// differently configured filters cannot be mixed. The largest representable value doubles as the
/// [`Metric::infinite`] sentinel; construction saturates into it.
pub struct OnesMetric<C: GeoConfig<Diff>>(C::BucketType);

impl<C: GeoConfig<Diff>> OnesMetric<C> {
    /// Wraps a raw one-bit count, saturating at [`Metric::infinite`].
    fn new(count: usize) -> Self {
        let max = C::BucketType::from_usize(usize::MAX).into_usize();
        Self(C::BucketType::from_usize(count.min(max)))
    }

    /// The wrapped one-bit count.
    fn get(self) -> usize {
        self.0.into_usize()
    }
}

impl<C: GeoConfig<Diff> + Default> Metric for OnesMetric<C> {
    fn zero() -> Self {
        Self(C::BucketType::from_usize(0))
    }

    fn infinite() -> Self {
        Self(C::BucketType::from_usize(usize::MAX))
    }

    fn abs_diff(&self, other: &Self) -> Self {
        Self::new(self.get().abs_diff(other.get()))
    }

    /// The calibrated size (item count) that this number of one-bits represents, obtained by
    /// inverting the closed-form approximation of the expected-buckets function.
    fn to_f32(&self) -> f32 {
        if *self == Self::infinite() {
            return f32::INFINITY;
        }
        let phi = C::default().phi_f64();
        estimate_count_approx(phi, self.get() as f64) as f32
    }

    /// The number of one-bits expected for the given calibrated size, obtained by evaluating the
    /// closed-form approximation of the forward expected-buckets function.
    fn from_f32(value: f32) -> Self {
        if !value.is_finite() {
            return Self::infinite();
        }
        let phi = C::default().phi_f64();
        let buckets = expected_diff_buckets_approx(phi, value.max(0.0) as f64).0;
        Self::new(buckets.round() as usize)
    }
}

// `Add` saturates into `infinite`, so it is a total operation.
impl<C: GeoConfig<Diff>> Add for OnesMetric<C> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.get() + rhs.get())
    }
}

// The following trait implementations are written by hand rather than derived so that they do not
// impose the corresponding bounds on the phantom configuration type `C`.
impl<C: GeoConfig<Diff>> Clone for OnesMetric<C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C: GeoConfig<Diff>> Copy for OnesMetric<C> {}

impl<C: GeoConfig<Diff>> PartialEq for OnesMetric<C> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<C: GeoConfig<Diff>> Eq for OnesMetric<C> {}

impl<C: GeoConfig<Diff>> PartialOrd for OnesMetric<C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<C: GeoConfig<Diff>> Ord for OnesMetric<C> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl<C: GeoConfig<Diff>> fmt::Debug for OnesMetric<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OnesMetric({})", self.get())
    }
}

/// A [`GeoDiffCount`] paired with its cached one-bit count.
///
/// Taking ownership of a filter and caching its number of one-bits lets the exact bit-count
/// ("ones") distance to other filters be computed repeatedly - as needed for nearest-neighbor
/// search - and, given a bound, abandoned early. The cached count (see [`Self::size`]) also yields
/// an O(1) reverse-triangle lower bound on the distance. Once the nearest neighbor is found,
/// [`Self::filter`] gives access to the underlying filter for a calibrated size estimate.
pub struct GeoDiffMetric<'a, C: GeoConfig<Diff>> {
    filter: GeoDiffCount<'a, C>,
    ones: OnesMetric<C>,
}

impl<'a, C: GeoConfig<Diff>> GeoDiffMetric<'a, C> {
    /// Takes ownership of `filter`, caching its number of one-bits.
    pub fn new(filter: GeoDiffCount<'a, C>) -> Self {
        let ones = OnesMetric::new(filter.msb.len() + filter.lsb.count_ones());
        Self { filter, ones }
    }

    /// The wrapped filter, e.g. to compute a calibrated size estimate once the nearest neighbor is
    /// known.
    pub fn filter(&self) -> &GeoDiffCount<'a, C> {
        &self.filter
    }
}

impl<C: GeoConfig<Diff> + Default> MetricSpace for GeoDiffMetric<'_, C> {
    type Metric = OnesMetric<C>;

    /// The size of the wrapped filter, measured as its number of one-bits. The reverse-triangle
    /// lower bound on the distance between two filters is `a.size().abs_diff(&b.size())`.
    fn size(&self) -> OnesMetric<C> {
        self.ones
    }

    /// The exact ones-distance (Hamming distance) to `other`, but abandoned as soon as at least
    /// `bound` differing bits have been counted, in which case [`Metric::infinite`] is returned.
    /// Otherwise (the distance is strictly below `bound`) the exact distance is returned. Pass
    /// [`Metric::infinite`] as the bound to always compute the exact distance.
    ///
    /// A reverse-triangle rejection using the cached one-bit counts (`|ones(a) - ones(b)| <=
    /// distance(a, b)`) discards far candidates in O(1). Otherwise the dense lower bits shared by
    /// both filters are XOR-counted block by block directly, which is much faster than the general
    /// bit-chunk merge; only the sparse upper region (the boundary block and everything above it,
    /// where the most-significant positions live) falls back to the merge. Counting from least to
    /// most significant reaches `bound` early for far-apart filters, whose differing bits are
    /// concentrated in the dense lower region.
    fn symmetric_diff_size(&self, other: &Self, bound: OnesMetric<C>) -> OnesMetric<C> {
        let (a, b) = (&self.filter, &other.filter);
        assert!(
            a.config == b.config,
            "combined filters must have the same configuration"
        );

        // O(1) reverse-triangle rejection using the cached one-bit counts.
        if self.ones.abs_diff(&other.ones) >= bound {
            return OnesMetric::infinite();
        }
        let bound = bound.get();

        // Complete lower blocks that are dense (least-significant bits only, no most-significant
        // positions) in both filters, and can therefore be xor-ed directly.
        let common_full = a.lsb.num_bits().min(b.lsb.num_bits()) / BITS_PER_BLOCK;

        let mut count = 0;
        let (a_blocks, b_blocks) = (a.lsb.blocks(), b.lsb.blocks());
        for i in 0..common_full {
            count += (a_blocks[i] ^ b_blocks[i]).count_ones() as usize;
            if count >= bound {
                return OnesMetric::infinite();
            }
        }

        // Sparse upper region: the boundary block and everything above it, where the filters are no
        // longer both dense and most-significant positions appear.
        for BitChunk { block, .. } in xor_bit_chunks(a.bit_chunks(), b.bit_chunks())
            .take_while(|BitChunk { index, .. }| *index >= common_full)
        {
            count += block.count_ones() as usize;
            if count >= bound {
                return OnesMetric::infinite();
            }
        }
        OnesMetric::new(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GeoConfig;
    use crate::diff_count::{xor, GeoDiffConfig7, GeoDiffCount7};
    use crate::{Count, Metric, MetricSpace};

    fn mix(x: u64) -> u64 {
        x.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    /// Builds a filter with `shared` hashes (indices `0..shared`) plus `extra` hashes offset by
    /// `extra_offset` so that two builds share only their common prefix.
    fn build(shared: usize, extra_offset: u64, extra: usize) -> GeoDiffCount7<'static> {
        let mut f = GeoDiffCount7::default();
        for i in 0..shared as u64 {
            f.push_hash(mix(i));
        }
        for i in 0..extra as u64 {
            f.push_hash(mix(extra_offset + i));
        }
        f
    }

    #[test]
    fn test_ones_metric() {
        for &(shared, a_only, b_only) in &[
            (0usize, 1usize, 1usize),
            (0, 200, 150),
            (5000, 5000, 5000),
            (100000, 100, 100),
        ] {
            let a = build(shared, 1 << 40, a_only);
            let b = build(shared, 1 << 41, b_only);

            let a_ones = a.iter_ones().count();
            let b_ones = b.iter_ones().count();
            let hamming = xor(&a, &b).iter_ones().count();

            let am = GeoDiffMetric::new(a);
            let bm = GeoDiffMetric::new(b);

            // Cached `size` (one-bit count) equals the exact number of set bits.
            assert_eq!(am.size(), OnesMetric::new(a_ones));
            assert_eq!(bm.size(), OnesMetric::new(b_ones));

            // The exact (uncapped) distance is the Hamming distance and is symmetric.
            let inf = OnesMetric::infinite();
            let d = am.symmetric_diff_size(&bm, inf);
            assert_eq!(d, OnesMetric::new(hamming));
            assert_eq!(d, bm.symmetric_diff_size(&am, inf));

            // Reverse-triangle lower bound, computed from the cached sizes: symmetric, `<= distance`.
            let lb = am.size().abs_diff(&bm.size());
            assert_eq!(lb, bm.size().abs_diff(&am.size()));
            assert!(lb <= d);

            // Capped distance: exact strictly below the bound, `infinite()` once reached.
            assert!(hamming > 0, "test cases have a non-empty diff");
            for &bound in &[hamming / 2, hamming, hamming + 1, hamming * 2 + 1] {
                let capped = am.symmetric_diff_size(&bm, OnesMetric::new(bound));
                if hamming < bound {
                    assert_eq!(capped, d, "exact below bound {bound}");
                } else {
                    assert_eq!(
                        capped,
                        OnesMetric::infinite(),
                        "infinite at/above bound {bound}"
                    );
                }
                // Symmetric in its arguments.
                assert_eq!(capped, bm.symmetric_diff_size(&am, OnesMetric::new(bound)));
            }

            // The lower bound and zero also trigger the O(1) rejection.
            assert_eq!(am.symmetric_diff_size(&bm, lb), OnesMetric::infinite());
            assert_eq!(
                am.symmetric_diff_size(&bm, OnesMetric::zero()),
                OnesMetric::infinite()
            );
        }
    }

    #[test]
    fn test_ones_metric_f32() {
        type M = OnesMetric<GeoDiffConfig7>;

        // `zero` and `infinite` map to the expected floating-point values, both ways.
        assert_eq!(M::zero().to_f32(), 0.0);
        assert_eq!(M::infinite().to_f32(), f32::INFINITY);
        assert_eq!(M::from_f32(f32::INFINITY), M::infinite());
        assert_eq!(M::from_f32(f32::NAN), M::infinite());
        assert_eq!(M::from_f32(0.0), M::zero());
        assert_eq!(M::from_f32(-5.0), M::zero());

        // `to_f32` (Newton inverse) matches the LUT-based calibrated estimate, and `from_f32` (the
        // forward function) round-trips the bucket count.
        let config: GeoDiffConfig7 = Default::default();
        for &buckets in &[1usize, 5, 20, 50, 100, 300] {
            let f = M::new(buckets).to_f32();
            let lut = config.expected_items(buckets);
            assert!(
                (f - lut).abs() <= 0.02 * lut.max(1.0) + 0.5,
                "to_f32 {f} should match the LUT estimate {lut} for {buckets} buckets"
            );
            let round_trip = M::from_f32(f).get();
            assert!(
                round_trip.abs_diff(buckets) <= 1,
                "from_f32 round-trip {round_trip} should recover {buckets} buckets"
            );
        }

        // `to_f32` is monotonically increasing in the bucket count.
        assert!(M::new(10).to_f32() < M::new(100).to_f32());
    }
}
