use once_cell::sync::Lazy;

use crate::config::EstimationLookup;
use crate::config::FixedConfig;
use crate::config::HashToBucketLookup;
use crate::config::Lookup;
use crate::config::Lookups;
use crate::Distinct;

/// Distinct count configuration with a relative error standard deviation of ~0.065.
/// Uses at most 168 bytes of memory.
//
// Precision evaluation:
//
//     scripts/accuracy -n 10000 geo_distinct/u16/b=7/bytes={104,112,128,136,144,152,160}/msb=10
//
// Most-significant bytes evaluation:
//
//     scripts/accuracy -n 10000 geo_distinct/u16/b=7/bytes=136/msb={8,16,32,64}
//
pub type GeoDistinctConfig7 = FixedConfig<Distinct, u16, 7, 136, 8>;

/// Distinct count configuration with a relative error standard deviation of ~0.0075.
/// Uses at most 9248 bytes of memory.
//
// Precision evaluation:
//
//     scripts/accuracy -n 10000 geo_distinct/u32/b=13/bytes={6144,7168,8192,9216,10240,11264,12288,13312,14336}/msb=128
//
// Most-significant bytes evaluation:
//
//     scripts/accuracy -n 10000 geo_distinct/u32/b=13/bytes=9216/msb={128,192,256,320,512,640}
//
pub type GeoDistinctConfig13 = FixedConfig<Distinct, u32, 13, 9216, 320>;

impl Lookups for Distinct {
    #[inline]
    fn get_lookups() -> &'static [Lazy<Lookup>] {
        &LOOKUPS
    }

    #[inline]
    fn new_lookup(b: usize) -> Lookup {
        build_lookup(b)
    }
}

/// Computes the expected number of filled buckets after inserting `items` many distinct items.
/// The second result is the first derivative of the function.
///
/// Note: We allow here floating numbers as inputs so that the inverse can more easily computed.
/// The implementation relies on estimating infinite sums. For practical reasons, the sums are
/// truncated after certain number of iterations which can lead to incorrect numbers when `phi`
/// or `items` are not within a certain range. Thus, don't blindly trust the results of this function
/// for non-tested inputs.
pub(crate) fn expected_distinct_buckets(phi: f64, items: f64) -> (f64, f64) {
    let mut correction = 0.0f64;
    let mut sum = 0.0f64;
    let mut factor = -1.0f64;
    let mut derivative = 0.0f64;
    for k in 1..60 {
        // For negative integer values, the binomial coefficient becomes zero for negative numbers.
        // We simply apply the same rule for non-integer inputs which might not mathematically 100% correct,
        // but seems to work just fine for our use case.
        if items + 1.0 - k as f64 <= 0.0001 {
            break;
        }
        factor *= (phi - 1.0) * (items + 1.0 - k as f64) / k as f64;
        sum += factor / (1.0 - phi.powi(k));
        correction += 1.0 / (items + 1.0 - k as f64);
        derivative += correction * factor / (1.0 - phi.powi(k));
    }
    (sum, derivative)
}

static LOOKUPS: [Lazy<Lookup>; 16] = [
    Lazy::new(|| build_lookup(0)),
    Lazy::new(|| build_lookup(1)),
    Lazy::new(|| build_lookup(2)),
    Lazy::new(|| build_lookup(3)),
    Lazy::new(|| build_lookup(4)),
    Lazy::new(|| build_lookup(5)),
    Lazy::new(|| build_lookup(6)),
    Lazy::new(|| build_lookup(7)),
    Lazy::new(|| build_lookup(8)),
    Lazy::new(|| build_lookup(9)),
    Lazy::new(|| build_lookup(10)),
    Lazy::new(|| build_lookup(11)),
    Lazy::new(|| build_lookup(12)),
    Lazy::new(|| build_lookup(13)),
    Lazy::new(|| build_lookup(14)),
    Lazy::new(|| build_lookup(15)),
];

fn build_lookup(b: usize) -> Lookup {
    Lookup {
        hash_to_bucket: HashToBucketLookup::new(b),
        estimation: EstimationLookup::new(b, &expected_distinct_buckets),
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{estimate_count, GeoConfig};

    use super::*;

    #[test]
    fn test_estimation_lut_7() {
        let c = GeoDistinctConfig7::default();
        let err = (0..600)
            .step_by(1)
            .map(|i| {
                let a = c.expected_items(i); // uses estimation lookup
                let b = estimate_count(c.phi_f64(), i as f64, expected_distinct_buckets).0 as f32;
                (a - b).abs() / a.max(1.0)
            })
            .reduce(f32::max)
            .expect("a value");
        let bound = 0.0012;
        assert!(
            (err - bound).abs() < 0.5e-4,
            "found max error {err}, expected {bound}"
        );
    }

    #[test]
    fn test_estimation_lut_13() {
        let c = GeoDistinctConfig13::default();
        let err = (0..24000)
            .step_by(1)
            .map(|i| {
                let a = c.expected_items(i); // uses estimation lookup
                let b = estimate_count(c.phi_f64(), i as f64, expected_distinct_buckets).0 as f32;
                (a - b).abs() / a.max(1.0)
            })
            .reduce(f32::max)
            .expect("a value");
        let bound = 0.000021;
        assert!(
            (err - bound).abs() < 0.5e-6,
            "found max error {err}, expected {bound}"
        );
    }
}
