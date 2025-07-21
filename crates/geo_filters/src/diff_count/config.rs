use once_cell::sync::Lazy;

use crate::build_hasher::UnstableDefaultBuildHasher;
use crate::config::EstimationLookup;
use crate::config::FixedConfig;
use crate::config::HashToBucketLookup;
use crate::config::Lookup;
use crate::config::Lookups;
use crate::Diff;

/// Diff count configuration with a relative error standard deviation of ~0.125.
//
// Precision evaluation:
//
//     scripts/accuracy -n 10000 geo_diff/u16/b=7/bytes={104,108,112,116,120}/msb=10
//
// Most-significant bytes evaluation:
//
//     scripts/accuracy -n 10000 geo_diff/u16/b=7/bytes=112/msb={8,12,16,20}
//
pub type GeoDiffConfig7<H = UnstableDefaultBuildHasher> = FixedConfig<Diff, u16, 7, 112, 12, H>;

/// Diff count configuration with a relative error standard deviation of ~0.015.
//
// Precision evaluation:
//
//     scripts/accuracy -n 10000 geo_diff/u32/b=13/bytes={4096,5120,6144,7138,8192,9216}/msb=128
//
// Most-significant bytes evaluation:
//
//     scripts/accuracy -n 1000 geo_diff/u32/b=13/bytes=7138/msb={128,192,256,384,512}
//
pub type GeoDiffConfig13<H = UnstableDefaultBuildHasher> = FixedConfig<Diff, u32, 13, 7138, 384, H>;

impl Lookups for Diff {
    #[inline]
    fn get_lookups() -> &'static [Lazy<Lookup>] {
        &LOOKUPS
    }

    #[inline]
    fn new_lookup(b: usize) -> Lookup {
        build_lookup(b)
    }
}

// This function simply iterates over all buckets and sums up their probabilities of being one.
// At some point, the chance of having multiple items hitting the same bucket becomes negligible.
// Then, the computation can be short-cut by approximating the exponentiation with a linear function.
// With this simplification, the infinite sum of those linear terms can be directly computed
// and the loop be aborted.
// Note: Increasing the constant to switch earlier into approximation mode will mostly increase the
// approximation error, but not reduce the runtime of the function by much.
fn expected_diff_buckets(phi: f64, items: f64) -> (f64, f64) {
    let mut sum = 0.0;
    let mut derivative = 0.0;
    for k in 0.. {
        let base = 1.0 - 2.0 * phi.powi(k) * (1.0 - phi);

        let bound = phi.powi(k) * items;
        if bound < 1.0 {
            sum += bound;
            derivative += phi.powi(k);
            break;
        } else {
            sum += 0.5 * (1.0 - base.powf(items));
            derivative += -0.5 * base.powf(items) * base.ln();
        }
    }
    (sum, derivative)
}

// In theory, this is the "faster" solution, since it needs only items many iterators.
// Since it is an alternating sequence, it is unfortunately numerically unstable when items is large.
#[allow(dead_code)]
fn expected_diff_buckets_fast(phi: f64, items: f64) -> (f64, f64) {
    let mut correction = 0.0f64;
    let mut sum = 0.0;
    let mut derivative = 0.0f64;
    let mut factor = -0.5;
    for k in 1..=items as i32 + 1 {
        if items + 1.0 - k as f64 <= 0.0001 {
            break;
        }
        factor *= (items + 1.0 - k as f64) / k as f64;
        factor *= -2.0 * (1.0 - phi);
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
        estimation: EstimationLookup::new(b, &expected_diff_buckets),
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{estimate_count, GeoConfig};

    use super::*;

    #[test]
    fn test_bit_from_hash() {
        let config = GeoDiffConfig7::<UnstableDefaultBuildHasher>::default();
        assert_eq!(config.hash_to_bucket(u64::MAX), 0);
        assert_eq!(
            config.hash_to_bucket(0) as usize,
            config.bits_per_level() * 65 - 1
        );
        // Note: The test fails when going down to more than 40 leading zeros, since don't
        // have the required 32 significant bits. As a result the rounding fails.
        // Also, this is the only range that is practically relevant. All smaller hash
        // values are only relevant for bit sets with more than trillions of entries!
        for bit in 0..(config.bits_per_level() * 40) {
            let lower_bound = 2f64.powf(64f64) * config.phi_f64().powf((bit + 1) as f64);
            // Note: due to rounding issues we are testing hash values slight above or below the
            // computed limit.
            assert_eq!(
                config.hash_to_bucket((lower_bound * 1.0000001) as u64) as usize,
                bit
            );
            assert_eq!(
                config.hash_to_bucket((lower_bound * 0.9999999) as u64) as usize,
                bit + 1
            );
        }
    }

    #[test]
    fn test_estimates() {
        let phi = 0.5f64.powf(1. / 128.);
        for i in (0..10).step_by(1) {
            let i = i as f64;
            let (a, _) = expected_diff_buckets(phi, i);
            let (b, _) = estimate_count(phi, a, expected_diff_buckets);
            let err = (i - b).abs() / i.max(1.0);
            // let c = estimate_count(0.5.powf(1./128.), );
            println!("{i} {a} {b} {err}");
        }

        println!("{phi}");
        println!("{:?}", expected_diff_buckets(phi, 100.0));
        println!("{:?}", expected_diff_buckets_fast(phi, 100.0));
        println!("{:?}", estimate_count(phi, 78.450, expected_diff_buckets));
    }

    #[test]
    fn test_estimation_lut_7() {
        let c = GeoDiffConfig7::<UnstableDefaultBuildHasher>::default();
        let err = (0..600)
            .step_by(1)
            .map(|i| {
                let a = c.expected_items(i); // uses estimation loookup
                let b = estimate_count(c.phi_f64(), i as f64, expected_diff_buckets).0 as f32;
                (a - b).abs() / a.max(1.0)
            })
            .reduce(f32::max)
            .expect("a value");
        let bound = 0.0028;
        assert!(
            (err - bound).abs() < 0.5e-4,
            "found max error {err}, expected {bound}"
        );
    }

    #[test]
    fn test_estimation_lut_13() {
        let c = GeoDiffConfig13::<UnstableDefaultBuildHasher>::default();
        let err = (0..24000)
            .step_by(100)
            .map(|i| {
                let a = c.expected_items(i); // uses estimation lookup
                let b = estimate_count(c.phi_f64(), i as f64, expected_diff_buckets).0 as f32;
                (a - b).abs() / a.max(1.0)
            })
            .reduce(f32::max)
            .expect("a value");
        let bound = 0.000022;
        assert!(
            (err - bound).abs() < 0.5e-6,
            "found max error {err}, expected {bound}"
        );
    }
}
