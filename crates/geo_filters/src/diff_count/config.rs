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

/// Diff count configuration with a relative error standard deviation of ~0.04.
//
// Precision evaluation:
//
//     scripts/accuracy -n 5000 geo_diff/u32/b=10/bytes={768,832,896,960,1024}/msb=64
//
// Most-significant bytes evaluation:
//
//     scripts/accuracy -n 5000 geo_diff/u32/b=10/bytes=896/msb={32,48,64,80,96,128}
//
pub type GeoDiffConfig10<H = UnstableDefaultBuildHasher> = FixedConfig<Diff, u32, 10, 896, 64, H>;

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
pub(super) fn expected_diff_buckets(phi: f64, items: f64) -> (f64, f64) {
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

/// Euler–Mascheroni constant.
const GAMMA: f64 = 0.577_215_664_901_532_9;
/// `e^gamma`, precomputed.
const E_GAMMA: f64 = 1.781_072_417_990_198;

// Closed-form approximation of [`expected_diff_buckets`] and its inverse that avoids the
// per-bucket summation (and the Newton iteration on it). It is exact in both limits and stays
// within ~0.35% of the exact model for the predefined configurations (b = 7, 10, 13).
//
// Writing `p_k = (1 - phi) phi^k`, the expected number of one-bits is
//     E(N) = 1/2 sum_k (1 - (1 - 2 p_k)^N) ≈ S · Ein(x) + 1/4 (1 - e^-x),
// with `x = 2(1 - phi) N`, `S = 1 / (2 ln(1/phi))`, and `Ein(x) = gamma + ln x + E1(x)` the
// entire exponential-integral bridge (`Ein(x) ≈ x` near 0, `≈ gamma + ln x` for large x). The
// special function `Ein` is replaced by a rational "scaling factor" `rho(x)` inside the logarithm,
//     Ein(x) ≈ ln(1 + e^gamma x rho(x)),  rho(x) = (e^-gamma + A x + B x^2) / (1 + C x + B x^2),
// where `rho(0) = e^-gamma` reproduces the linear small-N regime and `rho(inf) = 1` reproduces the
// logarithmic asymptote. The coefficients are universal (independent of phi), fitted so the
// rational matches `Ein` to < 0.07%.
const RHO_A: f64 = 0.384_417;
const RHO_B: f64 = 0.130_468;
const RHO_C: f64 = 0.442_202;

/// Closed-form approximation of [`expected_diff_buckets`]: the expected number of one-bits for
/// `items` items, together with its derivative with respect to `items`. See the comment above for
/// the derivation. Accurate to < 1% for the predefined configurations.
pub(super) fn expected_diff_buckets_approx(phi: f64, items: f64) -> (f64, f64) {
    if items <= 0.0 {
        return (0.0, 1.0);
    }
    let s = 0.5 / -phi.ln(); // = 1 / (2 ln(1/phi))
    let a = 2.0 * (1.0 - phi);
    let x = a * items;
    let e_neg_x = (-x).exp();

    // Y = e^gamma x rho(x) = (x + e^gamma A x^2 + e^gamma B x^3) / (1 + C x + B x^2)
    let num = x * (1.0 + E_GAMMA * RHO_A * x + E_GAMMA * RHO_B * x * x);
    let den = 1.0 + RHO_C * x + RHO_B * x * x;
    let y = num / den;
    let value = s * (1.0 + y).ln() + 0.25 * (1.0 - e_neg_x);

    // derivative w.r.t. items = a * d/dx [ s ln(1 + Y) + 1/4 (1 - e^-x) ]
    let num_prime = 1.0 + 2.0 * E_GAMMA * RHO_A * x + 3.0 * E_GAMMA * RHO_B * x * x;
    let den_prime = RHO_C + 2.0 * RHO_B * x;
    let y_prime = (num_prime * den - num * den_prime) / (den * den);
    let derivative = a * (s * y_prime / (1.0 + y) + 0.25 * e_neg_x);

    (value, derivative)
}

/// Closed-form approximation of the inverse of [`expected_diff_buckets`]: the number of items that
/// produce `ones` one-bits. Uses the analytic large-N seed of the model above followed by three
/// Newton steps on the (cheap) closed-form forward, which converge to full `f64` precision.
pub(super) fn estimate_count_approx(phi: f64, ones: f64) -> f64 {
    if ones <= 0.0 {
        return 0.0;
    }
    let s = 0.5 / -phi.ln();
    let a = 2.0 * (1.0 - phi);
    // Invert the large-N asymptote `m ≈ S (gamma + ln(a N)) + 1/4` for the initial guess.
    let mut items = ((ones - 0.25) / s - GAMMA).exp() / a;
    if !items.is_finite() || items <= 0.0 {
        items = ones;
    }
    for _ in 0..3 {
        let (m, dm) = expected_diff_buckets_approx(phi, items);
        items -= (m - ones) / dm;
        if items < f64::MIN_POSITIVE {
            items = f64::MIN_POSITIVE;
        }
    }
    items
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
    fn test_expected_diff_buckets_approx() {
        // The closed-form forward stays within 1% of the exact model for the predefined configs.
        for b in [7u32, 10, 13] {
            let phi = 0.5f64.powf(1.0 / (1u64 << b) as f64);
            let mut items = 1.0f64;
            loop {
                let exact = expected_diff_buckets(phi, items).0;
                if exact > 2000.0 {
                    break;
                }
                if exact > 0.3 {
                    let approx = expected_diff_buckets_approx(phi, items).0;
                    let rel = (approx - exact).abs() / exact;
                    assert!(
                        rel < 0.01,
                        "b={b} items={items}: exact={exact} approx={approx} rel={rel}"
                    );
                }
                items *= 1.5;
            }
        }
    }

    #[test]
    fn test_estimate_count_approx() {
        // The closed-form inverse recovers the item count within 1% of the exact model.
        for b in [7u32, 10, 13] {
            let phi = 0.5f64.powf(1.0 / (1u64 << b) as f64);
            let mut items = 1.0f64;
            loop {
                let ones = expected_diff_buckets(phi, items).0;
                if ones > 2000.0 {
                    break;
                }
                let est = estimate_count_approx(phi, ones);
                let rel = (est - items).abs() / items;
                assert!(
                    rel < 0.01,
                    "b={b} items={items} ones={ones}: est={est} rel={rel}"
                );
                items *= 1.5;
            }
        }
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
    fn test_estimation_lut_10() {
        let c = GeoDiffConfig10::<UnstableDefaultBuildHasher>::default();
        let err = (0..5000)
            .step_by(10)
            .map(|i| {
                let a = c.expected_items(i); // uses estimation lookup
                let b = estimate_count(c.phi_f64(), i as f64, expected_diff_buckets).0 as f32;
                (a - b).abs() / a.max(1.0)
            })
            .reduce(f32::max)
            .expect("a value");
        let bound = 0.00035;
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
