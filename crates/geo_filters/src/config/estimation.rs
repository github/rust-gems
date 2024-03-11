use super::phi_f64;

const LOOKUP_POINTS: usize = 32;

/// Computing estimations from observed number of filled buckets is quite expensive.
/// This struct stores precomputed data points and then allows fast interpolation between them.
/// Data points are equidistant such that the two enclosing data points can be easily found.
/// This scheme also results in a very low interpolation error for our function.
pub(crate) struct EstimationLookup {
    points: Vec<(f32, f32)>,
    step_size: usize,
    upper_bound: usize,
}

impl EstimationLookup {
    /// Create a new estimation lookup for the given bit count `b` (1 << b bits cover half the hash space),
    /// the expected-buckets function `f`, which should return the expected number of filled buckets for a
    /// given phi and item count, and the upper bound point count `upper_bound` for the number of steps.
    pub(crate) fn new<F: Fn(f64, f64) -> (f64, f64)>(b: usize, f: &F) -> Self {
        let phi = phi_f64(b);
        let step_size = step_size(b);
        let upper_bound = upper_bound(b);
        let steps = upper_bound.div_ceil(step_size);
        let points = (0..=steps)
            .map(|i| estimate_count(phi, (step_size * i) as f64, f))
            .map(|(u, du)| (u as f32, du as f32))
            .collect();
        Self {
            points,
            step_size,
            upper_bound,
        }
    }

    /// This function interpolates the desired estimate between two precomputed estimates.
    /// To achieve high quality and smooth interpolation results, we fit a cubic polynomial
    /// through those two precomputed estimates, such that the cubic preserves not only the
    /// estimate, but also the derivative for those two estimates.
    /// To simplify the math and to improve numeric stability, the data points are mapped
    /// onto the coordinates 0 and 1.
    pub(crate) fn interpolate(&self, ones: usize) -> f32 {
        debug_assert!(
            ones <= self.upper_bound,
            "given {} ones higher than upper bound {} ones for this estimator",
            ones,
            self.upper_bound
        );
        let i = ones / self.step_size;
        let x0 = (i * self.step_size) as f32;
        let x1 = x0 + self.step_size as f32;
        let dx = x1 - x0;
        let (u, du) = self.points[i];
        let (v, dv) = self.points[i + 1];
        let du_ = du * dx;
        let dv_ = dv * dx;
        let d = u;
        let c = du_;
        let a = dv_ - 2.0 * v + 2.0 * d + c;
        let b = 3.0 * v - dv_ - 3.0 * d - 2.0 * c;
        let x = (ones as f32 - x0) / dx;
        d + x * (c + x * (b + x * a))
    }
}

/// Computes the expected number of unique items needed to produce `ones` many filled buckets.
/// The second return value is the derivative of this function.
///
/// The implementation uses Newton's algorithm to invert the given function `f`, which should
/// return the number of expected buckets for a given phi and number of items.
pub(crate) fn estimate_count<F: Fn(f64, f64) -> (f64, f64)>(
    phi: f64,
    ones: f64,
    f: F,
) -> (f64, f64) {
    if ones == 0.0 {
        (0.0, 1.0)
    } else {
        let mut items = ones;
        let mut deriv = 1.0;
        // TODO explain why six iterations is enough
        for _ in 0..6 {
            let (expected_buckets, derivative) = f(phi, items);
            items -= (expected_buckets - ones) / derivative;
            deriv = 1.0 / derivative;
        }
        (items, deriv)
    }
}

#[inline]
fn step_size(b: usize) -> usize {
    ((5 << b) / LOOKUP_POINTS).max(1)
}

#[inline]
fn upper_bound(b: usize) -> usize {
    10 << b
}

pub(crate) fn assert_buckets_within_estimation_bound(b: usize, buckets: usize) {
    assert!(
        buckets < upper_bound(b),
        "{} buckets is larger than {} estimation upper bound",
        buckets,
        upper_bound(b)
    );
}
