use std::panic::{catch_unwind, resume_unwind, UnwindSafe};

use rand::{rngs::StdRng, SeedableRng as _};

/// Provides a seeded random number generator to tests which require some
/// degree of randomization. If the test panics the harness will print the
/// seed used for that run. You can then pass in this seed using the `TEST_SEED`
/// environment variable when running your tests.
pub fn prng_test_harness<F>(test_fn: F)
where
    F: Fn(StdRng) -> () + UnwindSafe,
{
    let seed = std::env::var("TEST_SEED")
        .map(|s| s.parse::<u64>().expect("Parse TEST_SEED to u64"))
        .unwrap_or_else(|_| rand::random());
    let rng = StdRng::seed_from_u64(seed);
    let maybe_panic = catch_unwind(move || {
        test_fn(rng);
    });
    if let Err(panic_info) = maybe_panic {
        eprintln!("Test failed! Reproduce with: TEST_SEED={}", seed);
        resume_unwind(panic_info);
    }
}
