use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

use rand::{rngs::StdRng, SeedableRng as _};

/// Provides a seeded random number generator to tests which require some
/// degree of randomization. If the test panics the harness will print the
/// seed used for that run. You can then pass in this seed using the `TEST_SEED`
/// environment variable when running your tests.
///
/// You can provide a number of `iterations` this harness will run with randomly
/// generated seeds. If a manual seed is provided via the environment then the test
/// is only ran once with this seed.
pub fn prng_test_harness<F>(iterations: usize, mut test_fn: F)
where
    F: FnMut(&mut StdRng),
{
    let maybe_manual_seed = std::env::var("TEST_SEED")
        .map(|s| s.parse::<u64>().expect("Parse TEST_SEED to u64"))
        .ok();
    let mut seed = 0;
    let maybe_panic = catch_unwind(AssertUnwindSafe(|| {
        if let Some(manual_seed) = maybe_manual_seed {
            seed = manual_seed;
            let mut rng = StdRng::seed_from_u64(seed);
            test_fn(&mut rng);
        } else {
            for _ in 0..iterations {
                seed = rand::random();
                let mut rng = StdRng::seed_from_u64(seed);
                test_fn(&mut rng);
            }
        }
    }));
    match maybe_panic {
        Ok(t) => t,
        Err(panic_info) => {
            eprintln!("Test failed! Reproduce with: TEST_SEED={seed}");
            resume_unwind(panic_info);
        }
    }
}
