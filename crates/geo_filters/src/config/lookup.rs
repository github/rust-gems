use crate::config::phi_f64;

pub(crate) struct HashToBucketLookup {
    b: usize,
    buckets: Vec<(usize, usize)>,
}

impl HashToBucketLookup {
    pub(crate) fn new(b: usize) -> Self {
        let mut buckets = vec![(0, 0); 2 << b];
        let mut last_filled_bucket = buckets.len();
        let phi = phi_f64(b);
        for bucket in 0..(1 << b) {
            let lower_bucket_limit = phi.powf((bucket + 1) as f64);
            let lower_hash_limit = ((lower_bucket_limit - 0.5) * 2.0f64.powf(33.0)) as usize;
            let lower_hash_bucket = lower_hash_limit >> (32 - b - 1);
            assert!(lower_hash_bucket < last_filled_bucket);
            while last_filled_bucket > lower_hash_bucket {
                last_filled_bucket -= 1;
                buckets[last_filled_bucket] = (bucket, lower_hash_limit);
            }
        }
        assert_eq!(last_filled_bucket, 0);
        Self { b, buckets }
    }

    pub(crate) fn lookup(&self, hash: u64) -> usize {
        debug_assert_eq!(2 << self.b, self.buckets.len());
        let levels = hash.leading_zeros() as usize;
        // Take the most significant non-zero 32 bits from the hash (and drop the first leading 1).
        let hash = (if levels > 31 {
            // Note: in this case we don't have 32 significant bits. So, we take the bits
            // that we actually have. This case is extremely unlikely to be hit and therefore the
            // resulting inaccuracies are not really relevant for us.
            hash << (levels - 31)
        } else {
            hash >> (31 - levels)
        } & 0xFFFFFFFF) as usize;
        // From those, the first B bits determine the bucket index in our lookup table.
        let idx = hash >> (32 - self.b - 1);
        let offset = (hash < self.buckets[idx].1) as usize;
        offset + self.buckets[idx].0 + (1 << self.b) * levels
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, RngCore};

    use crate::{
        config::{hash_to_bucket, phi_f64},
        test_rng::prng_test_harness,
    };

    use super::HashToBucketLookup;

    #[test]
    fn test_lookup_7() {
        prng_test_harness(1, |rnd| {
            let var = lookup_random_hashes_variance::<7>(rnd, 1 << 16);
            assert!(var < 1e-4, "variance {var} too large");
        });
    }

    #[test]
    fn test_lookup_13() {
        prng_test_harness(1, |rnd| {
            let var = lookup_random_hashes_variance::<13>(rnd, 1 << 16);
            assert!(var < 1e-4, "variance {var} too large");
        });
    }

    fn lookup_random_hashes_variance<const B: usize>(rnd: &mut StdRng, n: u64) -> f64 {
        let phi = phi_f64(B);
        let buckets = HashToBucketLookup::new(B);

        let mut var = 0.0;
        for _ in 0..n {
            let hash = rnd.next_u64();
            let estimate = buckets.lookup(hash) as f64;
            let real = hash_to_bucket(phi, hash) as f64;
            let err = estimate - real; // assume the mean = 0.0
            var += err.powf(2.0) / n as f64;
        }
        var
    }
}
