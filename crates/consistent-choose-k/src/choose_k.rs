use crate::{ConsistentHasher, ManySeqBuilder};

/// Implementation of a consistent choose k hashing algorithm.
/// It returns k distinct consistent hashes in the range `0..n`.
/// The hashes are consistent when `n` changes and when `k` changes!
/// I.e. one hash changes with probability `k/(n+1)` when `n` increases by one,
/// resp. one hash gets added when `k` is increased. Additionally, the returned `k` tuple
/// is guaranteed to be uniformly chosen from all possible `n-choose-k` tuples.
///
/// Also implements `Iterator` to yield the next sample when k is increased.
/// Note: since this hashing algorithm implements choose k semantics, all the returned samples are distinct.
/// Note: The `Iterator` won't output the samples ordered by position.
///
/// # Example
/// ```
/// use std::hash::{DefaultHasher, Hash};
/// use consistent_choose_k::ConsistentChooseKHasher;
///
/// let mut h = DefaultHasher::default();
/// 42u64.hash(&mut h);
/// let top3: Vec<usize> = ConsistentChooseKHasher::new(h, 100).take(3).collect();
/// assert_eq!(top3.len(), 3);
/// ```
pub struct ConsistentChooseKHasher<H: ManySeqBuilder> {
    builder: H,
    n: usize,
    samples: Vec<usize>,
}

impl<H: ManySeqBuilder> ConsistentChooseKHasher<H> {
    /// Create a new iterator for `n` nodes starting with k=0.
    ///
    /// Time: O(1)
    pub fn new(builder: H, n: usize) -> Self {
        Self {
            builder,
            n,
            samples: Vec::new(),
        }
    }

    /// Create with the choose-k set for `k` out of `n` nodes pre-built.
    ///
    /// Average time: O(k^2)
    pub fn new_with_k(builder: H, n: usize, k: usize) -> Self {
        assert!(n >= k, "n must be at least k");
        let mut iter = Self::new(builder, n);
        for i in 0..k {
            iter.samples.push(iter.get_sample(i, n));
        }
        for i in (0..k).rev() {
            let s = iter.samples[0..=i]
                .iter()
                .copied()
                .max()
                .expect("subslice 0..=i has at least one element");
            iter.samples[i] = s;
            for j in 0..i {
                if iter.samples[j] == s {
                    iter.samples[j] = iter.get_sample(j, s);
                }
            }
        }
        iter
    }

    /// Returns the `k` underlying samples.
    pub fn samples(&self) -> &[usize] {
        &self.samples
    }

    /// Converts self into the `k` underlying samples vector.
    pub fn into_samples(self) -> Vec<usize> {
        self.samples
    }

    /// Returns the current universe size.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the current sample size.
    pub fn k(&self) -> usize {
        self.samples.len()
    }

    /// (Average) time: O(1)
    fn get_sample(&self, k: usize, n: usize) -> usize {
        ConsistentHasher::new(self.builder.seq_builder(k))
            .into_prev(n - k)
            .expect("must not fail")
            + k
    }

    /// Decrements n to the largest sample and computes the new sample it is
    /// being replaced with. Returns the index of the new largest sample.
    ///
    /// Time: O(k)
    ///
    /// Panics: if `n` is already at most `k`.
    pub fn shrink_n(&mut self) -> usize {
        assert!(self.n > self.k());
        let n = *self.samples.last().expect("samples must not be empty");
        self.n = n;
        self.shrink_n_inner(n)
    }

    fn shrink_n_inner(&mut self, mut n: usize) -> usize {
        for i in (0..self.samples.len()).rev() {
            if self.samples[i] < n {
                // We are done!
                return i + 1;
            }
            // The new maximum over all sequences at position i is either
            // the sample of the sequence i or the maximum over all other sequences.
            // The latter is already known via self.samples[i-1].
            let si = self.get_sample(i, n);
            if i > 0 && self.samples[i - 1] > si {
                self.samples[i] = self.samples[i - 1];
            } else {
                self.samples[i] = si;
            }
            n = self.samples[i];
        }
        0
    }

    /// Grow the sample set by one element. Returns the index at which the new
    /// element was inserted (i.e. its rank position).
    ///
    /// Time: O(k)
    ///
    /// Panics: if `k` equals `n`.
    pub fn grow_k(&mut self) -> usize {
        assert!(self.k() < self.n);
        let k = self.samples.len();
        let sk = self.get_sample(k, self.n);
        if let Some(last) = self.samples.last().copied() {
            if last < sk {
                self.samples.push(sk);
                k
            } else {
                let i = self.shrink_n_inner(last);
                self.samples.push(last);
                i
            }
        } else {
            self.samples.push(sk);
            k
        }
    }
}

impl<H: ManySeqBuilder> Iterator for ConsistentChooseKHasher<H> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.samples.len() >= self.n {
            return None;
        }
        let idx = self.grow_k();
        Some(self.samples[idx])
    }
}

#[cfg(test)]
mod tests {
    use std::hash::{DefaultHasher, Hash};

    use super::*;

    fn hasher_for_key(key: u64) -> DefaultHasher {
        let mut hasher = DefaultHasher::default();
        key.hash(&mut hasher);
        hasher
    }

    #[test]
    fn test_ranking_matches_prev() {
        // Every prefix of the ranking must equal the sorted prev(n) set.
        for key in 0..200 {
            for n in 2..30 {
                let hasher = hasher_for_key(key);
                let full: Vec<usize> = ConsistentChooseKHasher::new(hasher.clone(), n).collect();
                assert_eq!(full.len(), n);
                for k in 1..=n {
                    let expected =
                        ConsistentChooseKHasher::new_with_k(hasher.clone(), n, k).into_samples();
                    let mut prefix = full[..k].to_vec();
                    prefix.sort();
                    assert_eq!(
                        prefix, expected,
                        "key={key} n={n} k={k}: ranking prefix mismatch"
                    );
                }
            }
        }
    }

    #[test]
    fn test_ranking_k_equals_n() {
        // When exhausted, the ranking contains all nodes 0..n.
        for key in 0..200 {
            for n in 1..20 {
                let hasher = hasher_for_key(key);
                let mut ranking: Vec<usize> = ConsistentChooseKHasher::new(hasher, n).collect();
                ranking.sort();
                let expected: Vec<usize> = (0..n).collect();
                assert_eq!(ranking, expected, "key={key} n={n}");
            }
        }
    }

    #[test]
    fn test_uniform_k() {
        const K: usize = 3;
        for k in 0..100 {
            let hasher = hasher_for_key(k);
            for n in K..1000 {
                let samples =
                    ConsistentChooseKHasher::new_with_k(hasher.clone(), n + 1, K).into_samples();
                assert!(samples.len() == K);
                for i in 0..K - 1 {
                    assert!(samples[i] < samples[i + 1]);
                }
                let next =
                    ConsistentChooseKHasher::new_with_k(hasher.clone(), n + 2, K).into_samples();
                for i in 0..K {
                    assert!(samples[i] <= next[i]);
                }
                let mut merged = samples.clone();
                merged.extend(next.clone());
                merged.sort();
                merged.dedup();
                assert!(
                    merged.len() == K || merged.len() == K + 1,
                    "Unexpected {samples:?} vs. {next:?}"
                );
            }
        }
        let mut stats = vec![0; 8];
        for i in 0..32 {
            let hasher = hasher_for_key(i + 32783);
            let samples =
                ConsistentChooseKHasher::new_with_k(hasher, stats.len(), 2).into_samples();
            for s in samples {
                stats[s] += 1;
            }
        }
        println!("{stats:?}");
        assert_eq!(stats, vec![10, 12, 6, 6, 6, 5, 9, 10]);
        // Test consistency when increasing k!
        for k in 1..10 {
            for n in k + 1..20 {
                for key in 0..1000 {
                    let hasher = hasher_for_key(key);
                    let set1 =
                        ConsistentChooseKHasher::new_with_k(hasher.clone(), n, k).into_samples();
                    let set2 = ConsistentChooseKHasher::new_with_k(hasher, n, k + 1).into_samples();
                    assert_eq!(set1.len(), k);
                    assert_eq!(set2.len(), k + 1);
                    let mut merged = set1.clone();
                    merged.extend(set2);
                    merged.sort();
                    merged.dedup();
                    assert_eq!(merged.len(), k + 1);
                }
            }
        }
    }

    #[test]
    fn test_shrink_n() {
        for k in 1..10 {
            for n in k + 1..30 {
                let mut iter = ConsistentChooseKHasher::new_with_k(DefaultHasher::new(), n, k);
                while *iter.samples.last().unwrap() > k {
                    let expected = ConsistentChooseKHasher::new_with_k(
                        DefaultHasher::new(),
                        *iter.samples.last().unwrap(),
                        k,
                    );
                    iter.shrink_n();
                    assert_eq!(iter.samples, expected.samples);
                }
            }
        }
    }

    #[test]
    fn test_grow_k() {
        for n in 1..30 {
            let mut iter = ConsistentChooseKHasher::new(DefaultHasher::new(), n);
            for k in 1..10.min(n) {
                let expected = ConsistentChooseKHasher::new_with_k(DefaultHasher::new(), n, k);
                iter.grow_k();
                assert_eq!(iter.samples, expected.samples);
            }
        }
    }
}
