/// Implement DistinctCount trait for existing HLL++ implementation so that we can benchmark.
///
/// To avoid that hashing differs, we instantiate the HLL++ implementation with a NOOP hasher.
use std::{
    cell::RefCell,
    hash::{BuildHasher, Hasher},
};

use hyperloglogplus::{HyperLogLog, HyperLogLogPlus};

use crate::{
    build_hasher::{DefaultBuildHasher, ReproducibleBuildHasher},
    Count, Distinct,
};

/// Uses at most 192 bytes.
/// The relative error has a standard deviation of ~0.065.
pub type HllConfig8 = FixedHllConfig<8>;
pub type Hll8 = Hll<HllConfig8>;

/// Uses at most 12300 bytes.
/// The relative error has a standard deviation of ~0.0081.
pub type HllConfig14 = FixedHllConfig<14>;
pub type Hll14 = Hll<HllConfig14>;

pub trait HllConfig {
    fn precision(&self) -> usize;
}

#[derive(Clone, Default, Eq, PartialEq)]
pub struct FixedHllConfig<const N: usize> {}

impl<const N: usize> HllConfig for FixedHllConfig<N> {
    #[inline]
    fn precision(&self) -> usize {
        N
    }
}

#[derive(Clone)]
pub struct VariableHllConfig {
    precision: usize,
}

impl VariableHllConfig {
    pub fn new(precision: usize) -> Self {
        Self { precision }
    }
}

impl HllConfig for VariableHllConfig {
    #[inline]
    fn precision(&self) -> usize {
        self.precision
    }
}

pub struct Hll<C: HllConfig> {
    config: C,
    inner: RefCell<HyperLogLogPlus<u64, BuildNoopHasher>>,
}

impl<C: HllConfig + Default> Default for Hll<C> {
    fn default() -> Self {
        Self::new(C::default())
    }
}

impl<C: HllConfig> Hll<C> {
    pub fn new(config: C) -> Self {
        let inner = HyperLogLogPlus::new(config.precision() as u8, BuildNoopHasher {}).expect("");
        Self {
            config,
            inner: inner.into(),
        }
    }
}

pub struct NoopHasher {
    hash: u64,
}

#[derive(Clone, Default)]
pub struct BuildNoopHasher {}

impl Hasher for NoopHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write(&mut self, _: &[u8]) {
        unimplemented!(
            "NoopHasher does not support arbitrary byte sequences. Use write_u64 instead"
        );
    }

    #[inline]
    fn write_u64(&mut self, hash: u64) {
        self.hash = hash
    }
}

impl BuildHasher for BuildNoopHasher {
    type Hasher = NoopHasher;

    fn build_hasher(&self) -> Self::Hasher {
        NoopHasher { hash: 0 }
    }
}

impl ReproducibleBuildHasher for BuildNoopHasher {}

impl<C: HllConfig> Count<Distinct> for Hll<C> {
    type BuildHasher = DefaultBuildHasher;

    fn push_hash(&mut self, hash: u64) {
        self.inner.borrow_mut().insert(&hash)
    }

    fn push_sketch(&mut self, _other: &Self) {
        unimplemented!()
    }

    fn size(&self) -> f32 {
        self.inner.borrow_mut().count() as f32
    }

    fn size_with_sketch(&self, _other: &Self) -> f32 {
        unimplemented!()
    }

    fn bytes_in_memory(&self) -> usize {
        (1 << self.config.precision()) * 6 / 8
    }
}
