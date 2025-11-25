mod batch_decoder;
mod elias_fano;

pub use batch_decoder::*;
pub use elias_fano::*;

#[cfg(target_arch = "x86_64")]
pub mod avx_batch_decoder;

/// Partitioned Elias-Fano structure.
pub struct PartitionedEliasFano {
    // Implementation details to be added
}

impl PartitionedEliasFano {
    pub fn new() -> Self {
        Self {}
    }
}
