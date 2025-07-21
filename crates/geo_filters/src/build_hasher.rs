use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher, Hasher as _};

use fnv::FnvBuildHasher;

/// Trait for a hasher factory that can be used to produce hashers
/// for use with geometric filters.
///
/// It is a super set of [`BuildHasher`], enforcing additional requirements
/// on the hasher builder that are required for the geometric filters and
/// surrounding code.
///
/// When performing operations across two different geometric filters,
/// the hashers must be equal, i.e. they must produce the same hash for the
/// same input.
pub trait ReproducibleBuildHasher: BuildHasher + Default + Clone {
    #[inline]
    fn debug_assert_hashers_eq() {
        // In debug builds we check that hash outputs are the same for
        // self and other. The library user should only have implemented
        // our build hasher trait if this is already true, but we check
        // here in case they have implemented the trait in error.
        debug_assert_eq!(
            Self::default().build_hasher().finish(),
            Self::default().build_hasher().finish(),
            "Hashers produced by ReproducibleBuildHasher do not produce the same output with the same input"
        );
    }
}

/// Note that this `BuildHasher` has a consistent implementation of `Default`
/// but is NOT stable across releases of Rust. It is therefore dangerous
/// to use if you plan on serializing the geofilters and reusing them due
/// to the fact that you can serialize a filter made with one version and
/// deserialize with another version of the hasher factor.
pub type UnstableDefaultBuildHasher = BuildHasherDefault<DefaultHasher>;

impl ReproducibleBuildHasher for UnstableDefaultBuildHasher {}
impl ReproducibleBuildHasher for FnvBuildHasher {}
