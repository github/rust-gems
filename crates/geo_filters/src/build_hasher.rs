use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher, Hasher as _};

/// Trait for a hasher factory that can be used to produce hashers
/// for use with geometric filters.
/// 
/// It is a super set of [`BuildHasher`], enforcing additional requirements
/// on the hasher builder that are required for the geometric filters and
/// surrounding code.
/// 
/// When performing operations across two different geometric filters,
/// the hashers must be equal, i.e. they must produce the same hash for the
/// same input. This is checked by the `hasher_eq` method.
pub trait GeoFilterBuildHasher: BuildHasher + Default + Clone + Send + Sync {
    fn hasher_eq(&self, other: &Self) -> bool {
        let v1 = self.build_hasher().finish();
        let v2 = other.build_hasher().finish();
        v1 == v2 
    }
}

impl<T> GeoFilterBuildHasher for T
where
    T: BuildHasher + Default + Clone + Send + Sync,
{
}

pub type DefaultBuildHasher = BuildHasherDefault<DefaultHasher>;