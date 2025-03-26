//! Configuration types for enabling/disabling features are compile time.
//! 
//! By disabling features, the compiler can generate faster code which can be important for certain use cases.
//! Certain implementations/conversion operations will only be available if the corresponding features were enabled.

/// Type-level boolean.
pub trait Bool {
    /// The value of the boolean.
    const VALUE: bool;
}
/// Type-level true.
pub struct True {}
/// Type-level false.
pub struct False {}
impl Bool for True {
    const VALUE: bool = true;
}
impl Bool for False {
    const VALUE: bool = false;
}

/// Configures which features should be enabled for a [`StringOffsets`] instance.
pub trait ConfigType {
    /// Whether to enable character conversions.
    type HasChars: Bool;
    /// Whether to enable UTF-16 conversions.
    type HasUtf16: Bool;
    /// Whether to enable line conversions.
    type HasLines: Bool;
    /// Whether to enable whitespace checks.
    type HasWhitespace: Bool;
}

/// Configuration type that enables all features.
pub struct AllConfig {}
impl ConfigType for AllConfig {
    type HasChars = True;
    type HasUtf16 = True;
    type HasLines = True;
    type HasWhitespace = True;
}

/// Configuration type that only enables line conversions.
pub struct OnlyLines {}
impl ConfigType for OnlyLines {
    type HasChars = False;
    type HasUtf16 = False;
    type HasLines = True;
    type HasWhitespace = False;
}
