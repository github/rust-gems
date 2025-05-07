use std::borrow::Cow;

use unicode_normalization::UnicodeNormalization;

/// Type which represents a normalized string.
/// This is to avoid calling normalize multiple times or forgetting to call normalization!
///
/// TODO: Annotate the type with the normalization type, once there are more than one.
pub struct NormalizedString<'a>(Cow<'a, str>);

impl<'a> NormalizedString<'a> {
    /// Returns the normalized inner str buffer.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// This function is unsafe, since the caller must ensure that the correct normalization
    /// was used. The normalization may vary by tokenizer. This mostly a backdoor which might
    /// be handy for certain optimizations or for testing.
    pub unsafe fn from_str(s: &'a str) -> NormalizedString<'a> {
        // SAFETY: This is safe if `s` is in fact correctly normalized already. The caller is
        // responsible for ensuring that.
        NormalizedString(Cow::Borrowed(s))
    }
}

/// Helper trait which converts string types into NormalizedString.
/// Calling normalize on a NormalizedString is a no-op.
pub trait Normalizable<'a> {
    fn normalize(self, nfc: bool) -> NormalizedString<'a>;
}

impl<'a> Normalizable<'a> for &'a str {
    fn normalize(self, nfc: bool) -> NormalizedString<'a> {
        if nfc {
            NormalizedString(self.nfc().collect())
        } else {
            NormalizedString(Cow::Borrowed(self))
        }
    }
}

impl<'a, T> Normalizable<'a> for &'a T
where
    T: AsRef<str>,
{
    fn normalize(self, nfc: bool) -> NormalizedString<'a> {
        self.as_ref().normalize(nfc)
    }
}

impl<'a> Normalizable<'a> for NormalizedString<'a> {
    fn normalize(self, _: bool) -> NormalizedString<'a> {
        self
    }
}
