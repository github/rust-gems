/// A value that has been found by the set reconciliation algorithm.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum DecodedValue<T> {
    /// A value that has been added
    Addition(T),
    /// A value that has been removed
    Deletion(T),
}

impl<T> DecodedValue<T> {
    /// Consume this `DecodedValue` to return the value
    pub fn into_value(self) -> T {
        match self {
            DecodedValue::Addition(v) => v,
            DecodedValue::Deletion(v) => v,
        }
    }

    /// Borrow the value within this decoded value.
    pub fn value(&self) -> &T {
        match self {
            DecodedValue::Addition(v) => v,
            DecodedValue::Deletion(v) => v,
        }
    }

    /// Returns true if this decoded value is a deletion
    pub fn is_deletion(&self) -> bool {
        matches!(self, DecodedValue::Deletion(_))
    }
}
