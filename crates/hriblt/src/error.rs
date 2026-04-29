/// Errors raised during set reconciliation
#[derive(thiserror::Error, Debug)]
pub enum SetReconciliationError {
    /// Expected hashers to match.
    #[error("coded symbol hasher mismatched")]
    MismatchedHasher,
    /// The provided split does not lie within the provided range
    #[error("split does not lie within the provided range")]
    SplitOutOfRange,
    /// Expected a provided range to follow on from the current range.
    #[error("provided coded symbol range did not follow on from previous range")]
    NonContiguousRanges,
    /// Expected ranges to be the same.
    #[error("provided coded symbol range did not match the previous range")]
    MismatchedRanges,
    /// The range of the provided coded symbols did not begin at zero.
    #[error("provided coded symbol range did not start from zero")]
    NotInitialRange,
    /// The length of the range and the length of the coded symbols do not match.
    #[error("number of coded symbols did not match the length of the provided range")]
    RangeLengthMismatch,
}
