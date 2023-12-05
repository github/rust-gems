//! Fast functions to read packed unsigned integers.
//!
//! This operation is performance-critical. Every read of a `u32` or `u64` field uses it. So does
//! every struct field access and every RepeatedField element access (via `PSVec::range`).

/// Read an unsigned integer of a given type `$ty` and number of input bytes `$n`.
///
/// `$n` must not be more than the number of bytes in `$ty`.
///
/// (This can't be written as a generic function because `BITS` and `from_le_bytes` are not part of
/// a convenient "UInt" trait.)
macro_rules! read_uint_case {
    ( $ty:ty, $bytes:expr, $n:literal ) => {{
        let mut tmp = [0; <$ty>::BITS as usize / 8];
        tmp[..$n].copy_from_slice(&$bytes[..$n]);
        <$ty>::from_le_bytes(tmp)
    }};
}

/// Read a little-endian `u128` from the beginning of the given byte buffer.
/// If `bytes.len() < 16`, the result is as if it were padded with zero bytes first.
pub fn read_u128_le(bytes: &[u8]) -> u128 {
    // Implementation note: This code here is equivalent to:
    //
    //     let n = bytes.len().min(16);
    //     let mut tmp = [0; 16];
    //     tmp[..n].copy_from_slice(&bytes[..n]);
    //     u128::from_le_bytes(tmp)
    //
    // which would be less code; but rustc can't optimize the `copy_from_slice`. It emits a runtime
    // call to `memcpy`, which copies the bytes one by one in a loop. This is slow.
    //
    // The `match` below is for speed. Rustc optimizes `copy_from_slice` beautifully when it knows
    // the number of bytes. Each arm of the `match` below passes a constant number of bytes to
    // `copy_from_slice`, and thus compiles to a few fast machine instructions.
    match bytes.len() {
        0 => read_uint_case!(u128, bytes, 0),
        1 => read_uint_case!(u128, bytes, 1),
        2 => read_uint_case!(u128, bytes, 2),
        3 => read_uint_case!(u128, bytes, 3),
        4 => read_uint_case!(u128, bytes, 4),
        5 => read_uint_case!(u128, bytes, 5),
        6 => read_uint_case!(u128, bytes, 6),
        7 => read_uint_case!(u128, bytes, 7),
        8 => read_uint_case!(u128, bytes, 8),
        9 => read_uint_case!(u128, bytes, 9),
        10 => read_uint_case!(u128, bytes, 10),
        11 => read_uint_case!(u128, bytes, 11),
        12 => read_uint_case!(u128, bytes, 12),
        13 => read_uint_case!(u128, bytes, 13),
        14 => read_uint_case!(u128, bytes, 14),
        15 => read_uint_case!(u128, bytes, 15),
        _ => read_uint_case!(u128, bytes, 16),
    }
}

/// Read a little-endian `u64` from the beginning of the given byte buffer.
/// If `bytes.len() < 8`, the result is as if it were padded with zero bytes first.
pub fn read_u64_le(bytes: &[u8]) -> u64 {
    match bytes.len() {
        0 => read_uint_case!(u64, bytes, 0),
        1 => read_uint_case!(u64, bytes, 1),
        2 => read_uint_case!(u64, bytes, 2),
        3 => read_uint_case!(u64, bytes, 3),
        4 => read_uint_case!(u64, bytes, 4),
        5 => read_uint_case!(u64, bytes, 5),
        6 => read_uint_case!(u64, bytes, 6),
        7 => read_uint_case!(u64, bytes, 7),
        _ => read_uint_case!(u64, bytes, 8),
    }
}

/// Read a little-endian `u32` from the beginning of the given byte buffer.
/// If `bytes.len() < 4`, the result is as if it were padded with zero bytes first.
pub fn read_u32_le(bytes: &[u8]) -> u32 {
    match bytes.len() {
        0 => read_uint_case!(u32, bytes, 0),
        1 => read_uint_case!(u32, bytes, 1),
        2 => read_uint_case!(u32, bytes, 2),
        3 => read_uint_case!(u32, bytes, 3),
        _ => read_uint_case!(u32, bytes, 4),
    }
}
