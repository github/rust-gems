# Hash Functions

This library has a trait, `HashFunctions` which is used to create the hashes required to place your symbol into the range of coded symbols.

The following documentation provides more details on this trait in particular. How and why this is done is explained in the `overview.md` documentation.

## Hash stability

When using HRIBLT in production systems it is important to consider the stability of your hash functions.

We provide a `DefaultHashFunctions` type which is a wrapper around the `DefaultHasher` type provided by the Rust standard library. Though the seed for this function is fixed, it should be noted that the hashes produces by this type are *not* guaranteed to be stable across different versions of the Rust standard library. As such, you should not use this type for any situation where clients might potentially be running on a binary built with an unspecified version of Rust.

We recommend you implement your own `HashFunctions` implementation with a stable hash function.

## Hash value hashing trick

If the value you're inserting into the encoding session is a high entropy random value, such as a cryptographic hash digest, you can recycle the bytes in that value to produce the coded symbol indexing hashes, instead of hashing that value again. This results in a constant-factor speed up.

For example if you were trying to find the difference between two sets of documents, instead of each coded symbol being the whole document it could instead just be a SHA1 hash of the document content. Since each SHA1 digest has 20 bytes of high entropy bits, instead of hashing this value five times again to produce the five coded symbol indices we can simply slice out five `u32` values from the digest itself.

This is a useful trick because hash values are often used as IDs for documents during set reconciliation since they are a fixed size, making serialization easy.
