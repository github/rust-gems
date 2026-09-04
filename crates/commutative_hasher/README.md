# Commutative Hasher

An order-independent hasher that hashes a large byte stream in any order, from any number of threads, and always get the
same digest.

The stream is split into fixed-size blocks. Each block is hashed, mapped to a point on the Ristretto group over
Curve25519, and all the points are summed. Because point addition is commutative and associative, blocks can be hashed
in any order, in parallel, and combined later.

To prevent collisions due to reordering, each block is hashed together with its byte offset and length, so a block only
ever contributes the same point at the same position in the stream. This means the block size chosen is critical; for
any data stream larger than one block, the hash value will be different for different block sizes. Each block must only
be added once; adding a block multiple times will result in a different hash value.

## Motivation

Verifying a large object usually means a linear pass with a conventional hash: bytes must arrive in order, on one
thread. That is a poor fit when data is uploaded or downloaded as ranged parts, sharded across workers, or written by a
pipeline that finishes chunks out of order.

`commutative_hasher` lets each worker hash the part it has access to, and lets the results be merged into a single
digest for the whole object, without buffering the object or serializing the work.

## Usage

Add `commutative_hasher` and `sha2 = 0.11` to your `Cargo.toml`.

The hashers are generic over any `digest::Digest` with a 64-byte output, since mapping into the Ristretto group requires
64 bytes of hash input. `Sha512` is the usual choice, but others would work too.

It is critical to make a future-proof choice of block size. The block size is difficult to change after the fact, as the
same data hashed with different block sizes will produce different results if the data is larger than the block size.

### Parallel

Use `ParallelHasher` to hash a stream of data out-of-order from multiple threads. A block size must be provided at
construction, and each block is hashed with something like `Sha512` first. All data must be provided to `update` in
multiples of the block size. Larger blocks are more efficient for the computation but limit the granularity of the
processed data.

```rust
use std::num::NonZeroUsize;
use commutative_hasher::ParallelHasher;
use rayon::prelude::*;
use sha2::Sha512;

const BLOCK_SIZE: NonZeroUsize = NonZeroUsize::new(4).unwrap();

fn hash_parallel() {
    let data = b"thisthatmoreverylong";
    let hasher = ParallelHasher::<Sha512>::new(BLOCK_SIZE);

    // Parts may be hashed in any order, on any thread.
    data.par_chunks(8)
        .enumerate()
        .try_for_each(|(i, chunk)| hasher.update(i * 8, chunk))
        .expect("hashing to succeed");

    // Same digest as the sequential example.
    assert_eq!(
        hasher.finalize().hex_digest(),
        "9258f9de43401cd6e8f55545754b84ac58257ec7779723c9790a986daab18206"
    );
}
```

### Sequential

`SequentialHasher` generates the same hashes as `ParallelHasher` but does not require that the data be provided in
multiples of the block size.

```rust
use std::num::NonZeroUsize;
use commutative_hasher::SequentialHasher;
use sha2::Sha512;

const BLOCK_SIZE: NonZeroUsize = NonZeroUsize::new(4).unwrap();

fn hash_sequential() {
    let mut hasher = SequentialHasher::<Sha512>::new(BLOCK_SIZE);
    hasher.update(b"this");
    hasher.update(b"that");
    hasher.update(b"moreverylong");
    let digest = hasher.finalize();

    assert_eq!(
        digest.hex_digest(),
        "9258f9de43401cd6e8f55545754b84ac58257ec7779723c9790a986daab18206"
    );

    // Or, for data you already have in memory:
    let digest = SequentialHasher::<Sha512>::digest_from_bytes(BLOCK_SIZE, b"thisthatmoreverylong");
}
```

### Digests

A `CommutativeHashDigest` is fundamentally a 32-byte compressed Ristretto point. It can be converted to bytes, a hex
string, serialized/deserialized, and stored inline in packed on-disk structures.

```rust
use commutative_hasher::CommutativeHashDigest;

fn digest() {
    ...
    let bytes: [u8; 32] = digest.to_bytes();
    let hex: String = digest.hex_digest();
    let parsed = CommutativeHashDigest::decode_hex_digest(&hex).expect("decode to succeed");
    assert_eq!(parsed, digest);
}
```
