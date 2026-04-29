# Hierarchical Rateless Bloom Lookup Tables

A novel algorithm for computing the symmetric difference between sets where the amount of data shared is proportional to the size of the difference in the sets rather than proportional to the overall size.

## Usage

Add the library to your `Cargo.toml` file.

```toml
[dependencies]
hriblt = "0.1"
```

Create two encoding sessions, one containing Alice's data, and another containing Bob's data. Bob's data might have been sent to you over a network for example.

The following example attempts to reconcile the differences between two such sets of `u64` integers, and is done from the perspective of "Bob", who has received some symbols from "Alice".

```rust
use hriblt::{DecodingSession, EncodingSession, DefaultHashFunctions};
// On Alice's computer

// Alice creates an encoding session...
let mut alice_encoding_session = EncodingSession::<u64, DefaultHashFunctions>::new(DefaultHashFunctions, 0..128);

// And adds her data to that session, in this case the numbers from 0 to 10.
for i in 0..=10 {
    alice_encoding_session.insert(i);
}

// On Bob's computer

// Bob creates his encoding session, note that the range **must** be the same as Alice's
let mut bob_encoding_session = EncodingSession::<u64, DefaultHashFunctions>::new(DefaultHashFunctions, 0..128);

// Bob adds his data, the numbers from 5 to 15.
for i in 5..=15 {
    bob_encoding_session.insert(i);
}

// "Subtract" Bob's coded symbols from Alice's, the remaining symbols will be the symmetric
// difference between the two sets, iff we can decode them. This is a commutative function so you
// could also subtract Alice's symbols from Bob's and it would still work.
let merged_sessions = alice_encoding_session.merge(bob_encoding_session, true);

let decoding_session = DecodingSession::from_encoding(merged_sessions);

assert!(decoding_session.is_done());

let mut diff = decoding_session.into_decoded_iter().map(|v| v.into_value()).collect::<Vec<_>>();

diff.sort();

assert_eq!(diff, [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]);

```
