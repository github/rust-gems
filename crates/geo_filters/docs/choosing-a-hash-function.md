# Choosing a hash function

## Reproducibility

This library uses hash functions to assign values to buckets deterministically. The same item
will hash to the same value, and modify the same bit in the geofilter.

When comparing geofilters it is important that the same hash functions, using the same seed
values, have been used for *both* filters. Attempting to compare geofilters which have been
produced using different hash functions or the same hash function with different seeds will
produce nonsensical results.

Similar to the Rust standard library, this crate uses the `BuildHasher` trait and creates
a new `Hasher` for every item processed.

To help prevent mistakes caused by mismatching hash functions or seeds we introduce a trait
`ReproducibleBuildHasher` which you must implement if you wish to use a custom hashing function.
By marking a `BuildHasher` with this trait you're asserting that `Hasher`s produced using
`Default::default` will hash identical items to the same `u64` value across multiple calls
to `BuildHasher::hash_one`.

The following is an example of some incorrect code which produces nonsense results:

```rust
use std::hash::RandomState;

// Implement our marker trait for `RandomState`.
// You should _NOT_ do this as `RandomState::default` does not produce
// reproducible hashers.
impl ReproducibleBuildHasher for RandomState {}

#[test]
fn test_different_hash_functions() {
    // The last parameter in this FixedConfig means we're using RandomState as the BuildHasher
    pub type FixedConfigRandom = FixedConfig<Diff, u16, 7, 112, 12, RandomState>;

    let mut a = GeoDiffCount::new(FixedConfigRandom::default());
    let mut b = GeoDiffCount::new(FixedConfigRandom::default());

    // Add our values
    for n in 0..100 {
        a.push(n);
        b.push(n);
    }

    // We have inserted the same items into both filters so we'd expect the
    // symmetric difference to be zero if all is well.
    let diff_size = a.size_with_sketch(&b);

    // But all is not well. This assertion fails!
    assert_eq!(diff_size, 0.0);
}
```

The actual value returned in this example is ~200. This makes sense because the geofilter
thinks that there are 100 unique values in each of the filters, so the difference is approximated
as being ~200. If we were to rerun the above example with a genuinely reproducable `BuildHasher`
then the resulting diff size would be `0`.

In debug builds, as part of the config's `eq` implementation, our library will assert that the `BuildHasher`s
produce the same `u64` value when given the same input but this is not enabled in release builds.

## Stability

Following from this, it might be important that your hash functions and seed values are stable, meaning,
that they won't change from one release to another.

The default function provided in this library is *NOT* stable as it is based on the Rust standard libraries
`DefaultHasher` which does not have a specified algorithm and may change across releases of Rust.

Stability is especially important to consider if you are using serialized geofilters which may have
been created in a previous version of the Rust standard library.

This library provides an implementation of `ReproducibleBuildHasher` for the `FnvBuildHasher` provided
by the `fnv` crate version `1.0`. This is a _stable_ hash function in that it won't change unexpectedly
but it doesn't have good diffusion properties. This means if your input items have low entropy (for
example numbers from `0..10000`) you will find that the geofilter is not able to produce accurate estimations.

## Uniformity and Diffusion

In order to produce accurate estimations it is important that your hash function is able to produce evenly
distributed outputs for your input items.

This property must be balanced against the performance requirements of your system as stronger hashing
algorithms are often slower.

Depending on your input data, different functions may be more or less appropriate. For example, if your input
items have high entropy (e.g. SHA256 values) then the diffusion of your hash function might matter less.

## Implementing your own `ReproducibleBuildHasher` type

If you are using a hash function that you have not implemented yourself you will not be able to implement
`ReproducibleBuildHasher` on that type directly due to Rust's orphan rules. The easiest way to get around this
is to create a newtype which proxies the underlying `BuildHasher`.

In addition to `BuildHasher` `ReproducibleBuildHasher` needs `Default` and `Clone`, which is usually implemented
on `BuildHasher`s, so you can probably just `#[derive(...)]` those. If your `BuildHasher` doesn't have those
traits then you may need to implement them manually.

Here is an example of how to use new types to mark your `BuildHasher` as reproducible.

```rust
#[derive(Clone, Default)]
pub struct MyBuildHasher(BuildHasherDefault<DefaultHasher>);

impl BuildHasher for MyBuildHasher {
    type Hasher = DefaultHasher;

    fn build_hasher(&self) -> Self::Hasher {
        self.0.build_hasher()
    }
}

impl ReproducibleBuildHasher for MyBuildHasher {}
```
