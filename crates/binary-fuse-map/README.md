# binary-fuse-map

A static, immutable **binary fuse filter+map**: it associates a set of distinct `u64` keys with
fixed-size values (up to 6 bytes) using only ~1.13 slots per key, and embeds a two-byte fingerprint
so lookups can also reject keys that were never inserted.

It is the map generalisation of the [binary fuse filter](https://arxiv.org/abs/2201.01174)
(Graf & Lemire, 2022). Every slot is a fixed 8-byte word holding an `N`-byte value plus a two-byte
fingerprint, and a key's payload is the XOR of its three slots:

```text
(value(k), fingerprint(k)) == slots[h0(k)] ^ slots[h1(k)] ^ slots[h2(k)]
```

## Properties

* **Compact**: ~1.13 slots/key; each slot is a fixed 8-byte word (value plus a two-byte
  fingerprint), so ~72 bits/key regardless of value width.
* **Fast `O(1)` lookups**: three segment-local aligned 64-bit slot loads, XOR-ed together, plus a
  two-byte fingerprint check.
* **Filter + map**: [`BinaryFuseMap::get`] returns `Some(value)` for inserted keys and rejects
  ~`65535/65536` of absent keys via the fingerprint (the rest are false positives returning an
  arbitrary value).
* **Static**: built once from all keys; not modifiable afterwards.

## Usage

```rust
use binary_fuse_map::BinaryFuseMap;

// Keys are pre-hashed to distinct u64s by the caller; values are fixed byte slices.
let keys: Vec<u64> = (0..1_000u64).map(|i| i.wrapping_mul(0x9E3779B97F4A7C15)).collect();
let values: Vec<[u8; 6]> = (0..1_000u64)
    .map(|i| {
        let mut v = [0u8; 6];
        v.copy_from_slice(&i.to_le_bytes()[..6]);
        v
    })
    .collect();

let map = BinaryFuseMap::<6>::try_construct(&keys, &values).expect("distinct keys");
assert_eq!(map.get(keys[42]), Some(values[42]));
```

## Benchmarks

Criterion benchmarks for small/medium sizes live in `benchmarks/`:

```sh
cargo bench -p binary-fuse-map-benchmarks
```

The headline large-scale construction measurement (criterion's repeated sampling is impractical at
hundreds of millions of keys) is an example:

```sh
cargo run --release --example large_construction -- 400000000
```

On a 16-core / 64 GiB machine this builds a 400-million-entry map (5-byte values) in ~33 s
(~12 M keys/s) using ~21 GiB of peak working memory, then serves ~52 M lookups/s.

## License

MIT
