# binary-fuse-map

A static, immutable **binary fuse filter+map**: it associates a set of distinct `u64` keys with
fixed-size values (up to 8 bytes) using only ~1.13 slots per key, and embeds a one-byte fingerprint
so lookups can also reject keys that were never inserted.

It is the map generalisation of the [binary fuse filter](https://arxiv.org/abs/2201.01174)
(Graf & Lemire, 2022). Every slot stores an `N`-byte value plus a fingerprint byte, and a key's
payload is the XOR of its three slots:

```text
(value(k), fingerprint(k)) == slots[h0(k)] ^ slots[h1(k)] ^ slots[h2(k)]
```

## Properties

* **Compact**: ~1.13 slots/key; each slot is `value_bytes + 1`. For 8-byte values that is ~81
  bits/key (64 value + 8 fingerprint + ~9 structural overhead).
* **Fast `O(1)` lookups**: three segment-local slot loads, XOR-ed with a single unaligned 64-bit
  read per slot, plus a one-byte fingerprint check.
* **Filter + map**: [`BinaryFuseMap::get`] returns `Some(value)` for inserted keys and rejects
  ~`255/256` of absent keys via the fingerprint (the rest are false positives returning an
  arbitrary value).
* **Static**: built once from all keys; not modifiable afterwards.

## Usage

```rust
use binary_fuse_map::BinaryFuseMap;

// Keys are pre-hashed to distinct u64s by the caller; values are fixed byte slices.
let keys: Vec<u64> = (0..1_000u64).map(|i| i.wrapping_mul(0x9E3779B97F4A7C15)).collect();
let values: Vec<[u8; 8]> = (0..1_000u64).map(|i| i.to_le_bytes()).collect();

let map = BinaryFuseMap::<8>::try_construct(&keys, &values).expect("distinct keys");
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

On a 16-core / 64 GiB machine this builds a 400-million-entry map (8-byte values) in ~33 s
(~12 M keys/s) using ~21 GiB of peak working memory, then serves ~52 M lookups/s.

## License

MIT
