# ribbon-map

A static, immutable **ribbon filter+map**: it associates a set of distinct `u64` keys with
fixed-size `[u8; N]` values, and embeds a small fingerprint so lookups can also reject most keys that
were never inserted.

It is a *map* built on a [ribbon retrieval](https://arxiv.org/abs/2103.02515) structure (Dillinger,
Hübschle-Schneider, Sanders & Walzer, 2021). A ribbon filter solves a banded linear system over
GF(2) so that a windowed dot product over a solution vector reproduces each key's fingerprint. This
crate keeps that machinery but changes what those bits mean.

## From a ribbon filter to a ribbon map

* **The diagonal is a perfect hash.** Ribbon construction is incremental Gaussian elimination; each
  key ends up owning a distinct pivot column — its place on the matrix diagonal. That pivot is used
  to index a side array holding the key's value.
* **Store the deviation, not a fingerprint.** To recover a key's pivot at query time, the retrieval
  structure stores the *deviation* `d = pivot − start`: how far the diagonal sits from the key's hash
  position. Processing keys in start order provably confines every pivot to the key's own width-`W`
  window, so for `W = 64` the deviation always fits in **6 bits**.
* **2 spare bits are a zero-check.** With `W = 64` the deviation uses 6 of an 8-bit retrieval word.
  The other **2 bits are forced to zero for every inserted key**, and the linear system's free
  variables — columns no key pivots on — are filled with **random values**. A present key always
  reads back zero there; an absent key reads pseudo-random bits and is rejected ~3/4 of the time,
  with no fingerprint hash to store.
* **Interleaved storage for `GF2P8AFFINEQB`.** The solution is stored bit-sliced, as the Ribbon
  paper proposes: the 8 bit-planes of each 64-column block sit in 8 consecutive `u64`s (one cache
  line), so retrieval-word bit `p` is `parity(coeff & plane_p_window)`. On x86 with GFNI+AVX-512 all
  8 parities become a single `GF2P8AFFINEQB`; elsewhere a portable scalar reduction runs. Both paths
  are validated bit-for-bit against each other.

```text
word[p] = parity(coeff & plane_p_window(start))   // 8 planes -> 1 GF2P8AFFINEQB on x86
        = deviation                               // top 2 bits are zero for present keys
value   = values[start + (word & 0x3F)]
```

## Properties

* **Filter + map**: [`RibbonMap::get`] returns `Some(value)` for inserted keys and rejects ~3/4 of
  absent keys via a 2-bit zero-check (the rest are false positives returning an arbitrary value).
  Widen the zero-checked prefix if you need stronger rejection.
* **Compact**: ~`1.18` slots/key (load factor ~0.85); each slot costs one byte of retrieval data
  plus `N` bytes of value, i.e. ~`(1 + N) · 9.4` bits/key. The value width `N` is **unbounded** —
  values live in their own array.
* **Fast `O(1)` lookups**: eight plane parities (one `GF2P8AFFINEQB` on x86, else a scalar
  reduction), a 2-bit zero-check, and a single value load.
* **Static**: built once from all keys; not modifiable afterwards.

## Usage

```rust
use ribbon_map::RibbonMap;

// Keys are pre-hashed to distinct u64s by the caller; values are fixed byte slices.
let keys: Vec<u64> = (0..1_000u64).map(|i| i.wrapping_mul(0x9E3779B97F4A7C15)).collect();
let values: Vec<[u8; 4]> = (0..1_000u32).map(|i| i.to_le_bytes()).collect();

let map = RibbonMap::<4>::try_construct(&keys, &values).expect("distinct keys");
assert_eq!(map.get(keys[42]), Some(values[42]));
```

## License

MIT
