# cuckoo-filter

A static **windowed cuckoo filter**: an approximate-membership structure that stores a one-byte
fingerprint per key. Instead of a classic cuckoo filter's "two buckets of four slots each", every
key gets **two hash locations that each expand into a *window* of `WINDOW` consecutive single
slots** (`WINDOW = 4` by default). Both windows are drawn from a single shared slot array, so a key
has up to `2 * WINDOW = 8` candidate slots. Overlapping windows of neighbouring keys share slots,
which gives the same cascading placement freedom that makes binary fuse filters dense — a key in a
crowded region can slide a few slots over.

## Addressing

A **single hash** is split with the classic cuckoo-filter XOR trick: `w0` is the low `b` bits of the
hash, the alternate window is `w1 = w0 ^ (top b bits)`, and the fingerprint is a middle byte
(`b = log2(num_windows)`). This costs one hash and a few bit ops per lookup, but requires the window
count to be a **power of two** so the XOR stays in range (see the sizing note below).

## Construction

Because the whole key set is known up front, the filter is built once by classic cuckoo
**random-walk**: each key is placed into a free slot of one of its two windows, evicting and
relocating incumbents on collision, up to a fixed kick limit. Construction tracks the owning key of
every slot (so no partial-key relocation is needed) and retries with a fresh seed if an attempt
fails, which is vanishingly unlikely below the load threshold.

## Properties

* **Two windows of `WINDOW` slots**, one-byte fingerprints (`0` marks empty). A lookup checks both
  windows (`2 * WINDOW = 8` slots), so an absent key is a false positive with probability
  ~`8/255` (~3%); use a wider fingerprint for a lower rate.
* **`O(1)` lookups**: a single hash, then the two windows are probed with a non-short-circuiting
  `|`, so their cache lines are fetched in parallel. Touching only two cache lines (vs three for a
  binary fuse filter) makes lookups faster than a fuse filter across sizes — see `benchmarks/`.
* **Power-of-two sizing**: the XOR addressing rounds the window count up to a power of two. Sized to
  fill that table (`n ≈ 0.92·2^b`) it costs nothing (~8.7 bits/key, load ≈ the cuckoo threshold);
  for an unlucky `n` just past a boundary the load can halve, so prefer capacities near `0.9·2^b`.
* **Static**: built once from a set of distinct pre-hashed `u64` keys.

## Benchmarks

`benchmarks/` compares construction and lookup throughput against the `binary-fuse-map` on the same
keys, at power-of-two window counts filled to ~0.92 (so both are at a fair ~8.7 bits/key):
`cargo bench -p cuckoo-filter-benchmarks`. Roughly, on random `u64` keys:

* **Lookup**: the cuckoo filter beats the fuse filter at every size — from ~10% in-cache to ~1.8×
  at 15 M keys (two parallel cache-line probes and one hash vs three scattered probes).
* **Build**: random-walk is faster than the fuse filter until ~15 M keys, then a little slower.

## Usage

```rust
use cuckoo_filter::CuckooFilter;

// Keys are pre-hashed to distinct u64s by the caller.
let keys: Vec<u64> = (0..1_000u64).map(|i| i.wrapping_mul(0x9E3779B97F4A7C15)).collect();

let filter = CuckooFilter::construct(&keys).expect("construction succeeds");
assert!(keys.iter().all(|&k| filter.contains(k)));
```
