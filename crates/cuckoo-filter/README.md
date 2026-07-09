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
hash and the alternate window is `w0 ^ offset` (`b = log2(num_windows)`, window count a power of
two). The `offset` uses only the low `min(b, SEGMENT_SHIFT)` bits, so **both windows fall in the
same `2^SEGMENT_SHIFT`-slot segment** — each segment is an independent windowed cuckoo. The
fingerprint is a byte from the bits above the position bits.

## Construction

The key set is known up front, so construction radix-sorts the keys into their segments and fills
each **cache-resident segment** with classic cuckoo **random-walk** (place into a free window slot,
else evict and relocate, bounded by a kick limit; retried with a fresh seed on failure). Since a
key's whole random walk stays inside one segment, the build touches only L2-resident memory and its
throughput stays roughly **flat as the filter grows past cache** (~80 M keys/s even at 60 M+ keys,
vs ~20 M/s for a global-table cuckoo or a fuse filter). The working table stores each key's hash
directly in its slot, so a relocation reads the victim's hash straight from the slot it swaps.

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

* **Lookup**: the cuckoo filter beats the fuse filter at large sizes (~2× at 15 M keys) and is
  competitive in cache (two parallel cache-line probes and one hash vs three scattered probes).
* **Build**: segment-local construction stays cache-resident, so throughput is roughly flat with
  size (~80 M keys/s) — several times faster than the fuse filter at tens of millions of keys,
  where a global-table cuckoo or fuse build falls to ~20 M/s on random DRAM access.

## Usage

```rust
use cuckoo_filter::CuckooFilter;

// Keys are pre-hashed to distinct u64s by the caller.
let keys: Vec<u64> = (0..1_000u64).map(|i| i.wrapping_mul(0x9E3779B97F4A7C15)).collect();

let filter = CuckooFilter::construct(&keys).expect("construction succeeds");
assert!(keys.iter().all(|&k| filter.contains(k)));
```
