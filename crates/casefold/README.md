# casefold

A **fast** Unicode simple case-folding library for Rust, backed by a **very
compact** (~1.7 KB) paged-bitmap + run-length table. It folds whole strings at
multiple GiB/s — several × faster than a `HashMap` fold table — while using
~10× less memory than that hash map of the same data.

`simple_fold(s: String) -> String` maps a string to its lower-case fold
form, as defined by the Unicode [CaseFolding.txt][cf] data file restricted to
the **simple** (1-to-1) folds (statuses `C` and `S`). Full multi-character
folds (`F`, e.g. `ß` → `ss`) and Turkic locale folds (`T`) are not supported.

[cf]: https://www.unicode.org/Public/UCD/latest/ucd/CaseFolding.txt

The output is always valid UTF-8. ASCII is lowercased in place in the input's
own buffer (one auto-vectorized pass); the multibyte tail is scanned and the
original allocation is returned untouched unless a character actually folds, so
text that never folds (CJK, Kana, Arabic, Hebrew, …) pays nothing. A second
buffer is allocated only on the first real fold, since folds can change UTF-8
length (U+212A KELVIN SIGN → `k`, U+023A Ⱥ → U+2C65 ⱥ).

```rust
use casefold::simple_fold;
assert_eq!(simple_fold("Hello, WORLD!".to_string()), "hello, world!");
assert_eq!(simple_fold("ÜBER".to_string()), "über");
```

## Why does this crate exist?

Unicode 16.0 defines 1484 simple-fold mappings. Common ways to store them:

| Representation                                        | Size        |
|-------------------------------------------------------|-------------|
| Naïve `[(u32, u32); 1484]`                            | ~11.6 KB    |
| `regex-syntax::unicode_tables::case_folding_simple`   | ~70 KB      |
| Go `unicode.SimpleFold` (orbit + ASCII + ranges)      | ~7.3 KB     |
| **This crate (paged bitmap + packed runs)**           | **1776 B**  |

That is **9.6 bits per fold entry** — a little over half of it the `BYTE_DELTA`
side table that powers the decode-free fold path; the index + run records alone
are ~4.4 bits per entry.

## How the encoding works

A few observations make the data both highly compressible and decode-free to
query:

1. **Folds come in runs.** Adjacent code points share a fold delta (`A`–`Z` all
   map with `+32`); a 1-bit `stride` flag also covers *alternating* runs like
   Latin-Extended `0x0100, 0x0102, …` where every second code point folds.
2. **Runs cluster in pages.** Splitting every run at 64-cp page boundaries (and
   wherever the byte delta changes) leaves 238 runs in just 59 of ~1960 pages.
   A page-presence bitmap plus a cumulative-popcount sidetable answers "does
   this page hold any run?" in one bit test — and since runs never cross a page,
   an unset bit is a *definitive* "no fold".
3. **A run is two clean bytes.** Both ends fit in 6 bits, split across
   `RUN_END_LOW[i]` (`end & 0x3F`, the scan key) and `RUN_START_STRIDE[i]`
   (`start & 0x3F | (stride−1) << 6`). The hot scan compares `RUN_END_LOW`
   byte-to-byte against `cp & 0x3F` — no mask, no shift, no code-point
   reconstruction — reading `RUN_START_STRIDE` only on a hit.
4. **Whole characters are rejected from their lead bytes.** For a 2-/3-byte
   sequence the page index `cp >> 6` is fixed by the first one or two bytes, so
   the bulk path probes `PAGE_BITMAP` straight from them and copies fold-free
   characters (CJK, Hangul, Kana, …) verbatim without ever assembling `cp`.
5. **Folding is a little-endian byte add.** The folded character is the source
   bytes read as a `u32` plus a per-run `BYTE_DELTA[i]` (a full 32 b, since low
   code-point bits land in the high word byte): a masked 4-byte load, one
   `wrapping_add`, one 4-byte store — no decode, no encode. Writing fewer/more
   bytes than were read handles length-changing folds (`K`→`k`, `Ⱥ`→`ⱥ`).

### Table layout (1776 B total)

| Component                                       | Bytes |
|-------------------------------------------------|------:|
| `PAGE_BITMAP[31]: u64` (1 bit per 64-cp page)   |   248 |
| `POPCNT_SAMPLES[32]: u8` (cumulative popcount)  |    32 |
| `PAGE_OFFSET[60]: u8` (per populated page)      |    60 |
| `RUN_END_LOW[238 + 8]: u8` (clean scan key, `end & 0x3F`; +8 SWAR pad) | 246 |
| `RUN_START_STRIDE[238]: u8` (`start & 0x3F` \| stride bit) | 238 |
| `BYTE_DELTA[238]: u32` (little-endian fold delta per run) | 952 |
| **Total**                                       | **1776** |

(Splitting runs at byte-delta boundaries raises the run count from 227 to 238.)
The data file is parsed at build time by `build.rs`, which emits the packed
`static` tables to `OUT_DIR/table.rs`.

### Lookup

`simple_fold` folds one multibyte character at byte offset `read` like so
(ASCII is already lowercased by the in-place pass, so it never reaches here):

```text
fold_char(bytes, read):
    page     = page index from bytes[read] (+1-2 continuation bytes)  # cp >> 6
    if PAGE_BITMAP bit for page is clear: copy `len` bytes verbatim   # no fold
    low      = bytes[read + len - 1] & 0x3F             # within-page offset
    idx      = run_in_page(page, low)                   # one bitmap test
    ss       = RUN_START_STRIDE[idx]                    #   + chunked scan
    start_lo = ss & 0x3F
    stride_b = ss >> 6
    if low < start_lo: copy verbatim                    # in a gap between runs
    if (low - start_lo) & stride_b != 0: copy verbatim  # in a stride-2 gap
    word = utf8_le(bytes[read..]) + BYTE_DELTA[idx]     # fold by byte add
    write word (dest_len bytes)
```

Test the character's `PAGE_BITMAP` bit (clear ⇒ no fold). On a hit, the dense
page index is `POPCNT_SAMPLES[page/64] + popcount(PAGE_BITMAP[page/64] & ((1 <<
(page%64)) − 1))`, and a short scan of `RUN_END_LOW[PAGE_OFFSET[dense] ..]`
finds the first end `>= low` — a raw `u8` compare, no masking. Because runs
never cross a page that run is the only candidate, and (since the scan
guarantees `low <= end_low`) membership is just `low >= start_low`, both 6-bit
offsets, no `cp` reconstruction. Pages hold ≤30 runs (~3.8 on average), so every
lookup touches only small, branch-predictable, cache-friendly arrays.

## Performance

The byte path returns the input allocation untouched unless a character folds;
otherwise it builds the output with a raw write cursor (bulk-copied unmodified
spans + masked `BYTE_DELTA` folds). Its within-page scan is a chunked SWAR scan
(8 `end_low` bytes at a time, branchless), whose latency the per-character
pipeline hides.

Throughput on an Apple M-series machine (criterion medians). The **true
case-folders** produce the same output as `simple_fold`:

| Workload (input size) | `simple_fold` | `simd_normalizer` | `HashMap` (byte path) |
|---|--:|--:|--:|
| Pure ASCII (5 700 B)                   | **40.8 GiB/s** |  1.21 GiB/s | 213 MiB/s |
| CJK, no folds (8 100 B)                | **2.95 GiB/s** |  1.97 GiB/s | 558 MiB/s |
| Symbols / Myanmar, no folds (9 000 B)  | **2.96 GiB/s** |  1.56 GiB/s | 410 MiB/s |
| Mixed BMP, all folding (8 800 B)       |   869 MiB/s    | **922 MiB/s**| 334 MiB/s |
| Length-changing folds (1 700 B)        | **1.26 GiB/s** |  716 MiB/s  | 233 MiB/s |

The standard-library routines below perform Unicode **lowercasing**, *not* case
folding — a different operation with different output (they diverge on e.g.
final-sigma, `İ`, `ß`; `to_ascii_lowercase`† leaves all multibyte sequences
untouched). They are included only as a throughput reference for the same
workloads, not as output-equivalent alternatives:

| Workload (input size) | `simple_fold` (fold) | `str::to_lowercase` | `chars().flat_map` | `to_ascii_lowercase`† |
|---|--:|--:|--:|--:|
| Pure ASCII (5 700 B)                   | **40.8 GiB/s** | 26.1 GiB/s | 383 MiB/s | 21.2 GiB/s |
| CJK, no folds (8 100 B)                | **2.95 GiB/s** | 473 MiB/s  | 369 MiB/s | 22.9 GiB/s |
| Symbols / Myanmar, no folds (9 000 B)  | **2.96 GiB/s** | 497 MiB/s  | 348 MiB/s | 22.9 GiB/s |
| Mixed BMP, all folding (8 800 B)       |   869 MiB/s    | 287 MiB/s  | 205 MiB/s | 21.1 GiB/s |
| Length-changing folds (1 700 B)        | **1.26 GiB/s** | 492 MiB/s  | 269 MiB/s | 15.9 GiB/s |

† `to_ascii_lowercase` is shown only as the "memcpy + ASCII-lowercase" speed
floor.

Against the true case-folders, `simple_fold` leads every workload except
all-folding mixed-BMP, where `simd-normalizer` edges ahead (922 vs 869 MiB/s).
Two highlights: no-fold text runs at GiB/s by probing `PAGE_BITMAP` and
returning the buffer as-is, and the compact table beats the `HashMap` by 3–5×
on the *identical* byte-level fold — plus an ASCII fast path the `HashMap`
lacks (40 GiB/s vs 213 MiB/s).

Reproduce with:

```
cargo bench -p casefold-benchmarks
```

## License

This crate is licensed under the MIT License. The vendored
`data/CaseFolding.txt` is part of the Unicode Character Database, redistributed
under the [Unicode terms of use](https://www.unicode.org/terms_of_use.html).
