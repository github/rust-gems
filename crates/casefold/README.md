# casefold

A compact Unicode simple case-folding table for Rust.

`simple_fold(s: String) -> String` maps a string to its lower-case fold
form, as defined by the Unicode [CaseFolding.txt][cf] data file restricted to
the **simple** (1-to-1) folds (statuses `C` and `S`). Full multi-character
folds (`F`, e.g. `ß` → `ss`) and Turkic locale folds (`T`) are not supported.

[cf]: https://www.unicode.org/Public/UCD/latest/ucd/CaseFolding.txt

`simple_fold` consumes a `String` and returns a `String` (the fold is always
valid UTF-8). ASCII is lowercased in place via the string's existing heap
allocation (a single auto-vectorized pass over the bytes), and the multibyte
tail is scanned for the first character that actually folds; if none does, the
original allocation is returned untouched — so text whose multibyte content
never folds (CJK, Kana, Arabic, Hebrew, …) pays nothing. A fresh buffer is
allocated only once a real fold is found, since simple folds can change UTF-8
length (e.g. U+212A KELVIN SIGN → `k`, or U+023A Ⱥ → U+2C65 ⱥ).

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
| `regex-syntax::unicode_tables::case_folding_simple`   | ~67 KB src  |
| Go `unicode.SimpleFold` (orbit + ASCII + ranges)      | ~7.3 KB     |
| **This crate (paged bitmap + packed runs)**           | **1776 B**  |

That is 9.6 bits per fold entry. A little over half of that is the
`BYTE_DELTA` side table that powers the decode-free bulk fold path (see
below); the index + run records alone are ~4.4 bits per entry.

## How the encoding works

Several observations make the data extraordinarily compressible *and* fast to
query, and let the bulk byte path fold without ever decoding a character:

1. **Most folds occur in runs.** Stretches of adjacent code points share the
   same delta to their fold, e.g. `U+0041..U+005A` (`A`–`Z`) all map with
   delta `+32`. A great many Latin-Extended pairs come in *alternating*
   sequences like `0x0100, 0x0102, 0x0104 …` where every second code point
   folds. A 1-bit `stride` flag lets the same run encoding cover both cases.

2. **Run ends cluster in a few 64-cp pages.** After splitting any run that
   crosses a 64-cp boundary (and, for the byte-delta fold, wherever the delta
   changes), just 238 runs live in only 59 of the ~1960 possible pages. A
   page-presence bitmap plus a cumulative popcount sidetable answers, in a
   single bit test, whether the page containing `cp` holds any run at all —
   and because every run lives entirely inside one page, an unset bit is a
   *definitive* "no fold". No cross-page successor scan is ever required.

3. **A run is two clean bytes.** Because every run is split to stay inside one
   64-cp page, *both* ends fit in 6 bits, and the record splits across two byte
   arrays: `RUN_END_LOW[i]` (`end & 0x3F`) and `RUN_START_STRIDE[i]`
   (`start & 0x3F | (stride−1) << 6`). The hot within-page scan compares
   `RUN_END_LOW` **byte-to-byte against `cp & 0x3F`** — no mask, no shift, and a
   dense `u8` array — and reads `RUN_START_STRIDE` only on a hit to confirm
   `cp & 0x3F >= start_low`. No code-point reconstruction anywhere. The fold
   itself is *not* stored here — it lives in the parallel `BYTE_DELTA` table
   (idea 5).

4. **The bulk path rejects whole characters from their first bytes.** For a
   2-/3-byte UTF-8 sequence the page index `cp >> 6` is fully determined by
   the first one or two bytes — only the final continuation byte carries the
   within-page offset `cp & 0x3F`. So `simple_fold` probes `PAGE_BITMAP`
   straight from `b0` (and `b1`); a clear page bit copies the character
   verbatim without assembling `cp` or scanning a run. This skips fold-free
   scripts (CJK, Hangul, Kana, Arabic, Hebrew, Indic) *and* the empty 64-cp
   pages inside otherwise-foldable blocks (e.g. Myanmar, punctuation/symbol
   blocks), reusing the very same `PAGE_BITMAP` the run lookup uses.

5. **Folding is a little-endian byte add.** On a little-endian machine the
   folded character is the source character's UTF-8 bytes — read as a `u32` —
   plus a per-run constant. `BYTE_DELTA[i]` stores that constant (a full 32 b,
   because the low code-point bits land in the high word byte and 3→1-byte
   shrinks must subtract whole bytes away). The bulk fold is therefore a
   masked 4-byte load, one `wrapping_add`, and a single 4-byte store — no
   UTF-8 decode, no encode — and it handles length-changing folds (`K`→`k`,
   `Ⱥ`→`ⱥ`) by simply writing fewer or more bytes than it read. Every
   length-preserving fold also works because runs are split wherever the byte
   delta would change (e.g. when the destination crosses a 64-cp boundary).

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

### Lookup algorithm

`simple_fold` folds one multibyte character at byte offset `read` like so
(ASCII is already lowercased by the in-place tier-1 pass, so it never reaches
here):

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

The scan walks `RUN_END_LOW` — a dense array of clean `end_low` bytes — and
stops at the first one `>= low`, a raw byte comparison with no masking. That
also guarantees `low <= end_low`, so membership is just `low >= start_low`,
both 6-bit within-page offsets, no `cp` reconstruction. The fold is a masked
little-endian load of the input bytes plus the run's `BYTE_DELTA`, written back
directly — no decode/encode.

`run_in_page(page, low)` returns the run whose `end_low` is the smallest one ≥
`low` *within that 64-cp page*, or "no fold". Because the build splits every run
at 64-cp boundaries, no other page can contain a run covering the character, so
the answer is either in this page or there is no fold:

1. `page = cp >> 6` — the 64-cp page containing the character.
2. Test bit `page % 64` of `PAGE_BITMAP[page / 64]`. **If clear, there is no
   fold** — the page is empty, a definitive "no fold".
3. The dense index of `page` is `POPCNT_SAMPLES[page/64] +
   popcount(PAGE_BITMAP[page/64] & ((1 << (page%64)) - 1))` — one load
   plus one masked popcount.
4. Scan `RUN_END_LOW[PAGE_OFFSET[dense] .. PAGE_OFFSET[dense+1]]` for the first
   byte that is ≥ `low` — a raw `u8` comparison, no masking. Pages hold at most
   30 runs (averaging ~3.8), so the search touches a small contiguous slice of a
   dense byte array and is branch-predictable.

Every access touches the bitmap (248 B), the popcount samples (32 B),
`PAGE_OFFSET` (60 B), `RUN_END_LOW` (238 B) and, on a hit, `RUN_START_STRIDE`
(238 B) and `BYTE_DELTA` (952 B) — all small, cache-friendly arrays.

## Performance

`simple_fold(s: String) -> String` consumes the input `String` and
auto-vectorizes a single in-place pass that lowercases ASCII and detects
whether any multibyte sequence is present. It then scans the multibyte tail
and **hands back the original allocation untouched unless a character actually
folds** — so pure-ASCII input and any text whose multibyte characters never
fold (CJK, Hangul, Kana, Arabic, Hebrew, Indic, symbols, …) avoid a second
buffer entirely. Once a real fold is found it allocates once, then builds the
output with a raw write cursor: unmodified spans are bulk-copied and each
folded character is a masked little-endian load + `BYTE_DELTA` add + 4-byte
store (no decode/encode). Its within-page run search uses a chunked SWAR scan
(8 `end_low` bytes at a time, branchlessly): the byte path's longer
per-character pipeline hides the SWAR latency and benefits from dropping the
scan's data-dependent branch.

Throughput on an Apple M-series machine (criterion medians), against the SIMD
`simd-normalizer` crate, the same byte path backed by a `HashMap` instead of
the table, and the standard library:

| Workload (input size) | `simple_fold` | `simd_normalizer` | HashMap (byte path) | `str::to_lowercase` | `chars().flat_map` | `to_ascii_lowercase`† |
|---|--:|--:|--:|--:|--:|--:|
| Pure ASCII (5 700 B)                   | **40.8 GiB/s** |   1.21 GiB/s |  213 MiB/s | 26.1 GiB/s | 383 MiB/s | 21.2 GiB/s |
| CJK, no folds (8 100 B)                | **2.95 GiB/s** |   1.97 GiB/s |  558 MiB/s | 473 MiB/s  | 369 MiB/s | 22.9 GiB/s |
| Symbols / Myanmar, no folds (9 000 B)  | **2.96 GiB/s** |   1.56 GiB/s |  410 MiB/s | 497 MiB/s  | 348 MiB/s | 22.9 GiB/s |
| Mixed BMP, all folding (8 800 B)       |   869 MiB/s    | **922 MiB/s**|  334 MiB/s | 287 MiB/s  | 205 MiB/s | 21.1 GiB/s |
| Length-changing folds (1 700 B)        | **1.26 GiB/s** |  716 MiB/s   |  233 MiB/s | 492 MiB/s  | 269 MiB/s | 15.9 GiB/s |

† `str::to_ascii_lowercase` is **not** a correct case-folder for non-ASCII —
it leaves every multibyte sequence untouched. It is shown only as the
"memcpy + ASCII-lowercase" speed floor. The two `to_lowercase` variants
perform Unicode *lowercasing*, which is not identical to case folding (they
diverge on e.g. final-sigma, `İ`, `ß`); this is an equal-workload throughput
comparison, not an output-equality one.

`simple_fold` leads every workload except all-folding mixed-BMP text,
where `simd-normalizer`'s wide-lane SIMD still edges ahead (922 vs 869 MiB/s) —
though the chunked SWAR run-scan closed most of that gap. Two things stand out:

* **The no-fold rows run at GiB/s** (CJK 3.0, symbols 3.0 — ~6× `str::
  to_lowercase`): the tail scan probes `PAGE_BITMAP` from the first one or two
  UTF-8 bytes, finds nothing folds, and returns the input buffer as-is — no
  second allocation, no byte copied.
* **The compact table beats a HashMap by 3–5×** on the *identical* byte-level
  fold (CJK 2.95 GiB/s vs 558 MiB/s, mixed-BMP 869 vs 334 MiB/s), and the
  HashMap has no ASCII fast path at all (213 MiB/s vs 40 GiB/s).

Reproduce with:

```
cargo bench -p casefold-benchmarks
```

## License

This crate is licensed under the MIT License. The vendored
`data/CaseFolding.txt` is part of the Unicode Character Database, redistributed
under the [Unicode terms of use](https://www.unicode.org/terms_of_use.html).
