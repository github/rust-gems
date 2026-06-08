# casefold

A compact Unicode simple case-folding table for Rust.

`simple_fold(c: char) -> char` maps every character to its lower-case fold
form, as defined by the Unicode [CaseFolding.txt][cf] data file restricted to
the **simple** (1-to-1) folds (statuses `C` and `S`). Full multi-character
folds (`F`, e.g. `ß` → `ss`) and Turkic locale folds (`T`) are not supported.

[cf]: https://www.unicode.org/Public/UCD/latest/ucd/CaseFolding.txt

```rust
use casefold::simple_fold;
assert_eq!(simple_fold('A'), 'a');
assert_eq!(simple_fold('Ä'), 'ä');
assert_eq!(simple_fold('Ω'), 'ω');
assert_eq!(simple_fold('漢'), '漢');     // no fold defined → identity
```

For bulk string folding the crate also exposes `fold_into_bytes`, which
consumes a `String` and returns a `Vec<u8>`. ASCII is lowercased in place via
the string's existing heap allocation (a single auto-vectorized pass over the
bytes), and the multibyte tail is scanned for the first character that
actually folds; if none does, the original allocation is returned untouched —
so text whose multibyte content never folds (CJK, Kana, Arabic, Hebrew, …)
pays nothing. A fresh buffer is allocated only once a real fold is found,
since simple folds can change UTF-8 length (e.g. U+212A KELVIN SIGN → `k`, or
U+023A Ⱥ → U+2C65 ⱥ).

```rust
use casefold::fold_into_bytes;
assert_eq!(fold_into_bytes("Hello, WORLD!".to_string()), b"hello, world!");
```

## Why does this crate exist?

Unicode 16.0 defines 1484 simple-fold mappings. Common ways to store them:

| Representation                                        | Size        |
|-------------------------------------------------------|-------------|
| Naïve `[(u32, u32); 1484]`                            | ~11.6 KB    |
| `regex-syntax::unicode_tables::case_folding_simple`   | ~67 KB src  |
| Go `unicode.SimpleFold` (orbit + ASCII + ranges)      | ~7.3 KB     |
| **This crate (paged bitmap + packed runs)**           | **1768 B**  |

That is 9.5 bits per fold entry. A little over half of that is the
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

3. **A run record fits in 16 bits.** Each run is packed into a single `u16` of
   `RUN_DATA`: `end_low` (6 b) | `stride − 1` (1 b) | `length − 1` (7 b). The
   fold itself is *not* stored here — it lives in the parallel `BYTE_DELTA`
   table (idea 5) — so the run record only has to drive the within-page scan
   and membership test. Pre-decrementing `length` and `stride` lets the
   membership test compute `start = end − ((length−1) << (stride−1))` with a
   shift instead of a multiply.

4. **The bulk path rejects whole characters from their first bytes.** For a
   2-/3-byte UTF-8 sequence the page index `cp >> 6` is fully determined by
   the first one or two bytes — only the final continuation byte carries the
   within-page offset `cp & 0x3F`. So `fold_into_bytes` probes `PAGE_BITMAP`
   straight from `b0` (and `b1`); a clear page bit copies the character
   verbatim without assembling `cp` or scanning a run. This skips fold-free
   scripts (CJK, Hangul, Kana, Arabic, Hebrew, Indic) *and* the empty 64-cp
   pages inside otherwise-foldable blocks (e.g. Myanmar, punctuation/symbol
   blocks), reusing the very same table the per-`char` lookup uses.

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

### Table layout (1768 B total)

| Component                                       | Bytes |
|-------------------------------------------------|------:|
| `PAGE_BITMAP[31]: u64` (1 bit per 64-cp page)   |   248 |
| `POPCNT_SAMPLES[32]: u8` (cumulative popcount)  |    32 |
| `PAGE_OFFSET[60]: u8` (per populated page)      |    60 |
| `RUN_DATA[238]: u16` (packed end_low/stride/length) | 476 |
| `BYTE_DELTA[238]: u32` (little-endian fold delta per run) | 952 |
| **Total**                                       | **1768** |

(Splitting runs at byte-delta boundaries raises the run count from 227 to 238.)
The data file is parsed at build time by `build.rs`, which emits the packed
`static` tables to `OUT_DIR/table.rs`.

### Lookup algorithm

```text
simple_fold(c):
    cp = c as u32
    if cp < 0x80: return ASCII-fast-path                # O(1)
    if cp > LAST_COVERED: return c                      # O(1) guard
    (idx, end) = successor(cp)?                         # one bitmap test
    packed   = RUN_DATA[idx]                            #   + ≤30-entry scan
    len_m1   = (packed >> 7) & 0x7F
    stride_b = (packed >> 6) & 1
    start    = end - (len_m1 << stride_b)               # shift, not multiply
    if cp < start: return c                             # in a gap between runs
    if (cp - start) & stride_b != 0: return c           # in a stride-2 gap
    word   = utf8_le(c) + BYTE_DELTA[idx]               # fold by byte add
    return decode(word)
```

The per-`char` path encodes `c` to its little-endian byte word, adds the
run's `BYTE_DELTA`, and decodes the result — the same delta the bulk byte
path applies directly to the input bytes.

`successor(cp)` returns the run whose `end_low` is the smallest one ≥ `cp`'s
low 6 bits *within `cp`'s own 64-cp page*, or `None`. Because the build splits
every run at 64-cp boundaries, no other page can contain a run covering `cp`,
so the answer is either in this page or there is no fold.

The successor lookup is:

1. `page = cp >> 6` — the 64-cp page containing `cp`.
2. Test bit `page % 64` of `PAGE_BITMAP[page / 64]`. **If clear, return
   `None`** — the page is empty, and that is a definitive "no fold".
3. The dense index of `page` is `POPCNT_SAMPLES[page/64] +
   popcount(PAGE_BITMAP[page/64] & ((1 << (page%64)) - 1))` — one load
   plus one masked popcount.
4. Linear-scan `RUN_DATA[PAGE_OFFSET[dense] .. PAGE_OFFSET[dense+1]]` for
   the first entry whose `& 0x3F` is ≥ `cp & 0x3F`. Pages hold at most 30
   runs (averaging ~3.8), so the search touches a small contiguous slice and
   is branch-predictable.

Every access touches the bitmap (248 B), the popcount samples (32 B),
`PAGE_OFFSET` (60 B), `RUN_DATA` (476 B) and, on a hit, `BYTE_DELTA` (952 B) —
all small, cache-friendly arrays.

## Performance

On Apple M-series, the per-`char` `simple_fold` against a `HashMap<u32, u32,
foldhash::fast::FixedState>` baseline (which costs ~17 KB for the same data;
`foldhash` is hashbrown 0.15's default hasher):

| Workload                                 | Casefold table     | HashMap        |
|------------------------------------------|-------------------:|---------------:|
| Sequential BMP scan (63 488 chars)       | **1 700 Melem/s**  |  356 Melem/s   |
| Random BMP (10 000 chars)                | **1 560 Melem/s**  |  573 Melem/s   |
| Random ASCII (10 000 chars)              | **3 090 Melem/s**  |  615 Melem/s   |
| Only-folds (every defined fold)          |     201 Melem/s    |  720 Melem/s   |

Casefold wins three of four workloads decisively (4.8×, 2.7×, and 5.0×
respectively) while using ~10× less memory than the `HashMap`. The
`only_folds` workload — every input is a code point that *does* fold — is
the worst case for the bitmap design (no early-out on the empty-bit test)
and the best case for the `HashMap` (every probe hits a stored key); since
`simple_fold` now folds via the byte-delta path (encode → add → decode),
this all-folds microbenchmark is its slowest case. Real text mixes folds with
non-folds (the random-BMP row), where the table is several × ahead.

### Bulk string conversion (`fold_into_bytes`)

`fold_into_bytes(s: String) -> Vec<u8>` consumes the input `String` and
auto-vectorizes a single in-place pass that lowercases ASCII and detects
whether any multibyte sequence is present. It then scans the multibyte tail
and **hands back the original allocation untouched unless a character actually
folds** — so pure-ASCII input and any text whose multibyte characters never
fold (CJK, Hangul, Kana, Arabic, Hebrew, Indic, symbols, …) avoid a second
buffer entirely. Once a real fold is found it allocates once, then builds the
output with a raw write cursor: unmodified spans are bulk-copied and each
folded character is a masked little-endian load + `BYTE_DELTA` add + 4-byte
store (no decode/encode).

Throughput on an Apple M-series machine (criterion medians), against the SIMD
`simd-normalizer` crate, the same byte path backed by a `HashMap` instead of
the table, and the standard library:

| Workload (input size) | `fold_into_bytes` | `simd_normalizer` | HashMap (byte path) | `str::to_lowercase` | `chars().flat_map` | `to_ascii_lowercase`† |
|---|--:|--:|--:|--:|--:|--:|
| Pure ASCII (5 700 B)                   | **40.4 GiB/s** |   1.21 GiB/s |  212 MiB/s | 25.4 GiB/s | 374 MiB/s | 23.0 GiB/s |
| CJK, no folds (8 100 B)                | **2.97 GiB/s** |   1.98 GiB/s |  562 MiB/s | 479 MiB/s  | 377 MiB/s | 25.0 GiB/s |
| Symbols / Myanmar, no folds (9 000 B)  | **3.03 GiB/s** |   1.55 GiB/s |  409 MiB/s | 499 MiB/s  | 350 MiB/s | 23.5 GiB/s |
| Mixed BMP, all folding (8 800 B)       |   742 MiB/s    | **931 MiB/s**|  330 MiB/s | 287 MiB/s  | 207 MiB/s | 23.2 GiB/s |
| Length-changing folds (1 700 B)        | **1.11 GiB/s** |  726 MiB/s   |  234 MiB/s | 493 MiB/s  | 271 MiB/s | 18.7 GiB/s |

† `str::to_ascii_lowercase` is **not** a correct case-folder for non-ASCII —
it leaves every multibyte sequence untouched. It is shown only as the
"memcpy + ASCII-lowercase" speed floor. The two `to_lowercase` variants
perform Unicode *lowercasing*, which is not identical to case folding (they
diverge on e.g. final-sigma, `İ`, `ß`); this is an equal-workload throughput
comparison, not an output-equality one.

`fold_into_bytes` leads every workload except all-folding mixed-BMP text,
where `simd-normalizer`'s wide-lane SIMD beats our scalar fold. Two things
stand out:

* **The no-fold rows run at GiB/s** (CJK 3.0, symbols 3.0 — ~6× `str::
  to_lowercase`): the tail scan probes `PAGE_BITMAP` from the first one or two
  UTF-8 bytes, finds nothing folds, and returns the input buffer as-is — no
  second allocation, no byte copied.
* **The compact table beats a HashMap by 3–5×** on the *identical* byte-level
  fold (CJK 2.97 GiB/s vs 562 MiB/s, mixed-BMP 742 vs 330 MiB/s), and the
  HashMap has no ASCII fast path at all (212 MiB/s vs 40 GiB/s).

Reproduce with:

```
cargo bench -p casefold-benchmarks
```

## License

This crate is licensed under the MIT License. The vendored
`data/CaseFolding.txt` is part of the Unicode Character Database, redistributed
under the [Unicode terms of use](https://www.unicode.org/terms_of_use.html).
