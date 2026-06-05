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

## Why does this crate exist?

Unicode 16.0 defines 1484 simple-fold mappings. Common ways to store them:

| Representation                                        | Size        |
|-------------------------------------------------------|-------------|
| Naïve `[(u32, u32); 1484]`                            | ~11.6 KB    |
| `regex-syntax::unicode_tables::case_folding_simple`   | ~67 KB src  |
| Go `unicode.SimpleFold` (orbit + ASCII + ranges)      | ~7.3 KB     |
| **This crate (paged bitmap + packed runs)**           | **1248 B**  |

That is 6.73 bits per fold entry — within a small factor of the
information-theoretic floor for 1484 arbitrary code-point pairs.

## How the encoding works

Three observations make the data extraordinarily compressible *and* fast to
query:

1. **Most folds occur in runs.** Stretches of adjacent code points share the
   same delta to their fold, e.g. `U+0041..U+005A` (`A`–`Z`) all map with
   delta `+32`. A great many Latin-Extended pairs come in *alternating*
   sequences like `0x0100, 0x0102, 0x0104 …` where every second code point
   folds. A 1-bit `stride` flag lets the same run encoding cover both cases.

2. **Run ends cluster in a few 64-cp pages.** After splitting any run that
   crosses a 64-cp boundary, just 227 runs live in only 59 of the ~1960
   possible pages. A page-presence bitmap plus a cumulative popcount sidetable
   answers, in a single bit test, whether the page containing `cp` holds any
   run at all — and because every run lives entirely inside one page, an
   unset bit is a *definitive* "no fold". No cross-page successor scan is
   ever required.

3. **A run fits in 32 bits.** Each run is packed into a single `u32` of
   `RUN_DATA`: `end_low` (6 b) | `stride − 1` (1 b) | `length` (7 b) |
   `delta` (18 b signed). 18-bit signed deltas easily cover Unicode's widest
   simple fold (Cherokee, +38864 — actually max |δ| in the data is 42561).
   The within-page scan reads the next entry's `end_low` from the same load
   that decodes the run on a hit, so there are no parallel arrays and no
   escape table.

### Table layout (1248 B total)

| Component                                       | Bytes |
|-------------------------------------------------|------:|
| `PAGE_BITMAP[31]: u64` (1 bit per 64-cp page)   |   248 |
| `POPCNT_SAMPLES[32]: u8` (cumulative popcount)  |    32 |
| `PAGE_OFFSET[60]: u8` (per populated page)      |    60 |
| `RUN_DATA[227]: u32` (packed end_low/stride/length/delta) | 908 |
| **Total**                                       | **1248** |

The data file is parsed at build time by `build.rs`, which emits a packed
`static` table to `OUT_DIR/table.rs`.

### Lookup algorithm

```text
simple_fold(c):
    cp = c as u32
    if cp < 0x80: return ASCII-fast-path                # O(1)
    if cp > LAST_COVERED: return c                      # O(1) guard
    (packed, end) = successor(cp)?                      # one bitmap test
    length = (packed >> 7) & 0x7F                       #   + ≤18-entry scan
    stride = ((packed >> 6) & 1) + 1
    start  = end - (length - 1) * stride
    if cp < start: return c                             # in a gap between runs
    if stride == 2 and (cp - start) is odd: return c    # in a stride-2 gap
    delta  = (packed as i32) >> 14                      # sign-extends 18 bits
    return char::from_u32(cp + delta)
```

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
`PAGE_OFFSET` (60 B) or `RUN_DATA` (908 B) — all small, cache-friendly
arrays. On a hit the matched entry is the *same* `u32` we just compared, so
decoding adds no extra memory traffic.

## Performance

On Apple M-series, comparing against a `HashMap<u32, u32, foldhash::fast::
FixedState>` baseline (which costs ~17 KB for the same data; `foldhash` is
hashbrown 0.15's default hasher):

| Workload                                 | Casefold table     | HashMap        |
|------------------------------------------|-------------------:|---------------:|
| Sequential BMP scan (63 488 chars)       | **1 900 Melem/s**  |  383 Melem/s   |
| Random BMP (10 000 chars)                | **1 550 Melem/s**  |  600 Melem/s   |
| Random ASCII (10 000 chars)              | **3 110 Melem/s**  |  624 Melem/s   |
| Only-folds (every defined fold)          |     339 Melem/s    |  722 Melem/s   |

Casefold wins three of four workloads decisively (5.0×, 2.6×, and 5.0×
respectively) while using ~14× less memory than the `HashMap`. The
`only_folds` workload — every input is a code point that *does* fold — is
the worst case for the bitmap design (no early-out on the empty-bit test)
and the best case for the `HashMap` (every probe hits a stored key); even
so, the table stays within a small constant factor.

Reproduce with:

```
cargo bench -p casefold-benchmarks
```

## License

This crate is licensed under the MIT License. The vendored
`data/CaseFolding.txt` is part of the Unicode Character Database, redistributed
under the [Unicode terms of use](https://www.unicode.org/terms_of_use.html).
