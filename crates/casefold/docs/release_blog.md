# Optimizing Case-folding to >40 GiB/s on a single core

Over the years we have optimized a lot of the hot paths in our code search and indexing pipelines. When possible we have also been trying to open source the things that generalize well in our [
`github/rust-gems`](https://github.com/github/rust-gems) repo. Case-folding is something that had been on the list for a while. As part of the open sourcing process we generally do a cleanup pass on the code, and in this case we did some validation as well, especially to ensure that the stdlib and compiler hadn't started beating our optimizations.

That cleanup and validation resulted in an even simpler and faster implementation than we had previously been using.

## How fast is it?

Criterion medians on an Apple M4 (single core,
`target-cpu=native`). Treat the absolute figures as illustrative, not portable: the whole design leans on auto-vectorization, SWAR, and little-endian byte arithmetic, so the numbers — and even the
*ratios* between rows — can shift substantially on a different microarchitecture (a wider or narrower vector unit, different memory bandwidth, a big-endian target, x86 vs ARM). The qualitative story holds; the exact GiB/s do not.

The other **true case-folders** —
`simd-normalizer` and the same byte path backed by a simple `HashMap` — produce identical output.
`str::to_lowercase` does
*not* in general, but on pure ASCII it coincides with the fold exactly, earning a spot on that row as the correct std-library baseline. The final column is
**not** a folder at all: it is
[
`simdutf`](https://github.com/simdutf/simdutf)'s UTF-8 → UTF-32 → UTF-8 round trip — decoding to code points with a state-of-the-art SIMD decoder and re-encoding them — included as the
*transcoding
tax* any folder that reconstructs code points must pay around its lookup (both buffer lengths assumed known, so only the two transcodes are timed; no folding happens in between):

| Workload (input size)                  | `simple_fold`  | `simd_normalizer` | `HashMap` (byte path) | `str::to_lowercase` | `simdutf` round-trip |
|----------------------------------------|---------------:|------------------:|----------------------:|--------------------:|---------------------:|
| Pure ASCII (5.7 KB)                    | **40.8 GiB/s** |        1.21 GiB/s |              213 MiB/s |          27.7 GiB/s |           9.33 GiB/s |
| CJK, no folds (8.1 KB)                 |  **2.95 GiB/s**|        1.97 GiB/s |              558 MiB/s |                  —  |           2.57 GiB/s |
| Symbols / Myanmar, no folds (9.0 KB)   |  **2.96 GiB/s**|        1.56 GiB/s |              410 MiB/s |                  —  |           2.00 GiB/s |
| Mixed BMP, all folding (8.8 KB)        |     869 MiB/s  |      **922 MiB/s**|              334 MiB/s |                  —  |           1.99 GiB/s |
| Length-changing folds (1.7 KB)         |  **1.26 GiB/s**|         716 MiB/s |              233 MiB/s |                  —  |           1.77 GiB/s |

The headline ASCII row is the workload that dominates real text, and it runs an order of magnitude faster than the SIMD-dispatching
`simd-normalizer` and ~200× faster than a
`HashMap` — purely because the common path is one branch-free vectorized sweep. The no-fold rows (CJK, symbols) run at GiB/s for the same reason: the page-bitmap probe rejects whole characters from their lead bytes and the original buffer is returned without a single byte copied. Even on the identical byte-level fold, the compact table beats a
`HashMap` by 3–5× at ~10× less memory; `simple_fold` only trails on all-folding mixed-BMP text, where
`simd-normalizer` edges ahead by a hair (922 vs 869 MiB/s).

The `simdutf` column is really about the **multibyte** rows. The ASCII figure
(9.33 GiB/s) is almost meaningless: nobody would transcode pure ASCII to 4-byte code units and straight back — there is nothing to gain and a 4× blow-up in memory traffic to pay, so the round trip there measures a step no sane folder takes. It is the non-ASCII path where a code-point-based folder is genuinely
*forced* to transcode, and that's where this number bites. Decode-
*and*-re-encode is the unavoidable envelope of every such design — ICU, Go's `unicode`, the
`regex` crate, CPython all decode UTF-8 to code points and re-encode the result — and even a world-class SIMD transcoder caps out at
**1.8–2.6 GiB/s** on multibyte input. That round trip is a *floor on the
competition*: any folder that decodes first has already spent this much transcoding before it looks a single character up. Yet
`simple_fold` beats it outright on the no-fold rows (2.95 vs 2.57, 2.96 vs 2.00 GiB/s) — the very rows where a decode-then-fold design would be paying the full transcoding tax — because it answers the real question,
*does this character
fold?*, straight from the raw bytes without ever decoding. Folding in byte space doesn't just beat the hash map's lookup; on multibyte text it beats the decode-and-re-encode that every code-point-based folder pays around the lookup.

The pure-ASCII row is the fairest fight of all: there `str::to_lowercase`
produces the **exact same bytes
** we do — a correct std-library baseline rather than a different operation — and even then the branch-free sweep is ~1.5× faster (40.8 vs 27.7 GiB/s), because
`to_lowercase` still scans for the first non-ASCII byte and allocates a fresh
`String` instead of folding in place. On multibyte inputs `to_lowercase` both diverges from the fold
*and* slows to roughly 290–500 MiB/s.

Let's walk through the evolution in detail.

## Why case-folding is even important?

Suppose a user searches for `straße` and your corpus contains `STRASSE`, or they type `İstanbul` and you stored
`istanbul`. To make these match you need a canonical form that erases case distinctions, letting two strings that "differ only in case" compare equal. That form is
**case folding**, and it shows up wherever text is *matched* rather than *displayed*:

- **Search engines** push every indexed term and every query token through the same fold, collapsing `Café`, `café`, and
  `CAFÉ` into one posting list. It runs on every token at index time and query time so it has to be fast and allocation-light.
- **Regex engines** implement the case-insensitive flag
  `(?i)` by folding the pattern's character classes (and comparing against folded input).
- **Identifiers and protocols
  ** use case-insensitive comparison of usernames, hostnames, file paths, HTTP headers, and so on.

### Folding is not lowercasing

It is tempting to reach for
`str::to_lowercase`, but lowercasing and folding are different operations with different goals:

- **Lowercasing** is for *display* and is locale- and context-sensitive: Greek final sigma `Σ` lowercases to
  `ς` at the end of a word and `σ` elsewhere; Turkish `I` lowercases differently than English `I`.
- **Case folding** is for
  *comparison* and is deliberately context-free and locale-independent, keeping the relation stable and symmetric. The Unicode Character Database ships an explicit
  [`CaseFolding.txt`](https://www.unicode.org/Public/UCD/latest/ucd/CaseFolding.txt)
  for exactly this.

These diverge on real characters — `ß`,
`İ`, final sigma — and lowercasing as a stand-in silently produces incorrect matches. This crate implements the **simple
** (1-to-1) folds — statuses `C` and `S` in [
`CaseFolding.txt`](https://www.unicode.org/Public/UCD/latest/ucd/CaseFolding.txt) — and deliberately
*not* the multi-character "full" folds (`ß` → `ss`) or Turkic locale folds.

That restriction aligns with the behavior of other common tools and regex engines like [ripgrep](https://github.com/BurntSushi/ripgrep).

We deal mostly with source code, so the text you fold is overwhelmingly ASCII. Since ASCII
*is* the common case, and making it run at memory speed is the single most important thing we can do. Everything else needs to keep the rare non-ASCII path from spoiling that.

## The counterintuitive core: don't stop early

The fold of an ASCII letter is trivial — `A..=Z` map to
`a..=z`, everything else is unchanged — making the ASCII pass really just "sweep the buffer, lowercase in place." Ask any LLM for it and you might get something like this:

```rust
let bytes = s.as_bytes_mut();
for (i, b) in bytes.iter_mut().enumerate() {
if * b > = 0x80 {
break; // non-ASCII at index i: hand the rest to the Unicode path
}
if b.is_ascii_uppercase() {
* b += 32; // 'A'..='Z' → 'a'..='z'
}
}
```

It looks ideal: do the cheap byte work, and the instant you hit a non-ASCII byte,
`break` and let the "real" Unicode path take over — "only do the cheap work until you have to." On an Apple M4 this runs at about
**3 GiB/s**. That sounds fine in isolation, but it is more than **15× short** of "optimal" because of the `if` branchs.

Let's delete every branch, line by line:

- **`if b >= 0x80 { break }`** → don't stop at all. OR every byte into an accumulator and test it
  *once*, after the loop:
  `high_bit_acc |= *b`. Same information (was there any non-ASCII byte?), zero branches in the body.
- **The `A..=Z` range test** → make it arithmetic. `b.wrapping_sub(b'A') < 26` is true exactly for
  `A..=Z` (any other byte wraps to `≥ 26`), yielding a 0/1 mask with no branch.
- **The conditional write** → fold the mask into the store.
  `| (is_upper << 5)` sets bit 5 — turning an upper-case letter lower-case and being a no-op on everything else — the byte is always written, never branched on.

What's left has no branch in its body and no early exit:

```rust
let mut high_bit_acc: u8 = 0;
for b in & mut bytes {
high_bit_acc |= * b;                       // detect any non-ASCII byte
let is_upper = b.wrapping_sub(b'A') < 26; // branchless A..=Z test
* b |= u8::from(is_upper) < < 5;            // set bit 5 → lowercase, else no-op
}
if high_bit_acc & 0x80 == 0 {
return bytes; // pure ASCII: already folded in place, no second buffer
}
```

A loop with no data-dependent control flow is trivially vectorizable: LLVM emits 16-byte-at-a-time NEON and the whole thing runs at >
**45 GiB/s** — essentially memory bandwidth. And we come out of the pass already knowing, from
`high_bit_acc`, whether there's any non-ASCII work left to do.

How much did each step matter? Measuring the cumulative ladder on pure ASCII (Apple M4, 5.7 KB buffer):

| version | throughput | vectorized? |
|---|--:|---|
| naive (break + branch test)           | 3.1 GiB/s  | no (0 vector instrs) |
| → branchless test/write, *keep* break | 2.6 GiB/s  | no (0 vector instrs) |
| → drop the early-exit `break`         | 7.6 GiB/s  | **partially** (25 vector instrs) |
| → branchless test + write (the loop)  | **46.9 GiB/s** | fully (41 vector instrs) |

The early-exit is what gates vectorization: keep the `break` but make the body perfectly branch-free and you still get *
*zero
** vector instructions (~2.6 GiB/s); a data-dependent loop exit is enough on its own to keep the loop scalar. Only once the
`break` is gone can the compiler vectorize. The final step — making the upper-case fold branchless — then turns a
*partially* vectorized loop (which still compiles the conditional store to a compare-blend-masked-store, ~7.6 GiB/s) into the straight-line arithmetic that hits memory bandwidth.

> [!Note]
> Branchless is a *pessimization* in scalar code.** Look again at the
> table: making the body branchless while *keeping* the `break` (2.6 GiB/s) is
> actually **slower** than the naive branchy loop (3.1 GiB/s). The asm explains
> why. The branchy version only stores a byte when it actually changes one — its
> conditional `strb` is skipped for every lowercase letter, digit and space (the
> vast majority of real text), and the well-predicted branch that guards it is
> nearly free. The branchless version replaces that rarely-taken store with an
> **unconditional `strb` every iteration**, writing back all ~5,700 bytes
> instead of just the handful of upper-case ones. Extra write traffic for no
> benefit. Branchless-write only *wins* once the loop vectorizes, because then the
> store becomes a single 16-byte vector write regardless of content and the
> per-byte cost disappears. The lesson: a branchless body is worth it **only** as
> the enabler for vectorization — on its own, in scalar code, it can cost you.

There's also a middle ground, and it's what standard library uses. Instead of testing one byte at a time,
`[u8]::is_ascii` scans a **machine word at a time** — on a 64-bit target it tests 16 bytes per iteration by OR-ing two
`u64` lanes and checking all their high bits with a single
`& 0x8080_8080_8080_8080` mask. You can build the ASCII fast path on top of that: chunk-scan to find the ASCII prefix, then run the branchless (vectorizable) convert over it. That keeps the early-exit ability — it still bails on the first non-ASCII block — while letting both halves go fast. The catch is that it reads the data
**twice** (once to scan, once to convert), landing at about **23 GiB/s
** — roughly half of the single-pass branchless sweep, and ~7× the naive break loop. A solid, general-purpose default; just not the absolute ceiling when you control the whole loop and can fold detection and conversion into one branch-free pass.

> [!Tip]
> Wouldn't *fusing* the two passes be faster?** It's the obvious next
> thought: keep the chunked early-exit but convert each 16-byte block right after
> you've confirmed it's ASCII, reading the data only *once*. Measured, it's
> **~2.6× slower** — 8.7 GiB/s versus the two-pass 23. The inner block convert
> still vectorizes to a single 16-byte op, but now there's a data-dependent
> early-exit branch *every 16 bytes*, and that branch pins the loop to one block
> at a time: the compiler can't unroll or software-pipeline across blocks, and each
> iteration pays the full load→test→branch→convert→store latency with nothing to
> hide it behind. Split into two passes, each one is clean: the scan is a
> branch-light, **store-free** word scan that races through memory, and the
> convert is the fully-vectorized branch-free sweep at ~47 GiB/s. Two fast,
> branch-free passes beat one branchy fused pass — even though the fused version
> touches the data half as many times. It's the same lesson one more time: in the
> hot loop, the branch is the enemy.

It is genuinely faster to
*unconditionally* sweep the entire buffer once, branch-free, and decide what to do afterwards than to try to stop early. Stripped of every branch, the loop becomes almost insultingly simple — a flat sequence of loads, an OR, an add, and stores over a contiguous buffer — and a loop that simple is a piece of cake for the compiler to vectorize into a 45 GiB/s racing car.

## Avoiding the heap

40 GiB/s also means doing zero unnecessary allocation. `simple_fold` takes the input `String` *by
value*, owning the heap buffer it can mutate and return it. If the OR-accumulator's high bit was clear, the input was pure ASCII — already folded in place — we hand the
**same allocation** straight back, no second buffer and no copy. Otherwise we
`memchr` to the first non-ASCII byte and scan the tail from there, leaving the output buffer
*unallocated* (a null write cursor) until we hit a character that folds to **different bytes
**. Text whose multibyte content never folds — CJK, Hangul, Kana, Arabic, Hebrew, symbols — also returns the original allocation untouched, never copying a byte.

Why a *second* buffer rather than rewriting in place like the ASCII pass? Because folding can make the string **longer
**: almost every fold preserves the UTF-8 length or shrinks it, but two outliers grow — U+023A (`Ⱥ`) and U+023E (
`Ɀ`) are 2 bytes each yet fold to 3-byte characters (`ⱥ`,
`ɀ`). Once one appears, the output no longer fits in the input's bytes, and we need somewhere new to write.

We allocate that buffer **once**, sized for the worst case, rather than growing it as more folds appear. Incremental
`reserve` calls would mean re-checking capacity, occasionally reallocating, copying everything written so far, and juggling extra length/capacity bookkeeping; a single up-front allocation lets a raw write cursor run straight to the end with none of that. (And since the cursor is null until that first growing/changing fold, it doubles as the "have we started building yet?" flag — the decision to allocate costs no extra state either.)

Sizing it needs a bound on growth, and those same two outliers give it: every 2 input bytes yield at most 3 output bytes, capping the output at
**1.5× the input** — exactly the capacity we reserve:

```rust
out = Vec::with_capacity(bytes.len() + bytes.len() / 2 + 4);
```

After that the loop writes through a raw pointer with no capacity checks and calls
`set_len` exactly once at the end. Two more details keep it branch-light. The run of unchanged bytes between two folds is moved with a single
`copy_nonoverlapping` rather than byte by byte. And each fold unconditionally writes all 4 bytes of a little-endian word before bumping the cursor by only the
*folded* length (1–4) — dropping a branch on the output length from the hot path, with the
`+ 4` in the reservation as the headroom that makes the final character's over-store safe.

## Making Unicode cheap too

When a character
*does* fold, we still don't want to fall off a cliff — decode UTF-8, hash, re-encode. Unicode 16.0 has 1484 simple-fold mappings, but they're a
*very* sparse and *very* structured relation. Four observations shrink them to
**1776 bytes** and let the fold run **without ever decoding a full character**.

But before any of that, the most important ingredient: even on the non-ASCII path, the overwhelming majority of characters
**do not fold
**. CJK, Hangul, Kana, Arabic, Hebrew, Indic scripts, emoji, punctuation, symbols — none of it folds. The hot operation, then, isn't really "fold this character," it's "
*does* this character fold?" — almost always no. The table has to make that **negative test
** as close to free as possible; the actual folding is the rare sub-case of an already-rare path. That priority is what shapes the layout below — the page bitmap ([idea 1](#idea-1-foldable-code-points-cluster-into-64-code-point-pages)) exists precisely so a non-folding character is rejected in a single bit test, straight from its leading UTF-8 bytes, without decoding or scanning anything.

This is exactly why a `HashMap<u32, u32>` is the
*wrong* shape for the job, not just a bigger one. A hash map is optimized for the **hit
**: it finds a present key in roughly one probe, and only spends extra work (more probes, full key comparison) when load factor or collisions bite. But our workload is dominated by
**misses** — characters that aren't in the table at all — and a miss is a hash map's
*least* favourite query: it still has to hash the key, jump to a bucket, and walk the probe sequence far enough to
*prove
absence*. We'd be paying the map's slow path on virtually every character and its fast path almost never. The bitmap inverts that: the common case (no fold) is a single bit test, and only the rare hit does any further work — the exact opposite of the hash map's bias, and the right one for this data.

### Idea 1: foldable code points cluster into 64-code-point "pages"

Foldable code points bunch together. Slice the code space into 64-code-point
"pages" and the ~1484 folds touch just **59** of ~1960 possible pages. A one-bit-per-page **presence bitmap
** answers the negative test on its own: a clear bit is a
*definitive* "no fold" — copy through, done — which is what makes fold-free scripts cheap. Only on a set bit do we consult a second structure, a
**cumulative-popcount side table
** that ranks the page (how many populated pages precede it) to find its slice of entries, storing nothing for the ~1900 empty pages.

Why **64
**? Six bits is exactly what makes the probe fall out of the UTF-8 bytes. A continuation byte carries 6 payload bits, which makes the within-page offset
`cp & 0x3F`
*literally the low 6 bits of the last
byte*. Indexing the bitmap as 64-bit words, the bit position is another 6 bits — straight from the second-to-last byte — leaving only the higher bits as the word index. So the bit index is always just the second-to-last byte masked with
`0x3F`, and the word index is
`0`, a nibble, or (only for four-byte sequences) two merged bytes — a tiny branch on the lead byte, no full code-point reconstruction:

```rust
let (word_idx, bit_idx, c_len) = if lead < 0xE0 {
    (0usize, lead & 0x1F, 2usize)                         // 2-byte: word 0
} else if lead < 0xF0 {
    ((lead & 0x0F) as usize, bytes[read + 1] & 0x3F, 3)   // 3-byte: word = nibble
} else {
    ((((lead & 0x07) as usize) < < 6) | (bytes[read + 1] & 0x3F) as usize, bytes[read + 2] & 0x3F, 4usize,)// 4-byte: merge 2 bytes
};
// reject without decoding: clear bit ⇒ no fold
if word_idx > = PAGE_BITMAP.len() || (PAGE_BITMAP[word_idx] >> bit_idx) & 1 == 0 {
    read += c_len;
    continue;
}
```

Because
`word_idx` depends only on the lead byte (and, for four-byte sequences, the first continuation byte), the bitmap load can be issued early.

### Idea 2: within a page, folds come in runs

A set page bit tells us
*something* on this page folds, but not which code points or to what. The obvious encoding is one entry per foldable code point — but that is both bulky and slow to search: a page can hold dozens of folds, and we'd have to scan them all to find the one matching the current code point. The structure of the data rescues us again. Adjacent code points overwhelmingly share the same delta to their fold:
`A`–`Z` all map `+32`, and Latin Extended is full of
*alternating* runs like
`0x0100, 0x0102, 0x0104, …` where every second code point folds. Instead of per-code-point entries we store **runs
** — start, end, stride, delta — and a 1-bit
`stride` flag covers both the contiguous and the every-other case. This interval compression collapses the ~1484 individual folds into just
**238
** runs across the 59 pages (≈4 per page), leaving the within-page search only a handful of entries to look at instead of dozens. This range-with-delta encoding (including the stride trick) is borrowed from Go's
`unicode` package, whose
[`CaseRange`](https://github.com/golang/go/blob/master/src/unicode/tables.go)
records store a `Lo`/`Hi` range plus per-case deltas, with an `UpperLower`
sentinel marking the alternating blocks. Runs are clipped at the 64-cp page boundaries so a run never straddles two pages — which is exactly what lets the page bitmap above treat a clear bit as a
*definitive* "no fold".

### Idea 3: a run record is two clean bytes

With both endpoints inside one page they fit in 6 bits, split across two arrays:
`RUN_END_LOW[i] = end & 0x3F` (the scan key) and
`RUN_START_STRIDE[i] = (start & 0x3F) | ((stride − 1) << 6)` (read only on a hit). Because each key is one clean byte, the within-page search can go
**wide**: rather than comparing `cp & 0x3F` against the runs one at a time, we load **8 `end_low` bytes into a
single `u64` and test all of them at once** with one branchless SWAR step —
`(chunk | 0x80…80) − broadcast(low) & 0x80…80` sets the top bit of every lane whose key is
`≥ cp & 0x3F`. A single bit-scan of that mask (the keys are sorted, so the first set lane is the run we want) finds the slot. A page holds ~4 runs on average; that one 8-wide compare almost always resolves the entire search in a single step. One unlucky page does hold 30 runs, which puts the compare inside a short loop that strides 8 keys at a time — but that loop trips at most a handful of times on exactly one page in all of Unicode, and never on the common ones. Either way: no per-run branch, and no code-point reconstruction anywhere.

```rust
/// Offset of the first run with `end_low >= low_v` in a page of `n` runs,
/// or `n` if none. Scans 8 `end_low` bytes at a time via SWAR.
#[inline]
fn scan_end_low(lo: usize, n: usize, low_v: u8) -> usize {
    const HIGH: u64 = 0x8080_8080_8080_8080;
    const ONES: u64 = 0x0101_0101_0101_0101;
    let bcast = (low_v as u64).wrapping_mul(ONES);
    let mut base = 0;
    while base < n {
        // RUN_END_LOW is padded by 8 bytes so this read is always in bounds.
        let chunk = u64::from_le_bytes(
            RUN_END_LOW[lo + base..lo + base + 8]
                .try_into()
                .expect("8-byte slice"),
        );
        // `(b | 0x80) - low_v` keeps its high bit iff `b >= low_v` (no
        // cross-lane borrow). The first set lane is the first run `>= low_v`.
        let ge = (chunk | HIGH).wrapping_sub(bcast) & HIGH;
        if ge != 0 {
            let j = base + (ge.trailing_zeros() / 8) as usize;
            return if j < n { j } else { n };
        }
        base += 8;
    }
    n
}
```

### Idea 4: folding is a little-endian byte addition

On a little-endian machine the folded character's UTF-8 bytes, read as a `u32`, equal the source bytes (as a
`u32`) plus a **per-run constant**. A parallel `BYTE_DELTA[i]` table then turns the whole fold into a masked load, one
`wrapping_add`, and a 4-byte store:

```rust
let word   = u32::from_le_bytes(next_four_bytes) & length_mask; // keep this char's bytes
let folded = word.wrapping_add(BYTE_DELTA[i]);                  // the fold, as one byte add
write_u32_le(dst, folded);                                     // store all 4 bytes...
dst += utf8_len(folded);                                       // ...advance by the folded length
```

Both lengths in that snippet — the `length_mask` for the source character and the *advance by the folded
length* for the destination — come from one more tiny trick. A UTF-8 sequence's length is fixed by the top four bits of its lead byte, letting the 16 possible lengths pack one nibble each into a single 64-bit constant (
`0x4322_1111_1111_1111`); the length is then a shift and a mask,
`(LEN_BITS >> (4 * (lead >> 4))) & 0xF` — no `if` chain, no table memory, nothing for the predictor to get wrong. (A
*count leading ones* — `(!lead).leading_zeros()`
— would also work, since a lead byte carries one leading 1-bit per byte of the sequence, but the nibble shift avoids the bit-complement.)

```rust
/// Number of bytes in the UTF-8 sequence whose lead byte is `lead`.
#[inline]
pub fn utf8_len(lead: u8) -> usize {
    const UTF8_LEN_BY_LEAD: u64 = 0x4322_1111_1111_1111;
    ((UTF8_LEN_BY_LEAD >> (4 * (lead >> 4))) & 0xF) as usize
}
```

Because we advance by the *folded* length, this even handles length-changing folds — U+212A KELVIN SIGN (3 bytes) →
`k` (1 byte), or U+023A `Ⱥ` (2 bytes) → U+2C65 `ⱥ` (3 bytes) — by writing fewer or more bytes than were read.[^overlong]
And it's the part I believe is genuinely new: every other folder I looked at — ICU, Go's `unicode`, Rust's
`regex`, CPython, glibc — decodes UTF-8 to a code point, applies the fold there, and re-encodes (even SIMD folders decode first). Doing the arithmetic in byte space skips both the decode and the encode, which is exactly why this path can outrun a hash map that already has the answer tabulated — the hash map still has to decode its key and encode its result.

[^overlong]: The byte-space arithmetic assumes the input is **well-formed, shortest-form UTF-8
** — every code point encoded with the minimal number of bytes. Reading the source bytes as a
`u32` and adding a per-run delta only lands on the correct folded encoding when the source is in canonical form; an
*overlong* encoding (a code point padded into more bytes than necessary, e.g. `/` as
`0xC0 0xAF`) has a different byte pattern and would break the
`length_mask` and the delta arithmetic. This is not a real restriction in Rust — `&str`/
`String` are guaranteed to hold valid UTF-8, which by definition rejects overlong sequences — but a caller feeding raw bytes from elsewhere must validate (or otherwise normalize) them first.

### The ASCII shortcut in the tail loop

One more shortcut rounds out the tail loop. Remember the first pass already lowercased every ASCII byte, so when the scan meets an ASCII byte in the tail it advances a single byte and moves on — no page probe, no table touch at all. And it doesn't copy that byte either: unmodified bytes (ASCII and non-folding multibyte alike) aren't moved one at a time. The scan just keeps walking until it reaches a character that actually folds, then flushes the whole unchanged run between the last fold and this one with a single
`copy_nonoverlapping`. Mixed text — CJK with ASCII spaces and punctuation, or code with the occasional accented identifier — therefore races through the ASCII filler and only consults the bitmap for genuine multibyte characters, copying in bulk rather than byte by byte.

### Putting it together: the whole table

| Component                                          | Bytes  |
|----------------------------------------------------|-------:|
| `PAGE_BITMAP` (1 bit per 64-cp page)               |    248 |
| `POPCNT_SAMPLES` (cumulative popcount)             |     32 |
| `PAGE_OFFSET` (per populated page)                 |     60 |
| `RUN_END_LOW` (scan key, `end & 0x3F`, +8 pad)     |    246 |
| `RUN_START_STRIDE` (`start & 0x3F` \| stride)      |    238 |
| `BYTE_DELTA` (little-endian fold delta per run)    |    952 |
| **Total**                                          | **1776** |

That's **9.6 bits per fold entry** — over half of it the
`BYTE_DELTA` side table we trade for the decode-free path; the index + run records alone are ~4.4 bits/entry.

Next to the obvious alternatives, that 1776 bytes is an order of magnitude or more smaller — and unlike most of them it never decodes a character:

| Representation                                      | Size       |
|-----------------------------------------------------|-----------:|
| Naïve `[(u32, u32); 1484]`                          | ~11.6 KB   |
| `regex-syntax`'s `case_folding_simple` table        | ~70 KB     |
| Go's `unicode.SimpleFold` (orbit + ASCII + ranges)  | ~7.3 KB    |
| A runtime `HashMap<u32, u32>`                        | ~17 KB     |
| **This crate (paged bitmap + packed runs)**         | **1776 B** |

## Takeaways

Case folding sounds solved — uppercase to lowercase, how hard can it be? Yet a task this basic hid two
*surprising* wins, in opposite directions.

On the common path, the win was doing **more** work: deleting the "obvious"
early-exit so the loop sweeps the whole buffer ran ~15× faster, because the branch we added to
*save* work was the very thing blocking vectorization.

On the rare path, the win was looking at the
*shape* of the data instead of reaching for a hash map: case folding is sparse, run-heavy, and page-clustered, and UTF-8's little-endian layout turns a code-point delta into a plain integer add — so a 1.7 KB table beats a hash map on both size and speed.

The meta-lesson: "basic" rarely means "fully explored." Measuring instead of guessing — and questioning the optimization everyone reaches for first — can still find an order of magnitude. That's the fun part.

The crate is [`casefold`](../README.md); the generated table and full design notes live alongside the source.
