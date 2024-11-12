# string-offsets

Offset calculator to convert between byte, char, and line offsets in a string.

Rust strings are UTF-8, but JavaScript has UTF-16 strings, and in Python, strings are sequences of
Unicode code points. It's therefore necessary to adjust string offsets when communicating across
programming language boundaries. [`StringOffsets`] does these adjustments.

Each `StringOffsets` value contains offset information for a single string. [Building the data
structure](StringOffsets::new) takes O(n) time and memory, but then each conversion is fast.

["UTF-8 Conversions with BitRank"](https://adaptivepatchwork.com/2023/07/10/utf-conversion/) is a
blog post explaining the implementation.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
string-offsets = "0.1"
```

Then:

```rust
use string_offsets::StringOffsets;

let s = "☀️hello\n🗺️world\n";
let offsets = StringOffsets::new(s);

// Find offsets where lines begin and end.
assert_eq!(offsets.line_to_utf8s(0), 0..12);  // note: 0-based line numbers

// Translate string offsets between UTF-8 and other encodings.
// This map emoji is 7 UTF-8 bytes...
assert_eq!(&s[12..19], "🗺️");
// ...but only 3 UTF-16 code units...
assert_eq!(offsets.utf8_to_utf16(12), 8);
assert_eq!(offsets.utf8_to_utf16(19), 11);
// ...and only 2 Unicode characters.
assert_eq!(offsets.utf8s_to_chars(12..19), 8..10);
```

See [the documentation](https://docs.rs/string-offsets/latest/string_offsets/struct.StringOffsets.html) for more.
