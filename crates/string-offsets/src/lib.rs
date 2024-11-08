//! Offset calculator to convert between byte, char, and line offsets in a string.
//!
//! See [`StringOffsets`] for details.
#![deny(missing_docs)]

use std::ops::Range;

mod bitrank;

use bitrank::{BitRank, BitRankBuilder};

/// Offset calculator to convert between byte, char, and line offsets in a string.
///
/// Rust strings are UTF-8, but JavaScript has UTF-16 strings, and in Python, strings are sequences
/// of Unicode code points. It's therefore necessary to adjust string offsets when communicating
/// across programming language boundaries. [`StringOffsets`] does these adjustments.
///
/// Each `StringOffsets` value contains offset information for a single string. [Building the
/// data structure](StringOffsets::new) takes O(n) time and memory, but then each conversion is fast.
///
/// ["UTF-8 Conversions with BitRank"](https://adaptivepatchwork.com/2023/07/10/utf-conversion/)
/// is a blog post explaining the implementation.
///
/// ## Converting offsets
///
/// The conversion methods follow a naming scheme that uses these terms for different kinds of
/// offsets:
///
/// - `utf8` - UTF-8 byte offsets (Rust style).
/// - `utf16` - UTF-16 code unit offsets (JavaScript style).
/// - `char` - Count of Unicode scalar values (Python style).
/// - `utf16_pos` - Zero-based line number and `utf16` offset within the line.
/// - `char_pos` - Zero-based line number and `char` offset within the line.
///
/// For example, [`StringOffsets::utf8_to_utf16`] converts a Rust byte offset to a number that will
/// index to the same position in a JavaScript string. Offsets are expressed as `usize` or [`Pos`]
/// values.
///
/// All methods accept arguments that are past the end of the string, interpreting them as pointing
/// to the end of the string.
///
/// ## Converting ranges
///
/// Some methods translate position *ranges*. These are expressed as `Range<usize>` except for
/// `line`, which is a `usize`:
///
/// - `line` - Zero-based line numbers. The range a `line` refers to is the whole line, including
///   the trailing newline character if any.
/// - `lines` - A range of line numbers.
/// - `utf8s` - UTF-8 byte ranges.
/// - `utf16s` - UTF-16 code unit ranges.
/// - `chars` - Ranges of Unicode scalar values.
///
/// When mapping offsets to line ranges, it is important to use a `_to_lines` function in order to
/// end up with the correct line range. We have these methods because if you tried to do it
/// yourself you would screw it up; use them! (And see the source code for
/// [`StringOffsets::utf8s_to_lines`] if you don't believe us.)
///
/// ## Complexity
///
/// Most operations run in O(1) time. A few require O(log n) time. The memory consumed by this
/// data structure is typically less than the memory occupied by the actual content. In the best
/// case, it requires ~45% of the content space.
pub struct StringOffsets {
    /// Vector storing, for every line, the byte position at which the line starts.
    line_begins: Vec<u32>,

    /// Encoded bitrank where the rank of a byte position corresponds to the line number to which
    /// the byte belongs.
    utf8_to_line: BitRank,

    /// Encoded bitrank where the rank of a byte position corresponds to the char position to which
    /// the byte belongs.
    utf8_to_char: BitRank,

    /// Encoded bitrank where the rank of a byte position corresponds to the UTF-16 encoded word
    /// position to which the byte belongs.
    utf8_to_utf16: BitRank,

    /// Marks, for every line, whether it consists only of whitespace characters.
    whitespace_only: Vec<bool>,
}

/// A position in a string, specified by line and column number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pos {
    /// Zero-indexed line number.
    pub line: usize,
    /// Zero-indexed column number. The units of this field depend on the method that produces the
    /// value. See [`StringOffsets::utf8_to_char_pos`], [`StringOffsets::utf8_to_utf16_pos`].
    pub col: usize,
}

// The actual conversion implementation between utf8, utf16, chars, and line numbers.
// New methods must follow the existing conventions:
//
// - All conversions saturate when the input is out of bounds.
// - Lines INCLUDE the terminating newline.
// - Line numbers and column numbers are 0-based.
// - `.xyz_to_lines(range)` methods behave like `.utf8_to_lines(the corresponding byte range)`.
//
// This last one is tricky, because in these methods, `range.begin` "rounds down" to the beginning
// of the line, but `range.end` "rounds up"; and because there are many corner cases.
//
// E.g.: The empty character range at the end of one line cannot be distinguished from the empty
// character range at the start of the subsequent line! This ambiguity is resolved by returning the
// line which starts with the empty character range.
//
// Question: Consider whether we should return an empty line range in this case which would
// probably be consistent from a mathematical point of view. But then we should also return empty
// line ranges for empty character ranges in the middle of a line...
impl StringOffsets {
    /// Create a new converter to work with offsets into the given string.
    pub fn new(content: &str) -> Self {
        new_converter(content.as_bytes())
    }

    /// Create a new converter to work with offsets into the given byte-string.
    ///
    /// If `content` is UTF-8, this is just like [`StringOffsets::new`]. Otherwise, the
    /// conversion methods will produce unspecified (but memory-safe) results.
    pub fn from_bytes(content: &[u8]) -> Self {
        new_converter(content)
    }

    /// Returns the number of Unicode characters on the specified line.
    pub fn line_chars(&self, line_number: usize) -> usize {
        let r = self.utf8s_to_chars(self.line_to_utf8s(line_number));
        r.end - r.start
    }

    /// Returns the number of lines in the string.
    pub fn lines(&self) -> usize {
        self.line_begins.len() - 1
    }

    /// Returns true if the specified line is empty except for whitespace.
    pub fn only_whitespaces(&self, line_number: usize) -> bool {
        self.whitespace_only
            .get(line_number)
            .copied()
            .unwrap_or(true)
    }

    /// Return the byte offset of the first character on the specified (zero-based) line.
    ///
    /// If `line_number` is greater than or equal to the number of lines in the text, this returns
    /// the length of the string.
    pub fn line_to_utf8_begin(&self, line_number: usize) -> usize {
        self.line_begins[line_number.min(self.lines())] as usize
    }

    /// UTF-16 offset of the first character of a line.
    ///
    /// That is, return the offset that would point to the start of that line in a UTF-16
    /// representation of the source string.
    pub fn line_to_utf16_begin(&self, line_number: usize) -> usize {
        self.utf8_to_utf16(self.line_to_utf8_begin(line_number))
    }

    /// UTF-32 offset of the first character of a line.
    ///
    /// That is, return the offset that would point to the start of that line in a UTF-32
    /// representation of the source string.
    pub fn line_to_char_begin(&self, line_number: usize) -> usize {
        self.utf8_to_char(self.line_to_utf8_begin(line_number))
    }

    /// UTF-8 offset of the first character of a line.
    pub fn line_to_utf8_end(&self, line_number: usize) -> usize {
        self.line_to_utf8_begin(line_number + 1)
    }

    /// UTF-16 offset one past the end of a line (the offset of the start of the next line).
    pub fn line_to_utf16_end(&self, line_number: usize) -> usize {
        self.utf8_to_utf16(self.line_to_utf8_end(line_number))
    }

    /// UTF-32 offset one past the end of a line (the offset of the start of the next line).
    pub fn line_to_char_end(&self, line_number: usize) -> usize {
        self.utf8_to_char(self.line_to_utf8_end(line_number))
    }

    /// UTF-8 offset one past the end of a line (the offset of the start of the next line).
    pub fn line_to_utf8s(&self, line_number: usize) -> Range<usize> {
        self.line_to_utf8_begin(line_number)..self.line_to_utf8_end(line_number)
    }

    /// UTF-32 offsets for the beginning and end of a line, including the newline if any.
    pub fn line_to_chars(&self, line_number: usize) -> Range<usize> {
        self.utf8s_to_chars(self.line_to_utf8s(line_number))
    }

    /// UTF-8 offsets for the beginning and end of a range of lines, including the newline if any.
    pub fn lines_to_utf8s(&self, line_numbers: Range<usize>) -> Range<usize> {
        self.line_to_utf8_begin(line_numbers.start)..self.line_to_utf8_begin(line_numbers.end)
    }

    /// UTF-32 offsets for the beginning and end of a range of lines, including the newline if any.
    pub fn lines_to_chars(&self, line_numbers: Range<usize>) -> Range<usize> {
        self.utf8s_to_chars(self.lines_to_utf8s(line_numbers))
    }

    /// Return the zero-based line number of the line containing the specified UTF-8 offset.
    /// Newline characters count as part of the preceding line.
    pub fn utf8_to_line(&self, byte_number: usize) -> usize {
        self.utf8_to_line.rank(byte_number)
    }

    /// Converts a UTF-8 offset to a zero-based line number and UTF-32 offset within the
    /// line.
    pub fn utf8_to_char_pos(&self, byte_number: usize) -> Pos {
        let line = self.utf8_to_line(byte_number);
        let line_start_char_number = self.line_to_char_begin(line);
        let char_idx = self.utf8_to_char(byte_number);
        Pos {
            line,
            col: char_idx - line_start_char_number,
        }
    }

    /// Converts a UTF-8 offset to a zero-based line number and UTF-16 offset within the
    /// line.
    pub fn utf8_to_utf16_pos(&self, byte_number: usize) -> Pos {
        let line = self.utf8_to_line(byte_number);
        let line_start_char_number = self.line_to_utf16_begin(line);
        let char_idx = self.utf8_to_utf16(byte_number);
        Pos {
            line,
            col: char_idx - line_start_char_number,
        }
    }

    /// Returns the range of line numbers containing the substring specified by the Rust-style
    /// range `bytes`. Newline characters count as part of the preceding line.
    ///
    /// If `bytes` is an empty range at a position within or at the beginning of a line, this
    /// returns a nonempty range containing the line number of that one line. An empty range at or
    /// beyond the end of the string translates to an empty range of line numbers.
    pub fn utf8s_to_lines(&self, bytes: Range<usize>) -> Range<usize> {
        // The fiddly parts of this formula are necessary because `bytes.start` rounds down to the
        // beginning of the line, but `bytes.end` "rounds up" to the end of the line. the final
        // `+1` is to produce a half-open range.
        self.utf8_to_line(bytes.start)
            ..self
                .lines()
                .min(self.utf8_to_line(bytes.end.saturating_sub(1).max(bytes.start)) + 1)
    }

    /// Returns the range of line numbers containing the substring specified by the UTF-32
    /// range `chars`. Newline characters count as part of the preceding line.
    pub fn chars_to_lines(&self, chars: Range<usize>) -> Range<usize> {
        self.utf8s_to_lines(self.chars_to_utf8s(chars))
    }

    /// Converts a UTF-8 offset to a UTF-32 offset.
    pub fn utf8_to_char(&self, byte_number: usize) -> usize {
        self.utf8_to_char.rank(byte_number)
    }

    /// Converts a UTF-8 offset to a UTF-16 offset.
    pub fn utf8_to_utf16(&self, byte_number: usize) -> usize {
        self.utf8_to_utf16.rank(byte_number)
    }

    /// Converts a UTF-32 offset to a UTF-8 offset.
    pub fn char_to_utf8(&self, char_number: usize) -> usize {
        let mut byte_number = char_number;
        for _ in 0..128 {
            let char_number2 = self.utf8_to_char(byte_number);
            if char_number2 == char_number {
                return byte_number;
            }
            byte_number += char_number - char_number2;
        }
        // If we couldn't find the char within 128 steps, then the char_number might be invalid!
        // This does not usually happen. For consistency with the rest of the code, we simply return
        // the max utf8 position in this case.
        if char_number > self.utf8_to_char.max_rank() {
            return self
                .line_begins
                .last()
                .copied()
                .expect("last entry represents the length of the file!")
                as usize;
        }
        let limit = *self.line_begins.last().expect("no line begins") as usize;
        // Otherwise, we keep searching, but are a bit more careful and add a check that we don't run into an infinite loop.
        loop {
            let char_number2 = self.utf8_to_char(byte_number);
            if char_number2 == char_number {
                return byte_number;
            }
            byte_number += char_number - char_number2;
            assert!(byte_number < limit);
        }
    }

    /// Converts a UTF-8 offset range to a UTF-32 offset range.
    pub fn utf8s_to_chars(&self, bytes: Range<usize>) -> Range<usize> {
        self.utf8_to_char(bytes.start)..self.utf8_to_char(bytes.end)
    }

    /// Converts a UTF-32 offset range to a UTF-8 offset range.
    pub fn chars_to_utf8s(&self, chars: Range<usize>) -> Range<usize> {
        self.char_to_utf8(chars.start)..self.char_to_utf8(chars.end)
    }
}

fn new_converter(content: &[u8]) -> StringOffsets {
    let n = content.len();
    let mut utf8_builder = BitRankBuilder::with_capacity(n);
    let mut utf16_builder = BitRankBuilder::with_capacity(n);
    let mut line_builder = BitRankBuilder::with_capacity(n);
    let mut line_begins = vec![0];
    let mut i = 0;
    let mut whitespace_only = vec![];
    let mut only_whitespaces = true; // true if all characters in the current line are whitespaces.
    while i < content.len() {
        // In case of invalid utf8, we might get a utf8_len of 0.
        // In this case, we just treat the single byte character.
        // In principle, a single incorrect byte can break the whole decoding...
        let c = content[i];
        let utf8_len = utf8_width(c).max(1);
        if i > 0 {
            utf8_builder.push(i - 1);
            utf16_builder.push(i - 1);
        }
        if utf8_to_utf16_width(&content[i..]) > 1 {
            utf16_builder.push(i);
        }
        if c == b'\n' {
            whitespace_only.push(only_whitespaces);
            line_begins.push(i as u32 + 1);
            line_builder.push(i);
            only_whitespaces = true; // reset for next line.
        } else {
            only_whitespaces &= matches!(c, b'\t' | b'\r' | b' ');
        }
        i += utf8_len;
    }
    if !content.is_empty() {
        utf8_builder.push(content.len() - 1);
        utf16_builder.push(content.len() - 1);
    }
    if line_begins.last() != Some(&(content.len() as u32)) {
        whitespace_only.push(only_whitespaces);
        line_begins.push(content.len() as u32);
        line_builder.push(content.len() - 1);
    }

    StringOffsets {
        line_begins,
        utf8_to_line: line_builder.finish(),
        whitespace_only,
        utf8_to_char: utf8_builder.finish(),
        utf8_to_utf16: utf16_builder.finish(),
    }
}

/// Returns the number of bytes a UTF-8 char occupies, given the first byte of the UTF-8 encoding.
/// Returns 0 if the byte is not a valid first byte of a UTF-8 char.
fn utf8_width(c: u8) -> usize {
    // Every nibble represents the utf8 length given the first 4 bits of a utf8 encoded byte.
    const UTF8_WIDTH: usize = 0x4322_0000_1111_1111;
    (UTF8_WIDTH >> ((c >> 4) * 4)) & 0xf
}

fn utf8_to_utf16_width(content: &[u8]) -> usize {
    let len = utf8_width(content[0]);
    match len {
        0 => 0,
        1..=3 => 1,
        4 => 2,
        _ => panic!("invalid utf8 char width: {}", len),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns true if, in a UTF-8 string, `b` indicates the first byte of a character.
    fn is_char_boundary(b: u8) -> bool {
        b as i8 >= -0x40 // NB: b < 128 || b >= 192
    }

    #[test]
    fn test_utf8_char_width() {
        for c in '\0'..=char::MAX {
            let mut dst = [0; 4];
            let len = c.encode_utf8(&mut dst).len();
            assert_eq!(len, utf8_width(dst[0]), "char: {:?} {len}", dst[0] >> 4);
        }

        for b in 0..=255u8 {
            if !is_char_boundary(b) {
                assert_eq!(utf8_width(b), 0, "char: {:?}", b >> 4);
            } else {
                assert!(utf8_width(b) > 0, "char: {:?}", b >> 4);
            }
        }
    }

    #[test]
    fn test_utf8_to_utf16_len() {
        for c in '\0'..=char::MAX {
            let mut dst = [0; 4];
            let _len = c.encode_utf8(&mut dst).len();
            assert_eq!(utf8_to_utf16_width(&dst), c.len_utf16());
        }

        for b in 0..=255u8 {
            if !is_char_boundary(b) {
                assert_eq!(utf8_to_utf16_width(&[b]), 0);
            }
        }
    }

    #[test]
    fn test_line_map() {
        let content = r#"a short line.
followed by another one.
no terminating newline!"#;
        let lines = StringOffsets::new(content);
        assert_eq!(lines.line_to_utf8s(0), 0..14);
        assert_eq!(&content[0..14], "a short line.\n");
        assert_eq!(lines.line_to_utf8s(1), 14..39);
        assert_eq!(&content[14..39], "followed by another one.\n");
        assert_eq!(lines.line_to_utf8s(2), 39..62);
        assert_eq!(&content[39..62], "no terminating newline!");
        assert_eq!(lines.utf8_to_line(0), 0);
        assert_eq!(lines.utf8_to_line(13), 0);
        assert_eq!(lines.utf8_to_line(14), 1);
        assert_eq!(lines.utf8_to_line(38), 1);
        assert_eq!(lines.utf8_to_line(39), 2);
        assert_eq!(lines.utf8_to_line(61), 2);
        assert_eq!(lines.utf8_to_line(62), 3); // <<-- this character is beyond the content.
        assert_eq!(lines.utf8_to_line(100), 3);
        assert_eq!(lines.utf8s_to_chars(4..10), 4..10);
        assert_eq!(lines.chars_to_utf8s(4..10), 4..10);

        assert_eq!(content.len(), 62);
        assert_eq!(lines.lines_to_utf8s(2..3), 39..62);
        assert_eq!(lines.lines_to_utf8s(2..4), 39..62);
        assert_eq!(lines.lines_to_chars(2..4), 39..62);
        assert_eq!(lines.utf8s_to_lines(39..62), 2..3);
        assert_eq!(lines.utf8s_to_lines(39..63), 2..3); // The "invalid" utf8 position results in a valid line position.
        assert_eq!(lines.char_to_utf8(62), 62);
        assert_eq!(lines.char_to_utf8(63), 62); // char 63 doesn't exist, so we map to the closest valid utf8 position.

        // Empty ranges
        assert_eq!(lines.utf8s_to_lines(0..0), 0..1);
        assert_eq!(lines.utf8s_to_lines(13..13), 0..1);
        assert_eq!(lines.utf8s_to_lines(14..14), 1..2);
        assert_eq!(lines.utf8s_to_lines(38..38), 1..2);
        assert_eq!(lines.utf8s_to_lines(39..39), 2..3);
        assert_eq!(lines.utf8s_to_lines(61..61), 2..3);
        assert_eq!(lines.utf8s_to_lines(62..62), 3..3);
        assert_eq!(lines.utf8s_to_lines(63..63), 3..3);
    }

    fn pos(line: usize, col: usize) -> Pos {
        Pos { line, col }
    }

    #[test]
    fn test_convert_ascii() {
        let content = r#"line0
line1"#;
        let lines = StringOffsets::new(content);
        assert_eq!(lines.utf8_to_char_pos(0), pos(0, 0));
        assert_eq!(lines.utf8_to_char_pos(1), pos(0, 1));
        assert_eq!(lines.utf8_to_char_pos(6), pos(1, 0));
        assert_eq!(lines.utf8_to_char_pos(7), pos(1, 1));
    }

    #[test]
    fn test_convert_unicode() {
        // Á - 2 bytes utf8
        let content = r#"❤️ line0
line1
✅ line2"#;
        let lines = StringOffsets::new(content);
        assert_eq!(lines.utf8_to_char_pos(0), pos(0, 0)); // ❤️ takes 6 bytes to represent in utf8 (2 code points)
        assert_eq!(lines.utf8_to_char_pos(1), pos(0, 0));
        assert_eq!(lines.utf8_to_char_pos(2), pos(0, 0));
        assert_eq!(lines.utf8_to_char_pos(3), pos(0, 1));
        assert_eq!(lines.utf8_to_char_pos(4), pos(0, 1));
        assert_eq!(lines.utf8_to_char_pos(5), pos(0, 1));

        assert_eq!(lines.utf8_to_char_pos(6), pos(0, 2)); // <space>
        assert_eq!(lines.utf8_to_char_pos(7), pos(0, 3)); // line
                                                          // ^

        assert_eq!(lines.utf8_to_char_pos(13), pos(1, 0)); // line
                                                           // ^

        assert_eq!(lines.utf8_to_char_pos(19), pos(2, 0)); // ✅ takes 3 bytes to represent in utf8 (1 code point)
        assert_eq!(lines.utf8_to_char_pos(20), pos(2, 0));
        assert_eq!(lines.utf8_to_char_pos(21), pos(2, 0));

        assert_eq!(lines.utf8_to_char_pos(22), pos(2, 1)); // <space>

        assert_eq!(lines.utf8_to_utf16_pos(0), pos(0, 0)); // ❤️ takes 4 bytes to represent in utf16 (2 code points)
        assert_eq!(lines.utf8_to_utf16_pos(1), pos(0, 0));
        assert_eq!(lines.utf8_to_utf16_pos(2), pos(0, 0));
        assert_eq!(lines.utf8_to_utf16_pos(3), pos(0, 1));
    }

    #[test]
    fn test_small() {
        // Á - 2 bytes utf8
        let content = r#"❤️ line0 ❤️Á 👋"#;
        let lines = StringOffsets::new(content);
        let mut utf16_index = 0;
        let mut char_index = 0;
        for (byte_index, char) in content.char_indices() {
            assert_eq!(lines.utf8_to_char(byte_index), char_index);
            assert_eq!(lines.utf8_to_utf16(byte_index), utf16_index);
            char_index += 1;
            utf16_index += char.len_utf16();
        }
        assert_eq!(lines.utf8_to_char(content.len()), char_index);
        assert_eq!(lines.utf8_to_utf16(content.len()), utf16_index);
    }

    #[test]
    fn test_variable_lengths() {
        let content = r#"❤️Á 👋"#;
        //                   ^~ utf8: 1 char, 4 bytes, utf16: 2 code units
        //                 ^~~~ utf8: 1 char, 1 byte, utf16: 1 code unit
        //                ^~~~~ utf8: 1 char, 2 bytes, utf16: 1 code unit
        //               ^~~~~~ utf8: 2 chars, 3 byte ea., utf16: 2 code units
        let lines = StringOffsets::new(content);

        // UTF-16 positions
        assert_eq!(lines.utf8_to_utf16_pos(0), pos(0, 0)); // ❤️
        assert_eq!(lines.utf8_to_utf16_pos(1), pos(0, 0));
        assert_eq!(lines.utf8_to_utf16_pos(2), pos(0, 0));
        assert_eq!(lines.utf8_to_utf16_pos(3), pos(0, 1));
        assert_eq!(lines.utf8_to_utf16_pos(5), pos(0, 1));
        assert_eq!(lines.utf8_to_utf16_pos(4), pos(0, 1));
        assert_eq!(lines.utf8_to_utf16_pos(6), pos(0, 2)); // Á
        assert_eq!(lines.utf8_to_utf16_pos(7), pos(0, 2));
        assert_eq!(lines.utf8_to_utf16_pos(8), pos(0, 3)); // <space>
        assert_eq!(lines.utf8_to_utf16_pos(9), pos(0, 4)); // 👋

        // These middle utf8 byte positions don't have valid mappings:
        // assert_eq!(lines.utf8_to_utf16_pos(10), pos(0, 4));
        // assert_eq!(lines.utf8_to_utf16_pos(11), pos(0, 5));
        //
        // 👋 in utf16: 0xd83d 0xdc4b
        // 👋 in utf8: 0xf0 0x9f 0x91 0x8b
        //                  ^    ^
        // It's not really defined where these inner bytes map to and it
        // doesn't matter because we would never report those byte offset as
        // they are in the middle of a character and therefore invalid.

        assert_eq!(lines.utf8_to_utf16_pos(12), pos(0, 5));

        // UTF-8 positions
        assert_eq!(lines.utf8_to_char_pos(0), pos(0, 0)); // ❤️
        assert_eq!(lines.utf8_to_char_pos(1), pos(0, 0));
        assert_eq!(lines.utf8_to_char_pos(2), pos(0, 0));
        assert_eq!(lines.utf8_to_char_pos(3), pos(0, 1));
        assert_eq!(lines.utf8_to_char_pos(4), pos(0, 1));
        assert_eq!(lines.utf8_to_char_pos(5), pos(0, 1));
        assert_eq!(lines.utf8_to_char_pos(6), pos(0, 2)); // Á
        assert_eq!(lines.utf8_to_char_pos(7), pos(0, 2));
        assert_eq!(lines.utf8_to_char_pos(8), pos(0, 3)); // <space>
        assert_eq!(lines.utf8_to_char_pos(9), pos(0, 4)); // 👋
        assert_eq!(lines.utf8_to_char_pos(10), pos(0, 4));
        assert_eq!(lines.utf8_to_char_pos(11), pos(0, 4));
        assert_eq!(lines.utf8_to_char_pos(12), pos(0, 4));
    }

    #[test]
    fn test_critical_input_len() {
        let content = [b'a'; 16384];
        let lines = StringOffsets::from_bytes(&content);
        assert_eq!(lines.utf8_to_utf16_pos(16384), pos(1, 0));
    }
}