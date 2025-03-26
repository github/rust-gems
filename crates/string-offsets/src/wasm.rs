use wasm_bindgen::prelude::*;

use crate::{AllConfig, Pos, StringOffsets as StringOffsetsImpl};

#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct StringOffsets(StringOffsetsImpl<AllConfig>);

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl StringOffsets {
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(content: &str) -> Self {
        Self(StringOffsetsImpl::new(content))
    }

    #[allow(unused_variables)]
    #[cfg_attr(feature = "wasm", wasm_bindgen(static_method_of = StringOffsets))]
    pub fn from_bytes(content: &[u8]) -> Self {
        Self(StringOffsetsImpl::from_bytes(content))
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(js_name = lines))]
    pub fn lines(&self) -> usize {
        self.0.lines()
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(js_name = lineToUtf8Begin))]
    pub fn line_to_utf8_begin(&self, line_number: usize) -> usize {
        self.0.line_to_utf8_begin(line_number)
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(js_name = lineToUtf8End))]
    pub fn line_to_utf8_end(&self, line_number: usize) -> usize {
        self.0.line_to_utf8_end(line_number)
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(js_name = utf8ToLine))]
    pub fn utf8_to_line(&self, byte_number: usize) -> usize {
        self.0.utf8_to_line(byte_number)
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(js_name = lineChars))]
    pub fn line_chars(&self, line_number: usize) -> usize {
        self.0.line_chars(line_number)
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(js_name = lineToCharBegin))]
    pub fn line_to_char_begin(&self, line_number: usize) -> usize {
        self.0.line_to_char_begin(line_number)
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(js_name = lineToCharEnd))]
    pub fn line_to_char_end(&self, line_number: usize) -> usize {
        self.0.line_to_char_end(line_number)
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(js_name = utf8ToCharPos))]
    pub fn utf8_to_char_pos(&self, byte_number: usize) -> Pos {
        self.0.utf8_to_char_pos(byte_number)
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(js_name = utf8ToChar))]
    pub fn utf8_to_char(&self, byte_number: usize) -> usize {
        self.0.utf8_to_char(byte_number)
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(js_name = charToUtf8))]
    pub fn char_to_utf8(&self, char_number: usize) -> usize {
        self.0.char_to_utf8(char_number)
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(js_name = utf8ToUtf16))]
    pub fn utf8_to_utf16(&self, byte_number: usize) -> usize {
        self.0.utf8_to_utf16(byte_number)
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(js_name = lineToUtf16Begin))]
    pub fn line_to_utf16_begin(&self, line_number: usize) -> usize {
        self.0.line_to_utf16_begin(line_number)
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(js_name = lineToUtf16End))]
    pub fn line_to_utf16_end(&self, line_number: usize) -> usize {
        self.0.line_to_utf16_end(line_number)
    }

    #[cfg_attr(feature = "wasm", wasm_bindgen(js_name = utf8ToUtf16Pos))]
    pub fn utf8_to_utf16_pos(&self, byte_number: usize) -> Pos {
        self.0.utf8_to_utf16_pos(byte_number)
    }
}
