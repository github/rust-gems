use wasm_bindgen::prelude::*;

use crate::{AllConfig, Pos, StringOffsets as StringOffsetsImpl};

#[wasm_bindgen]
pub struct StringOffsets(StringOffsetsImpl<AllConfig>);

#[wasm_bindgen]
#[allow(non_snake_case)]
impl StringOffsets {
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> Self {
        Self(StringOffsetsImpl::new(content))
    }

    #[allow(unused_variables)]
    #[wasm_bindgen(static_method_of = StringOffsets)]
    pub fn from_bytes(content: &[u8]) -> Self {
        Self(StringOffsetsImpl::from_bytes(content))
    }

    pub fn lines(&self) -> usize {
        self.0.lines()
    }

    pub fn lineToUtf8Begin(&self, line_number: usize) -> usize {
        self.0.line_to_utf8_begin(line_number)
    }

    pub fn lineToUtf8End(&self, line_number: usize) -> usize {
        self.0.line_to_utf8_end(line_number)
    }

    pub fn utf8ToLine(&self, byte_number: usize) -> usize {
        self.0.utf8_to_line(byte_number)
    }

    pub fn lineChars(&self, line_number: usize) -> usize {
        self.0.line_chars(line_number)
    }

    pub fn lineToCharBegin(&self, line_number: usize) -> usize {
        self.0.line_to_char_begin(line_number)
    }

    pub fn lineToCharEnd(&self, line_number: usize) -> usize {
        self.0.line_to_char_end(line_number)
    }

    pub fn utf8ToCharPos(&self, byte_number: usize) -> Pos {
        self.0.utf8_to_char_pos(byte_number)
    }

    pub fn utf8ToChar(&self, byte_number: usize) -> usize {
        self.0.utf8_to_char(byte_number)
    }

    pub fn charToUtf8(&self, char_number: usize) -> usize {
        self.0.char_to_utf8(char_number)
    }

    pub fn utf8ToUtf16(&self, byte_number: usize) -> usize {
        self.0.utf8_to_utf16(byte_number)
    }

    pub fn lineToUtf16Begin(&self, line_number: usize) -> usize {
        self.0.line_to_utf16_begin(line_number)
    }

    pub fn lineToUtf16End(&self, line_number: usize) -> usize {
        self.0.line_to_utf16_end(line_number)
    }

    pub fn utf8ToUtf16Pos(&self, byte_number: usize) -> Pos {
        self.0.utf8_to_utf16_pos(byte_number)
    }
}
