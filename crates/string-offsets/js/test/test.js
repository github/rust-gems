//@ts-check
const { StringOffsets } = require('../pkg/string_offsets.js');

describe('StringOffsets sanity checks', () => {
    test('basic ASCII text', () => {
        const text = "hello\nworld";
        const offsets = new StringOffsets(text);

        expect(offsets.lines()).toBe(2);
        expect(offsets.utf8_to_utf16(0)).toBe(0);
        expect(offsets.utf8_to_line(0)).toBe(0);
    });

    test('Unicode text', () => {
        const text = "☀️hello\n🗺️world";
        const offsets = new StringOffsets(text);

        expect(offsets.lines()).toBe(2);
        // ☀️ is 6 UTF-8 bytes and 3 UTF-16 code units
        expect(offsets.utf8_to_utf16(6)).toBe(2);
        // expect(offsets.utf16ToUtf8(3)).toBe(6);
        expect(offsets.utf8_to_utf16(0)).toBe(0);
    });
});
