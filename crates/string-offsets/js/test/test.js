//@ts-check
const assert = require('node:assert/strict');
const test = require('node:test');
const { StringOffsets } = require('../');

test('basic ASCII text', () => {
    const text = "hello\nworld";
    const offsets = new StringOffsets(text);

    assert.equal(offsets.lines(), 2);
    assert.equal(offsets.utf8ToUtf16(0), 0);
    assert.equal(offsets.utf8ToLine(0), 0);
});

test('Unicode text', () => {
    const text = "☀️hello\n🗺️world";
    const offsets = new StringOffsets(text);

    assert.equal(offsets.lines(), 2);
    // ☀️ is 6 UTF-8 bytes and 3 UTF-16 code units
    assert.equal(offsets.utf8ToUtf16(6), 2);
    assert.equal(offsets.utf8ToUtf16(0), 0);
});
