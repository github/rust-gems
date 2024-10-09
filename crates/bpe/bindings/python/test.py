#!/usr/bin/env python3

import bpe

cl100k = bpe.cl100k()

enc = cl100k.encode_via_backtracking("Hello, world!".encode())
print(enc)
cnt = cl100k.count("Hello, world!".encode())
print(cnt)
dec = cl100k.decode_tokens(enc).decode()
print(dec)
