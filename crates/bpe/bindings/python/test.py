#!/usr/bin/env python3

import bpe

tok = bpe.cl100k_base()

enc = tok.encode("Hello, world!")
print(enc)
cnt = tok.count("Hello, world!")
print(cnt)
dec = tok.decode(enc)
print(dec)
