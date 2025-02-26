#!/usr/bin/env python3

import bpe

tok = bpe.openai.cl100k_base()

## Use tokenizer

enc = tok.encode("Hello, world!")
print(enc)
cnt = tok.count("Hello, world!")
print(cnt)
dec = tok.decode(enc)
print(dec)

## Use underlying BPE instance

bpe = tok.bpe()
