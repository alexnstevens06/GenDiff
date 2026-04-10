# Arithmetic Coding & LLMzip — Reference Resources

## Arithmetic Coding Implementation

### Nayuki Reference Arithmetic Coding
- **Page**: https://www.nayuki.io/page/reference-arithmetic-coding
- **Repo**: https://github.com/nayuki/Reference-arithmetic-coding
- **Why relevant**: `core/AC/arithmeticcoding.py` is a direct derivative. The header says:
  > "Based on the algorithms from: https://www.nayuki.io/page/reference-arithmetic-coding"
  
  Modifications made in this repo over the upstream:
  1. `BitInputStream.position` — total bits dispensed counter
  2. `BitInputStream.unread(n)` — push back unconsumed lookahead bits
  3. `ArithmeticDecoder.bits_consumed` — exact bits-read accounting
  4. NumPy cumulative frequency tables instead of Python list-based ones

---

## The Lookahead Bits Problem

### What it is
The decoder constructor (`ArithmeticDecoder.__init__`) immediately reads **`STATE_SIZE` bits** (32 here)
from the input stream to prime its `code` register before it has decoded a single symbol.
These bits are not "consumed" by any symbol — they're the decoder's working window into the future
of the bitstream.

```python
for _ in range(self.STATE_SIZE):          # reads 32 bits immediately
    self.code = self.code << 1 | self._read_code_bit()
```

### Why this breaks naïve chunk concatenation
If you try to embed multiple AC-encoded blobs back-to-back in a single stream:

```
[blob A bits][blob B bits][blob C bits]
```

When you create a new `ArithmeticDecoder` to read blob B, it immediately
siphons 32 bits of lookahead — which may cross the blob A / blob B boundary,
stealing bits that belong to blob B's *content*.

This also means after blob A is fully decoded, up to 32 bits of blob B's data
have already been consumed and are sitting in the decoder's `code` register,
invisible to the caller.

### How this repo handles it
Two mechanisms work together:

1. **Isolated inner buffers** (`encode_segment_to_bitout` in `simple_grc.py`):
   Each INSERT segment is encoded into a private `BytesIO` blob. The blob's
   byte length is written as a 16-bit prefix. The decoder reads exactly that
   many bytes into a fresh local buffer and creates its `ArithmeticDecoder`
   against that isolated stream — so lookahead never bleeds across blob boundaries.

2. **Byte-alignment flush** (`_flush_bits`) before each blob:
   The outer `BitOutputStream` is padded to a byte boundary before writing
   the blob length + blob bytes. This ensures the decoder always starts
   reading a blob at a byte-aligned position in the outer stream.

3. **`unread(n)` on `BitInputStream`**:
   The `unread` method was added precisely to support the case where a decoder
   finishes and the caller needs to know how many bits were *actually* used
   (vs. the lookahead the decoder consumed but didn't need).

### Reference on the underlying theory
- Wikipedia — Arithmetic coding: https://en.wikipedia.org/wiki/Arithmetic_coding
- CMU lecture notes (Arithmetic Coding): https://www.cs.cmu.edu/~aarti/Class/10704/lec/Lec9.pdf
- Observable HQ interactive explainer: https://observablehq.com/@seldrid1/arithmetic-coding

---

## LLMzip (the base compression algorithm)

### Original paper
- **Title**: LLMZip: Lossless Text Compression using Large Language Models
- **arXiv**: https://arxiv.org/abs/2306.04050
- **Authors**: Chandra et al., 2023
- **Key result**: LLaMA-7B + arithmetic coding beats ZPAQ, BSC, paq8h on text8 (~1.47 bpc)
- **Core idea**: LLM next-token probabilities → arithmetic coding cumulative frequencies.
  Better model perplexity → closer to entropy → lower bpc.
