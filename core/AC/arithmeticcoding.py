#
# Arithmetic coding — rewritten for precise bit consumption tracking
#
# Based on the algorithms from:
#   https://www.nayuki.io/page/reference-arithmetic-coding
#   https://github.com/nayuki/Reference-arithmetic-coding
#
# Key improvements over the original Nayuki reference code:
#   1. The decoder tracks exactly how many bits it has consumed via the
#      `bits_consumed` property.
#   2. BitInputStream supports `unread(n)` to push unconsumed bits back,
#      so that after decoding the caller can hand the stream to the next
#      consumer without losing data.
#   3. BitInputStream supports `position` (total bits read) so that callers
#      can compute exactly how many bits a particular decode used.
#

import sys
import numpy as np

python3 = sys.version_info.major >= 3


# ---- Bit-oriented I/O streams ----

class BitInputStream:
    """
    A stream of bits read from an underlying byte stream (big-endian).
    
    Tracks the total number of bits dispensed via `position`.
    Supports `unread(n)` to logically push back the last n bits so that
    the next reader sees them.
    """

    def __init__(self, inp):
        self.input = inp
        self.currentbyte = 0
        self.numbitsremaining = 0
        self._position = 0          # total bits dispensed

    @property
    def position(self):
        """Total number of bits read (dispensed) from this stream."""
        return self._position

    def read(self):
        """
        Read one bit.  Returns 0 or 1 if a bit is available,
        or -1 if end of stream is reached.
        """
        if self.currentbyte == -1:
            return -1
        if self.numbitsremaining == 0:
            temp = self.input.read(1)
            if len(temp) == 0:
                self.currentbyte = -1
                return -1
            self.currentbyte = temp[0] if python3 else ord(temp)
            self.numbitsremaining = 8
        assert self.numbitsremaining > 0
        self.numbitsremaining -= 1
        self._position += 1
        return (self.currentbyte >> self.numbitsremaining) & 1

    def unread(self, n):
        """
        Push back `n` bits that were previously read.  This adjusts the
        internal pointer so those bits will be re-read by the next call(s)
        to `read()`.
        
        Only valid when the bits being pushed back are still within the
        current byte (i.e. n <= 8 - numbitsremaining, and we haven't
        crossed a byte boundary since those bits were read).
        
        For the arithmetic decoder use-case this is called at most once
        after finishing, pushing back the unused portion of the look-ahead.
        """
        self.numbitsremaining += n
        self._position -= n
        assert self.numbitsremaining <= 8

    def read_no_eof(self):
        """Read one bit, raising EOFError instead of returning -1."""
        result = self.read()
        if result != -1:
            return result
        else:
            raise EOFError()

    def close(self):
        self.input.close()
        self.currentbyte = -1
        self.numbitsremaining = 0


class BitOutputStream:
    """
    A stream where bits can be written to an underlying byte stream.
    Bits are written in big-endian order.  On close the current byte is
    padded with zeros.
    """

    def __init__(self, out):
        self.output = out
        self.currentbyte = 0
        self.numbitsfilled = 0

    def write(self, b):
        """Write a single bit (0 or 1)."""
        if b not in (0, 1):
            raise ValueError("Argument must be 0 or 1")
        self.currentbyte = (self.currentbyte << 1) | b
        self.numbitsfilled += 1
        if self.numbitsfilled == 8:
            towrite = bytes((self.currentbyte,)) if python3 else chr(self.currentbyte)
            self.output.write(towrite)
            self.currentbyte = 0
            self.numbitsfilled = 0

    def write_bits(self, bits):
        """Write an iterable of bits."""
        for bit in bits:
            self.write(bit)

    def close(self):
        """Pad to byte boundary and close the underlying stream."""
        while self.numbitsfilled != 0:
            self.write(0)
        self.output.close()


# ---- Arithmetic coding core ----

class ArithmeticCoderBase:
    """State and behaviours shared by encoder and decoder."""

    def __init__(self, statesize):
        self.STATE_SIZE = statesize
        self.MAX_RANGE = 1 << statesize          # 2^STATE_SIZE
        self.MIN_RANGE = (self.MAX_RANGE >> 2) + 2
        self.MAX_TOTAL = self.MIN_RANGE
        self.MASK = self.MAX_RANGE - 1
        self.TOP_MASK = self.MAX_RANGE >> 1
        self.SECOND_MASK = self.TOP_MASK >> 1

        self.low = 0
        self.high = self.MASK

    def update(self, cumul, symbol):
        """Narrow the [low, high] interval for the given symbol."""
        low = self.low
        high = self.high
        range_ = high - low + 1

        total = cumul[-1].item()
        symlow = cumul[symbol].item()
        symhigh = cumul[symbol + 1].item()

        self.low = low + symlow * range_ // total
        self.high = low + symhigh * range_ // total - 1

        # Shift out matching top bits
        while ((self.low ^ self.high) & self.TOP_MASK) == 0:
            self.shift()
            self.low = (self.low << 1) & self.MASK
            self.high = ((self.high << 1) & self.MASK) | 1

        # Underflow expansion
        while (self.low & ~self.high & self.SECOND_MASK) != 0:
            self.underflow()
            self.low = (self.low << 1) & (self.MASK >> 1)
            self.high = ((self.high << 1) & (self.MASK >> 1)) | self.TOP_MASK | 1

    def shift(self):
        raise NotImplementedError()

    def underflow(self):
        raise NotImplementedError()


# ---- Encoder ----

class ArithmeticEncoder(ArithmeticCoderBase):
    """Encodes symbols and writes to a bit output stream."""

    def __init__(self, statesize, bitout):
        super().__init__(statesize)
        self.output = bitout
        self.num_underflow = 0
        self._bits_written = 0

    def write(self, cumul, symbol):
        """Encode one symbol using the given cumulative frequency table."""
        self.update(cumul, symbol)

    def finish(self):
        """Flush the encoder state so that the output can be decoded."""
        self.output.write(1)
        self._bits_written += 1
        # Explicitly pad with 31 zeros to isolate the final fraction 
        # from any subsequent metadata in a continuous bitstream.
        for _ in range(31):
            self.output.write(0)
            self._bits_written += 1

    def shift(self):
        bit = self.low >> (self.STATE_SIZE - 1)
        self.output.write(bit)
        self._bits_written += 1
        for _ in range(self.num_underflow):
            self.output.write(bit ^ 1)
            self._bits_written += 1
        self.num_underflow = 0

    def underflow(self):
        self.num_underflow += 1

    @property
    def bits_written(self):
        """Total number of bits emitted by the encoder so far."""
        return self._bits_written


# ---- Decoder ----

class ArithmeticDecoder(ArithmeticCoderBase):
    """
    Reads from an arithmetic-coded bit stream and decodes symbols.

    On construction, reads STATE_SIZE bits to fill the code register
    (this is fundamental to the algorithm).  The `bits_consumed` property
    reports exactly how many bits the decoder has read from the input.
    """

    def __init__(self, statesize, bitin):
        super().__init__(statesize)
        self.input = bitin
        self.code = 0
        self._bits_consumed = 0
        for _ in range(self.STATE_SIZE):
            self.code = self.code << 1 | self._read_code_bit()

    @property
    def bits_consumed(self):
        """Total number of bits read from the input stream."""
        return self._bits_consumed

    def read(self, cumul, alphabet_size):
        """Decode the next symbol given a cumulative frequency table."""
        total = cumul[-1].item()
        range_ = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * total - 1) // range_

        # Binary search for the symbol
        start = 0
        end = alphabet_size
        while end - start > 1:
            middle = (start + end) >> 1
            if cumul[middle] > value:
                end = middle
            else:
                start = middle

        symbol = start
        self.update(cumul, symbol)
        return symbol

    def shift(self):
        self.code = ((self.code << 1) & self.MASK) | self._read_code_bit()

    def underflow(self):
        self.code = (self.code & self.TOP_MASK) | \
                    ((self.code << 1) & (self.MASK >> 1)) | self._read_code_bit()

    def _read_code_bit(self):
        """Read one bit from the input stream; EOF → 0."""
        temp = self.input.read()
        if temp == -1:
            temp = 0
        else:
            self._bits_consumed += 1
        return temp
