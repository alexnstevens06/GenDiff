"""Length-prefixed arithmetic-coded blocks on a ContinuousBitStream.

``ArithmeticEncoder.finish`` writes a terminating 1 and 31 zeros. That trailer
is enough when a later read past the end is treated as zero, which is the case
for a stream that ends. It does not line the reader up with a following field.
On random frequency tables the decoder reads as many as 13 bits past the
trailer, and that overshoot is not a constant, so rewinding by a fixed count
lands on the wrong bit.

A framed block stores the payload length in bits, then the finished code, then
``GUARD_ZEROS`` explicit zeros. Those zeros keep the lookahead inside the
block. After the symbols are decoded the reader seeks to the recorded end.
"""

from __future__ import annotations

from collections.abc import Callable

from .AC.arithmeticcoding import ArithmeticDecoder, ArithmeticEncoder
from .continuous_bitstream import ContinuousBitStream

GUARD_ZEROS = 64
LENGTH_BITS = 32


class FramedBlock:
    """An arithmetic block opened on a continuous stream."""

    def __init__(
        self,
        stream: ContinuousBitStream,
        start: int,
        nbits: int,
        decoder: ArithmeticDecoder,
    ):
        self.stream = stream
        self.start = start
        self.nbits = nbits
        self.decoder = decoder

    @property
    def end(self) -> int:
        return self.start + self.nbits


def encode_framed(
    stream: ContinuousBitStream,
    encode_symbols: Callable[[ArithmeticEncoder], None],
) -> int:
    """Encode symbols, finish the coder, and append a length-prefixed block.

    ``encode_symbols`` must not call ``finish``; this function does.
    Returns the payload length in bits, not counting the length field.
    """
    side = ContinuousBitStream()
    encoder = ArithmeticEncoder(32, side)
    encode_symbols(encoder)
    encoder.finish()
    side.write_bits([0] * GUARD_ZEROS)
    nbits = side.bit_position
    if nbits >= (1 << LENGTH_BITS):
        raise ValueError(
            f"framed block is {nbits} bits, above the {LENGTH_BITS}-bit length field"
        )
    stream.write_uint(nbits, LENGTH_BITS)
    side.bit_position = 0
    for _ in range(nbits):
        stream.write(side.read_no_eof())
    return nbits


def open_frame(stream: ContinuousBitStream) -> FramedBlock:
    """Read a framed block and return a decoder positioned at its payload."""
    nbits = stream.read_uint(LENGTH_BITS)
    start = stream.bit_position
    decoder = ArithmeticDecoder(32, stream)
    return FramedBlock(stream, start, nbits, decoder)


def close_frame(frame: FramedBlock) -> None:
    """Seek to the first bit after the framed payload."""
    if frame.stream.bit_position > frame.end:
        over = frame.stream.bit_position - frame.end
        raise RuntimeError(
            f"arithmetic decoder read {over} bits past the framed block; "
            "increase GUARD_ZEROS"
        )
    frame.stream.bit_position = frame.end
