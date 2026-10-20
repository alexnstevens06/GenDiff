import io
import random

import numpy as np

from core.AC import arithmeticcoding
from core.ac_frame import close_frame, encode_framed, open_frame
from core.continuous_bitstream import ContinuousBitStream


class _NonClosing(io.BytesIO):
    def close(self):
        pass


def _isolated_roundtrip(symbols, cumul, statesize=32):
    buf = _NonClosing()
    bitout = arithmeticcoding.BitOutputStream(buf)
    enc = arithmeticcoding.ArithmeticEncoder(statesize, bitout)
    for symbol in symbols:
        enc.write(cumul, symbol)
    enc.finish()
    bitout.close()
    bitin = arithmeticcoding.BitInputStream(io.BytesIO(buf.getvalue()))
    dec = arithmeticcoding.ArithmeticDecoder(statesize, bitin)
    return [dec.read(cumul, len(cumul) - 1) for _ in symbols]


def test_isolated_roundtrip_uniform_and_skewed():
    rng = random.Random(0)
    for alpha in (2, 3, 8, 32):
        cumul = np.arange(alpha + 1, dtype=np.uint64)
        for length in (0, 1, 5, 20):
            symbols = [rng.randrange(alpha) for _ in range(length)]
            assert _isolated_roundtrip(symbols, cumul) == symbols

    for _ in range(30):
        alpha = rng.choice((2, 5, 11))
        freqs = np.array([rng.randint(1, 400) for _ in range(alpha)], dtype=np.uint64)
        cumul = np.zeros(alpha + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(freqs)
        symbols = [rng.randrange(alpha) for _ in range(rng.randint(1, 40))]
        assert _isolated_roundtrip(symbols, cumul) == symbols


def test_framed_block_roundtrip_with_following_bits():
    """The decoder lookahead must not consume the bits that follow a block."""
    rng = random.Random(1)
    for _ in range(40):
        alpha = rng.choice((2, 3, 7))
        freqs = np.array([rng.randint(1, 200) for _ in range(alpha)], dtype=np.uint64)
        cumul = np.zeros(alpha + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(freqs)
        symbols = [rng.randrange(alpha) for _ in range(rng.randint(0, 25))]
        meta = [rng.randrange(2) for _ in range(24)]

        stream = ContinuousBitStream()

        def _write(enc, symbols=symbols, cumul=cumul):
            for symbol in symbols:
                enc.write(cumul, symbol)

        encode_framed(stream, _write)
        stream.write_bits(meta)

        stream.bit_position = 0
        frame = open_frame(stream)
        decoded = [frame.decoder.read(cumul, alpha) for _ in symbols]
        close_frame(frame)
        recovered = [stream.read() for _ in meta]
        assert decoded == symbols
        assert recovered == meta
