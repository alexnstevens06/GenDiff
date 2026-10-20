import difflib
import random

import pytest

from experiments.generative_reconstruction_compression.diff_ops import (
    LEN_BITS,
    MAX_RUN,
    OPCODE_EQUAL,
    coalesce_opcodes,
    write_count_opcode,
    write_uint,
)


class _Bits:
    def __init__(self):
        self.bits = []

    def write(self, bit):
        self.bits.append(bit)


def _read(bits, n, pos):
    value = 0
    for _ in range(n):
        value = (value << 1) | bits[pos]
        pos += 1
    return value, pos


def _apply(ref, tgt, ops):
    out = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            assert ref[i1:i2] == tgt[j1:j2]
            out.extend(ref[i1:i2])
        elif tag == "delete":
            assert j1 == j2
        elif tag in ("insert", "replace"):
            out.extend(tgt[j1:j2])
        else:
            raise AssertionError(tag)
    return out


def test_write_uint_rejects_overflow():
    sink = _Bits()
    with pytest.raises(ValueError):
        write_uint(sink, 1 << 16, 16)
    write_uint(sink, MAX_RUN, 16)
    assert len(sink.bits) == 16


def test_count_opcode_splits_runs_past_16_bits():
    sink = _Bits()
    total = MAX_RUN + 4465
    write_count_opcode(sink, OPCODE_EQUAL, total)
    pos = 0
    chunks = []
    while pos < len(sink.bits):
        _opcode, pos = _read(sink.bits, 2, pos)
        count, pos = _read(sink.bits, LEN_BITS, pos)
        chunks.append(count)
    assert chunks == [MAX_RUN, 4465]
    assert sum(chunks) == total


def test_coalesce_preserves_the_target_text():
    rng = random.Random(0)
    for _ in range(80):
        ref = [rng.randrange(5) for _ in range(rng.randint(0, 36))]
        tgt = list(ref)
        for _edit in range(rng.randint(0, 5)):
            if not tgt or rng.random() < 0.3:
                at = rng.randint(0, len(tgt))
                tgt[at:at] = [rng.randrange(5) for _ in range(rng.randint(1, 3))]
            elif rng.random() < 0.5:
                del tgt[rng.randrange(len(tgt))]
            else:
                tgt[rng.randrange(len(tgt))] = rng.randrange(5)
        ops = difflib.SequenceMatcher(None, ref, tgt, autojunk=False).get_opcodes()
        merged = coalesce_opcodes(ops, ref, tgt, min_equal=4)
        assert _apply(ref, tgt, ops) == tgt
        assert _apply(ref, tgt, merged) == tgt
