import pytest

from core.continuous_bitstream import ContinuousBitStream


def test_write_read_uint_across_byte_boundary():
    stream = ContinuousBitStream()
    stream.write_bits([1, 0, 1, 1])
    stream.write_uint(170, 8)
    assert stream.bit_position == 12
    raw = stream.get_bytes()
    assert raw[0] == 0b10111010
    assert raw[1] == 0b10100000

    stream.bit_position = 0
    assert [stream.read() for _ in range(4)] == [1, 0, 1, 1]
    assert stream.read_uint(8) == 170


def test_rewind_and_overwrite():
    stream = ContinuousBitStream()
    stream.write_uint(0b11110000, 8)
    stream.bit_position = 4
    stream.write_bits([0, 0, 0, 0])
    assert stream.get_bytes() == bytes([0b11110000])
    stream.bit_position = 0
    assert stream.read_uint(8) == 0b11110000


def test_seek_past_end_fills_the_gap_before_writing():
    stream = ContinuousBitStream()
    stream.bit_position = 16
    stream.write(1)
    assert stream.bit_position == 17
    assert list(stream.get_bytes()) == [0x00, 0x00, 0x80]


def test_read_past_end_returns_minus_one_and_read_uint_raises():
    stream = ContinuousBitStream(b"\x80")
    assert stream.read() == 1
    stream.bit_position = 8
    assert stream.read() == -1
    with pytest.raises(EOFError):
        stream.read_uint(1)


def test_write_uint_rejects_values_that_do_not_fit():
    stream = ContinuousBitStream()
    with pytest.raises(ValueError):
        stream.write_uint(4, 2)
    with pytest.raises(ValueError):
        stream.write_uint(-1, 8)
    with pytest.raises(ValueError):
        stream.read_uint(-1)


def test_rewind_rejects_negative_and_past_start():
    stream = ContinuousBitStream()
    stream.write(1)
    with pytest.raises(ValueError):
        stream.rewind(-1)
    with pytest.raises(ValueError):
        stream.rewind(2)
