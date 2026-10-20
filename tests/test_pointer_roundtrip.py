from experiments.generative_reconstruction_compression.pointer_grc import (
    bit_cost,
    decode_pointer_grczip,
    encode_pointer_grczip,
)
from tests.tiny_lm import make_llmzip


def test_bit_cost_includes_the_top_of_the_range():
    assert bit_cost(0) == 1
    assert bit_cost(1) == 1
    assert bit_cost(2) == 2
    assert bit_cost(4) == 3
    assert bit_cost(8) == 4
    for n in range(0, 64):
        width = bit_cost(n)
        assert n < (1 << width)


def test_pointer_roundtrip_including_power_of_two_spans():
    compressor = make_llmzip(max_window=4, max_position_embeddings=16)
    cases = [
        ("abcd", "abcd"),
        ("abcd", ""),
        ("abcdefgh", ""),
        ("ab", "abq"),
        ("abcd", "abcdq"),
        ("abcdefgh", "abcdefghq"),
        ("abcdefgh", "abcdqfgh"),
        ("abcdefgh", "abxyefghzz"),
        ("", "hello"),
        ("hello", "hullo"),
        ("the cat sat", "the dog sat here"),
    ]
    for reference, target in cases:
        blob = encode_pointer_grczip(reference, target, compressor)
        assert decode_pointer_grczip(reference, blob, compressor) == target
