import torch

from core.llmzip_utils import LLMzip
from tests.tiny_lm import ALPHABET, CharTokenizer, TinyCausalLM, make_llmzip


def test_default_device_is_cpu_when_cuda_is_unavailable():
    assert not torch.cuda.is_available()
    vocab = len(ALPHABET) + 1
    compressor = LLMzip(TinyCausalLM(vocab), CharTokenizer())
    assert compressor.device == "cpu"
    assert compressor.decode(compressor.encode("ab")) == "ab"


def test_roundtrip_empty_short_and_window_reset():
    # max_position_embeddings=8 forces cache eviction (max_pos_len is 7).
    compressor = make_llmzip(max_window=3, max_position_embeddings=8)
    samples = ["", "a", "hi", "the quick brown fox jumps"]
    for text in samples:
        assert compressor.decode(compressor.encode(text)) == text


def test_probs_to_cumul_is_strict_and_within_coder_range():
    compressor = make_llmzip()
    probs = torch.softmax(torch.randn(compressor.vocab_size), dim=0).numpy()
    cumul = compressor._probs_to_cumul(probs)
    assert len(cumul) == compressor.vocab_size + 1
    assert cumul[0] == 0
    assert int(cumul[-1]) < (1 << 30)
    assert all(int(cumul[i + 1]) > int(cumul[i]) for i in range(len(cumul) - 1))
