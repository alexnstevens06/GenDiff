"""A tiny deterministic causal model for codec round-trips.

The arithmetic coder and LLMzip loop are the code under test. This model only
supplies next-token logits so those paths can run without downloading a
language model.
"""

from __future__ import annotations

import numpy as np
import torch

from core.llmzip_utils import LLMzip

ALPHABET = "abcdefghijklmnopqrstuvwxyz "


class _Config:
    def __init__(self, vocab_size: int, max_position_embeddings: int):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings


class _Output:
    def __init__(self, logits: torch.Tensor, past: list[int]):
        self.logits = logits
        self.past_key_values = past


class TinyCausalLM:
    """Bigram model: the next-token logits depend on the last input id."""

    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int = 32,
        seed: int = 0,
    ):
        self.config = _Config(vocab_size, max_position_embeddings)
        rng = np.random.default_rng(seed)
        self._table = rng.standard_normal((vocab_size, vocab_size)).astype(np.float32)

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        del use_cache
        last = int(input_ids[0, -1].item())
        logits = torch.from_numpy(self._table[last]).to(dtype=torch.float32).view(1, 1, -1)
        if past_key_values is None:
            history = [int(token) for token in input_ids[0].tolist()]
        else:
            history = list(past_key_values) + [int(token) for token in input_ids[0].tolist()]
        return _Output(logits, history)


class CharTokenizer:
    """Invertible character tokenizer. Token 0 is EOS; characters are 1..N."""

    def __init__(self, alphabet: str = ALPHABET):
        self.alphabet = alphabet
        self.eos_token_id = 0
        self.bos_token_id = None
        self.pad_token_id = None

    def encode(self, text, return_tensors=None):
        try:
            ids = [self.alphabet.index(ch) + 1 for ch in text]
        except ValueError as exc:
            raise ValueError(f"character outside {self.alphabet!r}: {text!r}") from exc
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, tokens) -> str:
        chars = []
        for token in tokens:
            index = int(token)
            if index == self.eos_token_id:
                continue
            chars.append(self.alphabet[index - 1])
        return "".join(chars)


def make_llmzip(max_window: int = 4, max_position_embeddings: int = 8) -> LLMzip:
    vocab_size = len(ALPHABET) + 1
    model = TinyCausalLM(vocab_size, max_position_embeddings=max_position_embeddings)
    tokenizer = CharTokenizer()
    return LLMzip(model, tokenizer, device="cpu", max_window=max_window)
