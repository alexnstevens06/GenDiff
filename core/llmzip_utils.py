"""
LLMzip Utilities
----------------
This module provides a model-agnostic implementation of the LLMzip compression algorithm.
It uses standard Hugging Face Transformer interfaces for causal language modeling and 
integrates with arithmetic coding for entropy encoding.

ROCm Support:
-------------
This module includes environment overrides for AMD GPU support on Linux (ROCm).
Specifically, it sets HSA_OVERRIDE_GFX_VERSION=10.3.0 which is necessary for 
compatibility with many Navi-based GPUs (RX 6000 series).

Usage:
------
    from core.llmzip_utils import LLMzip
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("path/to/model")
    tokenizer = AutoTokenizer.from_pretrained("path/to/model")
    
    compressor = LLMzip(model, tokenizer)
    compressor.encode("Text to compress", "output.llmzip")
    text = compressor.decode("output.llmzip")
"""

import os
import sys
import torch
import numpy as np
import io
from typing import Optional, Tuple, Any, Union

# ROCm environment override for Navi 22/23/24 support
if sys.platform.startswith("linux"):
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

# Import arithmetic coding from the core directory
try:
    from .AC import arithmeticcoding
except ImportError:
    # Fallback if imported from outside core
    from core.AC import arithmeticcoding


class LLMzip:
    """
    A model-agnostic LLMzip compressor and decompressor.

    Attributes:
        model: A Hugging Face CausalLM model.
        tokenizer: A Hugging Face tokenizer.
        device: The device to run inference on (e.g., 'cuda', 'cpu').
        max_window: Maximum context window for KV cache re-computation.
        min_free_vram: Minimum VRAM buffer (bytes) before evicting cache.
        vocab_size: Determined vocabulary size of the model.
        max_pos_len: Maximum position embedding length of the model.

    VRAM Management:
        This class implements dynamic cache eviction during inference to prevent OOM.
        However, users should call the .cleanup() method manually when finished with
        heavy operations to explicitly clear the PyTorch VRAM buffer and release
        persisted states.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = 'cuda',
        max_window: int = 50,
        min_free_vram: int = 512 * 1024 * 1024
    ):
        """
        Initializes the LLMzip instance.

        Args:
            model: transformers.PreTrainedModel (CausalLM).
            tokenizer: transformers.PreTrainedTokenizer.
            device: String representing the device.
            max_window: How many tokens of context to use if KV cache is reset.
            min_free_vram: Minimum free VRAM in bytes to maintain (default 0.5 GB).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_window = max_window
        self.min_free_vram = min_free_vram
        self.vocab_size = self._resolve_vocab_size()

        # Max position limit: evict cache before exceeding model's max_position_embeddings
        config = self.model.config
        if hasattr(config, 'text_config'):
            config = config.text_config
        self.max_pos_len = int(
            getattr(config, 'max_position_embeddings', 32768) * 0.95)

        # Stabilize EOS ID
        self.eos_id = self.tokenizer.eos_token_id
        if self.eos_id is None:
            # Fallback if EOS is not defined (rare for causal LMs)
            self.eos_id = self.tokenizer.pad_token_id or 0

    def cleanup(self):
        """
        Explicitly clears VRAM by emptying the PyTorch cache.
        Call this after finished with encodes/decodes to release GPU memory.
        """
        if self.device == 'cuda' or 'cuda' in str(self.device):
            import torch
            torch.cuda.empty_cache()
            print("VRAM buffer cleared.")
        else:
            print("Cleanup: No CUDA device active.")

    def _resolve_vocab_size(self) -> int:
        """
        Extracts vocab_size from model config, handling architecture variations.

        Returns:
            The vocabulary size as an integer.
        """
        config = self.model.config
        if hasattr(config, 'vocab_size'):
            return config.vocab_size
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'vocab_size'):
            return config.text_config.vocab_size

        # Fallback: infer from the model's lm_head output dimension
        if hasattr(self.model, 'lm_head'):
            return self.model.lm_head.out_features

        raise ValueError(
            "Cannot determine vocab_size from model config or architecture")

    def _get_free_vram(self) -> float:
        """
        Checks available VRAM on CUDA devices.

        Returns:
            Free VRAM in bytes, or infinity for non-CUDA devices.
        """
        if self.device == 'cuda' and torch.cuda.is_available():
            try:
                free, _ = torch.cuda.mem_get_info()
                return free
            except Exception:
                return float('inf')
        return float('inf')

    def _vram_is_safe(self) -> bool:
        """Checks if current free VRAM is above the defined safety threshold."""
        return self._get_free_vram() >= self.min_free_vram

    def _evict_cache(self):
        """Standard cache eviction to prevent OOM on high-sequence lengths."""
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def _get_probs(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None
    ) -> Tuple[np.ndarray, Any]:
        """
        Performs a forward pass and returns normalized probabilities for the next token.

        Args:
            input_ids: Tensor of token IDs.
            past_key_values: KV cache to speed up inference.

        Returns:
            A tuple of (probabilities_numpy, updated_past_key_values).
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(
                logits, dim=-1).cpu().to(torch.float32).numpy()[0]

        # Sanitize probabilities to avoid numerical errors in arithmetic coding
        if np.isnan(probs).any() or np.isinf(probs).any():
            probs = np.ones_like(probs) / len(probs)
        probs_sum = np.sum(probs)
        if probs_sum <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum

        return probs, outputs.past_key_values

    def _probs_to_cumul(self, probs: np.ndarray) -> np.ndarray:
        """
        Converts probability distribution to cumulative frequencies for the encoder.

        Args:
            probs: Numpy array of token probabilities.

        Returns:
            Cumulative frequency table as a uint64 numpy array.
        """
        vocab_size = len(probs)
        scale = 1000000.0
        freqs = probs * scale
        freqs = np.nan_to_num(freqs, nan=1.0, posinf=1.0, neginf=1.0)
        freqs = freqs.astype(np.uint64)
        freqs = np.maximum(freqs, 1)

        total_freq = np.sum(freqs)
        # Handle overflow for the 32-bit arithmetic coding state
        if total_freq >= (1 << 30):
            freqs = (freqs / 2).astype(np.uint64)
            freqs = np.maximum(freqs, 1)

        cumul = np.zeros(vocab_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(freqs)
        return cumul

    def encode(self, text: str, output_file: Optional[str] = None) -> bytes:
        """
        Tokenizes text and encodes it into a compressed bitstream.

        Args:
            text: Input string to compress.
            output_file: Optional path where the .llmzip file will be saved.

        Returns:
            The compressed bitstream as bytes.

        Note:
            It is recommended to call .cleanup() after this method if you are 
            finished with heavy GPU operations.
        """
        input_ids = self.tokenizer.encode(
            text, return_tensors='pt').to(self.device)[0]

        # Append EOS token to signal end of stream during decoding
        eos = torch.tensor([self.eos_id], device=self.device)
        tokens = torch.cat([input_ids, eos])
        num_tokens = len(tokens)
        vocab_size = self.vocab_size

        print(f"Total tokens to encode: {num_tokens}")

        # Wrapper to prevent BitOutputStream from closing the BytesIO buffer
        class NonClosingBytesIO(io.BytesIO):
            def close(self):
                pass

        buffer = NonClosingBytesIO()
        bitout = arithmeticcoding.BitOutputStream(buffer)
        enc = arithmeticcoding.ArithmeticEncoder(32, bitout)

        # First token: uniform distribution (no context)
        uniform_cumul = np.arange(vocab_size + 1, dtype=np.uint64)
        enc.write(uniform_cumul, tokens[0].item())

        past_kv = None
        cache_len = 0

        for i in range(1, num_tokens):
            # Context maintenance: Reset if VRAM low or max position length reached
            if past_kv is not None and (not self._vram_is_safe() or cache_len >= self.max_pos_len):
                past_kv = None
                cache_len = 0
                self._evict_cache()

            if past_kv is None:
                # Partial context rebuild using sliding window
                start = max(0, i - self.max_window)
                context = tokens[start:i].unsqueeze(0)
                probs, past_kv = self._get_probs(context)
                cache_len = i - start
            else:
                # Efficient incremental update
                last_token = tokens[i - 1:i].unsqueeze(0)
                probs, past_kv = self._get_probs(last_token, past_kv)
                cache_len += 1

            cumul = self._probs_to_cumul(probs)
            enc.write(cumul, tokens[i].item())

            if i % 10 == 0:
                print(
                    f"Encoded {i}/{num_tokens} tokens (cache: {cache_len})", end='\r')
                sys.stdout.flush()

        enc.finish()
        bitout.close()  # Now pads with 0s but doesn't close the real buffer
        compressed_bytes = buffer.getvalue()
        buffer.seek(0)
        # buffer.real_close() or similar omitted, we'll let GC handle the underlying buffer
        # Or just:
        super(NonClosingBytesIO, buffer).close()

        if output_file:
            with open(output_file, 'wb') as f:
                f.write(compressed_bytes)

        print(f"\nEncoding complete.")
        self._print_stats(text, len(compressed_bytes), output_file)
        return compressed_bytes

    def decode(self, input_data: Union[str, bytes]) -> str:
        """
        Reads a compressed bitstream and reconstructs the original text.

        Args:
            input_data: Path to the .llmzip file (str) or raw bitstream (bytes).

        Returns:
            The decoded string.

        Note:
            It is recommended to call .cleanup() after this method if you are 
            finished with heavy GPU operations.
        """
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise FileNotFoundError(f"Input file {input_data} not found.")
            file_in = open(input_data, 'rb')
        elif isinstance(input_data, bytes):
            file_in = io.BytesIO(input_data)
        else:
            raise TypeError("input_data must be a file path (str) or bytes.")

        bitin = arithmeticcoding.BitInputStream(file_in)
        dec = arithmeticcoding.ArithmeticDecoder(32, bitin)

        vocab_size = self.vocab_size

        # First token: uniform distribution
        uniform_cumul = np.arange(vocab_size + 1, dtype=np.uint64)
        first_token = dec.read(uniform_cumul, vocab_size)

        decoded_tokens = [first_token]
        # Check if first token is EOS (empty text case)
        if first_token == self.eos_id:
            bitin.close()
            file_in.close()
            return ""

        past_kv = None
        cache_len = 0

        while True:
            # Context maintenance
            if past_kv is not None and (not self._vram_is_safe() or cache_len >= self.max_pos_len):
                past_kv = None
                cache_len = 0
                self._evict_cache()

            if past_kv is None:
                start = max(0, len(decoded_tokens) - self.max_window)
                context = torch.tensor(
                    [decoded_tokens[start:]], device=self.device)
                probs, past_kv = self._get_probs(context)
                cache_len = len(decoded_tokens) - start
            else:
                last_token = torch.tensor(
                    [[decoded_tokens[-1]]], device=self.device)
                probs, past_kv = self._get_probs(last_token, past_kv)
                cache_len += 1

            cumul = self._probs_to_cumul(probs)

            try:
                symbol = dec.read(cumul, vocab_size)
            except EOFError:
                break

            decoded_tokens.append(symbol)

            if symbol == self.eos_id:
                break

            if len(decoded_tokens) % 10 == 0:
                print(
                    f"Decoded {len(decoded_tokens)} tokens (cache: {cache_len})", end='\r')
                sys.stdout.flush()

        bitin.close()
        file_in.close()

        # Final cleanup: Remove BOS/EOS tokens
        self._filter_tokens(decoded_tokens)
        return self.tokenizer.decode(decoded_tokens)

    def _filter_tokens(self, tokens: list):
        """Removes BOS and EOS tokens from the list in-place."""
        if tokens and tokens[-1] == self.eos_id:
            tokens.pop()
        if tokens and self.tokenizer.bos_token_id is not None and tokens[0] == self.tokenizer.bos_token_id:
            tokens.pop(0)

    def _print_stats(self, original_text: str, comp_size: int, output_file: Optional[str] = None):
        """Prints compression statistics."""
        orig_size = len(original_text)
        bpc = (comp_size * 8) / orig_size if orig_size > 0 else 0
        print(f"Original size: {orig_size} chars")
        print(f"Compressed size: {comp_size} bytes")
        print(f"Bits per character (BPC): {bpc:.4f}")
        if output_file:
            print(f"Saved to: {output_file}")
