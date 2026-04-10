"""
Minimal Byte-Pair Encoding (BPE) tokenizer implemented from scratch.

This module provides a BPETokenizer class that implements the BPE algorithm
using only the Python standard library. The tokenizer operates at the byte
level (UTF-8 bytes form the initial vocabulary) and learns merge rules by
iteratively combining the most frequent adjacent token pair.

Typical usage:
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=512)
    ids = tokenizer.encode("hello world")
    text = tokenizer.decode(ids)
    assert text == "hello world"
"""

from __future__ import annotations
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BPETokenizer")


class BPETokenizer:
    """A minimal Byte-Pair Encoding tokenizer.

    The tokenizer starts with a base vocabulary of all 256 possible byte values
    (tokens 0–255) and learns merge operations by repeatedly combining the most
    frequent adjacent pair of tokens in the training corpus.  Each merge creates
    a new token whose byte representation is the concatenation of the two
    constituent tokens' bytes.

    Attributes:
        merges: Ordered list of ((left_id, right_id), new_id) tuples recording
            each merge operation in the order it was learned during training.
        vocab: Dictionary mapping every token ID to the ``bytes`` object it
            represents.  IDs 0–255 always map to the corresponding single byte.
    """

    def __init__(self) -> None:
        """Initialize the tokenizer with empty state.

        Call :meth:`train` to populate the vocabulary and merge list.
        """
        self.merges: list[tuple[tuple[int, int], int]] = []
        self.vocab: dict[int, bytes] = {}
        logger.debug("BPETokenizer instanced with empty state.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, corpus: str, vocab_size: int) -> None:
        """Train the BPE tokenizer on the given corpus.

        Starting from a 256-token byte-level vocabulary, repeatedly find the
        most frequent adjacent pair of tokens in the current tokenised
        representation of *corpus*, merge the pair into a new single token,
        and record the merge rule.  The process stops when the vocabulary
        reaches *vocab_size* or when no adjacent pairs remain.

        Args:
            corpus: The training text used to derive merge rules.
            vocab_size: Desired total vocabulary size (must be ≥ 256).

        Raises:
            ValueError: If *vocab_size* is less than 256.
        """
        if vocab_size < 256:
            logger.error("Attempted to train with vocab_size < 256")
            raise ValueError("vocab_size must be at least 256 (byte-level base)")

        logger.info("Starting BPE training with target vocab_size=%d", vocab_size)

        # Seed the vocabulary with every possible single byte
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []

        # Convert the entire corpus into a flat list of byte-level token IDs
        tokens: list[int] = list(corpus.encode("utf-8"))

        # Nothing to merge if the corpus has fewer than two bytes
        if len(tokens) < 2:
            return

        next_id: int = 256  # First available ID beyond the byte-level tokens

        while next_id < vocab_size:
            # --- Step 1: count every adjacent pair ---------------------------------
            pair_counts: dict[tuple[int, int], int] = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break  # No pairs left to merge (shouldn't happen unless len < 2)

            # --- Step 2: pick the most frequent pair -------------------------------
            # Ties are broken by whatever ``max`` returns (depends on tuple order).
            best_pair = max(pair_counts, key=pair_counts.get)

            # --- Step 3: create a new token for the merged pair --------------------
            new_id = next_id
            left_bytes = self.vocab[best_pair[0]]
            right_bytes = self.vocab[best_pair[1]]
            self.vocab[new_id] = left_bytes + right_bytes
            self.merges.append((best_pair, new_id))
            logger.debug("Merged %s + %s -> ID %d", left_bytes, right_bytes, new_id)

            # --- Step 4: apply the merge everywhere in the token list --------------
            tokens = self._merge_tokens(tokens, best_pair, new_id)

            next_id += 1

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of BPE token IDs.

        The input text is first converted to UTF-8 bytes (IDs 0–255) and
        then each learned merge rule is applied in the order it was
        discovered during training.  Applying merges in training order
        guarantees that the encoded output is consistent with the
        vocabulary.

        Args:
            text: The input string to encode.

        Returns:
            A list of integer token IDs.
        """
        logger.debug("Encoding input string of length %d", len(text))
        # Start from raw UTF-8 bytes
        tokens: list[int] = list(text.encode("utf-8"))

        # Replay every merge rule in the order it was learned
        for pair, new_id in self.merges:
            tokens = self._merge_tokens(tokens, pair, new_id)

        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decode a list of BPE token IDs back into a string.

        Each token ID is looked up in the vocabulary to retrieve the
        corresponding bytes; the bytes are concatenated and decoded as
        UTF-8.

        Args:
            ids: A list of integer token IDs previously produced by
                :meth:`encode`.

        Returns:
            The decoded Unicode string.

        Raises:
            KeyError: If any token ID is not present in the vocabulary.
        """
        logger.debug("Decoding sequence of %d tokens", len(ids))
        byte_parts: list[bytes] = []
        for token_id in ids:
            if token_id not in self.vocab:
                raise KeyError(
                    f"Token ID {token_id} is not in the vocabulary. "
                    "Was the tokenizer trained or loaded correctly?"
                )
            byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_tokens(
        tokens: list[int],
        pair: tuple[int, int],
        new_id: int,
    ) -> list[int]:
        """Replace all non-overlapping occurrences of *pair* with *new_id*.

        The scan proceeds left-to-right so that overlapping occurrences are
        handled deterministically: in ``[a, a, a]`` with pair ``(a, a)``,
        the leftmost two ``a`` tokens are merged first, producing
        ``[new_id, a]``.

        Args:
            tokens: Current list of token IDs.
            pair: The adjacent pair ``(left, right)`` to merge.
            new_id: The token ID that replaces each occurrence of *pair*.

        Returns:
            A new list of token IDs with every occurrence of *pair* replaced
            by *new_id*.
        """
        merged: list[int] = []
        i = 0
        while i < len(tokens):
            # Check whether the current position starts the target pair
            if (
                i + 1 < len(tokens)
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                merged.append(new_id)
                i += 2  # Advance past both elements of the merged pair
            else:
                merged.append(tokens[i])
                i += 1
        return merged


# ======================================================================
# Self-test / demonstration
# ======================================================================

if __name__ == "__main__":
    # --- Build a small training corpus ------------------------------------
    sentence = "the quick brown fox jumps over the lazy dog"
    corpus = (sentence + " ") * 10  # repeat with trailing spaces

    # --- Train the tokenizer ----------------------------------------------
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=300)

    # --- Encode and decode the same corpus ---------------------------------
    encoded: list[int] = tokenizer.encode(corpus)
    decoded: str = tokenizer.decode(encoded)

    # Round-trip must be lossless
    assert decoded == corpus, (
        f"Round-trip failed!\n"
        f"  Original : {corpus[:80]!r}…\n"
        f"  Decoded  : {decoded[:80]!r}…"
    )

    # --- Also test on unseen text -----------------------------------------
    test_text = "the fuzzy fox"
    test_encoded = tokenizer.encode(test_text)
    test_decoded = tokenizer.decode(test_encoded)
    assert test_decoded == test_text, "Round-trip on unseen text failed!"

    # --- Pretty-print a summary -------------------------------------------
    print("=" * 60)
    print("  BPE Tokenizer — Training Summary")
    print("=" * 60)
    logger.info("  Corpus length       : %s characters", len(corpus))
    logger.info("  Byte tokens (before): %s", len(corpus.encode('utf-8')))
    logger.info("  BPE tokens  (after) : %s", len(encoded))
    logger.info("  Compression ratio   : %.2f×", len(corpus) / len(encoded))
    logger.info("  Vocabulary size     : %d", len(tokenizer.vocab))
    logger.info("  Merge rules learned : %d", len(tokenizer.merges))

    # Show the first few merge rules
    print("  First 10 merges (pair → new_id):")
    for pair, new_id in tokenizer.merges[:10]:
        left = tokenizer.vocab[pair[0]]
        right = tokenizer.vocab[pair[1]]
        combined = tokenizer.vocab[new_id]
        print(
            f"    ({pair[0]:>3}, {pair[1]:>3}) → {new_id:>3}  "
            f"{left!r} + {right!r} → {combined!r}"
        )
    print()

    # Show a couple of multi-byte vocabulary entries
    multi_byte = {
        tid: val
        for tid, val in sorted(tokenizer.vocab.items())
        if len(val) > 1
    }
    print(f"  Sample multi-byte tokens (of {len(multi_byte)} total):")
    for tid in list(multi_byte)[:8]:
        print(f"    {tid:>3} → {multi_byte[tid]!r}")
    print()

    # Show encoding of the unseen test text
    print(f"  Unseen-text round-trip test:")
    print(f"    Input  : {test_text!r}")
    print(f"    Encoded: {test_encoded}")
    print(f"    Decoded: {test_decoded!r}")
    print()

    print("  ✓ All round-trip assertions passed!")
    print("=" * 60)