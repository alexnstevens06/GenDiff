"""
Generative Reconstruction Compression (GRC)
============================================
Compresses the *diff* between a reference program (LLM-generated from a
prompt + seed) and a revised version of that program.

.grczip file layout
-------------------
  [32-bit seed]                               unsigned integer, big-endian bits
  [16-bit prompt-blob length][prompt blob]    LLMzip AC segment (same as INSERT)
  ... diff opcode stream ...
  [EOS opcode]
  [0-7 padding bits to byte boundary]

Opcode table (2 bits per opcode)
---------------------------------
  (0,0) EQUAL  — keep N tokens from reference   [16-bit token count]
  (1,0) DELETE — skip N tokens in reference     [16-bit token count]
  (1,1) INSERT — AC-coded token segment         [16-bit blob length + blob bytes]
  (0,1) EOS    — end of opcode stream           [no payload]

Diff granularity
----------------
Diffs are computed at the *LLM token* level, not at the character or line level.
This ensures every INSERT segment starts and ends on a token boundary, keeping
the LLM's probability model perfectly aligned with the units being compressed.

Context-aware encoding
----------------------
Both encoder and decoder maintain a rolling `context_tokens` list of the token
IDs output so far (EQUAL tokens + previous INSERT tokens; deleted tokens are
excluded because they don't appear in the output).  Before encoding/decoding
each INSERT segment the model is warm-started on the most recent
`context_window` context tokens, giving it accurate probability distributions
for the first INSERT token instead of the flat uniform prior used when there is
no context.

INSERT blob protocol
--------------------
  1. Byte-align the outer BitOutputStream.
  2. Arithmetic-encode [insert_tokens ... EOS] into an inner buffer.
  3. Flush inner buffer to byte boundary (the AC decoder treats EOF as
     trailing zeros, so no extra padding is needed).
  4. Write 16-bit blob length + blob bytes into the outer stream.
"""

import io
import sys
import os
import difflib
from collections import deque

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import LLMzip, BitInputStream, BitOutputStream
from core.AC import arithmeticcoding


# ---------------------------------------------------------------------------
# Opcode table
# ---------------------------------------------------------------------------
OPCODE_EQUAL  = (0, 0)
OPCODE_DELETE = (1, 0)
OPCODE_INSERT = (1, 1)
OPCODE_EOS    = (0, 1)

LEN_BITS   = 16   # bit-width for EQUAL / DELETE token counts  (max 65535 tokens)
BLOB_BITS  = 16   # bit-width for INSERT blob byte count       (max 65535 bytes)
SEED_BITS  = 32   # bit-width for the random seed integer
STATE_SIZE = 32   # arithmetic coder state bits

# Default sliding context window (token count from previous output)
DEFAULT_CONTEXT_WINDOW = 100

# ---------------------------------------------------------------------------
# Globals (populated in main())
# ---------------------------------------------------------------------------
model      = None
tokenizer  = None
llmzip     = None


# ---------------------------------------------------------------------------
# Bit-level helpers
# ---------------------------------------------------------------------------

def write_uint(bitout: BitOutputStream, value: int, num_bits: int):
    """Write value as a big-endian num_bits-wide unsigned integer."""
    for i in range(num_bits - 1, -1, -1):
        bitout.write((value >> i) & 1)


def read_uint(bitin: BitInputStream, num_bits: int) -> int:
    """Read num_bits bits from bitin, return as a big-endian unsigned int."""
    value = 0
    for _ in range(num_bits):
        bit = bitin.read()
        if bit == -1:
            raise EOFError("Unexpected end of bitstream while reading uint")
        value = (value << 1) | bit
    return value


def _flush_bits(bitout: BitOutputStream):
    """Pad the current byte to a boundary without closing the underlying buffer."""
    while bitout.numbitsfilled != 0:
        bitout.write(0)


def _align_reader_to_byte(bitin: BitInputStream):
    """Discard remaining bits in the current byte to resync on a byte boundary."""
    if bitin.numbitsremaining > 0:
        for _ in range(bitin.numbitsremaining):
            bitin.read()


# ---------------------------------------------------------------------------
# AC segment encoder
# ---------------------------------------------------------------------------

def encode_segment_to_bitout(
    token_ids: list,
    bitout: BitOutputStream,
    context_tokens: list = None,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
):
    """
    Arithmetic-encode `token_ids` (with a trailing EOS) into `bitout`.

    If `context_tokens` is non-empty, the model is warm-started on the last
    `context_window` tokens before encoding the first INSERT token, giving the
    model accurate priors on all tokens rather than a flat uniform prior for
    the first token.

    Writes: [byte-align] [16-bit blob length] [blob bytes]
    """
    vocab_size = llmzip.vocab_size
    eos_id     = llmzip.eos_id
    tokens     = list(token_ids) + [eos_id]

    # 1) Byte-align the outer stream before the blob
    _flush_bits(bitout)

    # Inner non-closing buffer
    class _NC(io.BytesIO):
        def close(self): pass

    inner_buf    = _NC()
    inner_bitout = BitOutputStream(inner_buf)
    enc          = arithmeticcoding.ArithmeticEncoder(STATE_SIZE, inner_bitout)

    past_kv   = None
    cache_len = 0

    # Warm-start KV cache from context if available
    ctx = list(context_tokens[-context_window:]) if context_tokens else []
    if ctx:
        ctx_tensor = torch.tensor([ctx], device=llmzip.device)
        probs_first, past_kv = llmzip._get_probs(ctx_tensor)
        cache_len = len(ctx)
        cumul = llmzip._probs_to_cumul(probs_first)
        enc.write(cumul, tokens[0])
        start_i = 1
    else:
        # No context: flat uniform prior for the very first token
        uniform_cumul = np.arange(vocab_size + 1, dtype=np.uint64)
        enc.write(uniform_cumul, tokens[0])
        start_i = 1

    for i in range(start_i, len(tokens)):
        # VRAM guard
        if past_kv is not None and (
            not llmzip._vram_is_safe() or cache_len >= llmzip.max_pos_len
        ):
            past_kv   = None
            cache_len = 0
            llmzip._evict_cache()

        if past_kv is None:
            # Rebuild context from recently seen tokens
            recent_ctx = (ctx + tokens[:i])[-llmzip.max_window:]
            ctx_t = torch.tensor([recent_ctx], device=llmzip.device)
            probs, past_kv = llmzip._get_probs(ctx_t)
            cache_len = len(recent_ctx)
        else:
            prev_tok = torch.tensor([[tokens[i - 1]]], device=llmzip.device)
            probs, past_kv = llmzip._get_probs(prev_tok, past_kv)
            cache_len += 1

        cumul = llmzip._probs_to_cumul(probs)
        enc.write(cumul, tokens[i])

    enc.finish()

    # Byte-align the inner blob (no extra padding needed — the AC decoder
    # treats end-of-stream as trailing zeros, so BitOutputStream's natural
    # byte-alignment padding is sufficient)
    _flush_bits(inner_bitout)

    blob = inner_buf.getvalue()
    io.BytesIO.close(inner_buf)

    # 3) Write 16-bit blob length + blob into outer stream
    write_uint(bitout, len(blob), BLOB_BITS)
    for byte in blob:
        for shift in range(7, -1, -1):
            bitout.write((byte >> shift) & 1)


# ---------------------------------------------------------------------------
# AC segment decoder
# ---------------------------------------------------------------------------

def decode_segment_from_bitin(
    bitin: BitInputStream,
    context_tokens: list = None,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> list:
    """
    Decode one LLMzip-encoded segment from `bitin`.

    Mirrors encode_segment_to_bitout exactly:
      - Byte-aligns before reading
      - Reads 16-bit blob length, then blob bytes
      - Decodes token IDs until EOS (from an isolated buffer)
      - Returns the decoded token ID list (EOS stripped)
    """
    _align_reader_to_byte(bitin)

    blob_len   = read_uint(bitin, BLOB_BITS)
    blob_bytes = bytearray()
    for _ in range(blob_len):
        blob_bytes.append(read_uint(bitin, 8))

    local_bitin = BitInputStream(io.BytesIO(blob_bytes))
    dec         = arithmeticcoding.ArithmeticDecoder(STATE_SIZE, local_bitin)
    vocab_size  = llmzip.vocab_size
    eos_id      = llmzip.eos_id

    ctx = list(context_tokens[-context_window:]) if context_tokens else []
    past_kv   = None
    cache_len = 0

    if ctx:
        ctx_tensor = torch.tensor([ctx], device=llmzip.device)
        probs_first, past_kv = llmzip._get_probs(ctx_tensor)
        cache_len = len(ctx)
        cumul       = llmzip._probs_to_cumul(probs_first)
        first_token = dec.read(cumul, vocab_size)
    else:
        uniform_cumul = np.arange(vocab_size + 1, dtype=np.uint64)
        first_token   = dec.read(uniform_cumul, vocab_size)

    if first_token == eos_id:
        return []

    decoded = [first_token]
    past_kv_dec   = past_kv
    cache_len_dec = cache_len

    while True:
        if past_kv_dec is not None and (
            not llmzip._vram_is_safe() or cache_len_dec >= llmzip.max_pos_len
        ):
            past_kv_dec   = None
            cache_len_dec = 0
            llmzip._evict_cache()

        if past_kv_dec is None:
            recent_ctx = (ctx + decoded)[-llmzip.max_window:]
            ctx_t = torch.tensor([recent_ctx], device=llmzip.device)
            probs, past_kv_dec = llmzip._get_probs(ctx_t)
            cache_len_dec = len(recent_ctx)
        else:
            prev_tok = torch.tensor([[decoded[-1]]], device=llmzip.device)
            probs, past_kv_dec = llmzip._get_probs(prev_tok, past_kv_dec)
            cache_len_dec += 1

        cumul = llmzip._probs_to_cumul(probs)

        try:
            symbol = dec.read(cumul, vocab_size)
        except EOFError:
            break

        if symbol == eos_id:
            break

        decoded.append(symbol)

    return decoded


# ---------------------------------------------------------------------------
# Decoder — reconstruct target token list from reference + bitstream
# ---------------------------------------------------------------------------

def reconstruct_diffs(
    ref_tokens: list,
    bitin: BitInputStream,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> list:
    """
    Reconstruct the target token list by applying GRC opcodes to `ref_tokens`.

    Maintains a rolling `context_tokens` list (output tokens so far) that is
    passed to `decode_segment_from_bitin` for context-aware decoding.

    Args:
        ref_tokens:     Token ID list of the reference (LLM-generated) program.
        bitin:          BitInputStream over the diff portion of the .grczip file.
        context_window: Maximum number of context tokens to pass to each INSERT.

    Returns:
        The reconstructed target token ID list.
    """
    output_tokens   = []
    context_tokens  = deque(maxlen=context_window)
    ref_pos         = 0

    while True:
        b0 = bitin.read()
        if b0 == -1:
            break
        b1 = bitin.read()
        if b1 == -1:
            break

        opcode = (b0, b1)

        if opcode == OPCODE_EOS:
            break

        elif opcode == OPCODE_EQUAL:
            count  = read_uint(bitin, LEN_BITS)
            toks   = ref_tokens[ref_pos: ref_pos + count]
            output_tokens.extend(toks)
            context_tokens.extend(toks)
            ref_pos += count

        elif opcode == OPCODE_DELETE:
            count    = read_uint(bitin, LEN_BITS)
            ref_pos += count   # skip; not added to context

        elif opcode == OPCODE_INSERT:
            toks = decode_segment_from_bitin(
                bitin,
                context_tokens=list(context_tokens),
                context_window=context_window,
            )
            output_tokens.extend(toks)
            context_tokens.extend(toks)

        else:
            print(f"Warning: unknown opcode ({b0},{b1}) — aborting.", file=sys.stderr)
            break

    return output_tokens


# ---------------------------------------------------------------------------
# Opcode coalescing — merge small EQUAL runs into surrounding INSERTs
# ---------------------------------------------------------------------------

# Overhead per INSERT/REPLACE segment (in bits):
#   opcode (2) + blob-length prefix (16) + byte-alignment padding (~4)
#   + EOS token in AC stream (~15)  ≈ 37 bits
# Overhead of an EQUAL opcode: 2 + 16 = 18 bits
# Keeping INSERT|EQUAL(N)|INSERT as three ops costs ~92 bits of overhead.
# Merging into one INSERT costs ~37 bits + N × avg_bits_per_token (~5).
# Break-even: N ≈ (92 - 37) / 5 ≈ 11 tokens.
MIN_EQUAL_TOKENS = 12   # EQUAL runs shorter than this are absorbed into INSERTs


def coalesce_opcodes(
    opcodes: list,
    ref_tokens: list,
    tgt_tokens: list,
    min_equal: int = MIN_EQUAL_TOKENS,
) -> list:
    """
    Post-process difflib opcodes to merge small EQUAL regions into
    adjacent INSERT/REPLACE regions.

    An EQUAL region of fewer than `min_equal` tokens that sits between
    two "modifying" operations (insert, replace, or delete+insert) is
    absorbed: the tokens it covers are folded into a single larger
    REPLACE operation that spans the combined range.

    This dramatically reduces the number of AC segment blobs (and their
    per-segment overhead) at the cost of re-encoding a few tokens that
    were previously free EQUAL copies.

    Returns a new opcode list in the same (tag, i1, i2, j1, j2) format.
    """
    if not opcodes:
        return opcodes

    merged = list(opcodes)
    changed = True

    while changed:
        changed = False
        new = []
        i = 0
        while i < len(merged):
            # Look for pattern: [modifying] [small EQUAL] [modifying]
            if (i + 2 < len(merged)
                    and merged[i][0] in ("insert", "replace", "delete")
                    and merged[i + 1][0] == "equal"
                    and merged[i + 2][0] in ("insert", "replace", "delete")
                    and (merged[i + 1][2] - merged[i + 1][1]) < min_equal):

                # Merge all three into one replace spanning the full range
                a = merged[i]
                eq = merged[i + 1]
                b = merged[i + 2]

                # Combined ref range: from start of a to end of b
                ri1 = a[1]
                ri2 = b[2]
                # Combined tgt range: from start of a to end of b
                rj1 = a[3]
                rj2 = b[4]

                new.append(("replace", ri1, ri2, rj1, rj2))
                i += 3
                changed = True
            else:
                new.append(merged[i])
                i += 1
        merged = new

    # Clean up: adjacent replaces can be merged
    final = []
    for op in merged:
        if (final
                and final[-1][0] == "replace"
                and op[0] == "replace"
                and final[-1][2] == op[1]
                and final[-1][4] == op[3]):
            prev = final.pop()
            final.append(("replace", prev[1], op[2], prev[3], op[4]))
        else:
            final.append(op)

    return final


def _summarize_opcodes(opcodes, ref_tokens, tgt_tokens):
    """Print a brief summary of the opcode list for debugging."""
    counts = {}
    for tag, i1, i2, j1, j2 in opcodes:
        counts[tag] = counts.get(tag, 0) + 1
    equal_toks = sum(i2 - i1 for tag, i1, i2, j1, j2 in opcodes if tag == "equal")
    insert_toks = sum(j2 - j1 for tag, i1, i2, j1, j2 in opcodes
                      if tag in ("insert", "replace"))
    n_segments = sum(1 for tag, *_ in opcodes if tag in ("insert", "replace"))
    print(f"  Opcodes: {len(opcodes)} "
          f"({counts.get('equal',0)} equal, {counts.get('delete',0)} delete, "
          f"{counts.get('insert',0)} insert, {counts.get('replace',0)} replace)")
    print(f"  Equal tokens: {equal_toks} | Insert/Replace tokens: {insert_toks} | "
          f"AC segments: {n_segments}")


# ---------------------------------------------------------------------------
# .grczip writer
# ---------------------------------------------------------------------------

def write_grczip(
    bitout: BitOutputStream,
    seed: int,
    prompt: str,
    ref_tokens: list,
    tgt_tokens: list,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
):
    """
    Write a complete .grczip file into `bitout`.

    Layout:
      [SEED_BITS seed] [prompt blob] [diff opcodes] [EOS] [padding]
    """
    # --- Seed ---
    write_uint(bitout, seed, SEED_BITS)

    # --- Prompt (encoded as an LLMzip segment, no context) ---
    prompt_token_ids = llmzip.tokenizer.encode(prompt)
    print(f"Prompt tokens: {len(prompt_token_ids)}")
    encode_segment_to_bitout(prompt_token_ids, bitout, context_tokens=None)

    # --- Token-level diffs ---
    print(f"Reference tokens: {len(ref_tokens)}, Target tokens: {len(tgt_tokens)}")
    matcher = difflib.SequenceMatcher(None, ref_tokens, tgt_tokens, autojunk=False)

    raw_opcodes = matcher.get_opcodes()
    print(f"\n  Before coalescing:")
    _summarize_opcodes(raw_opcodes, ref_tokens, tgt_tokens)

    opcodes = coalesce_opcodes(raw_opcodes, ref_tokens, tgt_tokens)
    print(f"  After coalescing (min_equal={MIN_EQUAL_TOKENS}):")
    _summarize_opcodes(opcodes, ref_tokens, tgt_tokens)

    context_tokens = deque(maxlen=context_window)
    n_ops = len(opcodes)

    for idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        print(f"  [{idx+1}/{n_ops}] {tag:7}  ref[{i1}:{i2}]  tgt[{j1}:{j2}]", end='\r')

        if tag == "equal":
            toks = ref_tokens[i1:i2]
            for bit in OPCODE_EQUAL:
                bitout.write(bit)
            write_uint(bitout, len(toks), LEN_BITS)
            context_tokens.extend(toks)

        elif tag == "delete":
            toks = ref_tokens[i1:i2]
            for bit in OPCODE_DELETE:
                bitout.write(bit)
            write_uint(bitout, len(toks), LEN_BITS)
            # Deleted tokens NOT added to context

        elif tag == "insert":
            toks = tgt_tokens[j1:j2]
            for bit in OPCODE_INSERT:
                bitout.write(bit)
            encode_segment_to_bitout(
                toks, bitout,
                context_tokens=list(context_tokens),
                context_window=context_window,
            )
            context_tokens.extend(toks)

        elif tag == "replace":
            # Decompose: DELETE old tokens, INSERT new tokens
            del_toks = ref_tokens[i1:i2]
            ins_toks = tgt_tokens[j1:j2]

            for bit in OPCODE_DELETE:
                bitout.write(bit)
            write_uint(bitout, len(del_toks), LEN_BITS)

            for bit in OPCODE_INSERT:
                bitout.write(bit)
            encode_segment_to_bitout(
                ins_toks, bitout,
                context_tokens=list(context_tokens),
                context_window=context_window,
            )
            context_tokens.extend(ins_toks)

    print()  # newline after progress

    # --- EOS sentinel ---
    for bit in OPCODE_EOS:
        bitout.write(bit)

    _flush_bits(bitout)


# ---------------------------------------------------------------------------
# .grczip reader
# ---------------------------------------------------------------------------

def read_grczip(
    bitin: BitInputStream,
    ref_tokens: list,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> tuple:
    """
    Read and decode a .grczip file.

    Returns:
        (seed: int, prompt: str, reconstructed_tokens: list)
    """
    seed = read_uint(bitin, SEED_BITS)
    print(f"Seed: {seed}")

    prompt_token_ids = decode_segment_from_bitin(
        bitin, context_tokens=None, context_window=context_window
    )
    prompt = llmzip.tokenizer.decode(prompt_token_ids)
    print(f"Prompt ({len(prompt_token_ids)} tokens): {prompt[:80]}...")

    reconstructed = reconstruct_diffs(ref_tokens, bitin, context_window)
    return seed, prompt, reconstructed


# ---------------------------------------------------------------------------
# Main — benchmark test
# ---------------------------------------------------------------------------

def main():
    global model, tokenizer, llmzip

    context_window = DEFAULT_CONTEXT_WINDOW

    # --- Paths ---
    exp_dir = os.path.dirname(__file__)
    seed_path          = os.path.join(exp_dir, "nbody_seed.txt")
    prompt_path        = os.path.join(exp_dir, "nbody_prompt.txt")
    ref_path           = os.path.join(exp_dir, "nbody_candidate.py")
    tgt_path           = os.path.join(exp_dir, "nbody_candidate_revised.py")
    grczip_path        = os.path.join(exp_dir, "nbody.grczip")
    llmzip_path        = os.path.join(exp_dir, "nbody_revised.llmzip")

    # --- Load inputs ---
    seed   = int(open(seed_path).read().strip())
    prompt = open(prompt_path).read()
    ref_text = open(ref_path).read()
    tgt_text = open(tgt_path).read()

    # --- Load model ---
    model_name = "Qwen/Qwen3.5-4B-Base"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        llmzip = LLMzip(model, tokenizer, device=model.device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # --- Tokenise reference and target at LLM token level ---
    print("\nTokenising reference and target...")
    ref_tokens = llmzip.tokenizer.encode(ref_text)
    tgt_tokens = llmzip.tokenizer.encode(tgt_text)
    print(f"  Reference: {len(ref_tokens)} tokens ({len(ref_text)} chars)")
    print(f"  Target:    {len(tgt_tokens)} tokens ({len(tgt_text)} chars)")

    # ====================================================================
    # 1. Encode → .grczip
    # ====================================================================
    print(f"\n=== Encoding → {grczip_path} ===")
    out_buffer = io.BytesIO()
    bitout     = BitOutputStream(out_buffer)

    write_grczip(bitout, seed, prompt, ref_tokens, tgt_tokens, context_window)

    compressed = out_buffer.getvalue()
    with open(grczip_path, "wb") as f:
        f.write(compressed)
    print(f"Wrote {len(compressed)} bytes → {grczip_path}")

    # ====================================================================
    # 2. Decode ← .grczip  (round-trip verification)
    # ====================================================================
    print(f"\n=== Decoding ← {grczip_path} ===")
    in_buffer = io.BytesIO(compressed)
    bitin     = BitInputStream(in_buffer)

    _, _, reconstructed_tokens = read_grczip(bitin, ref_tokens, context_window)

    reconstructed_text = llmzip.tokenizer.decode(reconstructed_tokens)
    match = reconstructed_text == tgt_text
    print(f"Round-trip match: {match}")
    if not match:
        # Show first divergence point
        for i, (a, b) in enumerate(zip(reconstructed_text, tgt_text)):
            if a != b:
                print(f"  First diff at char {i}: got {repr(a)}, want {repr(b)}")
                print(f"  Context: ...{repr(tgt_text[max(0,i-20):i+20])}...")
                break

    # ====================================================================
    # 3. Baseline: LLMzip of the revised program
    # ====================================================================
    print(f"\n=== Baseline: LLMzip encoding of revised program ===")
    llmzip_bytes = llmzip.encode(tgt_text, output_file=llmzip_path)

    # ====================================================================
    # 4. Size comparison table
    # ====================================================================
    tgt_bytes = len(tgt_text.encode("utf-8"))
    ref_bytes = len(ref_text.encode("utf-8"))

    print("\n========== Size Comparison ==========")
    print(f"  Original revised file:   {tgt_bytes:>7} bytes  ({tgt_bytes*8/len(tgt_text):.2f} bits/char)")
    print(f"  LLMzip (revised):        {len(llmzip_bytes):>7} bytes  ({len(llmzip_bytes)*8/len(tgt_text):.2f} bits/char)")
    print(f"  GRC .grczip:             {len(compressed):>7} bytes  ({len(compressed)*8/len(tgt_text):.2f} bits/char)")

    # GRC breakdown (crude: seed=4B, everything else is prompt+diffs)
    seed_bytes   = SEED_BITS // 8
    print(f"\n  GRC breakdown:")
    print(f"    Seed field:    {seed_bytes} bytes (fixed)")
    print(f"    Prompt+diffs:  {len(compressed) - seed_bytes} bytes")
    print(f"    GRC vs LLMzip: {len(compressed)/len(llmzip_bytes)*100:.1f}% of LLMzip baseline size")
    print("=====================================")


if __name__ == "__main__":
    main()
