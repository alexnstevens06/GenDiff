import sys
import os
import io
import math
import torch
import numpy as np
import difflib

# Add root GenDiff to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from core import LLMzip
from core.AC import arithmeticcoding
from core.continuous_bitstream import ContinuousBitStream

# ---------------------------------------------------------------------------
# Core Parameters
# ---------------------------------------------------------------------------
DEFAULT_CONTEXT_WINDOW = 50

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def bit_cost(n: int) -> int:
    """Returns ceiling(log_2(n)). Minimum 1 bit if n <= 1."""
    if n <= 1: return 1
    return math.ceil(math.log2(n))

# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
def encode_pointer_grczip(
    seed_text: str,
    target_text: str,
    llmzip: LLMzip,
    context_window=DEFAULT_CONTEXT_WINDOW
) -> bytes:
    """
    Encodes the target text as a diff against the seed texts, using the Model
    and the Pointer-Relative Continuous BitStream paradigm.
    """
    ref_tokens = llmzip.tokenizer.encode(seed_text)
    tgt_tokens = llmzip.tokenizer.encode(target_text)

    # 1. Standard difflib resolution
    # (In high performance environments, sequence matchers use optimized Myers diff)
    matcher = difflib.SequenceMatcher(None, ref_tokens, tgt_tokens)
    ops = matcher.get_opcodes()

    stream = ContinuousBitStream()
    
    current_ref_idx = 0

    for tag, i1, i2, j1, j2 in ops:
        if tag == 'equal':
            continue

        # We have a patch! 
        stream.write(1) # HAS_PATCH = 1

        # Calculate dynamics
        rem_ref = len(ref_tokens) - current_ref_idx
        
        del_count = 0
        ins_count = 0
        if tag in ('delete', 'replace'):
            del_count = i2 - i1
        if tag in ('insert', 'replace'):
            ins_count = j2 - j1

        # 1. Delta Pointer Jump
        jump = i1 - current_ref_idx
        if rem_ref <= 0:
            jump_bits = 1
        else:
            jump_bits = bit_cost(rem_ref)
        stream.write_uint(jump, jump_bits)
        
        current_ref_idx += jump
        
        # 2. Delete Count
        new_rem_ref = len(ref_tokens) - current_ref_idx
        if new_rem_ref <= 0:
            del_bits = 1
        else:
            del_bits = bit_cost(new_rem_ref)
        stream.write_uint(del_count, del_bits)
        
        current_ref_idx += del_count
        
        # 3. Payload
        has_insertion = 1 if ins_count > 0 else 0
        stream.write(has_insertion)
        if has_insertion:
            insert_tokens = tgt_tokens[j1:j2]
            # Context warm-up (use preceding target tokens to match out_tokens during decode)
            context = tgt_tokens[max(0, j1 - context_window):j1]
            _encode_payload_segment(stream, insert_tokens, context, llmzip, context_window)

    # End of patches
    stream.write(0) # HAS_PATCH = 0
    return stream.get_bytes()

def _encode_payload_segment(
    stream: ContinuousBitStream,
    target_tokens: list,
    context_tokens: list,
    llmzip: LLMzip,
    context_window: int
):
    """Encodes an insertion using ArithmeticCoding on the continuous stream."""
    enc = arithmeticcoding.ArithmeticEncoder(32, stream)
    
    past_kv = None
    cache_len = 0
    
    eos_token = llmzip.eos_id
    vocab_size = llmzip.vocab_size

    tokens_to_encode = list(target_tokens) + [eos_token]

    ctx = list(context_tokens[-context_window:]) if context_tokens else []
    if ctx:
        ctx_tensor = torch.tensor([ctx], device=llmzip.device)
        probs_first, past_kv = llmzip._get_probs(ctx_tensor)
        cache_len = len(ctx)
        cumul = llmzip._probs_to_cumul(probs_first)
        enc.write(cumul, tokens_to_encode[0])
        start_i = 1
    else:
        uniform_cumul = np.arange(vocab_size + 1, dtype=np.uint64)
        enc.write(uniform_cumul, tokens_to_encode[0])
        start_i = 1

    for i in range(start_i, len(tokens_to_encode)):
        if past_kv is not None and (not llmzip._vram_is_safe() or cache_len >= llmzip.max_pos_len):
            past_kv = None
            cache_len = 0
            llmzip._evict_cache()

        if past_kv is None:
            recent_ctx = (ctx + tokens_to_encode[:i])[-llmzip.max_window:]
            ctx_t = torch.tensor([recent_ctx], device=llmzip.device)
            probs, past_kv = llmzip._get_probs(ctx_t)
            cache_len = len(recent_ctx)
        else:
            prev_tok = torch.tensor([[tokens_to_encode[i - 1]]], device=llmzip.device)
            probs, past_kv = llmzip._get_probs(prev_tok, past_kv)
            cache_len += 1

        cumul = llmzip._probs_to_cumul(probs)
        enc.write(cumul, tokens_to_encode[i])

    enc.finish()
    # No trailing zeros or byte-alignment necessary. The 31 zeroes explicit in
    # `enc.finish()` handles isolation flawlessly.


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
def decode_pointer_grczip(
    seed_text: str,
    bitstream_bytes: bytes,
    llmzip: LLMzip,
    context_window=DEFAULT_CONTEXT_WINDOW
) -> str:
    """
    Decodes the target text from a pointer-relative continuous bitstream.
    """
    ref_tokens = llmzip.tokenizer.encode(seed_text)
    stream = ContinuousBitStream(bitstream_bytes)
    
    out_tokens = []
    current_ref_idx = 0

    while True:
        try:
            has_patch = stream.read_no_eof()
        except EOFError:
            break
            
        if not has_patch:
            break
            
        rem_ref = len(ref_tokens) - current_ref_idx
        jump_bits = bit_cost(rem_ref) if rem_ref > 0 else 1
        jump = stream.read_uint(jump_bits)
        
        # 1. Implicit copy of skipped reference tokens
        out_tokens.extend(ref_tokens[current_ref_idx : current_ref_idx + jump])
        current_ref_idx += jump
        
        new_rem_ref = len(ref_tokens) - current_ref_idx
        del_bits = bit_cost(new_rem_ref) if new_rem_ref > 0 else 1
        del_count = stream.read_uint(del_bits)
        
        current_ref_idx += del_count
        
        has_insertion = stream.read_no_eof()
        if has_insertion:
            # Context warm-up
            context = out_tokens[-context_window:] if out_tokens else []
            decoded_tokens = _decode_payload_segment(stream, context, llmzip, context_window)
            out_tokens.extend(decoded_tokens)
            
    # Implicit copy tail
    if current_ref_idx < len(ref_tokens):
        out_tokens.extend(ref_tokens[current_ref_idx:])
        
    return llmzip.tokenizer.decode(out_tokens)

def _decode_payload_segment(
    stream: ContinuousBitStream,
    context_tokens: list,
    llmzip: LLMzip,
    context_window: int
) -> list:
    dec = arithmeticcoding.ArithmeticDecoder(32, stream)
    out_ids = []
    
    past_kv = None
    cache_len = 0
    vocab_size = llmzip.vocab_size
    eos_id = llmzip.eos_id
    
    ctx = list(context_tokens[-context_window:]) if context_tokens else []
    
    if ctx:
        ctx_tensor = torch.tensor([ctx], device=llmzip.device)
        probs_first, past_kv = llmzip._get_probs(ctx_tensor)
        cache_len = len(ctx)
        cumul = llmzip._probs_to_cumul(probs_first)
    else:
        uniform_cumul = np.arange(vocab_size + 1, dtype=np.uint64)
        cumul = uniform_cumul
        
    while True:
        token_id = dec.read(cumul, vocab_size)
        if token_id == eos_id:
            break
            
        out_ids.append(token_id)
        
        if past_kv is not None and (not llmzip._vram_is_safe() or cache_len >= llmzip.max_pos_len):
            past_kv = None
            cache_len = 0
            llmzip._evict_cache()

        if past_kv is None:
            recent_ctx = (ctx + out_ids)[-llmzip.max_window:]
            ctx_t = torch.tensor([recent_ctx], device=llmzip.device)
            probs, past_kv = llmzip._get_probs(ctx_t)
            cache_len = len(recent_ctx)
        else:
            prev_tok = torch.tensor([[token_id]], device=llmzip.device)
            probs, past_kv = llmzip._get_probs(prev_tok, past_kv)
            cache_len += 1

        cumul = llmzip._probs_to_cumul(probs)
        
    return out_ids

