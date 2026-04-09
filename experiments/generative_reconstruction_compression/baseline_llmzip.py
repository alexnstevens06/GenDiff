import os
import sys
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import LLMzip
from core import BitInputStream, ArithmeticDecoder


def main():
    MODEL_PATH = "Qwen/Qwen2.5-Coder-0.5B"
    TARGET_PATH = "experiments/generative_reconstruction_compression/nbody_candidate_revised.py"
    OUTPUT_PATH = "experiments/generative_reconstruction_compression/revised_baseline.llmzip"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    compressor = LLMzip(model, tokenizer, max_window=100)

    with open(TARGET_PATH, "r") as f:
        target_text = f.read()

    print(f"\n--- Encoding Baseline LLMzip ---")
    start_time = time.time()

    # Encode with baseline
    compressor.encode(target_text, OUTPUT_PATH)

    encoded_size = os.path.getsize(OUTPUT_PATH)
    print(f"\nEncoded AC Size: {encoded_size} bytes")
    print(f"Encoding Time:   {time.time() - start_time:.2f} seconds")

    print(f"\n--- Decoding First 1/5 of Baseline LLMzip ---")

    # Calculate 1/5th tokens
    input_ids = tokenizer.encode(target_text, return_tensors='pt')[0]
    total_tokens = len(input_ids)
    target_token_count = total_tokens // 5
    print(
        f"Total tokens in original: {total_tokens}. Decoding first {target_token_count} tokens...")

    start_time = time.time()

    # Manual decode loop for partial decoding
    file_in = open(OUTPUT_PATH, 'rb')
    bitin = arithmeticcoding.BitInputStream(file_in)
    dec = arithmeticcoding.ArithmeticDecoder(32, bitin)

    vocab_size = compressor.vocab_size
    uniform_cumul = np.arange(vocab_size + 1, dtype=np.uint64)
    first_token = dec.read(uniform_cumul, vocab_size)

    decoded_tokens = [first_token]
    past_kv = None
    cache_len = 0

    while len(decoded_tokens) < target_token_count:
        if past_kv is not None and (not compressor._vram_is_safe() or cache_len >= compressor.max_pos_len):
            past_kv = None
            cache_len = 0
            compressor._evict_cache()

        if past_kv is None:
            start_idx = max(0, len(decoded_tokens) - compressor.max_window)
            context = torch.tensor(
                [decoded_tokens[start_idx:]], device=compressor.device)
            probs, past_kv = compressor._get_probs(context)
            cache_len = len(decoded_tokens) - start_idx
        else:
            last_token = torch.tensor(
                [[decoded_tokens[-1]]], device=compressor.device)
            probs, past_kv = compressor._get_probs(last_token, past_kv)
            cache_len += 1

        cumul = compressor._probs_to_cumul(probs)
        try:
            symbol = dec.read(cumul, vocab_size)
        except EOFError:
            break

        decoded_tokens.append(symbol)

        if len(decoded_tokens) % 10 == 0:
            print(
                f"Decoded {len(decoded_tokens)}/{target_token_count} tokens...", end='\r')
            sys.stdout.flush()

    bitin.close()
    file_in.close()

    decoded_text = tokenizer.decode(decoded_tokens)
    print(f"\n\n--- Decoded Output snippet (1/5) ---")
    print(decoded_text)
    print("-" * 40)
    print(f"Decoding Time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
