"""Decode an LLMzip file back to text.

Run from the repository root:

    python core/decode_story.py --model Qwen/Qwen3.5-4B-Base \\
        --input ancient_rome.llmzip --output ancient_rome.decoded.txt
"""

import argparse
import os
import sys
import time

# The script lives in core/, so a direct `python core/decode_story.py`
# would not find the package unless the repository root is on the path.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.llmzip_utils import LLMzip


def main():
    parser = argparse.ArgumentParser(description="Decode text using LLMzip.")
    parser.add_argument("--model", type=str, required=True,
                        help="Hugging Face model id or local path")
    parser.add_argument("--input", type=str, required=True,
                        help="Input .llmzip file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output text file")
    parser.add_argument("--window-size", type=int, default=50,
                        help="Context window size (default: 50)")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 instead of float16")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.bf16:
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = "auto"

    print(f"Loading model from {args.model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    llmzip = LLMzip(model, tokenizer, device=device, max_window=args.window_size)

    if not os.path.exists(args.input):
        print(f"Input file {args.input} not found.")
        return

    print(f"Decompressing {args.input} to {args.output}...")
    start_time = time.time()
    decoded_text = llmzip.decode(args.input)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(decoded_text)
    end_time = time.time()
    print(f"Decompression finished in {end_time - start_time:.2f} seconds.")
    print(f"Decoded text saved to {args.output}")


if __name__ == "__main__":
    main()
