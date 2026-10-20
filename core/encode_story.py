"""Encode a text file with LLMzip.

Run from the repository root:

    python core/encode_story.py --model Qwen/Qwen3.5-4B-Base \\
        --input corpus/ancient_rome.txt --output ancient_rome.llmzip
"""

import argparse
import os
import sys
import time

# The script lives in core/, so a direct `python core/encode_story.py`
# would not find the package unless the repository root is on the path.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.llmzip_utils import LLMzip


def main():
    parser = argparse.ArgumentParser(description="LLMzip encoding wrapper")
    parser.add_argument("--model", type=str, required=True,
                        help="Hugging Face model id or local path")
    parser.add_argument("--input", type=str, required=True,
                        help="Input text file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .llmzip file")
    parser.add_argument("--window-size", type=int,
                        default=50, help="Context window")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.bf16:
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    llmzip = LLMzip(
        model,
        tokenizer,
        device=device,
        max_window=args.window_size,
    )

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    start = time.time()
    llmzip.encode(text, args.output)
    end = time.time()

    print(f"Encoded in {end - start:.2f}s")


if __name__ == "__main__":
    main()
