from llmzip_qwen import LLMzipQwen
import argparse
import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Override for Navi 22 (RX 6700 XT) support on ROCm
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

# Add experiments and core to path to find modules
sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../experiments"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="LLMzip encoding wrapper")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model")
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

    # Check if model supports standard loading (some multimodal models need specific classes)
    # For now, standard AutoModel
    dtype = torch.bfloat16 if args.bf16 else (
        torch.float16 if torch.cuda.is_available() else "auto")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model.eval()

    llmzip = LLMzipQwen(model, tokenizer, device="cuda" if torch.cuda.is_available(
    ) else "cpu", max_window=args.window_size)

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    start = time.time()
    llmzip.encode(text, args.output)
    end = time.time()

    print(f"Encoded in {end - start:.2f}s")


if __name__ == "__main__":
    main()
