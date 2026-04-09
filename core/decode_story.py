
import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmzip_qwen import LLMzipQwen

# Override for Navi 22 (RX 6700 XT) support on ROCm
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import argparse

def main():
    parser = argparse.ArgumentParser(description="Decode text using LLMzip.")
    parser.add_argument("--model", type=str, default="/home/alexn/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b", help="Path to the model")
    parser.add_argument("--input", type=str, default="short_story.llmzip", help="Input compressed file")
    parser.add_argument("--output", type=str, default="short_story_decoded.txt", help="Output text file")
    parser.add_argument("--window-size", type=int, default=50, help="Context window size (default: 50)")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 instead of float16")
    args = parser.parse_args()

    model_path = args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if device == "cuda" else "auto"),
            device_map=device,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize LLMzip
    llmzip = LLMzipQwen(model, tokenizer, device=device, max_window=args.window_size)
    
    # Ensure output directory exists
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Update default paths if they don't have a path
    if os.path.dirname(args.input) == "":
        args.input = os.path.join(output_dir, args.input)
    if os.path.dirname(args.output) == "":
        args.output = os.path.join(output_dir, args.output)

    input_file = args.input
    output_file = args.output
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return
    
    print(f"Decompressing {input_file} to {output_file}...")
    start_time = time.time()
    
    # Decode
    decoded_text = llmzip.decode(input_file)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(decoded_text)
    
    end_time = time.time()
    print(f"Decompression finished in {end_time - start_time:.2f} seconds.")
    print(f"Decoded text saved to {output_file}")

if __name__ == "__main__":
    main()
