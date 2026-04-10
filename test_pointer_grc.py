import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core import LLMzip
from experiments.generative_reconstruction_compression.pointer_grc import encode_pointer_grczip, decode_pointer_grczip

def run_e2e_test():
    print("Loading LLMzip model (this should be fast)...")
    # Use the model that is already cached from previous benchmark runs
    model_id = "Qwen/Qwen3.5-4B-Base"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llmzip = LLMzip(model, tokenizer, device=device)
    
    ref = "This is a simple reference text to test generative reconstruction methods."
    
    # Target 1: Pure deletion (tests has_insertion = 0)
    tgt1 = "This is a simple reference text methods."
    
    # Target 2: Skip + Replace + Insert + Skip
    tgt2 = "This is a simple NEW and shiny text to test amazing generative reconstruction methods!"
    
    targets = [("Pure Deletion", tgt1), ("Complex Replacements", tgt2)]
    
    for name, tgt in targets:
        print(f"\nTesting {name}...")
        compressed = encode_pointer_grczip(ref, tgt, llmzip)
        print(f"Compressed Size: {len(compressed)} bytes")
        
        reconstructed = decode_pointer_grczip(ref, compressed, llmzip)
        assert reconstructed == tgt, f"Mismatch!\nExpected: {tgt}\nGot: {reconstructed}"
        print("Reconstruction: SUCCESS")
        
    print("\nAll Pointer GRC E2E tests passed flawlessly!")

if __name__ == '__main__':
    run_e2e_test()
