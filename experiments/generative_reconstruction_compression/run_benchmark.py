"""
GRC Multi-Content Benchmark Runner
===================================
Loads Qwen 2.5-3B once, then for each content type:
  1. GRC-encodes (prompt + token-level diffs) → .grczip
  2. Verifies GRC round-trip
  3. LLMzip-encodes the revised content → .llmzip
  4. Records byte sizes and bits-per-character

Saves results to benchmark_results.json.
"""

import io
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core import LLMzip, BitInputStream, BitOutputStream
import experiments.generative_reconstruction_compression.simple_grc as grc

EXP_DIR = os.path.dirname(os.path.abspath(__file__))

CONTENT_ITEMS = [
    {
        "name":     "N-body Code",
        "label":    "nbody",
        "candidate": os.path.join(EXP_DIR, "nbody_candidate.py"),
        "revised":   os.path.join(EXP_DIR, "nbody_candidate_revised.py"),
        "prompt":    os.path.join(EXP_DIR, "nbody_prompt.txt"),
        "seed":      os.path.join(EXP_DIR, "nbody_seed.txt"),
        "grczip":    os.path.join(EXP_DIR, "nbody.grczip"),
        "llmzip":    os.path.join(EXP_DIR, "nbody_revised.llmzip"),
    },
    {
        "name":     "Short Story",
        "label":    "story",
        "candidate": os.path.join(EXP_DIR, "story_candidate.txt"),
        "revised":   os.path.join(EXP_DIR, "story_candidate_revised.txt"),
        "prompt":    os.path.join(EXP_DIR, "story_prompt.txt"),
        "seed":      os.path.join(EXP_DIR, "story_seed.txt"),
        "grczip":    os.path.join(EXP_DIR, "story.grczip"),
        "llmzip":    os.path.join(EXP_DIR, "story_revised.llmzip"),
    },
    {
        "name":     "Baking Recipe",
        "label":    "recipe",
        "candidate": os.path.join(EXP_DIR, "recipe_candidate.txt"),
        "revised":   os.path.join(EXP_DIR, "recipe_candidate_revised.txt"),
        "prompt":    os.path.join(EXP_DIR, "recipe_prompt.txt"),
        "seed":      os.path.join(EXP_DIR, "recipe_seed.txt"),
        "grczip":    os.path.join(EXP_DIR, "recipe.grczip"),
        "llmzip":    os.path.join(EXP_DIR, "recipe_revised.llmzip"),
    },
]

CONTEXT_WINDOW = 100


def load_model():
    model_name = "Qwen/Qwen3.5-4B-Base"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    llmzip = LLMzip(model, tokenizer, device=model.device)
    print("Model loaded.\n")
    return model, tokenizer, llmzip


def run_item(item: dict, llmzip_obj) -> dict:
    name = item["name"]
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    ref_text = open(item["candidate"], encoding="utf-8").read()
    tgt_text = open(item["revised"],   encoding="utf-8").read()
    prompt   = open(item["prompt"],    encoding="utf-8").read()
    seed     = int(open(item["seed"]).read().strip())

    ref_tokens = llmzip_obj.tokenizer.encode(ref_text)
    tgt_tokens = llmzip_obj.tokenizer.encode(tgt_text)

    print(f"  ref: {len(ref_tokens)} tokens ({len(ref_text)} chars)")
    print(f"  tgt: {len(tgt_tokens)} tokens ({len(tgt_text)} chars)")

    # --- GRC encode ---
    print("\n  [GRC] Encoding...")
    out_buf = io.BytesIO()
    bitout  = BitOutputStream(out_buf)
    grc.write_grczip(bitout, seed, prompt, ref_tokens, tgt_tokens, CONTEXT_WINDOW)
    grc_bytes = out_buf.getvalue()

    with open(item["grczip"], "wb") as f:
        f.write(grc_bytes)
    print(f"  [GRC] Wrote {len(grc_bytes)} bytes → {item['grczip']}")

    # --- GRC decode (round-trip verification) ---
    print("  [GRC] Decoding (round-trip check)...")
    in_buf = io.BytesIO(grc_bytes)
    bitin  = BitInputStream(in_buf)
    _, _, rt_tokens = grc.read_grczip(bitin, ref_tokens, CONTEXT_WINDOW)
    rt_text = llmzip_obj.tokenizer.decode(rt_tokens)
    match = rt_text == tgt_text
    print(f"  [GRC] Round-trip match: {match}")
    if not match:
        for i, (a, b) in enumerate(zip(rt_text, tgt_text)):
            if a != b:
                print(f"  [GRC] First diff at char {i}: {repr(a)} vs {repr(b)}")
                break

    # --- LLMzip encode ---
    print("  [LLMzip] Encoding...")
    llmzip_bytes_data = llmzip_obj.encode(tgt_text, output_file=item["llmzip"])
    llmzip_size = len(llmzip_bytes_data)
    print(f"  [LLMzip] Wrote {llmzip_size} bytes → {item['llmzip']}")

    # --- BPC ---
    n_chars = len(tgt_text)
    grc_bpc    = len(grc_bytes)    * 8 / n_chars
    llmzip_bpc = llmzip_size       * 8 / n_chars
    raw_bpc    = len(tgt_text.encode("utf-8")) * 8 / n_chars

    print(f"\n  Raw:    {raw_bpc:.3f} bpc")
    print(f"  LLMzip: {llmzip_bpc:.3f} bpc")
    print(f"  GRC:    {grc_bpc:.3f} bpc  ({grc_bpc/llmzip_bpc*100:.1f}% of LLMzip)")

    return {
        "name":        name,
        "label":       item["label"],
        "n_chars":     n_chars,
        "n_ref_tokens": len(ref_tokens),
        "n_tgt_tokens": len(tgt_tokens),
        "raw_bytes":   len(tgt_text.encode("utf-8")),
        "grc_bytes":   len(grc_bytes),
        "llmzip_bytes": llmzip_size,
        "raw_bpc":     raw_bpc,
        "grc_bpc":     grc_bpc,
        "llmzip_bpc":  llmzip_bpc,
        "rt_match":    match,
    }


def main():
    # Load any pre-existing results so we can skip items already benchmarked
    results_path = os.path.join(EXP_DIR, "benchmark_results.json")
    existing: dict = {}
    if os.path.exists(results_path):
        import json as _json
        for r in _json.load(open(results_path)):
            existing[r["label"]] = r
        print(f"Found {len(existing)} pre-existing result(s): {list(existing.keys())}")

    # Only items not already in the results file need the model
    pending = [item for item in CONTENT_ITEMS if item["label"] not in existing]

    if pending:
        model, tokenizer, llmzip_obj = load_model()
        # Inject into simple_grc globals so write_grczip / read_grczip work
        grc.model     = model
        grc.tokenizer = tokenizer
        grc.llmzip    = llmzip_obj
    else:
        print("All items already benchmarked — skipping model load.")
        llmzip_obj = None

    results = list(existing.values())
    for item in pending:
        result = run_item(item, llmzip_obj)
        results.append(result)

    out_path = os.path.join(EXP_DIR, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved → {out_path}")

    print("\n========== SUMMARY ==========")
    print(f"{'Content':<18} {'Raw BPC':>8} {'LLMzip BPC':>11} {'GRC BPC':>8} {'GRC/LZ%':>8}")
    print("-" * 58)
    for r in results:
        print(
            f"{r['name']:<18} {r['raw_bpc']:>8.3f} {r['llmzip_bpc']:>11.3f}"
            f" {r['grc_bpc']:>8.3f} {r['grc_bpc']/r['llmzip_bpc']*100:>7.1f}%"
        )
    print("=============================")


if __name__ == "__main__":
    main()
