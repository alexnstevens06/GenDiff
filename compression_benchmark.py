"""
gzip vs LLMzip benchmark
========================
Compresses the same text corpus with:
  1. gzip (levels 1, 6, 9)
  2. LLMzip (Qwen3.5-4B-Base, bfloat16, KV-cache)

Reports bytes, bits-per-character, and wall-clock time.
Saves a bar + timing chart to compression_benchmark.png.

Run from project root:
    python3 compression_benchmark.py

Corpus: enwik8 first 100KB slice (inline download if not cached)
        + the nbody candidate files in the repo for a code sample
"""

import gzip
import hashlib
import io
import os
import sys
import time
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import LLMzip

MODEL_NAME   = "Qwen/Qwen3.5-4B-Base"
PLOT_PATH    = "compression_benchmark.png"
ENWIK8_CACHE = "/tmp/enwik8_100k.txt"
ENWIK8_URL   = "https://mattmahoney.net/dc/enwik8"
SLICE_BYTES  = 100_000   # first 100 KB of enwik8


# ─────────────────────────────────────────────────────────────────────────────
# Corpus helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_corpus_files() -> dict:
    """Load all text files from the corpus directory."""
    corpora = {}
    corpus_dir = os.path.join(os.path.dirname(__file__), "corpus")
    if not os.path.exists(corpus_dir):
        return corpora
        
    for fname in sorted(os.listdir(corpus_dir)):
        if fname.endswith(".txt"):
            path = os.path.join(corpus_dir, fname)
            with open(path, "r") as f:
                corpora[fname] = f.read()
    return corpora


# ─────────────────────────────────────────────────────────────────────────────
# gzip
# ─────────────────────────────────────────────────────────────────────────────

def bench_gzip(text: str, level: int) -> tuple[int, float]:
    raw  = text.encode("utf-8")
    t0   = time.perf_counter()
    comp = gzip.compress(raw, compresslevel=level)
    return len(comp), time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
# LLMzip
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    print(f"  Loading {MODEL_NAME} (bfloat16)...")
    t0 = time.time()
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"  Model ready in {time.time()-t0:.1f}s  "
          f"({torch.cuda.memory_allocated(0)/1024**3:.2f} GiB VRAM)")
    return LLMzip(model, tok, device=model.device)


def bench_llmzip(compressor: LLMzip, text: str) -> tuple[int, float]:
    t0 = time.perf_counter()
    compressed = compressor.encode(text)
    return len(compressed), time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_corpus(name: str, text: str, compressor: LLMzip, results: list):
    n_chars = len(text)
    n_bytes_raw = len(text.encode("utf-8"))
    print(f"\n{'='*60}")
    print(f"  Corpus: {name}  ({n_chars} chars / {n_bytes_raw} raw bytes)")
    print(f"{'='*60}")

    def record(label, n_bytes, elapsed):
        bpc = n_bytes * 8 / n_chars
        ratio = n_bytes / n_bytes_raw
        print(f"  {label:<22}  {n_bytes:>8} B   {bpc:>6.3f} bpc   "
              f"{ratio*100:>6.1f}%   {elapsed:>6.2f}s")
        results.append(dict(
            corpus=name, method=label,
            bytes=n_bytes, bpc=bpc, ratio=ratio, time=elapsed,
        ))

    # gzip levels
    for lvl in (1, 6, 9):
        nb, t = bench_gzip(text, lvl)
        record(f"gzip -L{lvl}", nb, t)

    # LLMzip
    print(f"  LLMzip encoding {n_chars} chars...")
    nb, t = bench_llmzip(compressor, text)
    record("LLMzip (Qwen3.5-4B)", nb, t)
    compressor.cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: list):
    from collections import defaultdict
    by_corpus = defaultdict(list)
    for r in results:
        by_corpus[r["corpus"]].append(r)

    n_corpus = len(by_corpus)
    fig, axes = plt.subplots(1, n_corpus, figsize=(7 * n_corpus, 6))
    if n_corpus == 1:
        axes = [axes]

    BG   = "#0f0f1a"
    CARD = "#1a1a2e"
    fig.patch.set_facecolor(BG)

    palette = {
        "gzip -L1":           "#60a5fa",
        "gzip -L6":           "#3b82f6",
        "gzip -L9":           "#1d4ed8",
        "LLMzip (Qwen3.5-4B)": "#f97316",
    }

    for ax, (corpus, rows) in zip(axes, by_corpus.items()):
        ax.set_facecolor(CARD)
        labels = [r["method"] for r in rows]
        bpcs   = [r["bpc"]    for r in rows]
        colors = [palette.get(l, "#94a3b8") for l in labels]

        bars = ax.bar(labels, bpcs, color=colors, edgecolor="#334155",
                      linewidth=0.8, zorder=2)
        ax.set_title(corpus, color="#f8fafc", fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("Bits per character (lower = better)", color="#cbd5e1", fontsize=10)
        ax.tick_params(colors="#94a3b8", axis="both")
        ax.set_xticklabels(labels, rotation=20, ha="right", color="#cbd5e1", fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.yaxis.grid(True, color="#334155", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

        # Annotate bar tops
        for bar, bpc, row in zip(bars, bpcs, rows):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bpc:.3f}\n({row['time']:.1f}s)",
                    ha="center", va="bottom",
                    color="#e2e8f0", fontsize=8)

    fig.suptitle(
        f"Compression: gzip vs LLMzip ({MODEL_NAME.split('/')[-1]})",
        color="#f8fafc", fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, facecolor=BG, bbox_inches="tight")
    print(f"\n  Plot saved → {PLOT_PATH}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  COMPRESSION BENCHMARK: gzip vs LLMzip")
    print("="*60)

    # Corpora
    corpora = get_corpus_files()
    if not corpora:
        print("  WARNING: No .txt files found in corpus directory.")
        sys.exit(1)
        
    print(f"\n[Found {len(corpora)} corpus files]")
    for name, text in corpora.items():
        print(f"  {name}: {len(text)} chars")

    # Load model once
    print("\n[Model]")
    compressor = load_model()

    # Run
    results = []
    for name, text in corpora.items():
        run_corpus(name, text, compressor, results)

    # Summary table
    print("\n\n" + "="*60)
    print(f"  {'Corpus':<22} {'Method':<24} {'BPC':>7}  {'Ratio':>7}  {'Time':>6}")
    print("  " + "-"*58)
    for r in results:
        print(f"  {r['corpus']:<22} {r['method']:<24} "
              f"{r['bpc']:>7.3f}  {r['ratio']*100:>6.1f}%  {r['time']:>5.1f}s")
    print("="*60)

    plot_results(results)
    print("\n  Done.")
