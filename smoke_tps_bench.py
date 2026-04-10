"""
Smoke test + TPS benchmark for Qwen3.5-4B-Base on L4 GPU.

- Verifies model loads onto CUDA
- Generates 250 tokens autoregressively with a live KV cache
- Records per-token latency and plots TPS over time
"""

import time
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME  = "Qwen/Qwen3.5-4B-Base"
N_TOKENS    = 250
PROMPT      = (
    "The following is a detailed technical explanation of how arithmetic coding "
    "works as a lossless data compression algorithm. Arithmetic coding represents "
    "the entire message as a single number in the interval [0, 1). "
)
PLOT_PATH   = "tps_benchmark.png"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test():
    print("\n" + "="*60)
    print("  SMOKE TEST")
    print("="*60)

    assert torch.cuda.is_available(), "CUDA not available — aborting."
    gpu = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU  : {gpu}")
    print(f"  VRAM : {vram_total:.1f} GiB")
    print(f"  CUDA : {torch.version.cuda}")
    print(f"  torch: {torch.__version__}")

    print(f"\n  Loading tokenizer from {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  Tokenizer loaded in {time.time()-t0:.1f}s")

    print(f"  Loading model (bfloat16, device_map=auto)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    load_time = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    vram_used = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  Model loaded in {load_time:.1f}s | {n_params:.2f}B params | {vram_used:.2f} GiB VRAM")

    # Quick forward pass sanity check
    ids = tokenizer("Hello", return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model(ids)
    assert out.logits.shape[-1] > 0, "Logits shape unexpected"
    print(f"  Forward pass OK — logits shape {tuple(out.logits.shape)}")
    print("  ✓ Smoke test passed\n")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 2. TPS benchmark — 250 tokens, KV cache, record per-token latency
# ─────────────────────────────────────────────────────────────────────────────

def tps_benchmark(model, tokenizer, prompt: str, n_tokens: int):
    print("="*60)
    print("  TPS BENCHMARK  (KV-cache autoregressive decode)")
    print("="*60)
    print(f"  Prompt : {prompt[:80]}...")
    print(f"  Tokens : {n_tokens}\n")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    print(f"  Prompt tokens: {input_ids.shape[1]}")

    # Warm-up forward pass to prime CUDA kernels (not timed)
    with torch.no_grad():
        _ = model(input_ids, use_cache=True)
    torch.cuda.synchronize()

    # ── Autoregressive generation with KV cache ──────────────────────────────
    token_times   = []   # seconds per token
    generated_ids = []

    past_kv       = None
    cur_input     = input_ids           # first step: full prompt

    print("  Generating", end="", flush=True)

    for i in range(n_tokens):
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            out = model(cur_input, past_key_values=past_kv, use_cache=True)

        torch.cuda.synchronize()
        t_end = time.perf_counter()

        logits   = out.logits[:, -1, :]          # (1, vocab)
        next_tok = logits.argmax(dim=-1)          # greedy
        past_kv  = out.past_key_values
        cur_input = next_tok.unsqueeze(0)          # (1,1) for subsequent steps

        generated_ids.append(next_tok.item())
        token_times.append(t_end - t_start)

        if (i + 1) % 25 == 0:
            tps = 1.0 / np.mean(token_times[-25:])
            print(f" {i+1}tok({tps:.0f}t/s)", end="", flush=True)

    print()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\n  Generated text snippet:\n  {generated_text[:200]}...\n")

    # ── Stats ────────────────────────────────────────────────────────────────
    tps_per_token = [1.0 / t for t in token_times]
    # Skip token 0 (includes KV-cache miss warmup from prompt)
    tps_steady    = tps_per_token[1:]

    print(f"  Token 0 (cache-miss) : {tps_per_token[0]:.1f} t/s")
    print(f"  Tokens 1-{n_tokens}  ")
    print(f"    mean TPS : {np.mean(tps_steady):.1f}")
    print(f"    p50  TPS : {np.median(tps_steady):.1f}")
    print(f"    min  TPS : {np.min(tps_steady):.1f}")
    print(f"    max  TPS : {np.max(tps_steady):.1f}")

    vram_peak = torch.cuda.max_memory_allocated(0) / 1024**3
    print(f"  Peak VRAM: {vram_peak:.2f} GiB")

    return tps_per_token


# ─────────────────────────────────────────────────────────────────────────────
# 3. Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_tps(tps_per_token, out_path: str):
    tokens = np.arange(1, len(tps_per_token) + 1)
    tps    = np.array(tps_per_token)

    # 5-token rolling mean for a smooth trend line
    window = 5
    pad    = window // 2
    smooth = np.convolve(tps, np.ones(window) / window, mode="same")

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    # Raw per-token scatter
    ax.scatter(tokens, tps, color="#5b8dee", s=14, alpha=0.55, zorder=2, label="Per-token TPS")

    # Smooth trend
    ax.plot(tokens, smooth, color="#f4a261", linewidth=2.0, zorder=3, label=f"{window}-tok rolling mean")

    # Horizontal mean line (skip token 0)
    mean_steady = float(np.mean(tps[1:]))
    ax.axhline(mean_steady, color="#2dd4bf", linewidth=1.2, linestyle="--",
               label=f"Mean (tok 1–{len(tps)}): {mean_steady:.1f} t/s", zorder=4)

    # Shade token-0 (cache-miss region)
    ax.axvspan(0.5, 1.5, color="#ef4444", alpha=0.15, label="Cache-miss (tok 0)")

    ax.set_xlabel("Token index", color="#cbd5e1", fontsize=12)
    ax.set_ylabel("Tokens / second", color="#cbd5e1", fontsize=12)
    ax.set_title(
        f"Qwen3.5-4B-Base · L4 GPU · KV-cache autoregressive decode · {len(tps)} tokens",
        color="#f8fafc", fontsize=13, fontweight="bold", pad=14
    )

    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    ax.set_ylim(bottom=0)
    ax.set_xlim(0.5, len(tps) + 0.5)

    legend = ax.legend(framealpha=0.25, labelcolor="#e2e8f0", facecolor="#1e293b",
                       edgecolor="#475569", fontsize=10)

    # Annotate peak
    peak_idx = int(np.argmax(tps))
    ax.annotate(
        f"{tps[peak_idx]:.0f} t/s",
        xy=(tokens[peak_idx], tps[peak_idx]),
        xytext=(tokens[peak_idx] + 5, tps[peak_idx] + 5),
        arrowprops=dict(arrowstyle="->", color="#f4a261", lw=1.2),
        color="#f4a261", fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"\n  Plot saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, tokenizer = smoke_test()
    tps_series       = tps_benchmark(model, tokenizer, PROMPT, N_TOKENS)
    plot_tps(tps_series, PLOT_PATH)
    print("\n  Done.")
