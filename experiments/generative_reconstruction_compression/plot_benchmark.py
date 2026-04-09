"""
GRC Benchmark Plot
==================
Reads benchmark_results.json and generates a grouped bar chart
comparing LLMzip BPC vs GRC BPC for each content type.

Saves benchmark_bpc.png in the same directory.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

EXP_DIR  = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(EXP_DIR, "benchmark_results.json")
OUT_PATH  = os.path.join(EXP_DIR, "benchmark_bpc.png")

with open(JSON_PATH) as f:
    results = json.load(f)

labels      = [r["name"]       for r in results]
llmzip_bpcs = [r["llmzip_bpc"] for r in results]
grc_bpcs    = [r["grc_bpc"]    for r in results]

x     = np.arange(len(labels))
width = 0.34

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")

bars_lz = ax.bar(x - width / 2, llmzip_bpcs, width,
                 label="LLMzip (full program)",
                 color="#4e8cff", edgecolor="#2a4da0", linewidth=0.8, zorder=3)
bars_gr = ax.bar(x + width / 2, grc_bpcs, width,
                 label="GRC (seed + prompt + diffs)",
                 color="#ff6b6b", edgecolor="#a02a2a", linewidth=0.8, zorder=3)

# Value labels on bars
for bar in bars_lz:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.04,
            f"{h:.2f}", ha="center", va="bottom",
            color="#4e8cff", fontsize=9.5, fontweight="bold")

for bar in bars_gr:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.04,
            f"{h:.3f}", ha="center", va="bottom",
            color="#ff6b6b", fontsize=9.5, fontweight="bold")

# Raw (uncompressed) reference line — 8 bpc for UTF-8 ASCII
ax.axhline(8.0, color="#888", linestyle="--", linewidth=0.9, zorder=2, label="Raw UTF-8 (8 bpc)")

ax.set_xticks(x)
ax.set_xticklabels(labels, color="white", fontsize=12)
ax.set_ylabel("Bits per Character (BPC)\n(lower is better)", color="white", fontsize=11)
ax.set_title("Generative Reconstruction Compression vs LLMzip\nBits-per-Character by Content Type",
             color="white", fontsize=13, pad=14)

ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
ax.tick_params(axis="y", colors="white")
ax.tick_params(axis="x", colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")

ax.grid(axis="y", color="#333", linestyle="--", linewidth=0.6, zorder=1)
ax.set_ylim(0, max(llmzip_bpcs) * 1.25)

legend = ax.legend(fontsize=10, framealpha=0.3,
                   facecolor="#1a1d27", edgecolor="#555",
                   labelcolor="white", loc="upper right")

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {OUT_PATH}")
