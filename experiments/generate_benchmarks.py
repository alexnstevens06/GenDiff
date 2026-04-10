"""
Generate benchmark source files using Z.ai GLM 5.1 via OpenRouter.

Reads prompts from experiments/benchmark_programs/prompts/*.json
Writes generated Python files to experiments/benchmark_programs/

Prompt file format (JSON):
    {
        "seed": <int>,
        "system": "<system message>",
        "user": "<user message>"
    }

All 6 generations are launched concurrently via asyncio.gather.

Usage:
    python3 experiments/generate_benchmarks.py
    (OPENROUTER_API_KEY must be set in .env or the environment)
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openrouter import OpenRouter

load_dotenv()

MODEL = "z-ai/glm-5.1"

PROMPTS_DIR = Path(__file__).parent / "benchmark_programs" / "prompts"
OUT_DIR     = Path(__file__).parent / "benchmark_programs"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Prompt file loader
# ---------------------------------------------------------------------------

def load_prompt(path: Path) -> dict:
    """Load a JSON prompt file into a dict with keys: seed, system, user."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {
        "seed":   int(data["seed"]),
        "system": data["system"],
        "user":   data["user"],
    }


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code(text: str | None) -> str:
    """Pull the first fenced ```python … ``` block, or fall back to raw text."""
    if not text:
        return ""
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# ---------------------------------------------------------------------------
# Async generation of a single file
# ---------------------------------------------------------------------------

async def generate_one(client: OpenRouter, prompt_path: Path) -> Path:
    """Generate a single benchmark file asynchronously."""
    stem     = prompt_path.stem          # e.g. "01_quicksort_seed42"
    out_path = OUT_DIR / f"{stem}.py"

    if out_path.exists():
        print(f"  [skip] {out_path.name}  (already exists)")
        return out_path

    p = load_prompt(prompt_path)
    print(f"  [gen]  {out_path.name}  (seed={p['seed']}) — launched")

    t0 = time.monotonic()
    resp = await client.chat.send_async(
        model=MODEL,
        seed=p["seed"],
        messages=[
            {"role": "system", "content": p["system"]},
            {"role": "user",   "content": p["user"]},
        ],
    )
    elapsed = time.monotonic() - t0

    choice  = resp.choices[0]  # type: ignore[union-attr]
    content = choice.message.content
    if not content:
        finish = getattr(choice, "finish_reason", "unknown")
        raise RuntimeError(
            f"Empty content for {out_path.name} "
            f"(finish_reason={finish!r}, usage={resp.usage})"
        )
    code = extract_code(content)
    out_path.write_text(code, encoding="utf-8")

    tokens = getattr(resp.usage, "total_tokens", "?")
    print(f"  [done] {out_path.name}  ({elapsed:.1f}s, {tokens} tokens, {len(code)} bytes)")
    return out_path


# ---------------------------------------------------------------------------
# Main — launch all generations concurrently
# ---------------------------------------------------------------------------

async def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not found. Set it in .env or the environment."
        )

    prompt_files = sorted(PROMPTS_DIR.glob("*.json"))
    if not prompt_files:
        raise RuntimeError(f"No prompt files found in {PROMPTS_DIR}")

    print(f"Model  : {MODEL}")
    print(f"Prompts: {len(prompt_files)} files in {PROMPTS_DIR}")
    print(f"Output : {OUT_DIR}")
    print(f"Mode   : async (all {len(prompt_files)} launched concurrently)\n")

    client = OpenRouter(api_key=api_key)

    t0 = time.monotonic()
    results = await asyncio.gather(
        *[generate_one(client, pf) for pf in prompt_files],
        return_exceptions=True,
    )
    elapsed = time.monotonic() - t0

    print(f"\n{'─' * 60}")
    print(f"Finished in {elapsed:.1f}s total\n")

    ok, failed = [], []
    for pf, result in zip(prompt_files, results):
        if isinstance(result, Exception):
            failed.append((pf.name, result))
        else:
            ok.append(result)

    if ok:
        print(f"✓ {len(ok)} file(s) written to {OUT_DIR}/")
        for p in ok:
            print(f"   {p.name:50s}  {p.stat().st_size:>6} bytes")

    if failed:
        import traceback
        print(f"\n✗ {len(failed)} generation(s) failed:")
        for name, exc in failed:
            print(f"\n   ✗ {name}:")
            traceback.print_exception(type(exc), exc, exc.__traceback__, limit=5)


if __name__ == "__main__":
    asyncio.run(main())
