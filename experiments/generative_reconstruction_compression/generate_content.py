"""
Generalized content generator for GRC benchmarks.

Usage:
  python generate_content.py \
      --prompt-file <path/to/prompt.txt> \
      --output-base <basename>          \
      [--seed 8675309]                  \
      [--model moonshotai/kimi-k2.5]    \
      [--temperature 0.7]               \
      [--ext txt]

Writes:
  <output-base>_candidate.<ext>
  <output-base>_prompt.txt   (copy of the prompt)
  <output-base>_seed.txt     (the seed integer)

The OPENROUTER_API_KEY environment variable (or .env file) must be set.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from openrouter import OpenRouter

# Load .env from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def generate(
    prompt: str,
    model: str = "moonshotai/kimi-k2.5",
    seed: int = 8675309,
    temperature: float = 0.7,
) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    client = OpenRouter(api_key=api_key)

    messages = [{"role": "user", "content": prompt}]

    response = client.chat.send(
        model=model,
        messages=messages,
        seed=seed,
        temperature=temperature,
    )

    # Navigate response: try .choices directly, fall back to .object.choices
    choices = getattr(response, "choices", None) or response.object.choices
    return choices[0].message.content or ""


def strip_markdown(text: str) -> str:
    """Strip leading/trailing markdown code fences if present."""
    if text.startswith("```python"):
        text = text[9:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate content via Kimi/OpenRouter for GRC benchmarks")
    parser.add_argument("--prompt-file", required=True, help="Path to prompt text file")
    parser.add_argument("--output-base", required=True, help="Base path for output files (no extension)")
    parser.add_argument("--seed", type=int, default=8675309)
    parser.add_argument("--model", default="moonshotai/kimi-k2.5")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--ext", default="txt", help="Output file extension (txt, py, etc.)")
    parser.add_argument("--strip-markdown", action="store_true", help="Strip markdown code fences from output")
    args = parser.parse_args()

    with open(args.prompt_file, "r") as f:
        prompt = f.read()

    print(f"Generating with model={args.model}, seed={args.seed}...")
    text = generate(prompt, model=args.model, seed=args.seed, temperature=args.temperature)

    if args.strip_markdown:
        text = strip_markdown(text)

    candidate_path = f"{args.output_base}_candidate.{args.ext}"
    prompt_path    = f"{args.output_base}_prompt.txt"
    seed_path      = f"{args.output_base}_seed.txt"

    with open(candidate_path, "w") as f:
        f.write(text)
    with open(prompt_path, "w") as f:
        f.write(prompt)
    with open(seed_path, "w") as f:
        f.write(str(args.seed))

    nlines = len(text.splitlines())
    nchars = len(text)
    print(f"Generated {nlines} lines / {nchars} chars → {candidate_path}")
    print(f"Wrote {prompt_path} and {seed_path}")


if __name__ == "__main__":
    main()
