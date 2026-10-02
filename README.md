# GenDiff

Research code for generative reconstruction compression (GRC). `experiments/generative_reconstruction_compression/simple_grc.py` compresses a token-level diff between a reference text and a revised text. Inserted tokens are arithmetic-coded the same way as LLMzip. The file layout of a `.grczip` stream is described in that module's docstring.

## Layout

- `core/` holds the arithmetic coder (`core/AC/`) and the LLMzip wrappers `encode_story.py` and `decode_story.py`.
- `experiments/generative_reconstruction_compression/` holds the GRC encoder, `run_benchmark.py`, and `baseline_llmzip.py`.
- `experiments/benchmark_programs/` holds cached and augmented program samples.
- `corpus/` holds short text files.
- `compression_benchmark.py` compares gzip with LLMzip.
- `REFERENCES.md` notes the arithmetic coder and the LLMzip paper.

## Running

`compression_benchmark.py` says to run this from the project root:

```sh
python3 compression_benchmark.py
```

`core/encode_story.py` and `core/decode_story.py` take `--model`, `--input`, `--output`, `--window-size`, and `--bf16`.

`simple_grc.py`'s `main` reads `nbody_seed.txt`, `nbody_prompt.txt`, `nbody_candidate.py`, and `nbody_candidate_revised.py` from its own directory. Those files are not in this repository.
