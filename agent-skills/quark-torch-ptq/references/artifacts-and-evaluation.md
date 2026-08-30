# Artifact and evaluation checks

## Structural checks

- Output path differs from the float source.
- `config.json`, tokenizer assets, and weight shards/indexes expected by the export format exist.
- Every shard referenced by an index is present and readable.
- Quantization metadata is present and consistent with the requested scheme.
- Non-weight auxiliary files required by the loader were preserved.

Run `scripts/inspect_artifacts.py` for a dependency-free first pass. Its `ok` status means the directory is structurally plausible, not numerically correct.

## Consumer load and smoke inference

Use the exact runtime that will consume the artifact: Transformers, vLLM, SGLang, llama.cpp/GGUF, or another target. Record its version and loading arguments. Run a fixed short prompt/input and retain the output and error log.

## Quality evaluation

Compare float and quantized models using:

- identical model/tokenizer revision;
- identical dataset split, preprocessing, sequence length, and random seed;
- the same metric implementation;
- enough samples to make the stated conclusion credible.

Report baseline, quantized value, absolute/relative delta, sample count, and whether the result meets the user-approved threshold. Perplexity, lm-eval tasks, ROUGE/METEOR, or domain metrics may be appropriate; choose based on the deployment goal.

## Runtime evaluation

If the goal is performance, measure on target hardware after warmup:

- peak GPU and host memory;
- prefill and decode throughput/latency as applicable;
- model load time and output size;
- runtime flags such as tensor parallelism, batch size, and sequence length.

Do not compare measurements collected with different runtime settings without calling out the difference.
