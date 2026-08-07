---
name: quark-torch-ptq
description: Plan, run, validate, or debug AMD Quark post-training quantization for PyTorch and Hugging Face models. Use for LLM FP8, INT8, INT4, MX, AWQ, GPTQ, SmoothQuant, KV-cache quantization, Quark export, quantized-model evaluation, or a failed quark.torch workflow. Covers prerequisites through deployable artifacts without relying on other skills.
---

# Quantize PyTorch Models with AMD Quark

Turn a PyTorch or Hugging Face model and deployment goal into an evidence-backed Quark PTQ run. Do not treat artifact creation as proof of accuracy or runtime compatibility.

## Workflow

### 1. Intake

Collect:

- model ID or local path, exact revision, architecture, parameter count, and access requirements;
- target runtime/hardware and required export format;
- precision/scheme request, or the user's latency, memory, and accuracy goal;
- calibration dataset, sample count, sequence length, and redistribution rights;
- available host RAM, GPU count/memory, disk, and time budget;
- float baseline metric and acceptable regression.

Inspect local files and hardware rather than assuming them. If installation is incomplete, use the install and verification procedure in [references/prerequisites.md](references/prerequisites.md).

### 2. Choose a release-matched recipe

Read [references/recipe-selection.md](references/recipe-selection.md). Prefer a Quark example that explicitly supports the model family and export target. Pin a Quark release or commit and use the example shipped with that same release; do not mix `latest` examples with an older installed package.

Produce a plan with model/revision, scheme, algorithms, calibration input, device placement, output/export format, evaluation, expected disk/memory cost, and exact command. Mark every unverified assumption.

### 3. Confirm before expensive or mutating work

Get explicit confirmation before package changes, gated model downloads, calibration-data downloads, generated scripts, quantization, export, or evaluation. Show output paths and whether an existing path could be overwritten.

### 4. Preflight and execute

1. Verify Quark/PyTorch imports and backend alignment.
2. Verify model and calibration access before allocating accelerators.
3. In a Quark examples checkout or release bundle, inspect the command surface:

   ```bash
   python quantize_quark.py --help
   ```

4. Run the approved command with a persistent log and capture package versions, revision, arguments, and accelerator visibility.
5. Preserve the float model; write quantized output to a distinct directory.

### 5. Validate in layers

Read [references/artifacts-and-evaluation.md](references/artifacts-and-evaluation.md), then:

1. Inspect exported artifacts:

   ```bash
   python scripts/inspect_artifacts.py \
     --source-model-dir /path/to/float-model \
     --quantized-model-dir /path/to/output \
     --output validation_artifacts.json
   ```

2. Load the artifact with the intended consumer, not only Quark.
3. Run one deterministic smoke inference.
4. Compare the agreed accuracy metric against the float baseline on the same data and preprocessing.
5. Measure memory/latency in the target runtime if performance motivated quantization.

Report structural, load, numerical, quality, and runtime results separately. A missing test is `not-run`, never a pass.

## Rules

- Do not invent a quantization scheme from model size alone. Tie the choice to supported model/format/runtime evidence.
- Keep calibration and evaluation data distinct when reporting accuracy.
- Record tokenizer revision and `trust_remote_code` decisions.
- For multi-GPU runs, verify every checkpoint shard is readable before GPU allocation. Use explicit placement and capture it.
- Never overwrite the source model. Treat large outputs and model downloads as cost-bearing actions.
- If a process dies and only a cancellation error remains, inspect earlier logs for the first GPU/OOM/compiler failure.
- Never call an output deployable until its target loader and one smoke inference pass.

## Deliverable

Return a concise run record containing the pinned inputs, exact command, output location, logs, validation matrix, metric comparison, limitations, and smallest recovery step for any failure.
