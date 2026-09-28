# Mixed Precision Planner

This directory contains the fine-grained MI355X MVP, using [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base) as its reference model. It supports dense Qwen3 and Llama models, `native`/`fp8`/`ptpc_fp8`, Weight-MSE profiling, fixed-budget `PuLP`/CBC Top-K search, PPL selection, exact QConfig compilation, and Quark safetensors export/reload. Qwen3 MoE, Qwen3-Next, and Qwen3.5 hybrid models are outside this MVP.

Install Quark with its CLI dependencies:

```bash
pip install "amd-quark[cli]"
```

Run the complete Python example:

```bash
python examples/torch/experimental/mixed_precision_planner/run_planner.py \
  --model-dir Qwen/Qwen3-0.6B-Base \
  --strategy examples/torch/experimental/mixed_precision_planner/strategy.json \
  --export
```

The same workflow is available through `quark-cli`:

```bash
quark-cli torch-mixed-precision run \
  --model-dir Qwen/Qwen3-0.6B-Base \
  --strategy examples/torch/experimental/mixed_precision_planner/strategy.json
```

Hub model IDs are resolved once to a commit-specific local snapshot shared by profiling, evaluation, and export.
Use `--model-revision` to request a particular Hub revision.
The saved token manifests are directly replayable; set the calibration and PPL `revision` fields to concrete dataset
revisions when they must also be regenerated from Strategy alone.

Replace `run` with one of these stages that can be replayed independently:

- `decision-space`
- `sensitivity-profile`
- `search`
- `select-best-plan`
- `build-qconfig`

The example independently declares a `no_fusion` deployment backend and selects the HF evaluator. Deployment alone defines the Decision Space; changing the evaluator does not rewrite its units or invalidate profiling and search artifacts. A vLLM evaluator currently requires a Decision Space built for the vLLM deployment backend because its fused QKV and gate/up modules do not preserve the independent schemes and scale domains of `no_fusion`. Its profile scores those physical fused tensors, and the MVP currently requires tensor parallel size 1 so its scale domains remain exact. Incompatible bindings fail before evaluation instead of silently shrinking the search space or rewriting an assignment.

The HF evaluator reuses the profiling model for its baseline and loads one fresh model per candidate. Export adds one fresh HF load, measures a separate HF source baseline before quantization, and compares fresh-process HF reload PPL only with that baseline. A Plan baseline is used only within its selected evaluator and is never mixed with reload metrics from another backend.

Top-K search requires candidates to differ on the configured highest-impact Decision Space units. The selected unit IDs
and minimum difference count are recorded in `candidates.json`.

The existing `experimental/torch/mix_precision` package remains the Roofline search implementation. Both paths reuse Quark's vLLM fake-quant worker, but they do not share planner artifacts or runtime state.
