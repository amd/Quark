# Skills Catalog

A quick reference for every user-facing Quark skill currently shipped under `.claude/skills/`. Use this page to find the right entry point for a task; for the deeper rules behind layering, naming, and artifact handoff, see [architecture.md](architecture.md), [interaction-contract.md](interaction-contract.md), and [artifact-contracts.md](artifact-contracts.md).

## Skill Table

Skills are grouped by **backend scope** (`shared` / `torch` / `onnx`) and **layer** (`L0` foundation, `L1` atomic, `L2` workflow, `L3` recipe).

### Backend-agnostic (shared)

| Skill | Layer | Purpose | Typical trigger |
|-------|-------|---------|-----------------|
| `quark-env-preflight` | L0 | Collect OS, Python, GPU, CUDA/ROCm, container state into `env_context.json` before any install or PTQ planning step. | "check my environment", "what GPU do I have", "is my setup ready for Quark" |
| `quark-install` | L1 | Install or verify the AMD Quark package (`amd-quark`) and its core dependencies. | "install Quark", "pip install amd-quark", `ModuleNotFoundError: quark` |

### Torch backend (`quark-torch-*`)

| Skill | Layer | Purpose | Typical trigger |
|-------|-------|---------|-----------------|
| `quark-torch-install` | L1 | Install or verify the correct PyTorch build for the user's accelerator (CUDA / ROCm / CPU) before Quark. | "install PyTorch", "torch version mismatch", `torch.cuda.is_available()` returns `False` |
| `quark-torch-model-intake` | L1 | Inspect a HuggingFace transformers / safetensors checkpoint and produce `model_analysis.json` (architecture, quant targets, risks). | "analyze my model", "is this model supported", "what architecture is this" |
| `quark-torch-export` | L1 | Export a planned or completed Torch PTQ run to HF safetensors / GGUF / ONNX. | "export model", "save quantized model", "convert to GGUF" |
| `quark-torch-debug` | L1 | Diagnose Torch-side failures: install errors, CUDA OOM, `transformers` / `accelerate` stack traces, unexpected PTQ results. | "PTQ failed", "CUDA out of memory", any `quark.torch` / `torch._dynamo` stack trace |
| `quark-torch-result-validator` | L1 | Validate exported HF safetensors + `config.json` via aux-file alignment, MD5 byte-identity on excluded tensors, config diff, and header pattern checks. | "validate quantization result", "check quantized model output", "verify exported weights" |
| `quark-torch-llm-eval` | L1 | End-to-end LLM accuracy evaluation on AMD ROCm using `vLLM` / `SGLang` / `ATOM` with `lm-eval` / `lighteval` / `evalscope`. | "evaluate this model", "run gsm8k/mmlu", "measure perplexity" |
| `quark-torch-file2file-quantization` | L1 | Low-memory file2file quantization for very large safetensors LLMs that cannot be loaded whole; supports external `LLMTemplate` registration, `WeightConverter`, and progressive (two-step) quantization. | "run file2file quantization", "quantize without loading the model", "file2file for DeepSeek/Qwen/MoE" |
| `quark-quantization-result-validator` | L1 | Validate a completed Quark export (HF safetensors) via four lightweight checks: auxiliary file alignment, excluded-tensor MD5 byte-identity, `config.json` deep comparison, and safetensors header dtype/pattern summary. | "validate quantization result", "check quantized model output", "verify exported weights" |
| `quark-torch-ptq` | L2 | End-to-end Torch LLM PTQ pipeline: intake → plan → script → optional execution. Stops at the quantized output. | "quantize my model", "run PTQ end to end", "quantize Llama/Qwen/Mistral with FP8/INT4" |
| `quark-torch-llm-ptq-eval` | L3 | Full PTQ lifecycle: delegates PTQ to `quark-torch-ptq`, then mandatory validation via `quark-torch-result-validator`, then opt-in accuracy eval via `quark-torch-llm-eval`. | "quantize and validate", "quantize and evaluate", "PTQ end to end with accuracy check" |

### ONNX backend (`quark-onnx-*`)

| Skill | Layer | Purpose | Typical trigger |
|-------|-------|---------|-----------------|
| `quark-onnx-install` | L1 | Install or verify the correct ONNX Runtime build (`onnxruntime` / `onnxruntime-gpu` / `onnxruntime-rocm`) and matching `onnx` package. | "install onnxruntime", "onnxruntime-gpu vs onnxruntime", "providers list missing CUDAExecutionProvider" |
| `quark-onnx-model-intake` | L1 | Inspect a `.onnx` graph (opset, IR version, I/O shapes, op-type histogram, quantizable-op count, deployment-target compatibility). | "analyze my ONNX model", "what opset is this", "is my model NPU-compatible", "is my model larger than 2 GB" |
| `quark-onnx-debug` | L1 | Diagnose ONNX-side failures: install, calibration, custom-op compile (BFP / MX / Extended), ORT provider mismatch, silent CPU fallback, OOM. | "quantize_static failed", "custom op library load failed", "CUDAExecutionProvider not available", any `quark.onnx` / `onnxruntime` stack trace |
| `quark-onnx-result-validator` | L1 | Validate the quantized `model.onnx` (+ optional `model.onnx_data`): aux-file alignment, initializer MD5 byte-identity (inline `raw_data` + external byte ranges), metadata equality after stripping quant-only opsets/Quark domains, and fuzzy QDQ / `com.amd.quark` op-pattern summary. | "validate ONNX quantization result", "did QDQ insertion happen", "are the non-quantized weights byte-identical" |
| `quark-onnx-ptq` | L2 | End-to-end ONNX-to-ONNX PTQ pipeline for vision/CNN models: intake → plan → calibration script → manifest → confirmed execution. | "quantize my .onnx", "quantize yolov8/resnet50 with XINT8/A8W8/BFP16/MXFP*", "weights-only INT4 for my .onnx LLM" |
| `quark-onnx-autosearch-pro` | L3 | AutoSearchPro recipe — drives `quark.onnx.AutoSearchPro` (Optuna) to search activation/weight spec, calibration method, CLE, AdaRound / AdaQuant, FastFinetune params. Ships built-in presets `ADVANCED_SEARCH`, `XINT8_SEARCH`, `A8W8_SEARCH`, `A16W8_SEARCH`. | "auto search", "tune quantization", "find the best quant config", "two-stage search" |

## Notes and Tips When Calling Skills (ONNX backend)

These three rules cover the most common routing mistakes on the `quark-onnx-*` side.

### 1. Confirm the input really is `.onnx` before routing to an ONNX skill

The trigger for every `quark-onnx-*` skill is the input artifact, not the user's wording. Treat a request as ONNX only when one of these is true:

- the path ends in `.onnx` (with or without a sibling `.onnx_data` external-weights file),
- the user explicitly mentions `onnxruntime`, `quantize_static`, `ModelQuantizer`, ORT execution providers, or a Quark ONNX custom op (`BFPQuantizeDequantize`, `MXQuantizeDequantize`, `Extended*`),
- or the stack trace / log line names `quark.onnx`, `onnxruntime`, or `onnx`.

If the user only says "quantize my model" with no path and no ONNX cue, **ask before routing**. Never feed a HuggingFace repo id or a `*.safetensors` checkpoint into `quark-onnx-ptq` / `quark-onnx-autosearch-pro` — those skills assume a real `.onnx` graph and will fail at intake. Conversely, do not feed a `.onnx` file into `quark-torch-ptq`.

### 2. Pick the right ONNX layer: intake (L1) vs. workflow (L2) vs. recipe (L3)

Match the skill layer to the request shape — escalating one layer at a time avoids both over- and under-orchestration:

- **Single fact about a `.onnx` graph** (opset, op-type histogram, NPU compatibility, >2 GB external-data check, "does this already have QDQ") → `quark-onnx-model-intake` (L1).
- **Full ONNX-to-ONNX PTQ pipeline** (intake → plan → calibration script → manifest → confirmed execution) for vision/CNN models or weights-only INT4 LLMs → `quark-onnx-ptq` (L2). It already chains the L1 atomic skills and inserts the mandatory confirm checkpoint before calibration.
- **Quant config search** ("auto search", "tune", "find the best config", AdaRound/AdaQuant/FastFinetune sweeps, or any of the `ADVANCED_SEARCH` / `XINT8_SEARCH` / `A8W8_SEARCH` / `A16W8_SEARCH` presets) → `quark-onnx-autosearch-pro` (L3). It is a recipe, not a workflow — do not fall back to `quark-onnx-ptq` and hand-roll a sweep.

When the request spans more than one of these (e.g. "analyze my .onnx and then quantize it"), start at the L2 workflow — it owns the intake → plan → execute handoff and will reuse the L1 skills internally.

### 3. ONNX-specific failure routing: install vs. debug vs. validate

ONNX failures fall into three buckets that map to different skills — getting this right saves a full diagnostic loop:

- **Missing or mismatched runtime** — `onnxruntime` import fails, `onnxruntime-gpu` vs `onnxruntime` confusion, providers list missing `CUDAExecutionProvider` / `ROCMExecutionProvider`, silent CPU fallback when GPU was expected, or `quark-install` reports ONNX Runtime is missing → `quark-onnx-install`. Do this **before** retrying the PTQ flow; the calibration scripts assume the correct ORT build.
- **Runtime/quantization error** — any stack trace mentioning `quark.onnx` / `onnxruntime` / `onnx`, `quantize_static` failure, calibration crash, custom-op load failure (`BFPQuantizeDequantize`, `MXQuantizeDequantize`, `Extended*`), `model.onnx` >2 GB / external-data not found, AdaRound divergence, GPTQ-ONNX / QuaRot failure, or NPU power-of-2 scale issue → `quark-onnx-debug`. Do not route these to `quark-torch-debug` — the two are not interchangeable.
- **Post-run inspection of `model.onnx`** (with or without `model.onnx_data`) — "did QDQ insertion happen", "are non-quantized initializers byte-identical", "are the Quark domain ops present" → `quark-onnx-result-validator`. The torch validator (`quark-torch-result-validator`) does not understand ONNX initializers or QDQ patterns; do not substitute it.
