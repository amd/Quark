# Quark Agent Skills

Skill system for Quark task routing, PTQ planning, workflow orchestration, and governance. Skills here are auto-discovered by Claude Code via the entry stubs in `.claude/skills/`.

## Goals

- Organize skills by capability tier instead of historical file layout.
- Standardize every user-facing skill on the same interaction contract.
- Ship two fully connected MVP paths: `Torch + LLM PTQ` (HuggingFace / safetensors input) and `ONNX + CNN PTQ` (`.onnx` input).
- Keep knowledge traceable to existing Quark docs, CLI behavior, and source entry points (both `quark/torch/` and `quark/onnx/`).
- Add a minimal governance loop so skill drift is detectable.

## Directory Layout

Every layer is subdivided by **backend scope**: `shared/`, `torch/`, `onnx/`. Skills under `<scope>/` carry the matching name infix (`quark-torch-*` / `quark-onnx-*` / bare `quark-*`).

```text
.claude/
├── skills/                          # User-facing entry stubs (1:1 with implementations)
└── skills-impl/                     # This directory — actual skill bodies
    ├── l0-foundation/
    │   └── shared/                  # quark-env-preflight, quark-workspace-validate
    ├── l1-atomic/
    │   ├── shared/                  # quark-install
    │   ├── torch/                   # quark-torch-router, quark-torch-install,
    │   │                            # quark-torch-model-intake, quark-torch-quant-plan,
    │   │                            # quark-torch-llm-eval, quark-torch-export,
    │   │                            # quark-torch-debug, quark-torch-result-validator,
    │   │                            # quark-torch-file2file-quantization
    │   └── onnx/                    # quark-onnx-router, quark-onnx-install,
    │                                # quark-onnx-model-intake, quark-onnx-quant-plan,
    │                                # quark-onnx-debug, quark-onnx-result-validator
    ├── l2-workflows/
    │   ├── torch/                   # quark-torch-llm-ptq-workflow
    │   └── onnx/                    # quark-onnx-ptq-workflow (vision/CNN)
    ├── l3-recipes/
    │   ├── torch/                   # quark-torch-llm-ptq-eval (PTQ + validate + eval recipe)
    │   └── onnx/                    # quark-onnx-autosearch-pro (AutoSearchPro / Optuna recipe)
    ├── meta/
    │   ├── shared/                  # quark-skill-creator
    │   ├── torch/                   # quark-torch-skill-sync, quark-torch-doc-drift-check,
    │   │                            # quark-torch-eval-runner
    │   └── onnx/                    # quark-onnx-skill-sync, quark-onnx-doc-drift-check,
    │                                # quark-onnx-eval-runner
    └── shared/                      # contracts/, templates/ (cross-skill resources, not skills)
```

Skill-system narrative docs live in [`docs/agent_skills/`](../../docs/agent_skills/README.md). Sample prompts live in [`examples/agent_skills/prompts/`](../../examples/agent_skills/prompts/). To add or modify a skill, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Layering Rules

- `.claude/skills/`: thin SKILL.md stubs Claude Code auto-discovers. Each stub references its real implementation via `Read and follow the instructions in .claude/skills-impl/...`.
- `l0-foundation`: environment detection, workspace checks, version and path validation.
- `l1-atomic`: single-responsibility skills with explicit inputs, outputs, and recovery behavior.
- `l2-workflows`: generic orchestrators (PTQ, QAT) that chain atomic skills, manage checkpoints, and hand off artifacts.
- `l3-recipes`: customer- or deployment-specific compositions that target a named context (e.g. PTQ for InferenceMax models) by configuring and combining L2 workflows.
- `meta`: sync, eval, regression, and drift checks that maintain the skill system itself. Orthogonal to L0-L3 — operates on skills, not in the user-facing data flow.
- `shared`: reusable contracts, templates, schemas, and conventions; no business-specific PTQ logic.

## MVP Skills

### User-facing (entry points in `.claude/skills/`)

| Skill | Backend | Triggers on |
|-------|---------|-------------|
| `quark-torch-ptq` | torch | "quantize my model", "FP8", "INT4", full PTQ pipeline |
| `quark-onnx-ptq` | onnx | "quantize my .onnx" (vision/CNN), "quantize yolov8/resnet50 with XINT8/A8W8/BFP16" |
| `quark-onnx-autosearch-pro` | onnx | "auto search", "tune quantization", "find best quant config", `ADVANCED_SEARCH` / `XINT8_SEARCH` / `A8W8_SEARCH` / `A16W8_SEARCH` presets |
| `quark-torch-install` | torch | "install PyTorch", "torch version mismatch" |
| `quark-onnx-install` | onnx | "install onnxruntime", "onnxruntime-gpu vs onnxruntime", "onnx version mismatch" |
| `quark-torch-model-intake` | torch | "analyze my model", "is this model supported" |
| `quark-onnx-model-intake` | onnx | "analyze my ONNX model", "what opset is this", "is my model NPU-compatible" |
| `quark-torch-export` | torch | "export model", "convert to GGUF/ONNX (output)" |
| `quark-torch-debug` | torch | torch tracebacks, "PTQ failed", CUDA OOM |
| `quark-onnx-debug` | onnx | "Quark ONNX error", "onnxruntime error", "quantize_static failed", "CUDAExecutionProvider not available", "custom op library load failed" |
| `quark-torch-result-validator` | torch | "validate quantized output", "verify exported weights" |
| `quark-onnx-result-validator` | onnx | "validate ONNX quantization result", "check quantized .onnx output", "verify ONNX initializers", "did QDQ insertion happen" |
| `quark-torch-llm-eval` | torch | "evaluate this model", "run gsm8k/mmlu" |
| `quark-torch-file2file-quantization` | torch | "file2file", large safetensors, low-memory quant |
| `quark-install` | shared | "install Quark", "pip install amd-quark" |
| `quark-env-preflight` | shared | "check my environment", "what GPU do I have" |

### Internal (no entry point, referenced by workflows)

- `quark-workspace-validate` (L0/shared) — path validation
- `quark-torch-router` (L1/torch) — Torch-side intent routing
- `quark-onnx-router` (L1/onnx) — ONNX-side intent routing — produces `session_context.json` with `constraints.backend = "onnx"`
- `quark-torch-quant-plan` (L1/torch) — Torch scheme selection
- `quark-onnx-quant-plan` (L1/onnx) — ONNX preset / calibration / algorithm selection (matches `quark.onnx` QConfig + `algo_config` surface)
- `quark-torch-llm-ptq-workflow` (L2/torch) — Torch LLM PTQ orchestrator (backs `quark-torch-ptq`)
- `quark-onnx-ptq-workflow` (L2/onnx) — ONNX-to-ONNX PTQ orchestrator for vision / CNN models (backs `quark-onnx-ptq`)
- `quark-torch-skill-sync` / `quark-torch-doc-drift-check` / `quark-torch-eval-runner` (meta/torch) — Torch-side governance
- `quark-onnx-skill-sync` / `quark-onnx-doc-drift-check` / `quark-onnx-eval-runner` (meta/onnx) — ONNX-side mirrors of the torch governance trio. `quark-onnx-doc-drift-check` is a read-only fact-checker for user-facing ONNX guidance (presets, calibration methods, custom-op names, QConfig fields, ORT install matrix, AutoSearchPro presets); `quark-onnx-skill-sync` audits the broader `quark/onnx/`, `examples/onnx/`, `tutorials/onnx/`, `docs/source/onnx/`, and `tools/ci/install_onnxruntime.sh` surface and is the only one of the three allowed to apply fixes; `quark-onnx-eval-runner` smoke-tests the ONNX skill family across the four contract categories (routing, planning, artifact, recovery) and includes a cross-backend isolation guard.
- `quark-skill-creator` (meta/shared) — authoring assistant for new skills, enforces format contract

## Core Contracts

- `session_context.json`
- `model_analysis.json`
- `quant_plan.json`
- `run_manifest.yaml`
- `validation_report.md`

JSON schemas live under [`shared/contracts/`](shared/contracts/), narrative docs (interaction/artifact/skill-format contracts) live under [`docs/agent_skills/`](../../docs/agent_skills/).

## Upstream Alignment

Skills in this directory reference Quark source files using **repo-root-relative paths** (e.g., `tools/ci/install_torch.sh`, `docs/source/install.rst`). Since this skill system now lives inside the Quark repository, the upstream source of truth is the surrounding repo itself — keep skill behavior aligned to current Quark docs, examples, requirements, and source entry points.
