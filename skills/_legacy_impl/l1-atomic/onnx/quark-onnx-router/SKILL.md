---
name: quark-onnx-router
description: >
  Route Quark ONNX user goals to the correct atomic skill. Use when a user describes an ONNX
  quantization task in plain language — such as "install onnxruntime", "analyze my .onnx model",
  "choose a preset for my YOLO model", "plan ONNX PTQ", "quantize this .onnx with XINT8/BFP16/MXFP4",
  "validate my quantized .onnx", "debug a failed ONNX quantization", or any request that involves
  Quark's ONNX-to-ONNX flow. This is the ONNX-side entry point — trigger it whenever the user's
  intent involves Quark ONNX and the correct downstream skill is not immediately obvious.
layer: l1-atomic
primary_artifact: session_context.json
source_knowledge:
  - docs/source/install.rst
  - docs/source/onnx/basic_usage_onnx.rst
  - docs/source/onnx/onnx_examples.rst
  - examples/onnx/model_support.md
---

# quark-onnx-router

## Purpose

Translate a user's natural-language ONNX quantization goal into the smallest correct skill boundary. The router exists because Quark's ONNX flow spans install, intake, planning, execution, debug, and validation — picking the wrong skill wastes time and produces wrong artifacts. The router is the sole producer of `session_context.json` for the ONNX backend; every other artifact (env, workspace, install results, model analysis, quant plan) is produced by its own owning skill, and the router only carries forward references to those files.

## Inputs

- User goal stated in natural language
- `env_context.json` (optional, when routing depends on hardware / execution-provider facts)
- `workspace_context.json` (optional, when routing depends on validated `.onnx` / `.onnx_data` paths)

## Outputs: session_context.json

Carries the user goal, selected workflow (or atomic skill if no workflow applies), constraints with `backend = "onnx"`, `*_ref` pointers to other artifacts, and unresolved questions.

Schema: [`session_context.schema.json`](../../../shared/contracts/session_context.schema.json)

```json
{
  "user_goal": "Quantize ./models/yolov8n.onnx with XINT8 and validate against the FP32 baseline",
  "workflow": "quark-onnx-ptq-workflow",
  "constraints": {
    "backend": "onnx",
    "offline": false,
    "execution_mode": "interactive_execute"
  },
  "env_context_ref": null,
  "workspace_context_ref": null,
  "onnx_install_result_ref": null,
  "quark_install_result_ref": null,
  "model_analysis_ref": null,
  "quant_plan_ref": null,
  "open_questions": [
    "Deployment target (CPU / CUDA / ROCm / NPU CNN / NPU Transformer) not yet confirmed — affects preset gating in quark-onnx-quant-plan"
  ]
}
```

Set each `*_ref` field to the path of the corresponding artifact once its producer skill has run. The router never embeds hardware, workspace, install, model, or plan facts inline — those belong in their owning artifacts. Always set `constraints.backend = "onnx"` so downstream skills and any future cross-backend orchestrator can disambiguate from the Torch flow.

## CRITICAL ROUTING RULE

**Vision / CNN `.onnx` PTQ requests MUST route to `quark-onnx-ptq-workflow`.**

Signals: model name contains `yolo`, `resnet`, `mobilenet`, `efficientnet`, etc.; inputs are image tensors `[N, C, H, W]`; preset is `XINT8` / `A8W8` / `A16W8` / `BF16` / `BFP16`; user mentions mAP / Prec@1 / NPU CNN / Ryzen AI.

Examples:

- "quantize yolov8n.onnx with XINT8 for Ryzen AI" → `quark-onnx-ptq-workflow`
- "quantize resnet50 .onnx with A8W8" → `quark-onnx-ptq-workflow`
- "run ONNX PTQ on my image-classification model" → `quark-onnx-ptq-workflow`

**NEVER invoke `quark.onnx.ModelQuantizer.quantize_model(...)` directly** from within the router. Execution belongs inside the workflow, after a confirmed `quant_plan.json` and a generated script the user has reviewed (the workflow's 4-step flow: intake → plan → manifest+script → confirmed execution).

## Skill Map

| User Intent | Target Skill | Why |
|-------------|-------------|-----|
| **Quantize a .onnx vision/CNN model** (YOLO / ResNet / MobileNet / …) | `quark-onnx-ptq-workflow` | Vision-specific: image data reader, CNN presets (XINT8 / A8W8 / BFP16), `EnableNPUCnn`, mAP / Prec@1 eval |
| Full end-to-end ONNX PTQ: from `.onnx` to quantized `.onnx` | `quark-onnx-ptq-workflow` | Multi-step workflow orchestration |
| Install ONNX Runtime, fix CPU/GPU variant, missing CUDA/ROCm EP | `quark-onnx-install` | ONNX Runtime install is separate from Quark package install |
| Install Quark, set up Quark dependencies | `quark-install` | Backend-neutral Quark package install (assumes ORT already set up) |
| Inspect a `.onnx` model, check opset / IR / op-type histogram, NPU compatibility | `quark-onnx-model-intake` | Model facts are prerequisites for planning |
| Choose preset (XINT8 / A8W8 / BFP16 / MX* / MatMulNBits), calibration method, algorithm | `quark-onnx-quant-plan` | Planning is separate from execution |
| Debug a failed ONNX quantization, ORT EP error, custom-op load failure, calibration crash | `quark-onnx-debug` | Error recovery has its own diagnostic flow |
| Validate quantized `.onnx`, verify QDQ insertion, check non-quantized initializers | `quark-onnx-result-validator` | Post-quantization byte-level + structural validation |
| Check environment, detect GPU, verify setup | `quark-env-preflight` | L0 fact collection (shared backend-neutral) |
| Validate `.onnx` / `.onnx_data` paths, output directory | `quark-workspace-validate` | L0 path validation (shared backend-neutral) |

## Routing Logic

1. **Confirm the backend.** If the user mentions `.onnx`, `quantize_static`, `ModelQuantizer`, an ONNX Runtime execution provider, opset, or QDQ, treat the request as ONNX. If they mention HuggingFace, safetensors, `transformers`, `torch._dynamo`, or `quantize_quark.py`, hand off to `quark-torch-router` instead — never silently mix backends.

2. **Extract the core intent.** Strip filler and figure out what the user actually needs. "I want to quantize yolov8n.onnx with XINT8 for CPU" → vision PTQ → `quark-onnx-ptq-workflow`.

3. **Check for L0 prerequisites.** Before routing to an L1 skill, check if downstream skills need facts that are missing:
   - If hardware / accelerator / execution-provider facts are needed but unknown → run `quark-env-preflight` first.
   - If the `.onnx` (and any sibling `.onnx_data`) path or output directory needs validation → run `quark-workspace-validate` first.

4. **Pick the smallest fit.** If the user only wants to inspect a `.onnx` graph, route to `quark-onnx-model-intake` — do not start the full workflow. If they only want to install ORT, route to `quark-onnx-install`. But if they say "quantize my .onnx", use `quark-onnx-ptq-workflow`.

5. **Handle ambiguity honestly.** If the goal is unclear, record the likely routes in `open_questions` and ask. Example: "I want to use Quark with my ONNX model" — does that mean inspect, plan, quantize end-to-end, or validate an already-quantized file?

## Rules

- **Do not guess deployment targets, EP availability, or paths.** If the deployment target (CPU / CUDA / ROCm / NPU CNN / NPU Transformer) is unknown, do not assume CPU — gate it as an open question. Preset gating in `quark-onnx-quant-plan` depends on this.
- **Do not load `.onnx` files in the router.** The router only reads paths and string intent; loading the model is `quark-onnx-model-intake`'s job (especially relevant for >2 GB models with external data).
- **Explain the routing.** Tell the user why you chose a particular skill: "Since you want to quantize a .onnx, I'll start with model intake, then we'll pick a preset, then I'll generate a script you confirm before running."
- **Reuse existing context.** If a `session_context.json` already exists from a previous step, read it and carry forward — do not start from scratch.
- **Refuse cross-backend mixing.** Never feed a `.onnx` into `quark-torch-*`; never feed safetensors into `quark-onnx-*`. If the user's input doesn't match this router's backend, hand off to `quark-torch-router`.

## Interaction Flow

1. **Listen**: Restate the user's goal in one sentence to confirm understanding and confirm the backend is ONNX.
2. **Assess**: Check what facts are already known vs. missing. Do L0 skills need to run first? Is the deployment target known?
3. **Route**: Name the target skill (or first link in the chain) and explain why it is the right fit. For PTQ requests, state the chain explicitly and that execution will only happen after the user confirms the generated script.
4. **Hand off**: Produce the initial `session_context.json` (with `constraints.backend = "onnx"`) and pass control to the chosen skill.

## Recovery

- If the goal is ambiguous, present the 2-3 most likely interpretations and ask the user to pick.
- If required inputs are missing, list them explicitly — do not route to a downstream skill with known gaps that will immediately fail (e.g. starting `quark-onnx-quant-plan` without a `model_analysis.json`).
- If L0 facts are unresolved, hand off a partial `session_context.json` with the gaps documented rather than forcing a premature routing decision.
- If the user pasted a Torch traceback or a HuggingFace path, do not try to repair-route — hand off to `quark-torch-router` and stop.

## Examples

### Example 1: Clear vision PTQ request
>
> User: "Quantize ./models/yolov8n.onnx with XINT8 for Ryzen AI deployment, output to ./output/yolov8n-xint8.onnx"
> Router: → `quark-onnx-ptq-workflow` (vision task family: YOLO + XINT8 + NPU CNN)
> Explain: "I'll quantize your YOLOv8 ONNX model with the vision PTQ workflow: 4 steps (intake → plan → generated script + manifest → confirmed execution)."

### Example 2: ONNX Runtime install request
>
> User: "I need to install onnxruntime-rocm 7.1"
> Router: → `quark-onnx-install` (clear ORT install intent, EP stated)

### Example 3: Debug request
>
> User: "`quantize_static` is failing with 'CUDAExecutionProvider not available'"
> Router: → `quark-onnx-debug` (ORT EP error, established diagnostic flow)

### Example 4: Validation-only request
>
> User: "Here's the quantized output — did the QDQ insertion actually happen?"
> Router: → `quark-onnx-result-validator`

### Example 5: Ambiguous request
>
> User: "I want to use Quark with my ONNX model"
> Router: Ask — "Do you want to: (a) inspect the graph's opset and op-types, (b) plan and run a full PTQ, (c) install ONNX Runtime / Quark first, or (d) validate an already-quantized output?"

### Example 6: Cross-backend mistake
>
> User: "Quantize Qwen/Qwen3-8B with FP8" (HuggingFace repo id, no `.onnx`)
> Router: Hand off → `quark-torch-router` and stop. Do not attempt to route this through the ONNX flow.
