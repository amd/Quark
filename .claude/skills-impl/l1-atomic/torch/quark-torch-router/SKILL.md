---
name: quark-torch-router
description: >
  Route Quark user goals to the correct atomic skill or workflow. Use when a user describes a Quark task in plain language —
  such as "install Quark", "quantize a model", "analyze my model", "build a PTQ plan", "export the quantized model",
  "debug a failed run", "run the full PTQ pipeline", "run model quantization", "install Quark", or any request that involves AMD Quark
  model quantization. This is the entry point skill — trigger it whenever the user's intent involves Quark and the
  correct downstream skill is not immediately obvious.
layer: l1-atomic
primary_artifact: session_context.json
source_knowledge:
  - docs/source/install.rst
  - examples/torch/language_modeling/llm_ptq/README.md
---

# quark-torch-router

## Purpose

Translate a user's natural-language goal into the smallest correct skill boundary. The router exists because Quark has a layered skill system — picking the wrong skill wastes time and produces wrong artifacts. The router is the sole producer of `session_context.json`; every other artifact (env, workspace, install results) is produced by its own owning skill, and the router only carries forward references to those files.

## Inputs

- User goal stated in natural language
- `env_context.json` (optional, when routing depends on hardware facts)
- `workspace_context.json` (optional, when routing depends on validated paths)

## Outputs: session_context.json

Carries the user goal, selected workflow, constraints, `*_ref` pointers to other artifacts, and unresolved questions.

Schema: [`session_context.schema.json`](../../../shared/contracts/session_context.schema.json)

```json
{
  "user_goal": "Quantize Qwen/Qwen3-8B with FP8 and export to HuggingFace format",
  "workflow": "quark-torch-llm-ptq-workflow",
  "constraints": {
    "offline": false,
    "execution_mode": "interactive_execute"
  },
  "env_context_ref": null,
  "workspace_context_ref": null,
  "pytorch_install_result_ref": null,
  "quark_install_result_ref": null,
  "open_questions": [
    "GPU type and CUDA/ROCm version not yet confirmed — quark-env-preflight needed"
  ]
}
```

Set each `*_ref` field to the path of the corresponding artifact once its producer skill has run. The router never embeds hardware, workspace, or install facts inline — those belong in their owning artifacts.

## CRITICAL ROUTING RULE

**Any request that involves quantizing a model MUST route to `quark-torch-llm-ptq-workflow`.**

This includes:

- "quantize X with Y" → workflow
- "quantize X to FP8/INT4/..." → workflow
- "run PTQ on X" → workflow
- "run quantization on X" → workflow
- "help me quantize" → workflow

**NEVER run `quantize_quark.py` directly** without going through the workflow's 4-step flow (intake → plan → manifest → confirmed execution).

## Skill Map

| User Intent | Target Skill | Why |
|-------------|-------------|-----|
| **Quantize a model** (any scheme, any model) | `quark-torch-llm-ptq-workflow` | **Must use the 4-step workflow with checkpoints** |
| Full end-to-end PTQ: from model to quantized output | `quark-torch-llm-ptq-workflow` | Multi-step workflow orchestration |
| Install PyTorch, set up torch, fix torch version | `quark-torch-install` | PyTorch installation is separate from Quark package installation |
| Install Quark, set up Quark dependencies, check Quark packages | `quark-install` | Quark package installation, assumes PyTorch already set up |
| Inspect a model, check architecture, validate model path | `quark-torch-model-intake` | Model facts are prerequisites for planning |
| Choose quantization scheme, build a quant plan | `quark-torch-quant-plan` | Planning is separate from execution |
| Export quantized model, package for deployment | `quark-torch-export` | Export is a post-quantization step |
| Debug a failed run, fix an error, diagnose issues | `quark-torch-debug` | Error recovery has its own diagnostic flow |
| Check environment, detect GPU, verify setup | `quark-env-preflight` | L0 fact collection only |
| Validate paths, check model directory | `quark-workspace-validate` | L0 path validation only |

## Routing Logic

1. **Extract the core intent.** Strip away filler and figure out what the user actually needs. "I want to quantize Llama-2 with INT4" → the intent is PTQ → route to `quark-torch-llm-ptq-workflow`.

2. **Check for L0 prerequisites.** Before routing to an L1 skill, check if the downstream skill needs facts that are missing:
   - If hardware/accelerator facts are needed but unknown → run `quark-env-preflight` first
   - If model path or output directory needs validation → run `quark-workspace-validate` first

3. **Pick the smallest fit.** If the user only wants to check their model's architecture, route to `quark-torch-model-intake` — do not send them through the full workflow. But if they say "quantize my model", use `quark-torch-llm-ptq-workflow`.

4. **Handle ambiguity honestly.** If the goal is unclear, record the likely routes in `open_questions` and ask the user. Example: "I want to set up Quark" — does that mean install, or install + run PTQ?

## Rules

- **Do not guess hardware facts or paths.** If you do not know the GPU type, do not assume CUDA. If you do not know the model path, do not invent one. Unresolved gaps go into `open_questions`.
- **Explain the routing.** Tell the user why you chose a particular skill: "Since you want to quantize a model, I'll use the PTQ workflow which goes through 4 steps: model intake → quant plan → manifest → confirmed execution."
- **Reuse existing context.** If a `session_context.json` already exists from a previous step, read it and carry forward — do not start from scratch.

## Interaction Flow

1. **Listen**: Restate the user's goal in one sentence to confirm understanding.
2. **Assess**: Check what facts are already known vs. missing. Do L0 skills need to run first?
3. **Route**: Name the target skill and explain why it is the right fit. For PTQ requests, state that you will follow the 4-step flow with checkpoints.
4. **Hand off**: Produce the initial `session_context.json` and pass control to the chosen skill.

## Recovery

- If the goal is ambiguous, present the 2-3 most likely interpretations and ask the user to pick.
- If required inputs are missing, list them explicitly — do not route to a downstream skill with known gaps that will immediately fail.
- If L0 facts are unresolved, hand off a partial `session_context.json` with the gaps documented rather than forcing a premature routing decision.

## Examples

### Example 1: Clear PTQ request
>
> User: "Help me quantize Qwen/Qwen3-8B to FP8, output to ./output/qwen3-8b-fp8"
> Router: → `quark-torch-llm-ptq-workflow`
> Explain: "I'll quantize your model through a 4-step workflow: (1) model intake, (2) quantization plan, (3) command generation, (4) execution after your confirmation."

### Example 2: Quark install request
>
> User: "How do I install Quark? I'm on Ubuntu with ROCm 7.1"
> Router: → `quark-install` (clear Quark install intent, hardware stated — PyTorch setup handled by `quark-torch-install` if needed)

### Example 3: PyTorch install request
>
> User: "I need to install PyTorch for ROCm 7.1"
> Router: → `quark-torch-install` (clear PyTorch install intent, accelerator stated)

### Example 4: Ambiguous request
>
> User: "I want to use Quark with my model"
> Router: Ask — "Do you want to: (a) inspect your model's architecture, (b) quantize it, or (c) set up Quark first?"
