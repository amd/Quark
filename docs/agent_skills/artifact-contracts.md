# Artifact Contracts

## Run-scoped ownership

Artifact ownership is scoped to one workflow or run directory. Within that
scope, one producing skill or workflow owns an artifact throughout its
lifecycle and may create and update it across steps. Other producers must not
write the same path concurrently. Separate runs may use the same filename
without sharing ownership.

## Contract resolution

- A self-contained entry's top-level `SKILL.md` defines output paths and
  lifecycle; directly owned runtime contracts live beside that entry.
- For a transitional entry, the public stub routes and the delegated
  `_legacy_impl` body defines artifacts and lifecycle.
- Contracts under
  [`skills/_legacy_impl/shared/contracts/`](../../skills/_legacy_impl/shared/contracts/)
  are compatibility contracts for delegated and internal legacy
  implementations. Resolve a contract from the producing skill's profile; a
  familiar filename alone does not select a global schema.

## Current shared and Torch primary artifacts

This inventory intentionally leaves ONNX artifact details to a later
backend-specific update.

### Shared

| Producer | Primary artifact |
|---|---|
| [`quark-env-preflight`](../../skills/quark-env-preflight/SKILL.md) | `env_context.json` |
| [`quark-install`](../../skills/quark-install/SKILL.md) | `quark_install_result.json` |
| [`quark-create-shapeshifter-pass`](../../skills/quark-create-shapeshifter-pass/SKILL.md) | `shapeshifter_pass.py` |

### Torch

| Producer | Primary artifact |
|---|---|
| [`quark-torch-ptq`](../../skills/quark-torch-ptq/SKILL.md) | `model_analysis.json`, `quant_plan.json`, `run_manifest.yaml`, and the confirmed quantized-model directory |
| [`quark-torch-quant-perf`](../../skills/quark-torch-quant-perf/SKILL.md) | `session_report.md` and the structured `session_breakdown.json` |
| [`quark-torch-install`](../../skills/quark-torch-install/SKILL.md) | `pytorch_install_result.json` |
| [`quark-torch-model-intake`](../../skills/quark-torch-model-intake/SKILL.md) | `model_analysis.json` |
| [`quark-torch-result-validator`](../../skills/quark-torch-result-validator/SKILL.md) | `validation_report.md` |
| [`quark-torch-llm-eval`](../../skills/quark-torch-llm-eval/SKILL.md) | `$EVAL_STATE_DIR/eval_report.md` |
| [`quark-torch-file2file-quantization`](../../skills/quark-torch-file2file-quantization/SKILL.md) | `run_manifest.yaml`, generated wrapper or conversion scripts, and the quantized checkpoint |
| [`quark-torch-shrink-model`](../../skills/quark-torch-shrink-model/SKILL.md) | `shrink_result.md` and the destination model, or JSON-only test output |

Legacy internal routing, workspace validation, and planning may also produce
`session_context.json`, `workspace_context.json`, and `quant_plan.json` using
the transitional shared contracts. These are internal handoffs, not additional
public entries.
