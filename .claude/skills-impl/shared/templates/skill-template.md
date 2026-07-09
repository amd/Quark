# Skill Template

Place the skill at `.claude/skills-impl/<layer>/<scope>/<skill-name>/SKILL.md`. The skill name MUST match the parent `<scope>/` directory:

- `torch/quark-torch-*`
- `onnx/quark-onnx-*`
- `shared/quark-*` (no infix)

For user-facing skills, also add a thin entry stub at `.claude/skills/<skill-name>/SKILL.md`. The stub's `description` MUST include backend keywords so the harness can route by prompt-match (torch: `PyTorch` / `HuggingFace safetensors` / `transformers`; onnx: `.onnx` / `onnxruntime`).

```yaml
---
name: <skill-name>             # matches the <scope> infix rule above
description: <trigger phrases + backend keywords>
layer: <l0-foundation | l1-atomic | l2-workflows | l3-recipes | meta>
backend: <shared | torch | onnx>   # must equal the parent <scope>/ directory
primary_artifact: <artifact file name or report name>
source_knowledge:
  - <Quark upstream doc or source file>
---
```

## Purpose

One paragraph describing the stable responsibility boundary of the skill.

## Inputs

- required inputs
- optional defaults

## Outputs: <primary_artifact filename>

Schema: [`<primary_artifact>.schema.json`](../../../shared/contracts/<primary_artifact>.schema.json)

```json
{ "...": "example matching the schema" }
```

## Interaction Flow

1. Intake
2. Route
3. Plan
4. Confirm
5. Execute or Summarize

## Recovery

- common failure modes
- recovery advice

## Notes

- traceability to upstream docs, CLI, or code
- guardrails that should not be silently skipped
