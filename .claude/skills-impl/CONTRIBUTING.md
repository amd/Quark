# Contributing a Skill

## Skill Directory Structure

Each skill lives in its own directory under the appropriate layer **and backend scope**:

```text
.claude/skills-impl/<layer>/<scope>/<skill-name>/
                              ↑
                       shared | torch | onnx
└── SKILL.md
```

Layers: `l0-foundation`, `l1-atomic`, `l2-workflows`, `l3-recipes`, `meta` (orthogonal — skill-system governance).

User-facing skills also need a thin entry stub at `.claude/skills/<skill-name>/SKILL.md` that points at the implementation via `Read and follow the instructions in .claude/skills-impl/<layer>/<scope>/<skill-name>/SKILL.md`. **Stub names are 1:1 with the implementation directory** — no dispatcher stubs.

## Choosing Scope

The `<scope>` decision determines both the directory and the skill name infix:

| Scope | Use when | Naming | Examples |
|-------|----------|--------|----------|
| `torch/` | Skill operates on PyTorch / HuggingFace safetensors models, uses `quark.torch`, `transformers`, `torch._dynamo`, etc. | `quark-torch-<verb>` | `quark-torch-ptq`, `quark-torch-debug` |
| `onnx/` | Skill operates on `.onnx` graphs, uses `onnxruntime` or ONNX-specific quantization paths | `quark-onnx-<verb>` | `quark-onnx-ptq`, `quark-onnx-export` |
| `shared/` | Skill is truly backend-agnostic — installs the `amd-quark` package itself, validates host env / paths, or governs the skill format. Should NOT import torch-only or onnx-only modules. | `quark-<verb>` (no infix) | `quark-install`, `quark-env-preflight`, `quark-skill-creator` |

**`<scope>` applies to every layer including `meta/`.** Drift checks, sync, and smoke-tests target backend-specific upstream source files, so a torch-side `quark-torch-skill-sync` and an onnx-side `quark-onnx-skill-sync` are distinct skills. Only governance that operates on the skill format itself (e.g. `quark-skill-creator`) belongs in `meta/shared/`.

The skill name's backend infix MUST match the parent `<scope>/` directory. CI lint will reject mismatches.

### Entry stub `description` keywords

Because Claude Code's harness routes by `description` prompt-matching, every backend-specific entry stub MUST include unambiguous backend keywords in its `description` field:

- **torch stubs**: include at least one of `PyTorch`, `HuggingFace safetensors`, `transformers`, `torch._dynamo`, or a torch-side framework name (`vLLM`, `SGLang`, etc.); add a `Not for .onnx — use quark-onnx-*` pointer when ambiguity exists.
- **onnx stubs**: include `.onnx`, `onnxruntime`, or `ONNX model` so the harness can distinguish from torch.

## What Goes in a SKILL.md

Every SKILL.md needs two parts:

**1. YAML Frontmatter:**

```yaml
---
name: <skill-name>             # must match the <scope> infix rule above
description: <when to trigger and what it does>
layer: <l0-foundation | l1-atomic | l2-workflows | l3-recipes | meta>
backend: <shared | torch | onnx>   # must equal the parent <scope>/ directory
primary_artifact: <main output file>
source_knowledge:
  - <upstream Quark doc or source file, repo-root-relative>
---
```

**2. Required Sections:**

| Section | What to Write |
|---------|---------------|
| `## Purpose` | One paragraph — what this skill does |
| `## Inputs` | What the skill needs before it can run |
| `## Outputs` | What the skill produces |
| `## Interaction Flow` | Steps the skill walks through |
| `## Recovery` | Common failures and how to recover |

Beyond these, add whatever sections your skill needs (examples, decision tables, rules, etc.).

## Contracts to Follow

These define how skills interact with each other — read them before writing a new skill:

- [Interaction Contract](../../docs/agent_skills/interaction-contract.md) — the five-stage flow (Intake, Route, Plan, Confirm, Execute)
- [Artifact Contracts](../../docs/agent_skills/artifact-contracts.md) — the five canonical handoff artifacts
- [Skill Format Contract](../../docs/agent_skills/skill-format-contract.md) — why each frontmatter field and section is required

The skill template is at [`shared/templates/skill-template.md`](shared/templates/skill-template.md).

For architecture, governance, and validation docs, see [`docs/agent_skills/`](../../docs/agent_skills/README.md).

## Length Budgets

Skills load into Claude's context, so keep them tight:

- **SKILL.md ≤ 315 lines.** Move detail into reference files alongside SKILL.md.
- **`description` ≤ 100 words.** It loads in baseline context on every session.
