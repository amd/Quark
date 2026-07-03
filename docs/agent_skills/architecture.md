# Architecture And Naming Specification

## Purpose

This document freezes the executable architecture for the greenfield Quark skill system. New skills must satisfy the layer criteria below before they are added.

## Design Anchors

- User intent is expressed as a goal, not as a specific skill name.
- Routing happens before execution.
- High-cost or high-risk actions require a visible confirmation step.
- Artifacts are first-class contracts between skills.
- Governance is part of the product, not a later add-on.

## Layer Admission Rules

### L0 Foundation

Use `l0-foundation` only when the capability is about environment sensing or safety checks.

Allowed responsibilities:

- detect platform, Python, GPU, Quark version, and workspace shape
- validate paths, model references, and required files
- inspect upstream references before downstream planning starts

Not allowed:

- choosing a PTQ scheme
- generating workflow artifacts beyond raw environment facts
- orchestrating multi-step user flows

### L1 Atomic

Use `l1-atomic` only when the skill performs one stable action with a single primary output.

Required characteristics:

- single responsibility
- explicit input checklist
- explicit primary artifact
- bounded recovery guidance
- reusable from a workflow without rewriting its prompt

Examples:

- `quark-install` (shared)
- `quark-torch-router`
- `quark-torch-install`
- `quark-torch-model-intake`
- `quark-torch-quant-plan`
- `quark-torch-export`
- `quark-torch-debug`
- `quark-onnx-router`
- `quark-onnx-install`
- `quark-onnx-model-intake`
- `quark-onnx-quant-plan`
- `quark-onnx-debug`
- `quark-onnx-result-validator`

### L2 Workflows

Use `l2-workflows` only when the skill coordinates multiple atomic skills and manages checkpoints.

Required characteristics:

- consumes and passes standard artifacts
- defines ordering and branching rules
- exposes confirm checkpoints before risky execution
- can summarize a manual runbook if execution is skipped

Examples:

- `quark-torch-llm-ptq-workflow`
- `quark-onnx-ptq-workflow`

### L3 Recipes

Use `l3-recipes` when the skill targets a specific deployment context (customer model family, downstream runtime, hardware profile) by composing one or more L2 workflows with context-specific configuration. Recipes are concrete applications, not new techniques.

Required characteristics:

- targets a named context (a specific model family, customer, or deployment target)
- composes existing L2 workflows; does not duplicate atomic logic
- carries the context-specific configuration that L2 workflows leave open
- documents the deployment scenario it is built for

Examples:

- `quark-onnx-autosearch-pro` (ONNX AutoSearchPro recipe — composes the ONNX PTQ workflow with Optuna-driven hyperparameter search and built-in presets such as `ADVANCED_SEARCH`, `XINT8_SEARCH`, `A8W8_SEARCH`, `A16W8_SEARCH`)
- (Torch-side recipe slot still open — first deployment-targeted recipe will land here, e.g. `quark-torch-inferencemax-ptq-recipe`)

### Meta (orthogonal to L0-L3)

Use `meta` only when the capability protects correctness or freshness of the skill system. This layer is orthogonal to the L0-L2 stack — it operates on skills as artifacts, not in the user-facing data flow.

Required characteristics:

- targets skill quality, drift, or evaluation
- may inspect multiple skills or upstream references
- must produce a governance artifact or report
- should not be the first skill a normal product user sees

Examples:

- `quark-torch-skill-sync` — broad audit of `quark/torch/` source vs. each torch skill's `source_knowledge`; classifies drift as mechanical / semantic / breaking and is the only torch meta skill allowed to apply fixes
- `quark-torch-doc-drift-check` — read-only fact-checker that verifies user-facing torch guidance (CLI flags, scheme list, model templates, install matrix) still matches `docs/source/` and `examples/torch/`
- `quark-torch-eval-runner` — manual four-category smoke test (routing, planning, artifact, recovery) for the torch skill family
- `quark-onnx-skill-sync` — ONNX-side mirror of `quark-torch-skill-sync`; audits `quark/onnx/`, `examples/onnx/`, `tutorials/onnx/`, `docs/source/onnx/`, and `tools/ci/install_onnxruntime.sh` against ONNX skill assumptions (presets, calibration methods, custom-op registry, AutoSearchPro presets, QConfig surface). Only ONNX meta skill allowed to apply fixes.
- `quark-onnx-doc-drift-check` — ONNX-side mirror of `quark-torch-doc-drift-check`; read-only fact-check of user-facing ONNX guidance (preset names, calibration methods, custom-op names, ORT install matrix, AutoSearchPro presets, contract fields). Hands off to `quark-onnx-skill-sync` for any fix.
- `quark-onnx-eval-runner` — ONNX-side mirror of `quark-torch-eval-runner`; runs the same four-category protocol against the ONNX skill family and adds a cross-backend isolation guard (ONNX prompts must never route to `quark-torch-*`).

The torch and onnx meta skills are governed independently — each set targets only the upstream paths and skills of its own backend. Cross-cut findings (e.g. a `shared/contracts/` schema change) are surfaced but never applied across backends from a single meta skill.

## Naming Rules

- Use lowercase kebab-case for skill identifiers.
- Prefix all skills with `quark-`.
- **Backend-specific skills carry a backend infix**: `quark-<backend>-<verb>` (e.g. `quark-torch-ptq`, `quark-onnx-ptq`). Backend-agnostic shared skills use the bare form `quark-<verb>` (e.g. `quark-install`, `quark-env-preflight`).
- Prefer verbs or action-bearing nouns after the prefix / backend infix: `quark-install`, `quark-torch-debug`.
- Reserve `*-workflow` suffix for L2 orchestrators only.
- Reserve `*-recipe` suffix for L3 recipes that target a specific deployment context.
- Reserve `*-sync`, `*-eval-*`, and `*-drift-*` style names for `meta` governance skills.
- Keep names aligned to user goals, not implementation details.

## Directory Rules

Each skill lives in its own directory, grouped first by layer and then by **backend scope**:

```text
.claude/skills-impl/<layer>/<scope>/<skill-name>/
                              ↑
                       shared | torch | onnx
└── SKILL.md
```

`<scope>` applies to **every layer including `meta/`** — drift checks, sync, and smoke-tests target backend-specific upstream source and CLI flags, so they must be partitioned by backend just like atomic skills. Only truly backend-agnostic skills (e.g. `quark-install` which installs the `amd-quark` package itself, format/template tooling) live under `shared/`.

The skill name's backend infix must match the parent `<scope>/` directory:

- `<scope>/torch/quark-torch-*/`
- `<scope>/onnx/quark-onnx-*/`
- `<scope>/shared/quark-*/` (no infix)

Optional adjacent files are allowed for examples, checklists, and helper notes, but cross-skill shared conventions and contract schemas live under `.claude/skills-impl/shared/`.

## Release Gate For MVP

The MVP is considered ready only when:

1. The `Torch + LLM PTQ` path can route, intake, plan, and hand off a run manifest.
2. The `ONNX + CNN PTQ` path can route, intake, plan, and hand off a run manifest, with `quark-onnx-result-validator` available for post-quantization inspection of `model.onnx` (+ optional `model.onnx_data`).
3. All five standard artifacts have canonical definitions and examples (shared across both backends; ONNX-specific artifacts such as the ONNX `quant_plan.json` and `run_manifest.yaml` reuse the same schemas).
4. The four evaluation classes each have at least one runnable task definition per active backend (torch and onnx).
5. Drift detection has a documented path from upstream change to affected skill for both `quark/torch/` and `quark/onnx/` source trees, backed by parallel meta trios: `quark-torch-skill-sync` / `quark-torch-doc-drift-check` / `quark-torch-eval-runner` on the torch side, and `quark-onnx-skill-sync` / `quark-onnx-doc-drift-check` / `quark-onnx-eval-runner` on the onnx side.
