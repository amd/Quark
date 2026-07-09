# Layer Decision

Authoritative source: `docs/agent_skills/architecture.md` "Layer Admission Rules". This file is the practical decision tree authors walk during the Route phase of `quark-skill-creator`. When you cannot decide between two layers, prefer the lower one (closer to L0) and flag for review — the cost of an over-scoped skill is much higher than an under-scoped one.

## Decision tree

```text
Does the skill operate on the skill system itself?
├── yes → meta
└── no
    │
    Does it coordinate other skills and manage checkpoints?
    ├── yes
    │   │
    │   Is it generic across deployments?
    │   ├── yes → l2-workflows
    │   └── no, targets a named customer/model/runtime → l3-recipes
    │
    └── no
        │
        Does it only read environment / validate paths
        (no installs, no PTQ decisions, no artifact generation
         beyond raw facts)?
        ├── yes → l0-foundation
        └── no, single responsibility with one primary artifact → l1-atomic
```

## Layer rubrics

### `l0-foundation`

**Allowed:** detect platform/Python/GPU/Quark version, validate paths and model references, inspect upstream references before downstream planning starts.

**Forbidden:** PTQ scheme selection, workflow artifacts beyond raw environment facts, multi-step user flows.

**Examples:** `quark-env-preflight`, `quark-workspace-validate`.

### `l1-atomic`

**Required:** single responsibility, explicit input checklist, explicit primary artifact, bounded recovery, reusable from a workflow without rewriting its prompt.

**Examples:** `quark-torch-router`, `quark-torch-install`, `quark-install`, `quark-torch-model-intake`, `quark-torch-quant-plan`, `quark-torch-export`, `quark-torch-debug`, `quark-torch-llm-eval`.

### `l2-workflows`

**Required:** consumes and passes standard artifacts, defines ordering and branching, exposes confirm checkpoints before risky execution, can summarize a manual runbook if execution is skipped.

**Examples:** `quark-torch-llm-ptq-workflow`.

### `l3-recipes`

**Required:** targets a named context (model family, customer, deployment), composes existing L2 workflows, carries the context-specific configuration L2 leaves open, documents the deployment scenario.

**Examples:** placeholder — first recipe will be `quark-inferencemax-ptq-recipe` style.

### `meta`

**Required:** targets skill quality / drift / evaluation, may inspect multiple skills or upstream references, must produce a governance artifact or report, should not be the first skill a normal product user sees.

**Examples:** `quark-torch-skill-sync`, `quark-torch-doc-drift-check`, `quark-torch-eval-runner`, `quark-skill-creator`.

## Naming rules (per `docs/agent_skills/architecture.md`)

- Lowercase kebab-case, always `quark-` prefix.
- Verbs or action-bearing nouns after the prefix (`quark-install`, `quark-torch-debug`).
- Reserved suffixes:
  - `*-workflow` for L2 orchestrators only
  - `*-recipe` for L3 recipes
  - `*-sync`, `*-eval-*`, `*-drift-*`, `*-creator` for meta governance
- Names align to user goals, not implementation details (`quark-install`, not `quark-pip-runner`).

## Common misplacements (caught in review)

- **Routing or drift logic in `l1-atomic`** — anything that operates on the skill system itself belongs in `meta`. `quark-torch-router` is the deliberate exception (it routes user intent, not skills) and is documented as such.
- **PTQ scheme selection in `l0-foundation`** — that is `quark-torch-quant-plan`'s job (l1-atomic).
- **Workflow orchestration in `l1-atomic`** — if a skill consumes the output of two atomic skills and decides what to do next, it is L2 by construction.
- **Atomic skill that duplicates an existing one's responsibility** — find the existing skill, extend it, or absorb the new behavior into a workflow. Do not create a parallel atomic.
