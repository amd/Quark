# Format Rules (detailed)

The full set of rules that `validate_skill.py` enforces and that the SKILL.md authoring flow checks before commit. Pulled together from `docs/agent_skills/skill-format-contract.md`, `docs/agent_skills/interaction-contract.md`, `docs/agent_skills/artifact-contracts.md`, `docs/agent_skills/governance.md`, and `.claude/skills-impl/CONTRIBUTING.md` so authors do not need to read all of them in sequence every time.

## Frontmatter — five required fields

| Field | Requirement | Why |
|-------|-------------|-----|
| `name` | matches `^[a-z][a-z0-9-]{0,63}$` | identifier across routing, logging, governance |
| `description` | ≤ 100 words, third-person, includes WHAT + WHEN + trigger phrases | the trigger surface; Claude routes on this string |
| `layer` | one of `l0-foundation`, `l1-atomic`, `l2-workflows`, `l3-recipes`, `meta` | enforces what the skill may depend on (see `references/layer-decision.md`) |
| `primary_artifact` | concrete filename (e.g. `validation_report.md`, `quant_plan.json`) | downstream skills know what to expect |
| `source_knowledge` | list of repo-root-relative upstream Quark paths | traceability when upstream changes; checked by `quark-torch-doc-drift-check` |

No other frontmatter keys are added. `compatibility`, `license`, `metadata`, etc. are not part of this project's contract. The format is governed by **prose**, not by a JSON schema in `shared/contracts/` — that directory holds artifact contracts, not skill-file contracts.

## Length budgets

- `SKILL.md` body ≤ **315 lines**. Spill detail into sibling files (this `references/` directory, `agents/`, `scripts/`).
- `description` ≤ **100 words**. Loaded into baseline context every session — every word competes for space against other skills.
- References one level deep. Do not chain `references/foo.md` → `references/sub/bar.md`; partial reads truncate.

## Required sections

In order (additional sections allowed afterward — `Examples`, `Rules`, `Notes`, `Anti-Patterns` are common):

1. `## Purpose` — one paragraph describing the stable responsibility boundary.
2. `## Inputs` — required + optional inputs, including artifacts produced by upstream skills.
3. `## Outputs` — primary artifact name, side effects. Reference the schema in `shared/contracts/` if applicable.
4. `## Interaction Flow` — five stages from `docs/agent_skills/interaction-contract.md`: Intake → Route → Plan → Confirm → Execute or Summarize.
5. `## Recovery` — what to do when the skill cannot continue. Must return blocking reason, missing precondition, smallest unblocking action.

## Source knowledge boundary

From `docs/agent_skills/governance.md` Upstream Validation Boundary: the surrounding Quark repo is the only accepted upstream. `source_knowledge` paths must be:

- Repo-root-relative (no `/`, no `http://`, no `..`).
- Resolvable in the deployed Quark host repo. In this standalone skills repo many upstream paths (`docs/source/install.rst`, `quark/torch/...`) do not exist locally — `validate_skill.py` warns rather than fails for paths that match the upstream directory pattern.
- Authoritative for the behavior the skill describes — minimal but sufficient. `quark-torch-doc-drift-check` re-verifies these references on every governance run.

## Entry stub rule

User-facing skills (most L1 atomic, all L2 workflows, all L3 recipes) get a thin stub at `.claude/skills/<name>/SKILL.md` that Claude Code auto-discovers:

```markdown
---
name: <skill-name>
description: <copy of the description from the implementation>
---

Read and follow the instructions in `.claude/skills-impl/<layer>/<name>/SKILL.md`.
```

Internal skills do **not** get a stub — they are invoked by other skills (e.g., `quark-torch-router` consumes `quark-workspace-validate`) or by the `/skill` command. All five `meta/` skills are internal: `quark-torch-skill-sync`, `quark-torch-doc-drift-check`, `quark-torch-eval-runner`, `quark-skill-creator`, and any future governance addition.

## Artifact rule

The eight canonical artifacts are listed in `docs/agent_skills/artifact-contracts.md` — `session_context.json`, `env_context.json`, `workspace_context.json`, `pytorch_install_result.json`, `quark_install_result.json`, `model_analysis.json`, `quant_plan.json`, `run_manifest.yaml`, plus the per-producer `validation_report.md`. Each has exactly one producer. A new skill that needs a new artifact must:

1. Add a JSON schema under `.claude/skills-impl/shared/contracts/<name>.schema.json`.
2. Add a producer/consumer entry in `docs/agent_skills/artifact-contracts.md`.
3. Get review — this changes the inter-skill contract surface, not just one skill.

`quark-skill-creator` stops the scaffold and surfaces this as a governance question rather than letting authors invent artifacts unilaterally.
