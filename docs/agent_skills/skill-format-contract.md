# Skill Format Contract

Every `SKILL.md` must follow the template at `.claude/skills-impl/shared/templates/skill-template.md`. This contract explains what is checked and why.

## Frontmatter (required fields)

| Field | Why |
|-------|-----|
| `name` | Used as the skill identifier across routing, logging, and evaluation. MUST carry the backend infix matching its `backend`/`<scope>/` (e.g. `quark-torch-*`, `quark-onnx-*`, or bare `quark-*` for shared) |
| `description` | The trigger surface — Claude's harness picks a skill by prompt-matching this field. Backend-specific skills MUST include unambiguous backend keywords (torch: `PyTorch` / `HuggingFace safetensors` / `transformers`; onnx: `.onnx` / `onnxruntime`) so harness routing is deterministic |
| `layer` | Must be one of `l0-foundation`, `l1-atomic`, `l2-workflows`, `l3-recipes`, `meta`. Determines what a skill is allowed to depend on (see `architecture.md`) |
| `backend` | Must be one of `shared`, `torch`, `onnx`. Must equal the parent `<scope>/` directory in the path. Drives CI lint that prevents `shared` skills from importing torch-only or onnx-only modules, and prevents `torch` meta skills from referencing onnx upstream paths (and vice versa) |
| `primary_artifact` | Names the main output artifact so downstream skills know what to expect |
| `source_knowledge` | Links to upstream Quark docs or source files this skill is derived from. Keeps skills traceable when upstream changes |

## Required Sections

| Section | Purpose |
|---------|---------|
| `## Purpose` | One paragraph — what this skill does and why it exists as a separate unit |
| `## Inputs` | What the skill needs before it can run (artifacts, user info, environment facts) |
| `## Outputs` | What the skill produces (artifacts, reports, side effects) |
| `## Interaction Flow` | The steps this skill walks through, following the five-stage interaction contract |
| `## Recovery` | What to do when the skill fails — common errors, fallback actions |

## Length Budgets

Skills load into Claude's context, so two hard caps apply:

- **`SKILL.md` body ≤ 315 lines.** Spill detail into sibling `references/`, `agents/`, or `scripts/` files. Caps the per-skill load cost.
- **`description` ≤ 100 words.** Loaded into baseline context every session, so every word competes against other skills' triggers.

Both caps are enforced as **blocking** by `.claude/skills-impl/meta/shared/quark-skill-creator/scripts/validate_skill.py` and reiterated in `CONTRIBUTING.md`.

## Placeholder rule

`primary_artifact` must be a concrete filename (e.g. `validation_report.md`, `quant_plan.json`). Any value containing `<...>` angle brackets, `<TBD>`, or bare `TODO` is a blocking failure — these tokens are unresolved placeholders that should not ship.

## Validation

`validate_skill.py` performs the mechanical pass (frontmatter, length, sections, placeholder, source_knowledge resolvability). Authors should run it before every commit and capture the resulting report (see `quark-skill-creator` Phase 5). Contractual / semantic review is a separate sub-agent pass — see `.claude/skills-impl/meta/shared/quark-skill-creator/agents/reviewer.md`.
