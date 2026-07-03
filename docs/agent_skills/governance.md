# Governance & Validation

How the Quark skill system stays correct and aligned with the surrounding Quark repository. Since this skill system now lives **inside** Quark, the host repo itself is the upstream source of truth — there is no external sibling repo to chase.

This doc covers two layers: the **governance loop** (when and how to react to upstream change) and the **validation boundary** (what counts as upstream truth).

## 1. Governance Loop

### Trigger Events

- Quark documentation changes (e.g., `docs/source/install.rst`)
- Quark CLI, template, or PTQ-script changes (e.g., `examples/torch/language_modeling/llm_ptq/quantize_quark.py`, `quark/torch/quantization/config/template.py`)
- A contract schema changes (`.claude/skills-impl/shared/contracts/`)
- An MVP skill changes

### Loop

1. Run `quark-torch-doc-drift-check` to identify likely documentation or contract drift.
2. If drift is confirmed, run `quark-torch-skill-sync` to classify affected skills and update dependency references.
3. Run `quark-torch-eval-runner` on the four MVP evaluation classes.
4. Publish a `validation_report.md` summarizing pass, fail, and follow-up work.

### Required Outputs

- affected skill list
- changed upstream references
- evaluation classes executed
- blocking failures and next actions

## 2. Upstream Validation Boundary

The surrounding Quark repository is the **only accepted upstream source of truth**. Skill `source_knowledge` entries reference repo-root-relative paths.

### Allowed Upstream Evidence

- Quark docs under `docs/`
- Quark examples under `examples/`
- Quark requirements files (`requirements.txt`, `examples/torch/language_modeling/llm_ptq/requirements.txt`)
- Quark source entry points under `quark/`
- Quark CI/install scripts under `tools/`

### Eval Evidence Rule

Each eval task should point to live Quark facts (installation guides, PTQ scripts, requirements files, configuration templates, evaluation entry points). Evidence should be minimal but sufficient:

- choose files that are authoritative for the behavior under test
- list only the facts that materially support the task
- avoid broad or redundant evidence lists

Suggested mapping for the four MVP eval classes:

- `routing` tasks → Quark PTQ docs and examples
- `planning` tasks → template definitions and PTQ script arguments
- `artifact` tasks → workflow examples and output-related code paths
- `recovery` tasks → real Quark dependency constraints or documented runtime requirements
