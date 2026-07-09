---
name: quark-torch-skill-sync
description: >
  Detect upstream Quark changes that affect the skill system and classify required updates. Use when Quark docs, CLI flags,
  quantization templates, model support, or source behavior may have drifted from the skill contracts. Trigger for
  "check if skills are up to date", "sync skills with Quark", "has Quark changed", "update skills after Quark upgrade",
  or when debugging reveals a mismatch between skill instructions and actual Quark behavior.
layer: meta
primary_artifact: validation_report.md
source_knowledge:
  - docs/source/install.rst
  - quark/torch/quantization/config/template.py
  - examples/torch/language_modeling/llm_ptq/quantize_quark.py
---

# quark-torch-skill-sync

## Purpose

Audit upstream Quark source against the skill system's assumptions and report which skills or contracts need updates. This is important because Quark is actively developed — new quantization schemes, model architectures, CLI flags, and API changes can silently invalidate skill instructions. A skill that references a removed flag or an outdated version constraint will produce wrong guidance.

## Inputs

- Quark upstream source files
- Existing skill SKILL.md files

## Outputs: validation_report.md

Lists which skills or contracts need updates after a Quark upstream change.

Schema: [`validation_report.schema.json`](../../../shared/contracts/validation_report.schema.json)

```markdown
# Skill Sync Report

## What to Check

### Source Knowledge Dependencies

Each skill declares `source_knowledge` files. Check that these files still exist and that the skill's content matches the current source:

| Skill | Source Files to Check |
|-------|----------------------|
| `quark-env-preflight` | `docs/source/install.rst`, `pyproject.toml` |
| `quark-torch-install` | `tools/ci/install_torch.sh`, `docs/source/install.rst` |
| `quark-install` | `docs/source/install.rst`, `requirements.txt` |
| `quark-torch-model-intake` | `quark/torch/utils/llm/model_preparation.py`, `quark/torch/quantization/config/template.py` |
| `quark-torch-quant-plan` | `quark/torch/quantization/config/template.py`, `examples/torch/language_modeling/llm_ptq/README.md` |
| `quark-torch-export` | `quark/torch/export/api.py`, `examples/contrib/llm_eval/llm_eval.py` |
| `quark-torch-debug` | `quark/torch/utils/llm/compatibility.py`, `docs/source/install.rst` |
| `quark-torch-llm-ptq-workflow` | `examples/torch/language_modeling/llm_ptq/quantize_quark.py` |
| `quark-torch-llm-ptq-eval` | `examples/torch/language_modeling/llm_ptq/quantize_quark.py`, `examples/torch/language_modeling/llm_ptq/example_quark_torch_llm_ptq.rst` |

### Key Facts to Verify

1. **Supported Python versions** — check `pyproject.toml` `requires-python` field
2. **PyTorch version matrix** — check `tools/ci/install_torch.sh` for supported combinations
3. **Quantization schemes** — check `quark/torch/quantization/config/template.py` for the scheme list
4. **Supported model architectures** — check `quark/torch/utils/llm/model_preparation.py` for the model template list
5. **CLI arguments** — check `quantize_quark.py` argparse definitions for new/removed/changed flags
6. **Export formats** — check `quark/torch/export/api.py` for supported formats
7. **Compatibility checks** — check `quark/torch/utils/llm/compatibility.py` for transformer version constraints

### Drift Classification

- **Mechanical drift**: A version number changed, a flag was renamed, a new scheme was added. Fix is straightforward — update the affected table or list in the skill.
- **Semantic drift**: A workflow pattern changed, a concept was redefined, a behavior assumption is no longer true. Requires careful skill rewriting.
- **Breaking drift**: A skill references something that was removed entirely. The skill will produce wrong guidance until fixed.

## Audit Process

1. **Discover dependencies**: Read `source_knowledge` from each skill's frontmatter.
2. **Check for changes**: Compare the current source files against what the skills assume. Look for:
   - New entries not mentioned in skills (e.g., new quantization scheme)
   - Removed entries still mentioned in skills (e.g., deprecated flag)
   - Changed values (e.g., version constraint updated)
3. **Classify each drift**: mechanical, semantic, or breaking
4. **Report**: Produce a `validation_report.md` with the findings

## Rules

- **Do not auto-fix breaking drift** — report it and require human confirmation before applying changes.
- **Mechanical drift can be flagged for batch update** — version bumps and new scheme additions are safe to apply after review.
- **If no prior baseline exists**, run calibration mode: record the current state of all source files as the baseline for future comparisons.

## Summary
- Checked: 7 skills, 12 source files
- Mechanical drift: 3 findings
- Semantic drift: 0 findings
- Breaking drift: 1 finding

## Breaking Drift
### quark-install: Python version constraint changed
- **Skill says**: Python 3.11-3.13
- **Source says**: Python 3.11-3.14 (pyproject.toml updated)
- **Impact**: Users on Python 3.14 will be told it's unsupported when it is now supported
- **Fix**: Update Python version table in quark-install SKILL.md

## Mechanical Drift
### quark-torch-quant-plan: New scheme added
- **Source**: template.py now includes `nvfp4` scheme
- **Impact**: Users asking about NVFP4 won't see it in the scheme list
- **Fix**: Add `nvfp4` row to the scheme table

### quark-torch-model-intake: New model template
- **Source**: model_preparation.py now supports `phi4`
- **Fix**: Add `phi4` to the supported models table
```

## Interaction Flow

1. **Collect**: Identify all source knowledge dependencies across skills.
2. **Audit**: Check each source file for changes relevant to skill content.
3. **Classify**: Tag each finding as mechanical, semantic, or breaking.
4. **Report**: Present findings in a structured report.
5. **Confirm**: Get approval before applying any fixes.

## Recovery

- If a source file no longer exists (was moved or deleted), this is breaking drift. Report the missing file and suggest where the information might have moved to.
- If the upstream repository is not accessible, report the blocked audit and which skills could not be verified.
