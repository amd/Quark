---
name: quark-torch-doc-drift-check
description: >
  Compare skill contracts and guidance against current Quark documentation and source entry points. Use when maintainers
  need to verify that installation docs, artifact schemas, PTQ planning assumptions, CLI flags, or supported model lists
  still match upstream Quark reality. Trigger for "check doc drift", "are skills still accurate", "verify against Quark docs",
  or after a Quark release when upstream documentation may have changed.
layer: meta
primary_artifact: validation_report.md
source_knowledge:
  - docs/source/install.rst
  - examples/torch/language_modeling/llm_ptq/README.md
  - quark/torch/quantization/config/template.py
---

# quark-torch-doc-drift-check

## Purpose

Provide a lightweight governance check focused on documentation accuracy. While `quark-torch-skill-sync` audits source code changes broadly, this skill focuses specifically on whether the skills' user-facing guidance (install commands, scheme lists, model names, CLI flags) still matches what Quark's own docs and source say. Think of this as a fact-checker for the skill system.

## Inputs

- Quark upstream docs and source code (read-only)
- Current skill SKILL.md files

## Outputs: validation_report.md

Lists where current skill guidance has drifted from upstream Quark docs and source.

Schema: [`validation_report.schema.json`](../../../shared/contracts/validation_report.schema.json)

````markdown
# Documentation Drift Report

## What to Check

### 1. CLI Flags Still Exist
The skills reference specific `quantize_quark.py` flags. Verify they are still in the argparse definitions:

**Critical flags to verify:**
- `--quant_scheme` — and that all 21 scheme names listed in `quark-torch-quant-plan` are still valid
- `--quant_algo` — and that all algorithm names are still valid
- `--model_export` — and that `hf_format`, `onnx`, `gguf` are still the options
- `--kv_cache_dtype` — still accepts `fp8`
- `--multi_gpu` — still accepts `auto` and `balanced`
- `--file2file_quantization` — still exists for ultra-large models
- `--trust_remote_code` — still defaults to True

### 2. Model Templates Still Exist
Check that `quark/torch/quantization/config/template.py` still contains all model templates listed in `quark-torch-model-intake`:

```python
# Key check: model_type values in the template registry
# Expected: llama, mistral, qwen, qwen2, qwen3, deepseek, phi, phi3, etc.
```

### 3. Contract Fields Match Workflow Needs

Verify that the JSON schemas in `.claude/skills-impl/shared/contracts/` match what the workflow actually produces:

- `session_context.schema.json` — does the schema match the `session_context.json` structure skills emit?
- `quant_plan.schema.json` — does it include all fields from `quark-torch-quant-plan`'s decision table?
- `run_manifest.schema.json` — does it match `quark-torch-llm-ptq-workflow`'s manifest output?

### 4. Install Docs Match

Compare `quark-torch-install`'s PyTorch version matrix with:

- `tools/ci/install_torch.sh` PyTorch version matrix
- `docs/source/install.rst` installation instructions

Compare `quark-install`'s dependency tables with:

- `pyproject.toml` Python version constraint
- `requirements.txt` core dependency versions
- `docs/source/install.rst` installation instructions

### 5. Examples Don't Contradict

Check that the example prompts in `examples/agent_skills/prompts/` would still route correctly with current skill descriptions. Verify skill output formats against the schemas in `.claude/skills-impl/shared/contracts/` directly.

## Checking Process

```bash
# Extract current scheme list from source
grep -oP "register_scheme\(['\"](\w+)['\"]" quark/torch/quantization/config/template.py

# Extract CLI flags from quantize_quark.py
grep -oP "add_argument\(['\"]--(\w+)['\"]" examples/torch/language_modeling/llm_ptq/quantize_quark.py

# Extract model templates
grep -oP "model_type=['\"](\w+)['\"]" quark/torch/quantization/config/template.py

# Check Python version constraint
grep "requires-python" pyproject.toml
```

## Rules

- **This is a read-only check** — report findings but do not modify skills. Modifications go through `quark-torch-skill-sync`.
- **Focus on user-facing facts** — version numbers, command syntax, and option lists matter most because users will copy-paste them.
- **Flag removed items as critical** — a skill that recommends a deleted flag will cause immediate user failures.
- **Flag new items as informational** — a new scheme that is not yet documented in skills is a gap, not an error.

## Checks Performed

- CLI flags: 15 checked, 14 match, 1 drift
- Scheme list: 21 checked, 21 match
- Model templates: 36 checked, 34 match, 2 new (not in skills)
- Install versions: 8 checked, 7 match, 1 drift

## Findings

### CRITICAL: CLI flag renamed

- `--custom_mode` in skills → now `--export_mode` in source
- **Impact**: Copy-paste commands in quark-torch-export will fail
- **Action**: Update quark-torch-export SKILL.md

### INFO: New model templates not in skills

- `phi4` and `command_r` added to template.py
- **Impact**: Users asking about these models won't get model-specific guidance
- **Action**: Add to quark-torch-model-intake supported models table

### INFO: PyTorch 2.10.0 added for CUDA 13.0

- Skills already list 2.10.0 as supported — no action needed

````

## Interaction Flow

1. **Select scope**: Full check or focused on specific skills/facts?
2. **Run checks**: Compare skill content against upstream source and docs.
3. **Classify findings**: Critical (broken guidance), warning (outdated), info (gap in coverage).
4. **Report**: Produce the drift report.
5. **Hand off**: If fixes are needed, route to `quark-torch-skill-sync` for the actual updates.

## Recovery

- If upstream docs are inaccessible, report which checks could not be performed.
- If drift is detected, hand off to `quark-torch-skill-sync` with the specific findings so it can apply targeted fixes.
