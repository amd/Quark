---
name: quark-torch-eval-runner
description: >
  Manually verify that the Quark Agent Skills system behaves correctly across the four contract categories
  (routing, planning, artifact, recovery). Use when maintainers need to confirm that routing, planning,
  artifact generation, or error recovery skills still work as expected. Trigger for "verify the skills",
  "smoke-test routing", "check skill behavior", or before tagging a release. This is a governance tool for
  skill maintainers, not for end users running model evaluation.
layer: meta
primary_artifact: validation_report.md
source_knowledge:
  - examples/torch/language_modeling/llm_ptq/example_quark_torch_llm_ptq.rst
---

# quark-torch-eval-runner

## Purpose

Walk a maintainer through manual verification of the skill system across the four contract categories: routing, planning, artifact, and recovery. Run this after modifying any skill, after a Quark upgrade, or before tagging a release.

## Inputs

- The current skill files under `.claude/skills-impl/`
- The entry stubs under `.claude/skills/`
- The contract schemas under `.claude/skills-impl/shared/contracts/`
- The example prompts under `examples/agent_skills/prompts/`

## Outputs: validation_report.md

A markdown report recording per-category pass/fail and concrete evidence for each finding.

Schema: [`validation_report.schema.json`](../../../shared/contracts/validation_report.schema.json)

```markdown
# Skill Verification Report

## Summary
| Category | Cases Run | Pass | Fail |
|----------|-----------|------|------|
| routing  | N         | N    | 0    |
| planning | N         | N    | 0    |
| artifact | N         | N    | 0    |
| recovery | N         | N    | 0    |

## Failures
### <category> / <case name>
- **Expected**: ...
- **Got**: ...
- **Impact**: ...
- **Fix**: ...
```

## Manual Verification Protocol

For each category below, run **at least one case** and record the result in the report. The example prompts in `examples/agent_skills/prompts/` are good starting cases; add more as the skill set grows.

### 1. Routing

**Goal:** verify that `quark-torch-router` (and Claude's auto-routing via descriptions in `.claude/skills/`) maps natural-language goals to the correct downstream skill.

**Manual procedure:**

1. Pick a user-style prompt (e.g., from `examples/agent_skills/prompts/torch_llm_ptq.md` or a freshly invented one).
2. In a fresh Claude Code session at the Quark repo root, paste the prompt.
3. Observe which skill Claude invokes first.
4. Compare against the expected target skill. Examples of expected mappings:
   - "Quantize Llama-2-7B with FP8 and export to HuggingFace format" → `quark-torch-ptq` (which loads `quark-torch-llm-ptq-workflow`)
   - "What GPU do I have?" → `quark-env-preflight`
   - "pip install amd-quark fails" → `quark-install` or `quark-torch-debug`
   - "Convert my quantized model to GGUF" → `quark-torch-export`

**Pass criteria:** the first skill invoked matches the expected target.

### 2. Planning

**Goal:** verify that `quark-torch-quant-plan` produces internally consistent plans for typical inputs.

**Manual procedure:**

1. Construct (or take from a prior session) a `model_analysis.json` for a representative model (e.g., Qwen3-8B).
2. Hand it to `quark-torch-quant-plan` with a target scheme (e.g., `fp8`).
3. Inspect the produced `quant_plan.json` for:
   - `global_scheme` matches the requested scheme
   - `exclude_layers` is non-empty and includes `lm_head` for LLMs
   - No conflicting options (e.g., `kv_cache_dtype: fp8` requires the FP8 scheme path)
   - `requires_confirmation` is set when the plan deviates from defaults

**Pass criteria:** the plan validates against `quant_plan.schema.json` and contains no internal contradictions.

### 3. Artifact

**Goal:** verify that workflow output artifacts conform to their JSON schemas.

**Manual procedure:**

1. Take the `quant_plan.json` from the planning case.
2. Run `quark-torch-llm-ptq-workflow` to produce a `run_manifest.yaml`.
3. Validate the manifest against `.claude/skills-impl/shared/contracts/run_manifest.schema.json` (use any JSON-schema validator, e.g., the `jsonschema` Python package, or eyeball required fields).
4. Spot-check that:
   - All required fields are present
   - Referenced artifact paths exist or are clearly marked as to-be-produced
   - Export configuration is included (e.g., `export.formats`)

**Pass criteria:** schema validation passes; required fields are present and consistent.

### 4. Recovery

**Goal:** verify that `quark-torch-debug` correctly diagnoses known error patterns.

**Manual procedure:**

1. Pick a known error scenario. Examples:
   - `AttributeError: 'PreTrainedTokenizerFast' object has no attribute 'get_max_length'` (transformers version mismatch)
   - CUDA out-of-memory during a 70B-parameter quantization
   - Missing `lm_head` exclusion causing accuracy collapse
2. Present the error to `quark-torch-debug`.
3. Verify the diagnosis:
   - Root cause is correctly identified
   - A concrete fix command or config change is provided
   - The affected version range or environment condition is stated

**Pass criteria:** the diagnosis names the actual root cause and suggests a fix that would actually work.

## Rules

- **Run at least one case per category before any release** — even small wording changes can shift routing behavior.
- **A failing case means the skill is broken, not the procedure** — investigate the skill first. Only update the expected behavior if the skill change was intentional.
- **Report all results**, not just failures. A clean run is positive evidence and worth recording.
- **Document new cases inline in the report.** As the skill set grows, add cases that cover new triggers.

## Interaction Flow

1. **Select scope:** which categories to verify — all four, or a subset affected by a recent change?
2. **Pick or write cases:** use prompts from `examples/agent_skills/prompts/` or write minimal new ones.
3. **Run each case manually** following the protocol above.
4. **Record results** in `validation_report.md` using the template.
5. **Hand off failures** to `quark-torch-skill-sync` if they look like upstream drift, or directly to the affected skill's owner if it's a content bug.

## Recovery

- If a case can't run because a prerequisite artifact is missing (e.g., no `model_analysis.json` for the planning case), report the missing producer skill and stop — do not fabricate the input.
- If Claude Code itself is misbehaving (skill not discovered, stub not loading), that's an infrastructure issue separate from skill quality — record it distinctly in the report.
