---
name: quark-onnx-eval-runner
description: >
  Manually verify that the Quark ONNX skill family behaves correctly across the four contract categories
  (routing, planning, artifact, recovery). Use when maintainers need to confirm that ONNX routing, planning,
  artifact generation, or error recovery skills still work as expected. Trigger for "verify the ONNX skills",
  "smoke-test ONNX routing", "check ONNX skill behavior", or before tagging a release that touches `quark-onnx-*`
  skills. This is a governance tool for skill maintainers, not for end users running ONNX model accuracy
  evaluation — for the latter use the upstream Quark ONNX evaluation tooling.
layer: meta
backend: onnx
primary_artifact: validation_report.md
source_knowledge:
  - examples/onnx/yolo_quantization/quantize_yolo.py
  - examples/onnx/model_support.md
  - docs/source/onnx/basic_usage_onnx.rst
  - docs/source/onnx/user_guide_config_description.rst
  - docs/source/onnx/user_guide_auto_search_pro.rst
  - quark/onnx/quantization/config/custom_config.py
---

# quark-onnx-eval-runner

## Purpose

Walk a maintainer through manual verification of the ONNX skill family across the four contract
categories: **routing**, **planning**, **artifact**, and **recovery**. Run this after modifying any
`quark-onnx-*` skill, after a Quark ONNX upgrade, or before tagging a release.

## Inputs

- The current ONNX skill files under `.claude/skills-impl/{l1-atomic,l2-workflows,l3-recipes}/onnx/`
- The entry stubs under `.claude/skills/quark-onnx-*`
- The contract schemas under `.claude/skills-impl/shared/contracts/`
- The example prompts under `examples/agent_skills/prompts/` (add ONNX-specific cases as the
  catalog grows)

## Outputs: validation_report.md

A markdown report recording per-category pass/fail and concrete evidence for each finding.

Schema: [`validation_report.schema.json`](../../../shared/contracts/validation_report.schema.json)

```markdown
# ONNX Skill Verification Report

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

For each category below, run **at least one case** and record the result in the report. As ONNX
prompts are not yet enumerated in `examples/agent_skills/prompts/`, the cases below double as the
seed catalog — add more as the ONNX skill set grows.

### 1. Routing

**Goal:** verify that `quark-onnx-router` (and Claude's auto-routing via the descriptions in
`.claude/skills/quark-onnx-*`) maps natural-language ONNX goals to the correct downstream skill
and never silently routes ONNX requests through a torch skill.

**Manual procedure:**

1. Pick a user-style prompt that names a `.onnx` artifact or ONNX-specific vocabulary.
2. In a fresh Claude Code session at the Quark repo root, paste the prompt.
3. Observe which skill Claude invokes first.
4. Compare against the expected target skill. Examples of expected mappings:
   - "Quantize my `./models/yolov8n.onnx` to XINT8 for AMD NPU CNN" → `quark-onnx-ptq`
     (which loads `quark-onnx-ptq-workflow`)
   - "Run AutoSearchPro on this `.onnx` with the `XINT8_SEARCH` preset" → `quark-onnx-autosearch-pro`
   - "Analyze this `.onnx` — what opset is it, is it NPU-compatible, is it already QDQ?" →
     `quark-onnx-model-intake`
   - "Validate my quantized `model.onnx` — did QDQ insertion happen, are the non-quantized
     initializers byte-identical?" → `quark-onnx-result-validator`
   - "`onnxruntime-gpu` import fails, `CUDAExecutionProvider` not in providers list" →
     `quark-onnx-install` (or `quark-onnx-debug` if the user already attempted install)
   - "`quantize_static` failed with custom-op library load failure for `BFPQuantizeDequantize`"
     → `quark-onnx-debug`
   - "Is `onnxruntime-rocm` installed correctly? Show me the install matrix" → `quark-onnx-install`

5. **Cross-backend guard:** also run one **negative** prompt that mentions a `.onnx` path and
   confirm Claude does **not** route to `quark-torch-*` (e.g., "quantize `./models/foo.onnx` with
   FP8" must not land on `quark-torch-ptq`).

**Pass criteria:** the first skill invoked matches the expected target, and no ONNX prompt is
routed to a torch skill.

### 2. Planning

**Goal:** verify that `quark-onnx-quant-plan` produces internally consistent plans for typical
ONNX inputs and that the deployment-target gates are respected.

**Manual procedure:**

1. Construct (or take from a prior session) a `model_analysis.json` produced by
   `quark-onnx-model-intake` for a representative model (e.g., YOLOv8n exported at opset 17,
   Conv-heavy, 6.2 MB inline).
2. Hand it to `quark-onnx-quant-plan` with a target preset (e.g., `XINT8`) and a deployment
   target (e.g., `AMD NPU CNN`).
3. Inspect the produced `quant_plan.json` for:
   - `preset` matches the requested preset
   - `activation_spec` and `weight_spec` are consistent with the preset (e.g., both `XInt8Spec`
     for `XINT8`)
   - `EnableNPUCnn=True` is set when the target is AMD NPU CNN
   - `use_external_data_format` is `True` iff the model is >2 GB
   - `algo_config` is a non-empty list when CLE or AdaRound was requested or recommended
   - `exclude` is a list (may be empty) and never contains an op the plan also quantizes
   - `requires_confirmation` is set when the plan deviates from preset defaults
   - **Negative gate:** the plan refuses incompatible combos (e.g., `BFP16` + `AMD NPU CNN`)
     rather than silently downgrading

**Pass criteria:** the plan validates against `quant_plan.schema.json`, contains no internal
contradictions, and explicitly rejects unsupported deployment-target / preset combinations.

### 3. Artifact

**Goal:** verify that ONNX workflow output artifacts conform to their JSON schemas and that the
generated standalone script + manifest produced by `quark-onnx-ptq-workflow` agree with each
other.

**Manual procedure:**

1. Take the `quant_plan.json` from the planning case.
2. Run `quark-onnx-ptq-workflow` to produce a `run_manifest.yaml` and the standalone
   `<name>_ptq.py` script in the user's working directory.
3. Validate the manifest against `.claude/skills-impl/shared/contracts/run_manifest.schema.json`
   (use any JSON-schema validator, e.g., the `jsonschema` Python package).
4. Spot-check that:
   - All required fields are present in the manifest
   - The manifest's `command` references the generated script path and uses `python3`
   - The manifest's resolved `QConfig` matches the plan's `preset` / `algo_config` /
     `EnableNPUCnn` / `use_external_data_format`
   - The generated script imports only from `quark.onnx` and the standard ORT calibration API
     (no editing of upstream `examples/onnx/` or `quark/onnx/` files)
   - Output paths reflect any `.onnx_data` sidecar when external data is enabled

**Pass criteria:** schema validation passes; the manifest's resolved config matches the plan;
the generated script is self-contained in the user's working directory.

### 4. Recovery

**Goal:** verify that `quark-onnx-debug` correctly diagnoses known ONNX-side error patterns and
that handoffs to `quark-onnx-install` happen for runtime/provider issues.

**Manual procedure:**

1. Pick a known ONNX error scenario. Examples:
   - `RuntimeError: CUDAExecutionProvider not in available providers` after installing
     `onnxruntime` (CPU build) instead of `onnxruntime-gpu`.
   - Custom-op library load failure for `BFPQuantizeDequantize` or `MXQuantizeDequantize`
     (missing C++ build, ABI mismatch).
   - `model.onnx` >2 GB and the run fails with "external data not found" because
     `use_external_data_format` was not set or the sibling `.onnx_data` was not staged.
   - OOM during calibration on a vision model with `num_calib_data=1000` and `batch_size=4`.
   - AdaRound divergence with default learning rate.
   - NPU CNN run fails because activation scales are not power-of-two.
2. Present the error (full traceback) to `quark-onnx-debug`.
3. Verify the diagnosis:
   - Root cause is correctly identified
   - A concrete fix command or config change is provided
   - For runtime/provider issues, the skill explicitly hands off to `quark-onnx-install`
     instead of silently swapping execution providers
   - For OOM, the skill suggests the documented ladder: reduce `num_calib_data` → drop
     `batch_size` to 1 → move calibration to CPU (`OptimDevice="cpu"`)
   - For custom-op load failures, the skill names the expected library path / build step
     rather than recommending the user ignore the op

**Pass criteria:** the diagnosis names the actual root cause, suggests a fix that would actually
work, and respects the "never silently fall back to CPU" rule from `quark-onnx-ptq-workflow`.

## Rules

- **Run at least one case per category before any ONNX release** — even small wording changes
  in preset names or custom-op names can shift routing or break generated scripts.
- **A failing case means the skill is broken, not the procedure** — investigate the skill first.
  Only update the expected behavior if the skill change was intentional.
- **Report all results**, not just failures. A clean run is positive evidence and worth recording.
- **Document new cases inline in the report.** As the ONNX skill set grows (e.g. new deployment
  target, new preset, new AutoSearchPro preset), add cases that cover the new triggers.
- **Cross-backend isolation is part of the protocol.** Every routing case must include a guard
  that confirms ONNX requests stay on `quark-onnx-*` skills.

## Interaction Flow

1. **Select scope:** which categories to verify — all four, or a subset affected by a recent change?
2. **Pick or write cases:** start from the examples above; add ONNX prompt files under
   `examples/agent_skills/prompts/` as the catalog grows.
3. **Run each case manually** following the protocol above in a fresh Claude Code session.
4. **Record results** in `validation_report.md` using the template.
5. **Hand off failures** to `quark-onnx-skill-sync` if they look like upstream drift, to
   `quark-onnx-doc-drift-check` if they look like stale user-facing facts, or directly to the
   affected skill's owner if it's a content bug.

## Recovery

- If a case can't run because a prerequisite artifact is missing (e.g., no `model_analysis.json`
  for the planning case), report the missing producer skill and stop — do not fabricate the input.
- If the ONNX custom-op binaries are not built locally, the recovery case for the custom-op load
  failure may produce a real failure rather than a simulated one — note this distinction in the
  report so future runs don't confuse genuine environment gaps with skill bugs.
- If Claude Code itself is misbehaving (ONNX skill not discovered, stub not loading), that's an
  infrastructure issue separate from skill quality — record it distinctly in the report.
- If an ONNX prompt routes to a torch skill (or vice versa), this is a routing-category failure
  and must block the release — cross-backend mis-routing produces wrong artifacts silently.
