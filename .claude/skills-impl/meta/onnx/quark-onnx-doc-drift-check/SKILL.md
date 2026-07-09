---
name: quark-onnx-doc-drift-check
description: >
  Compare ONNX skill contracts and guidance against current Quark ONNX documentation and source entry points. Use when
  maintainers need to verify that ONNX install docs, custom-op registry, QConfig fields, preset and calibration lists,
  AutoSearchPro presets, deployment-target gates, or example-script invocations still match upstream Quark ONNX reality.
  Trigger for "check ONNX doc drift", "are ONNX skills still accurate", "verify against Quark ONNX docs",
  "fact-check quark-onnx-* skills", or after a Quark release when upstream ONNX documentation may have changed.
layer: meta
backend: onnx
primary_artifact: validation_report.md
source_knowledge:
  - docs/source/install.rst
  - docs/source/onnx/basic_usage_onnx.rst
  - docs/source/onnx/user_guide_config_description.rst
  - docs/source/onnx/appendix_full_quant_config_features.rst
  - docs/source/onnx/user_guide_auto_search_pro.rst
  - docs/source/onnx/gpu_usage_guide.rst
  - docs/source/onnx/onnx_examples.rst
  - quark/onnx/quantization/config/custom_config.py
  - quark/onnx/quantization/config/algorithm.py
  - quark/onnx/quantization/config/config.py
  - quark/onnx/calibration/methods.py
  - quark/onnx/operators/custom_ops/__init__.py
  - quark/onnx/quantization/auto_search/auto_search_pro.py
  - examples/onnx/yolo_quantization/quantize_yolo.py
  - examples/onnx/model_support.md
  - tools/ci/install_onnxruntime.sh
---

# quark-onnx-doc-drift-check

## Purpose

Provide a lightweight ONNX-side governance check focused on documentation accuracy. While
`quark-onnx-skill-sync` audits source-code changes broadly and applies the actual updates, this
skill focuses specifically on whether the ONNX skills' **user-facing guidance** — preset names,
calibration methods, custom-op names, QConfig field names, AutoSearchPro presets, ORT install
matrices, example-script invocations — still matches what Quark's own ONNX docs and source say.
Think of this as a fact-checker for the `quark-onnx-*` skill family.

## Inputs

- Quark upstream ONNX docs and source code (read-only)
- Current ONNX skill `SKILL.md` files under `.claude/skills-impl/{l1-atomic,l2-workflows,l3-recipes}/onnx/`
- ONNX-side shared schemas under `.claude/skills-impl/shared/contracts/`

## Outputs: validation_report.md

Lists where current ONNX skill guidance has drifted from upstream Quark ONNX docs and source.

Schema: [`validation_report.schema.json`](../../../shared/contracts/validation_report.schema.json)

````markdown
# ONNX Documentation Drift Report

## What to Check

### 1. Quantization Presets Still Exist

Skills reference specific preset names in their decision tables. Verify each is still defined in
`quark/onnx/quantization/config/custom_config.py`:

**Critical presets to verify** (referenced by `quark-onnx-quant-plan`, `quark-onnx-ptq-workflow`,
`quark-onnx-autosearch-pro`):

- `XINT8` (and the `EnableNPUCnn=True` companion flag)
- `A8W8`, `A16W8`
- `BF16`, `BFP16`
- `MX*` family (e.g. `MXINT8`, `MXFP8_E4M3`, `MXFP8_E5M2`, `MXFP4`, `MXFP6_E3M2`, `MXFP6_E2M3`)
- Weights-only INT4 path (e.g. `MatMulNBits`)

### 2. Calibration Methods Still Exist

Check that `quark/onnx/calibration/methods.py` (and `calibrators.py`) still contain every
calibration method named in `quark-onnx-quant-plan`:

```python
# Expected calibration methods
# MinMax, Percentile, Entropy, Distribution, MinMSE, LayerwisePercentile
```

### 3. Algorithm Configs Still Exist

Verify that `quark/onnx/quantization/config/algorithm.py` still exposes every algorithm config
class referenced by the skills:

- `CLEConfig`, `BiasCorrectionConfig`
- `AdaRoundConfig`, `AdaQuantConfig`, `FastFinetuneConfig`
- `SmoothQuantConfig`, `QuaRotConfig`, `GPTQConfig`

### 4. QConfig Fields Still Exist

The `QConfig` surface is copied into skill decision tables, generated scripts, and the workflow
example. Verify every field is still present in `config.py` / `custom_config.py`:

- `global_config`, `algo_config`, `exclude`
- `EnableNPUCnn`, `EnableNPUTransformer`
- `use_external_data_format`
- `calibration_method`, `OptimDevice`

### 5. Custom-Op Registry Still Matches

Skills reference Quark ONNX custom ops by name in error messages, debug guidance, and the
`com.amd.quark` opset domain check used by `quark-onnx-result-validator`. Verify each is still
registered in `quark/onnx/operators/custom_ops/__init__.py`:

- `BFPQuantizeDequantize`
- `MXQuantizeDequantize`
- `Extended*` family
- The `com.amd.quark` domain string

### 6. AutoSearchPro Presets Still Exist

Check that the `quark-onnx-autosearch-pro` recipe's preset names still exist in
`quark/onnx/quantization/auto_search/auto_search_pro.py`:

- `ADVANCED_SEARCH`, `XINT8_SEARCH`, `A8W8_SEARCH`, `A16W8_SEARCH`

### 7. ONNX Runtime Install Matrix Matches

Compare `quark-onnx-install`'s ORT package/version matrix with:

- `tools/ci/install_onnxruntime.sh` (the authoritative matrix)
- `docs/source/install.rst` installation instructions
- `docs/source/onnx/gpu_usage_guide.rst` (GPU/EP guidance)
- `requirements.txt` core ONNX deps (`onnx`, `onnxslim`, `onnxscript`)

Verify the supported `(accelerator, EP, package, version)` tuples cited by the skill still match.
Common EP names that must be consistent: `CPUExecutionProvider`, `CUDAExecutionProvider`,
`ROCMExecutionProvider`, `VitisAIExecutionProvider`.

### 8. Deployment-Target Gates Match

`quark-onnx-ptq-workflow` lists a deployment-target compatibility table (CPU / CUDA / ROCm /
AMD NPU CNN / AMD NPU Transformer). Verify the gating logic still matches what
`custom_config.py` accepts (e.g. BFP16 forbidden on NPU CNN, XINT8 + `EnableNPUCnn=True`).

### 9. Contract Fields Match Workflow Needs

Verify that the JSON schemas in `.claude/skills-impl/shared/contracts/` match what the ONNX
workflow actually produces:

- `session_context.schema.json` — matches the ONNX `session_context.json` with `constraints.backend = "onnx"`?
- `quant_plan.schema.json` — includes ONNX-specific fields used by `quark-onnx-quant-plan` (preset, calibration_method, algo_config, EnableNPUCnn, use_external_data_format, exclude_nodes)?
- `run_manifest.schema.json` — matches `quark-onnx-ptq-workflow`'s manifest (generated script path, exact `python3` command, resolved `QConfig`)?

### 10. Examples Don't Contradict

Check that the example invocation in `examples/onnx/yolo_quantization/quantize_yolo.py` and the
walkthrough in `quark-onnx-ptq-workflow/example-xint8-yolov8n.md` still agree on:

- the `QConfig` field names used in the worked example,
- the `ModelQuantizer(...).quantize_model(...)` argument order,
- the imports from `quark.onnx`,
- the calibration data reader pattern from `tutorials/onnx/ryzen_ai/yolov8/`.

### 11. >2 GB External-Data Rule

Verify the `>2 GB → use_external_data_format=True` rule cited by `quark-onnx-model-intake` and
the workflow still matches what `quark/onnx/quantization/api.py` / `input_check.py` enforce.

## Checking Process

```bash
# Extract preset list from custom_config.py
grep -oP "(?:class|^)\s*([A-Z][A-Za-z0-9_]+(?:Spec|Config))\b" \
  quark/onnx/quantization/config/custom_config.py

# Extract algorithm config classes
grep -oP "^class\s+([A-Za-z0-9_]+Config)\b" \
  quark/onnx/quantization/config/algorithm.py

# Extract calibration method names
grep -oP "(?:class|name\s*=\s*['\"])([A-Z][A-Za-z]+)\b" \
  quark/onnx/calibration/methods.py

# Extract custom-op registrations
grep -oP "register.*['\"]([A-Za-z0-9_]+)['\"]" \
  quark/onnx/operators/custom_ops/__init__.py

# Extract AutoSearchPro presets
grep -oP "['\"]([A-Z_]+_SEARCH)['\"]" \
  quark/onnx/quantization/auto_search/auto_search_pro.py

# Extract ORT install matrix
grep -oP "onnxruntime[a-z-]*==?[0-9.]+" tools/ci/install_onnxruntime.sh

# Check Python version constraint
grep "requires-python" pyproject.toml
```

## Rules

- **This is a read-only check** — report findings but do not modify skills. Modifications go through
  `quark-onnx-skill-sync`.
- **Focus on user-facing facts** — preset names, calibration-method names, custom-op names, QConfig
  field names, ORT versions, and example-script signatures matter most because users will copy-paste
  them into scripts the workflow will run.
- **Flag removed items as critical** — a skill that recommends a deleted preset, calibration method,
  custom op, or QConfig field will cause immediate user failures at script generation or runtime.
- **Flag new items as informational** — a new preset, calibration method, or AutoSearchPro preset
  that is not yet documented in skills is a coverage gap, not an error.
- **Stay in the ONNX scope** — never touch `quark-torch-*` skills. Cross-cut findings (e.g. a shared
  `validation_report.schema.json` field) are surfaced but deferred to the torch maintainer.

## Checks Performed

- Quantization presets: 11 checked, 10 match, 1 new (not in skills)
- Calibration methods: 6 checked, 6 match
- Algorithm configs: 8 checked, 7 match, 1 drift
- QConfig fields: 8 checked, 8 match
- Custom-op registry: 4 checked, 4 match
- AutoSearchPro presets: 4 checked, 4 match
- ORT install matrix: 5 rows checked, 4 match, 1 drift
- Deployment-target gates: 5 checked, 5 match

## Findings

### CRITICAL: Algorithm config renamed

- `AdaRoundConfig` in skills → now `AdaroundConfig` in `algorithm.py` (case change)
- **Impact**: Generated scripts in `quark-onnx-ptq-workflow` and `quark-onnx-autosearch-pro` will
  fail at `from quark.onnx import AdaRoundConfig` (ImportError)
- **Action**: Update the import + plan tables in `quark-onnx-ptq-workflow` and
  `quark-onnx-autosearch-pro` SKILL.md, then re-validate the YOLOv8 worked example

### CRITICAL: ORT package pinning drift

- Skills recommend `onnxruntime-rocm==1.19.2`
- `tools/ci/install_onnxruntime.sh` now pins `onnxruntime-rocm==1.20.0`
- **Impact**: Copy-paste install commands in `quark-onnx-install` will downgrade users on a
  freshly installed environment
- **Action**: Update the ROCm row in `quark-onnx-install`'s ORT install matrix

### INFO: New preset not in skills

- `MXFP6_E3M2` added to `custom_config.py`
- **Impact**: Users asking about MXFP6 won't see it in `quark-onnx-quant-plan`'s preset table
- **Action**: Add `MXFP6_E3M2` row to the preset table; consider whether AutoSearchPro should expose
  a matching `MXFP6_SEARCH`

### INFO: New calibration parameter

- `LayerwisePercentile` now accepts a `percentile_per_layer` mapping
- **Impact**: Skill mentions `LayerwisePercentile` but not the new parameter
- **Action**: Optional — extend `quark-onnx-quant-plan` calibration row when a real user request lands

````

## Interaction Flow

1. **Select scope**: Full check or focused on specific ONNX skills / facts (e.g. only AutoSearchPro presets)?
2. **Run checks**: Compare ONNX skill content against upstream source and docs using the `grep`-driven extraction patterns above.
3. **Classify findings**: Critical (broken guidance), warning (outdated), info (gap in coverage).
4. **Report**: Produce the drift report as `validation_report.md`.
5. **Hand off**: If fixes are needed, route to `quark-onnx-skill-sync` for the actual updates — never patch skills from this check.

## Recovery

- If upstream ONNX docs are inaccessible, report which checks could not be performed and which ONNX skills are therefore unverified.
- If `quark/onnx/operators/custom_ops/build_custom_ops.py` cannot be parsed (e.g. moved or renamed), mark `quark-onnx-install` and `quark-onnx-debug` as potentially affected even when no string drift is observed — custom-op load failures are runtime-only.
- If drift is detected, hand off to `quark-onnx-skill-sync` with the specific findings so it can apply targeted fixes; do not edit skills from this check.
- If a finding spans both backends (e.g. a `shared/contracts/` schema change), report it but defer cross-cut fixes to the torch maintainer rather than editing torch skills from here.
