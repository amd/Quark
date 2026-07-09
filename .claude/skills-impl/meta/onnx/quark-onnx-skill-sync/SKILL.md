---
name: quark-onnx-skill-sync
description: >
  Detect upstream Quark ONNX changes that affect the ONNX skill family and classify required updates. Use when Quark
  ONNX docs, custom-op registry, quantization config presets, calibration methods, AutoSearchPro presets, ONNX Runtime
  install matrix, or source behavior under `quark/onnx/` may have drifted from the skill contracts. Trigger for
  "check if ONNX skills are up to date", "sync ONNX skills with Quark", "has Quark ONNX changed",
  "update ONNX skills after Quark upgrade", or when ONNX debugging reveals a mismatch between skill instructions and
  actual `quark.onnx` behavior.
layer: meta
backend: onnx
primary_artifact: validation_report.md
source_knowledge:
  - docs/source/install.rst
  - docs/source/onnx/basic_usage_onnx.rst
  - docs/source/onnx/user_guide_config_description.rst
  - docs/source/onnx/appendix_full_quant_config_features.rst
  - docs/source/onnx/user_guide_auto_search_pro.rst
  - quark/onnx/quantization/config/custom_config.py
  - quark/onnx/quantization/config/algorithm.py
  - quark/onnx/quantization/config/config.py
  - quark/onnx/quantization/api.py
  - quark/onnx/quantization/auto_search/auto_search_pro.py
  - quark/onnx/calibration/methods.py
  - quark/onnx/operators/custom_ops/__init__.py
  - quark/onnx/operators/custom_ops/build_custom_ops.py
  - tools/ci/install_onnxruntime.sh
  - examples/onnx/yolo_quantization/quantize_yolo.py
  - examples/onnx/model_support.md
---

# quark-onnx-skill-sync

## Purpose

Audit upstream Quark ONNX source against the assumptions baked into the `quark-onnx-*` skill family
and report which skills or contracts need updates. ONNX-side drift is dangerous because the
quantization presets, calibration methods, custom-op registry, deployment-target gates, and
AutoSearchPro presets are referenced by name in skill decision tables — a renamed `QConfig` field,
a removed preset, a new calibration method, or a new custom op silently produces wrong guidance.

## Inputs

- Quark upstream ONNX source files (`quark/onnx/`, `examples/onnx/`, `tutorials/onnx/`, `docs/source/onnx/`, `tools/ci/install_onnxruntime.sh`)
- Existing ONNX skill `SKILL.md` files under `.claude/skills-impl/{l1-atomic,l2-workflows,l3-recipes}/onnx/`
- ONNX-side shared schemas under `.claude/skills-impl/shared/contracts/`

## Outputs: validation_report.md

Lists which ONNX skills or contracts need updates after a Quark ONNX upstream change.

Schema: [`validation_report.schema.json`](../../../shared/contracts/validation_report.schema.json)

```markdown
# ONNX Skill Sync Report

## What to Check

### Source Knowledge Dependencies

Each ONNX skill declares `source_knowledge` files. Check that these files still exist and that the
skill's content matches the current source:

| Skill | Source Files to Check |
|-------|----------------------|
| `quark-onnx-install` | `tools/ci/install_onnxruntime.sh`, `docs/source/install.rst`, `requirements.txt`, `quark/onnx/operators/custom_ops/build_custom_ops.py` |
| `quark-onnx-router` | `docs/source/install.rst`, `docs/source/onnx/basic_usage_onnx.rst`, `docs/source/onnx/onnx_examples.rst`, `examples/onnx/model_support.md` |
| `quark-onnx-model-intake` | `quark/onnx/__init__.py`, `quark/onnx/quantization/api.py`, `quark/onnx/quantization/config/custom_config.py`, `quark/onnx/quantization/input_check.py`, `quark/onnx/operators/custom_ops/build_custom_ops.py`, `docs/source/onnx/basic_usage_onnx.rst` |
| `quark-onnx-quant-plan` | `quark/onnx/quantization/config/custom_config.py`, `quark/onnx/quantization/config/algorithm.py`, `quark/onnx/quantization/config/config.py`, `quark/onnx/calibration/methods.py`, `docs/source/onnx/user_guide_config_description.rst`, `docs/source/onnx/appendix_full_quant_config_features.rst` |
| `quark-onnx-debug` | `quark/onnx/quantization/quantize.py`, `quark/onnx/quantization/api.py`, `quark/onnx/quantization/input_check.py`, `quark/onnx/calibration/calibrators.py`, `quark/onnx/operators/custom_ops/build_custom_ops.py`, `quark/onnx/quantizers/registry.py`, `docs/source/onnx/gpu_usage_guide.rst` |
| `quark-onnx-result-validator` | `quark/onnx/quantization/api.py`, `quark/onnx/quantization/config/custom_config.py`, `quark/onnx/operators/custom_ops/__init__.py`, `examples/onnx/yolo_quantization/quantize_yolo.py` |
| `quark-onnx-ptq-workflow` | `examples/onnx/yolo_quantization/quantize_yolo.py`, `tutorials/onnx/ryzen_ai/yolov8/`, `tutorials/onnx/ryzen_ai/resnet50/`, `docs/source/onnx/basic_usage_onnx.rst`, `docs/source/onnx/user_guide_config_description.rst`, `quark/onnx/quantization/config/custom_config.py` |
| `quark-onnx-autosearch-pro` | `quark/onnx/quantization/auto_search/auto_search_pro.py`, `quark/onnx/quantization/auto_search/qconfig_mapping.py`, `quark/onnx/quantization/auto_search/config_generator.py`, `examples/onnx/auto_search/auto_search_pro_model.py`, `docs/source/onnx/user_guide_auto_search_pro.rst` |

### Key Facts to Verify

1. **ONNX Runtime install matrix** — check `tools/ci/install_onnxruntime.sh` for the supported
   `(accelerator, EP, ORT package, ORT version)` combinations cited by `quark-onnx-install`.
2. **Python and core dependency versions** — `pyproject.toml` `requires-python`, `requirements.txt`
   (`onnx`, `onnxruntime*`, `onnxslim`, `onnxscript`) versions cited by `quark-onnx-install` / `quark-install`.
3. **Quantization presets** — check `quark/onnx/quantization/config/custom_config.py` for the
   preset list (`XINT8`, `A8W8`, `A16W8`, `BF16`, `BFP16`, `MX*`, `MXFP*`, weights-only INT4 …)
   referenced by `quark-onnx-quant-plan` and `quark-onnx-ptq-workflow`.
4. **QConfig surface** — fields cited by skill decision tables: `global_config`, `algo_config`,
   `EnableNPUCnn`, `EnableNPUTransformer`, `use_external_data_format`, `exclude`,
   `calibration_method`, `OptimDevice`. Verify each still exists in `config.py` / `custom_config.py`.
5. **Calibration methods** — `MinMax`, `Percentile`, `Entropy`, `Distribution`, `MinMSE`,
   `LayerwisePercentile` cited by `quark-onnx-quant-plan` are still present in
   `quark/onnx/calibration/methods.py` (and `calibrators.py`).
6. **Algorithm configs** — `CLEConfig`, `BiasCorrectionConfig`, `AdaRoundConfig`, `AdaQuantConfig`,
   `SmoothQuantConfig`, `QuaRotConfig`, `GPTQConfig`, `FastFinetuneConfig` cited by the plan and
   AutoSearchPro recipe still exist in `quark/onnx/quantization/config/algorithm.py`.
7. **Custom-op registry** — `BFPQuantizeDequantize`, `MXQuantizeDequantize`, the `Extended*`
   family, and the `com.amd.quark` opset domain are still registered under
   `quark/onnx/operators/custom_ops/__init__.py`. The build pipeline in
   `build_custom_ops.py` still matches `quark-onnx-install` / `quark-onnx-debug` expectations.
8. **Deployment-target gates** — `EnableNPUCnn=True`, `EnableNPUTransformer=True`, and the
   `CPU` / `CUDA` / `ROCm` execution-provider gates referenced by `quark-onnx-ptq-workflow`
   still match what `custom_config.py` accepts.
9. **AutoSearchPro presets** — `ADVANCED_SEARCH`, `XINT8_SEARCH`, `A8W8_SEARCH`, `A16W8_SEARCH`
   cited by `quark-onnx-autosearch-pro` are still defined in
   `quark/onnx/quantization/auto_search/auto_search_pro.py` and exposed by `config_generator.py`.
10. **External-data threshold** — the `>2 GB` rule that triggers
    `use_external_data_format=True` in `quark-onnx-model-intake` and the workflow still matches
    what `api.py` / `input_check.py` enforce.
11. **Already-quantized detection** — the QDQ / `com.amd.quark` domain checks that intake uses
    to stop before re-quantizing still match `input_check.py`.
12. **YOLOv8 example pin** — `examples/onnx/yolo_quantization/quantize_yolo.py` still matches
    the `quark-onnx-ptq-workflow` worked example (`example-xint8-yolov8n.md`).

### Drift Classification

- **Mechanical drift**: a preset name, calibration-method name, custom-op name, ORT version,
  or QConfig field rename. Straightforward — update the affected table or list in the skill.
- **Semantic drift**: a workflow pattern changed (e.g. AutoSearchPro now requires a two-stage
  search by default; CLE moved from `algo_config` to a pre-pass); a deployment-target gate
  was redefined; a custom-op signature changed. Requires careful skill rewriting.
- **Breaking drift**: a referenced preset, calibration method, custom op, AutoSearchPro preset,
  ORT package, or `QConfig` field was removed entirely. The skill will produce wrong guidance
  until fixed.

## Audit Process

1. **Discover dependencies**: Read `source_knowledge` from each ONNX skill's frontmatter and
   union with the table above.
2. **Check for changes**: Compare current source against what the skills assume. Look for:
   - New presets / algorithms / calibration methods not mentioned in skills (e.g. a new
     `MXFP6` preset or a new `FastFinetune` knob).
   - Removed presets / methods / custom ops still mentioned in skills (e.g. deprecated
     `BFP16Spec` field).
   - Renamed QConfig fields (e.g. `EnableNPUCnn` → `enable_npu_cnn`).
   - Changed value ranges (e.g. AutoSearchPro trial budget defaults).
   - Custom-op binary names / load paths that drift from `build_custom_ops.py`.
   - ORT EP names changing (`CUDAExecutionProvider` / `ROCMExecutionProvider` / `VitisAIExecutionProvider`).
3. **Classify each drift**: mechanical, semantic, or breaking.
4. **Report**: Produce a `validation_report.md` with the findings.

## Rules

- **Do not auto-fix breaking drift** — report it and require human confirmation before applying
  changes.
- **Mechanical drift can be flagged for batch update** — preset additions, calibration-method
  name normalizations, and ORT version bumps are safe to apply after review.
- **If no prior baseline exists**, run calibration mode: record the current state of all ONNX
  source files as the baseline for future comparisons.
- **Never edit ONNX source under audit** — `quark/onnx/`, `examples/onnx/`, `tutorials/onnx/`,
  `docs/source/onnx/`, and `tools/ci/install_onnxruntime.sh` are read-only from this skill.
  Drift is reported, not patched at the source.
- **Stay in the ONNX scope** — do not touch `quark-torch-*` skills. Torch-side drift belongs to
  `quark-torch-skill-sync`; if a finding crosses scopes (e.g. a shared schema field), surface it
  but defer the cross-cut change to the torch maintainer.

## Summary
- Checked: 8 ONNX skills, 17 source files, 4 contract schemas
- Mechanical drift: 2 findings
- Semantic drift: 1 finding
- Breaking drift: 1 finding

## Breaking Drift
### quark-onnx-quant-plan: removed calibration method
- **Skill says**: `Distribution` is a supported calibration method
- **Source says**: `Distribution` removed from `quark/onnx/calibration/methods.py` (replaced by `LayerwisePercentile`)
- **Impact**: Users selecting `Distribution` will hit `ValueError` at plan-to-script translation
- **Fix**: Drop `Distribution` row from `quark-onnx-quant-plan` calibration table; update workflow examples

## Semantic Drift
### quark-onnx-autosearch-pro: search now two-stage by default
- **Source**: `auto_search_pro.py` now runs a coarse + fine pass; old single-stage flag deprecated
- **Impact**: Skill's "single-shot search" framing is now misleading; trial budget interpretation changed
- **Fix**: Rewrite the "Search budget" section and the preset table to reflect two-stage semantics

## Mechanical Drift
### quark-onnx-quant-plan: new preset added
- **Source**: `custom_config.py` now exposes `MXFP6_E3M2` preset
- **Impact**: Users asking about MXFP6 won't see it in the preset list
- **Fix**: Add `MXFP6_E3M2` row to the preset table

### quark-onnx-install: ORT version bump
- **Source**: `tools/ci/install_onnxruntime.sh` now pins `onnxruntime-rocm==1.20.0` (was 1.19.2)
- **Impact**: Skill recommends 1.19.2; users on 1.20.0 are told to downgrade
- **Fix**: Update ROCm row in the ORT install matrix
```

## Interaction Flow

1. **Collect**: Identify all source-knowledge dependencies across ONNX skills (union of frontmatter `source_knowledge` + the table above).
2. **Audit**: Check each source file for changes relevant to ONNX skill content (presets, algo configs, calibration methods, custom ops, ORT matrix, AutoSearchPro presets, deployment-target gates).
3. **Classify**: Tag each finding as mechanical, semantic, or breaking.
4. **Report**: Present findings in a structured `validation_report.md`.
5. **Confirm**: Get approval before applying any fixes; never auto-apply breaking drift.

## Recovery

- If a source file no longer exists (was moved or deleted), this is breaking drift. Report the missing file and suggest where the information might have moved to (e.g. `custom_config.py` → `config/presets/`).
- If `tools/ci/install_onnxruntime.sh` is missing or replaced, surface the gap to `quark-onnx-install` so its ORT matrix is not silently stale.
- If the ONNX custom-op build pipeline (`build_custom_ops.py`) has changed, mark `quark-onnx-install` and `quark-onnx-debug` as potentially affected even when no string drift is observed — custom-op load failures are runtime-only.
- If the upstream Quark repository is not accessible, report the blocked audit and which ONNX skills could not be verified.
- If drift crosses backends (e.g. a `shared/contracts/` schema field used by both torch and onnx skills), report it but hand off the cross-cut change to the torch maintainer rather than editing torch skills from here.
