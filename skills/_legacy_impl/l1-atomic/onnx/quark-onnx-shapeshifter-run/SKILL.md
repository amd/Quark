---
name: quark-onnx-shapeshifter-run
description: >
  Apply existing ShapeShifter graph passes to an .onnx model via the quark-cli shapeshifter CLI or
  a ShapeShifter YAML. Trigger for "run ShapeShifter on my .onnx", "apply an onnx_ pass", "fold
  batch norm / simplify / convert opset / fuse LayerNorm on my ONNX model", "preprocess my .onnx
  before quantization", "postprocess my quantized .onnx for XINT8/NPU". Operates on .onnx only. NOT
  authoring a new pass (use quark-create-shapeshifter-pass), NOT full quantization (use
  quark-onnx-ptq), NOT PyTorch models.
layer: l1-atomic
backend: onnx
primary_artifact: shapeshifter_config.yaml
source_knowledge:
  - docs/source/quark_shapeshifter.rst
  - docs/source/quark_shapeshifter_onnx_passes.rst
  - quark/shapeshifter/engine.py
  - quark/shapeshifter/utils.py
  - quark/experimental/cli/shapeshifter.py
---

# quark-onnx-shapeshifter-run

## Purpose

Apply one or more **existing** ShapeShifter ONNX passes to an `.onnx` model by authoring a
ShapeShifter YAML config and running the `quark-cli shapeshifter` CLI. ShapeShifter is Quark's
pass-based graph-transformation framework; this skill covers *invoking* its built-in `onnx_*`
passes (fold BatchNorm, simplify, convert opset, fuse LayerNorm/GELU, align scales, XINT8/NPU
adaptation, etc.) as a standalone file→file transform — separate from authoring a new pass
(`quark-create-shapeshifter-pass`) and from a full quantization run (`quark-onnx-ptq`). It exists so
graph preprocessing/postprocessing can be run and inspected on its own, and so the driving YAML is
a reusable, reviewable artifact.

## Inputs

- **Required**: path to an `.onnx` model (optionally with an adjacent `.onnx_data` external-weights
  file).
- **Required**: the pass(es) to apply and their config keys. If the user names an effect ("fold
  batch norm") rather than a pass name, map it to the pass via the catalog in
  `docs/source/quark_shapeshifter_onnx_passes.rst`.
- **Required**: an output `.onnx` path.
- **Optional**: whether these are preprocessing (float model) or postprocessing (quantized Q/DQ
  model) passes — affects which passes are valid and the recommended order.

## Outputs: shapeshifter_config.yaml

The primary artifact is the ShapeShifter YAML config that drives the run — reusable and reviewable.
Side effect: the transformed model written to the user's output `.onnx` path.

```yaml
input_model_path: /path/to/model.onnx
passes:
  onnx_convert_opset_version:
    target_opset_version: 21
  onnx_simplify:
    simplify: true
  onnx_fold_batch_norm:
    fold_batch_norm: true
output_model_path: /path/to/model_out.onnx
```

No JSON schema — this is a ShapeShifter CLI config (see `quark/shapeshifter/utils.py`), not a
cross-skill contract artifact.

## Interaction Flow

1. **Intake** — confirm the input `.onnx` path exists, capture the requested transformation(s) and
   the output path. Map effect words to concrete pass names.
2. **Route** — verify every requested pass is an ONNX pass (`onnx_*`) and exists in
   `quark/shapeshifter/passes/`. If the user asked to *create* a pass, hand off to
   `quark-create-shapeshifter-pass`; if they asked to *quantize*, hand off to `quark-onnx-ptq`.
3. **Plan** — present the YAML config (pass order + config keys) before writing. Confirm the config
   key for each pass (many passes are a silent no-op without their enable flag — see Recovery).
4. **Confirm** — get approval before running the CLI (it writes the output file).
5. **Execute or Summarize** — write `shapeshifter_config.yaml`, run the CLI, then verify the output
   model loads and the transformation took effect. Summarize what changed.

## Invoking the CLI

Write the YAML (see Outputs), then:

```bash
quark-cli shapeshifter shapeshifter_config.yaml
```

The CLI loads the model, runs each pass **in the order listed** (each on the previous pass's
output), and writes `output_model_path`. JSON configs also work. An explicit model config is
equivalent to the flat form and clearer when in doubt:

```yaml
input_model_config:
  model_type: onnx        # discriminator
  input_model_path: /path/to/model.onnx
passes: { ... }
output_model_path: /path/to/model_out.onnx
```

## Pass Selection

Full catalog + config keys: `docs/source/quark_shapeshifter_onnx_passes.rst`. Common choices:

| Intent | Pass | Config key |
|--------|------|-----------|
| Constant-fold / clean graph | `onnx_simplify` | `simplify: true` |
| Upgrade opset | `onnx_convert_opset_version` | `target_opset_version: 21` |
| Fold BatchNorm into Conv/Gemm | `onnx_fold_batch_norm` | `fold_batch_norm: true` |
| Fuse LayerNorm / GELU | `onnx_fuse_layer_norm` / `onnx_fuse_gelu` | `fuse_layer_norm: true` / `fuse_gelu: true` |
| Layout NCHW→NHWC | `onnx_convert_nchw_to_nhwc` | `convert_nchw_to_nhwc: true` |
| Cross-layer equalization | `onnx_cross_layer_equalization` | `cross_layer_equalization: true` |
| Align Q/DQ scales (quantized) | `onnx_align_scale` | `align_scale: [Concat, MaxPool]` |
| XINT8/NPU adapt (quantized) | `onnx_xint8_adjust` / `onnx_xint8_simulate` | `xint8_adjust: true` / `xint8_simulate: true` |

**Preprocessing** passes run on the **float** model (before quantization); **postprocessing**
passes (`onnx_align_scale`, `onnx_adjust_bias_scale`, `onnx_xint8_*`, bfloat16 passes) expect a
**quantized Q/DQ** model. Do not run postprocessing passes on a float model.

**Ordering tip:** put opset conversion and `onnx_simplify` first (some fusions require a newer
opset and a cleaner graph), then folding/fusion, then per-node initializer passes.

## Relationship to quantization

If the goal is to run these passes **as part of** quantization rather than standalone, they can be
driven from the quantizer instead via `extra_options={"ShapeShifterYaml": "config.yaml"}` with
`preprocess_passes:` / `postprocess_passes:` groups — that path belongs to `quark-onnx-ptq`. Use
this skill only for the standalone file→file transform.

## Recovery

- **Pass ran but nothing changed (silent no-op)** — most passes require their enable flag; e.g.
  `onnx_convert_clip_to_relu` needs `convert_clip_to_relu: true`. A missing/`false` flag logs a
  warning and returns the model unchanged. Re-check the config key against the docs.
- **`Pass '<name>' is not registered`** — misspelled pass name or a `pytorch_*` name. List valid
  names with `ls quark/shapeshifter/passes/`; this skill is ONNX-only.
- **`ValueError: ... is not an ONNX pass`** — a `pytorch_*` pass slipped into the config. All passes
  in one run must be ONNX.
- **Postprocessing pass errors on a float model** — `onnx_align_scale` / `onnx_xint8_*` need a
  quantized Q/DQ model. Quantize first (`quark-onnx-ptq`), then apply.
- **CLI errors / onnxruntime traceback** — hand off to `quark-onnx-debug` with the exact message.
- **Output opset too low for a fusion** — `onnx_fuse_gelu` (opset ≥ 20) and `onnx_fuse_layer_norm`
  (opset ≥ 17) auto-skip on lower opsets; add `onnx_convert_opset_version` earlier in the list.

## Notes

- CLI wrapper: `quark/experimental/cli/shapeshifter.py` (the deprecated `onnx-adapter` alias still
  works). Config loader: `quark/shapeshifter/utils.py` (`LoadConfigFromFileOrDict` — accepts a dict,
  JSON string, or YAML/JSON file path). Execution loop + model-type detection:
  `quark/shapeshifter/engine.py`.
- Passes run sequentially in listed order; there is no parallelism (deterministic by design).
- For an in-memory (no-disk) transform in Python, the `shapeshifter()` API takes a `model=` arg and
  returns the transformed `ModelProto` — but this skill is the CLI file→file path.
- Do not silently skip output verification; confirm the model loads and the intended nodes changed.
