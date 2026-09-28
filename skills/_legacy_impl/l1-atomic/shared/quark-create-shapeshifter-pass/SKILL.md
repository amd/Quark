---
name: quark-create-shapeshifter-pass
description: >
  Author a new ShapeShifter graph-transformation pass for AMD Quark (ONNX or PyTorch) so it
  conforms to the pass framework's conventions and auto-registers. Use when a developer says
  "add a ShapeShifter pass", "create a new onnx_ or pytorch_ pass", "write a custom Quark graph
  transform", or "contribute a community ShapeShifter pass". Walks through choosing backend and
  name, subclassing ONNXPass/PytorchPass with the register_pass decorator, implementing
  _default_config and _run_for_config, adding a per-pass test, and documenting it. Developer-facing
  authoring tool, not an end-user quantization step.
layer: l1-atomic
backend: shared
primary_artifact: shapeshifter_pass.py
source_knowledge:
  - docs/source/quark_shapeshifter.rst
  - docs/source/quark_shapeshifter_onnx_passes.rst
  - docs/source/quark_shapeshifter_torch_passes.rst
  - quark/shapeshifter/pass_base.py
  - quark/shapeshifter/engine.py
  - quark/shapeshifter/pass_config.py
  - quark/shapeshifter/passes/onnx_convert_clip_to_relu.py
  - quark/shapeshifter/passes/pytorch_remove_dropout.py
---

# quark-create-shapeshifter-pass

## Purpose

Author a new ShapeShifter transformation pass that is correct on the first try. ShapeShifter is
Quark's pass-based graph-transformation framework (`quark/shapeshifter/`): each pass is a
self-contained unit that takes a model, applies one transformation, and returns it. Passes
auto-register by filename via a `@register_pass` decorator, so a new pass is usable from the CLI,
the Python API, and the ONNX quantizer's `ShapeShifterYaml` with zero manual wiring — **if** it
follows the naming, subclassing, and config conventions exactly. This skill replays those
conventions in authoring order and flags the non-obvious traps (defaults are not auto-applied;
class/file naming is load-bearing) so the developer focuses on the transformation logic.

## Inputs

- **Required**: the transformation the pass performs (one sentence); the backend it targets
  (ONNX `onnx.ModelProto` **or** PyTorch callable — a pass is one or the other, never both).
- **Required**: a pass name in `snake_case`, prefixed `onnx_` or `pytorch_` (this becomes the
  filename, the registry key, and the YAML key — pick it carefully).
- **Optional**: config parameters the pass exposes (name, type, default, whether required);
  whether it is a **core** pass (`quark/shapeshifter/passes/`) or a **community** pass
  (`quark/contrib/shapeshifter_community_passes/`); a minimal model that exercises it (for the test).

## Outputs: shapeshifter_pass.py

The primary artifact is the new pass module, written to
`quark/shapeshifter/passes/<pass_name>.py` (core) or
`quark/contrib/shapeshifter_community_passes/<pass_name>.py` (community). `<pass_name>` matches the
chosen name (e.g. `onnx_drop_identity.py`). Side-effect artifacts:

- `test/test_for_cli/test_shapeshifter_<pass_name>.py` — one test per pass (project convention).
- A documentation entry in `docs/source/quark_shapeshifter_onnx_passes.rst` (ONNX) or
  `docs/source/quark_shapeshifter_torch_passes.rst` (PyTorch).

No JSON schema — this artifact is Quark source code, not a cross-skill handoff artifact.

## Interaction Flow

1. **Intake** — capture the transformation, the backend, the pass name, and any config params.
   If the developer described the transform in prior conversation, extract these first.
2. **Route** — confirm backend (ONNX vs PyTorch) and core-vs-community placement; derive the
   file path, class name, and YAML key from the name.
3. **Plan** — present the file path, class skeleton, `_default_config` params, and the
   `_run_for_config` outline **before writing**. Confirm the name does not collide with an
   existing registry entry (`ls quark/shapeshifter/passes/`).
4. **Confirm** — get explicit approval before writing source files under `quark/`.
5. **Execute or Summarize** — write the pass, the test, and the doc entry; run the test and
   confirm the pass registers. Summarize what was created and how to invoke it.

## Backend & Placement Decision

| Question | Choose |
|----------|--------|
| Operates on `onnx.ModelProto` (a graph)? | ONNX pass → subclass `ONNXPass`, prefix `onnx_` |
| Operates on a PyTorch callable (`nn.Module`, function)? | PyTorch pass → subclass `PytorchPass`, prefix `pytorch_` |
| Officially maintained / production? | Core → `quark/shapeshifter/passes/` |
| Contributed / experimental? | Community → `quark/contrib/shapeshifter_community_passes/` |

A workflow must be **all-ONNX or all-PyTorch**; the Engine raises `ValueError` on mixing.

## Naming Rules (load-bearing)

- **Filename = pass name = registry key = YAML key.** `onnx_drop_identity.py` → pass name
  `onnx_drop_identity`, used verbatim under `passes:` in YAML. There is no separate name string.
- Prefix **`onnx_`** or **`pytorch_`**.
- Files starting with `_` (e.g. `__init__.py`) are **skipped** by discovery.
- **Class name must end in `Pass`** (only such classes are discovered / exported). Convention:
  `ONNX<Thing>Pass` / `Pytorch<Thing>Pass`.
- Duplicate pass names raise `ValueError` at import (with a core-vs-community conflict message).

## Authoring the Pass

Two required methods (see `quark/shapeshifter/pass_base.py`):

- `_default_config(self) -> dict[str, PassConfigParam]` — declare each config param with
  `PassConfigParam(type_, default_value, required, description)`; end with
  `config.update(self.config)` (the convention every pass follows).
- `_run_for_config(self, model, config) -> model` — the transformation. ONNX: takes/returns
  `onnx.ModelProto`. PyTorch: takes/returns any `Callable`.

**ONNX skeleton** (`quark/shapeshifter/passes/onnx_drop_identity.py`):

```python
#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any

import onnx
from onnx import ModelProto
from onnxruntime.quantization.onnx_model import ONNXModel  # helpers: remove_nodes, etc.

from quark.common.utils.log import ScreenLogger
from quark.shapeshifter.pass_base import ONNXPass, register_pass
from quark.shapeshifter.pass_config import PassConfigParam

logger = ScreenLogger(__name__)


@register_pass
class ONNXDropIdentityPass(ONNXPass):
    """Remove Identity nodes from the graph."""

    def _default_config(self) -> dict[str, PassConfigParam]:
        config = {
            "drop_identity": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Whether to remove Identity nodes.",
            ),
        }
        config.update(self.config)
        return config

    def _run_for_config(self, model: ModelProto, config: dict[str, Any]) -> ModelProto:
        # config is the RAW user dict from YAML — read values directly and
        # supply your own default (see Critical Gotcha below).
        if not config.get("drop_identity", True):
            logger.warning("onnx_drop_identity: drop_identity is False, skipping.")
            return model
        onnx_model = ONNXModel(model)
        # ... transform onnx_model.model, then clean up ...
        onnx_model.topological_sort()
        return onnx_model.model
```

**PyTorch skeleton** mirrors this: subclass `PytorchPass`, `_run_for_config(self, model, config)`
returns the (possibly mutated) callable. See `quark/shapeshifter/passes/pytorch_remove_dropout.py`.

## Critical Gotcha: defaults are NOT auto-applied

The Engine calls `_run_for_config` **directly** with the raw YAML dict — it never calls `run()`
or `_default_config()` (see `quark/shapeshifter/engine.py`, the pass-execution loop:
`pass_instance._run_for_config(model, pass_config)`). Consequences the pass author MUST handle:

- The `config` argument is exactly what the user wrote (e.g. `{"drop_identity": True}`), **not** a
  dict of `PassConfigParam` objects.
- **Apply defaults yourself** inside `_run_for_config`, e.g. `config.get("drop_identity", True)`.
  Do not assume `_default_config()` populated anything.
- Read **flat** values: `config["drop_identity"]` — not `config.get("x", {}).get("value", ...)`.
  (`_default_config` still documents the schema and is good practice; it is just not the runtime
  source of defaults today.)

## Registration & Usage (zero-config)

No manual registration. `quark/shapeshifter/passes/__init__.py` imports every non-`_` module via
`pkgutil`, triggering `@register_pass`, which adds the class to the global `REGISTRY`. Importing
`quark.shapeshifter` registers the pass. Then it works everywhere:

**CLI** — `quark-cli shapeshifter config.yaml`:

```yaml
input_model_path: /path/in.onnx
passes:
  onnx_drop_identity:
    drop_identity: true
output_model_path: /path/out.onnx
```

**Python API**:

```python
from pathlib import Path
from quark.shapeshifter import shapeshifter, RunConfig, ONNXModelConfig

cfg = RunConfig(
    input_model_config=ONNXModelConfig(input_model_path=Path("in.onnx")),
    passes={"onnx_drop_identity": {"drop_identity": True}},
    output_model_path="out.onnx",
)
shapeshifter(cfg)                                  # file-based → returns None
new_model = shapeshifter(RunConfig(passes={"onnx_drop_identity": {}}), model=proto)  # in-memory
```

**Inside the ONNX quantizer** — add it to a `ShapeShifterYaml` under `preprocess_passes:`
(runs on the float model before quantization) or `postprocess_passes:` (runs on the quantized
Q/DQ model), passed via `extra_options={"ShapeShifterYaml": "config.yaml"}`.

## Test & Docs (required to ship)

- **Test**: `test/test_for_cli/test_shapeshifter_<pass_name>.py`. Follow
  `test_shapeshifter_onnx_convert_clip_to_relu_pass.py`: build a tiny model with `onnx.helper`,
  write a YAML, run `cli(["shapeshifter", yaml_path])`, assert on the output graph. Use
  `@use_temporary_directory` from `quark.common.utils.testing_utils`.
- **Docs**: add an option-by-option entry to
  `docs/source/quark_shapeshifter_onnx_passes.rst` (ONNX) or `..._torch_passes.rst` (PyTorch),
  matching the existing style. CI (`.github/workflows/ci_build_and_unittest_cli.yml`) already
  triggers on `quark/shapeshifter/**` and these doc paths.

## Recovery

- **"Pass not registered" / KeyError on pass name** — filename ≠ YAML key, file starts with `_`,
  the class name does not end in `Pass`, or `@register_pass` is missing. Check all four.
- **`ValueError: ... already registered`** — name collides with a core or community pass. Rename.
- **`ValueError: ... is not an ONNX/PyTorch pass`** — the workflow mixes backends, or the class
  subclasses the wrong base. All passes in one run must share a backend.
- **Config value ignored / pass is a silent no-op** — you relied on `_default_config()` for
  defaults, or read `config["x"]["value"]`. Read flat values with an explicit default in
  `_run_for_config` (see Critical Gotcha).
- **`register_pass could not find __file__`** — the class was defined in a REPL/exec context; it
  must live in a real `.py` file under a discovered directory.

## Notes

- Framework source: `quark/shapeshifter/pass_base.py` (bases, `REGISTRY`, `register_pass`),
  `quark/shapeshifter/engine.py` (execution loop — proves defaults are not auto-applied),
  `quark/shapeshifter/pass_config.py` (`PassConfigParam`), `quark/shapeshifter/passes/__init__.py`
  (discovery). Worked examples: `onnx_convert_clip_to_relu.py`, `pytorch_remove_dropout.py`.
- Adding conventions live in `docs/source/quark_shapeshifter.rst` ("Adding New Passes").
- `primary_artifact` is `.py` source (not a canonical handoff artifact), so `validate_skill.py`
  emits a non-blocking WARN for it — expected.
- Do not silently skip the test or doc entry; both are required by CI and project convention.
