# MINCE: Monte-Carlo Informed N-sizing for Compact Evaluation

MINCE cuts LLM benchmark evaluation time by evaluating a small, **frozen** subset
of items instead of the full benchmark, while keeping the subset score close to
the full-benchmark score.

Given a bf16 model's per-item evaluation logs, MINCE:

1. **Sizes** a representative subset (`n*`) via a Monte-Carlo drift sweep,
2. **Freezes** that subset into a reproducible, ID-based artifact plus an
   lm-eval `--samples` map, and
3. lets you **reuse** the frozen subset to evaluate downstream model variants
   within a bounded, quantified accuracy drift.

## Layout

- **`quark/contrib/mince/`** — the importable package (`quark.contrib.mince`):
  `config`, `data_loader`, `mince_metrics`, `montecarlo`, `selection`, `subset`.
  Unit tests live in `quark/contrib/mince/test/`.
- **`quark/contrib/mince/docs/`** — the documentation, including the end-to-end
  walkthrough (`example_quark_torch_mince.rst`).
- **`examples/contrib/mince/`** — the runnable CLIs (`size.py`, `freeze.py`,
  `validate.py`, `extract_inputs.py`) and example requirements.

## Getting started

See the walkthrough in
[`quark/contrib/mince/docs/example_quark_torch_mince.rst`](docs/example_quark_torch_mince.rst)
for the full `size -> freeze -> score -> validate` flow, and install the example
dependencies with `pip install -r examples/contrib/mince/requirements.txt`.

Run the unit tests from the repo root:

```bash
python -m pytest quark/contrib/mince/test -q
```

## Support and contact

MINCE is a community contribution in Quark's `contrib` area. It is maintained
on by its author.

- **Author / maintainer:** [`@devledas`](https://github.com/devledas) (GitHub)
- **Questions, bugs, or feedback:** please open a GitHub issue on the AMD Quark
  repository and tag `@devledas`.

## License

Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
