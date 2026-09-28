# Shapeshifter Community Passes

Community-contributed [Shapeshifter](../../../docs/source/quark_shapeshifter.rst)
graph-transformation passes. These extend Shapeshifter beyond the core passes and
are automatically discovered and registered to the same global registry, using the
same implementation contract as core passes.

- **ONNX passes** transform `onnx.ModelProto` graphs (`onnx_` prefix).
- **PyTorch passes** transform PyTorch callables (`pytorch_` prefix).

## Layout

- Pass modules live directly in this directory
  (`quark/contrib/shapeshifter_community_passes/`).
- Tests live in `test/`.
- Documentation lives in `docs/` and is auto-published under the
  **Contributions** section of the Quark documentation.

## Getting started

See the documentation landing page at
[`docs/index.rst`](docs/index.rst) for how to author, register, test, and
document a community pass.

## Support and contact

Shapeshifter community passes are a contribution in Quark's `contrib` area. They
are **not officially supported** by the Quark Core Team and are maintained by
their original authors. For questions, bugs, or feedback, please open a GitHub
issue on the AMD Quark repository.

## License

Copyright (C) 2025 - 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
