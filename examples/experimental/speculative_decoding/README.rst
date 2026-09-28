.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

Speculative Decoding (EAGLE-3) on AMD Instinct
==============================================

Train an EAGLE-3 draft for any Hugging Face causal LM from scratch, export it to
a vLLM-loadable draft, deploy it with vLLM speculative decoding on ROCm, and
measure the speedup — end to end, on a single 8x AMD Instinct node.

Usage
-----

For the validated Qwen3-8B quick start, see
https://quark.docs.amd.com/latest/eagle3_quick_start.html.

For the portable large-model baseline and generic domain-manifest workflow, see
https://quark.docs.amd.com/latest/eagle3_best_recipe.html.

For the Quark-native library API behind this feature, see
https://quark.docs.amd.com/latest/eagle3.html.

Layout
------

Canonical assets now live under ``eagle3/``:

* ``eagle3/common`` contains shared orchestration and utilities.
* ``eagle3/qwen3_8b_quick_start`` owns the validated Qwen3-8B configs.
* ``eagle3/minimax_m3_best_recipe`` contains the public large-model adapter,
  smoke manifest, and replaceable full-manifest template.

All orchestration lives under ``eagle3/``. Invoke a profile through its own ``run.sh``, or through the
``quark.experimental.speculative_decoding`` CLI.

License
-------

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT.
