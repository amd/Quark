..  Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

AMD Quark — EAGLE-3
===================

An AMD/ROCm-native, Quark-integrated pipeline to train EAGLE-3 draft models from
scratch, synthesize on-policy training data, export to a vLLM-loadable draft,
deploy on vLLM, and evaluate acceptance length and throughput.

Validated CLI
-------------

Prepare the environment once, then run the full 8-GPU Qwen3-8B recipe:

.. code-block:: bash

   python3 -m quark.experimental.speculative_decoding.setup --base_model Qwen/Qwen3-8B
   python3 -m quark.experimental.speculative_decoding.run --base_model Qwen/Qwen3-8B

The default CLI generates on-policy data, trains through the packaged TorchSpec
runner, exports a vLLM draft, and validates real served acceptance length and
speedup. The programmatic ``qsd.train`` API remains the single-GPU reference
trainer.

Documentation
-------------

Please refer to https://quark.docs.amd.com/latest/eagle3.html.

For an overview of speculative decoding in AMD Quark, see
https://quark.docs.amd.com/latest/speculative_decoding.html.

License
-------

Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT.
