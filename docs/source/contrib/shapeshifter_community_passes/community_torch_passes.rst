..  Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

Community PyTorch Passes
========================

Community-contributed PyTorch passes operate on PyTorch models (any Python
callable: ``torch.nn.Module``, functions, or custom callables) and use the
``pytorch_`` pass-name prefix, exactly like core PyTorch passes. For the
implementation contract, naming conventions, and worked examples, see
:doc:`PyTorch Model Passes </quark_shapeshifter_torch_passes>` and
:doc:`Adding New Passes </quark_shapeshifter>`.

No community PyTorch passes are currently shipped. When you contribute one,
document it here with its pass name, configuration parameters, and behavior,
matching the style of the core PyTorch pass reference.
