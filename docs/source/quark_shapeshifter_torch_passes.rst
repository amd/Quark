.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

PyTorch Model Passes
====================

PyTorch passes operate on PyTorch models (any Python callable: ``torch.nn.Module``, functions, or custom callables). These passes enable direct transformations on PyTorch models without converting to ONNX. Pass names use the ``pytorch_`` prefix.

.. note::

    **New in this release:** PyTorch model support is newly added. Currently includes example passes for model optimization and pre-export preparation. More PyTorch passes will be added in future releases.

**Model Optimization**

*  **pytorch_remove_dropout**:

   -  **remove_dropout**: (bool). Remove all dropout layers (``nn.Dropout``, ``nn.Dropout2d``, ``nn.Dropout3d``) from the model by replacing them with ``nn.Identity`` layers. This is useful for pre-export preparation since dropout is typically disabled during inference anyway. The pass recursively traverses the model and replaces all dropout variants.

**Pre-Export Preparation**

*  **pytorch_trace_model**:

   -  **input_shapes**: (dict, required). Input tensor shapes for tracing. Key: input name, Value: shape as list. Example: ``{'input': [1, 3, 224, 224]}``. Multiple inputs can be specified with different names.

   -  **input_dtypes**: (dict, optional). Input tensor dtypes. Key: input name, Value: dtype string (``'float32'``, ``'float16'``, ``'int64'``, ``'int32'``, ``'float64'``). Defaults to ``float32`` if not specified for an input.

   This pass converts a PyTorch model to a TorchScript traced model using ``torch.jit.trace``. Tracing can improve inference performance and enable static analysis. The pass creates dummy inputs based on the specified shapes and dtypes, then traces the model. Useful for preparing models for deployment or ONNX export.

Community-Contributed Passes
-----------------------------

Community-contributed passes extend Shapeshifter's capabilities beyond the core passes. These passes are maintained in ``quark/contrib/shapeshifter_community_passes/`` and follow the same patterns and registration system as core passes.

See :doc:`quark_shapeshifter` for information about creating community passes.
