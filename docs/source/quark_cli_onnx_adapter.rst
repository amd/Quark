.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Quark ONNX Adapter
==================

**AMD Quark** includes a power graph transformation tool called `Quark ONNX Adapter`. It can perform graph transformation of preprocessing like constant folding, operator fusion, removal of redundant nodes, streamlining input and output nodes, and optimizing the graph structure.

.. note::

    In this release, `Quark ONNX Adapter` only supports graph transformation of preprocessing, and in the next release, it will also support graph transformation of postprocessing on quantized models. Specific optimizations for quantized models include platform-specific optimizations, such as those for deployment on NPU platforms, to enhance the implementation on specific hardware platforms.


Passes of Quark ONNX Adapter
----------------------------

Passes are the blocks of `Quark ONNX Adapter`. As shown below, it supports 20 preprocessing passes. For example **onnx_convert_bn_to_conv** is a pass name, and **convert_bn_to_conv** is its corresponding parameter.

*  **onnx_convert_bn_to_conv**:

   -  **convert_bn_to_conv**: (bool). Whether to convert BatchNormalization operations to Conv operations.

*  **onnx_convert_clip_to_relu**:

   -  **convert_clip_to_relu**: (bool). Whether to convert Clip operations to Relu operations.

*  **onnx_convert_fp16_to_fp32**:

   -  **convert_fp16_to_fp32**: (bool). Whether to convert the input model from FP16 to FP32.

*  **onnx_convert_nchw_to_nhwc**:

   -  **convert_nchw_to_nhwc**: (bool or list[str]). Whether to convert the input model from NCHW to NHWC. If setting True, conversion applies to all nodes; if providing a list of input or output names, such as ["input_1", "output_2",], shape conversion applies only to the listed inputs or outputs.

*  **onnx_convert_opset_version**:

   -  **target_opset_version**: (int). Convert opset version of the input model to the target like 21.

*  **onnx_convert_reduce_mean_to_global_avg_pool**:

   -  **convert_reduce_mean_to_global_avg_pool**: (bool). Whether to convert ReduceMean operations to GlobalAveragePool operations.

*  **onnx_convert_split_to_slice**:

   -  **convert_split_to_slice**: (bool). Whether to convert Split operations to Slice operations.

*  **onnx_copy_bias_init**:

   -  **shared_bias_op_types**: (list[str]). Specifies the node op_types to run duplicating bias initializers in the model for separate quantization use across different nodes. For example: ["Conv", "ConvTranspose", "Gemm"].

*  **onnx_copy_shared_init**:

   -  **shared_init_op_types**: (list[str]). Specifies the node op_types to run duplicating initializers in the model for separate quantization use across different nodes. For example: ["Conv", "ConvTranspose", "Gemm"].

*  **onnx_fix_shapes**:

   -  **input_output_name_shapes**: (dict[str, list[int]]). Model input/output name & input/output shapes to replace shape of. For example: {'input_1': [1, 224, 224, 3], 'input_2': [1, 96, 96, 3], 'output_1' :[1, 1000], 'output_2':[1, 10]}.

*  **onnx_fold_batch_norm_after_concat**:

   -  **fold_batch_norm_after_concat**: (bool). Whether to fold BatchNormalization operations into Conv, ConvTranspose and Gemm operations before Concat operations.

*  **onnx_fold_batch_norm**:

   -  **fold_batch_norm**: (bool). Whether to fold BatchNormalization operations into Conv, ConvTranspose and Gemm operations.

*  **onnx_fuse_gelu**:

   -  **fuse_gelu**: (bool). Whether to fuse a bunch of separate Gelu operations into a single Gelu operation.

*  **onnx_fuse_instance_norm**:

   -  **fuse_instance_norm**: (bool). Whether to fuse a bunch of separate InstanceNormalization operations into one single InstanceNormalization operation.

*  **onnx_fuse_l2_norm**:

   -  **fuse_l2_norm**: (bool). Whether to fuse a bunch of separate L2Norm operations into one single LpNormalization operation with p=2.

*  **onnx_fuse_layer_norm**:

   -  **fuse_layer_norm**: (bool). Whether to fuse a bunch of LayerNormalization operations into a single LayerNormalization operation.

*  **onnx_optimize_with_ort**:

   -  **optimize_with_ort**: (bool). Whether to optimize the input ONNX model with official ONNXRuntime.

*  **onnx_simplify**:

   -  **simplify**: (bool). Whether to simplify the input model.

*  **onnx_split_large_kernel_pool**:

   -  **split_large_kernel_pool**: (bool). Whether to split large kernel pooling operations into multiple smaller poolings.

*  **onnx_remove_input_init**:

   -  **remove_input_init**: (bool). Whether to remove initializers from the model input.

Usage of Quark ONNX Adapter
----------------------------

The `Quark ONNX Adapter` takes a YAML file like the one shown below as input. You can include multiple passes in the same YAML file.

.. code-block:: yaml

    input_model_path: /path/to/input_model.onnx
    passes:
      onnx_convert_opset_version:
        target_opset_version: 21
      onnx_simplify:
        simplify: true
    output_model_path: /path/to/output_model.onnx

And use the following command line.

.. code-block:: bash

   quark-cli onnx-adapter demo.yaml

.. note::

    You mus set **SkipPreprocess** as True if ONNX Adapter is used before quantization. For more details about this parameter, please refer to :doc:`Full List of Quantization Config Features <onnx/appendix_full_quant_config_features>`
