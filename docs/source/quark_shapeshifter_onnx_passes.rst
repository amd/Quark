.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

ONNX Model Passes
==================

ONNX passes operate on ONNX graph models (``onnx.ModelProto``) and include both preprocessing passes (model preparation before quantization) and postprocessing passes (quantized model optimization). Pass names use the ``onnx_`` prefix.

Preprocessing Passes
^^^^^^^^^^^^^^^^^^^^

**Operator Conversion**

*  **onnx_convert_bn_to_conv**:

   -  **convert_bn_to_conv**: (bool). Convert BatchNormalization nodes into equivalent 1×1 depthwise Conv nodes. The pass folds the BN parameters (gamma, beta, running mean, running variance) into convolution weights and bias, which can improve compatibility with downstream quantization and hardware backends that do not natively support BatchNormalization. Only 4D inputs with all required BN parameters as initializers are converted.

*  **onnx_convert_clip_to_relu**:

   -  **convert_clip_to_relu**: (bool). Replace eligible Clip nodes with Relu nodes. A Clip node is converted only when its lower bound is non-negative (min >= 0), making it functionally equivalent to Relu. The related min/max initializers or Constant nodes are automatically cleaned up. This is useful for simplifying the graph before quantization on platforms that support Relu but not Clip.

*  **onnx_convert_reduce_mean_to_global_avg_pool**:

   -  **convert_reduce_mean_to_global_avg_pool**: (bool). Replace ReduceMean nodes that operate on spatial axes [2, 3] with GlobalAveragePool nodes. This conversion targets the common pattern of spatial averaging in CNNs and can improve compatibility with hardware backends that have dedicated GlobalAveragePool support but limited ReduceMean support.

*  **onnx_convert_split_to_slice**:

   -  **convert_split_to_slice**: (bool). Replace each Split node with a sequence of equivalent Slice nodes. For each output of the original Split, a separate Slice node with explicit starts, ends, axes, and steps is created. This transformation improves compatibility with backends that support Slice but not Split, or that benefit from the more explicit slicing representation.

*  **onnx_split_large_kernel_pool**:

   -  **split_large_kernel_pool**: (bool). Split GlobalAveragePool nodes with large spatial inputs (H×W > 512) into a smaller AveragePool followed by a GlobalAveragePool. The pass factorizes the spatial dimensions to find a suitable smaller kernel size. This is necessary for certain hardware backends (e.g., NPU) that have limits on the maximum pooling kernel size.

**Operator Fusion**

*  **onnx_fuse_gelu**:

   -  **fuse_gelu**: (bool). Detect and replace multi-node GELU patterns (e.g., x * 0.5 * (1 + erf(x / sqrt(2)))) with a single Gelu operator using the ONNX Runtime graph transformer. This reduces the number of nodes in the graph, simplifies analysis, and can improve runtime performance. Requires opset version >= 20; the pass is automatically skipped for models with lower opset versions.

*  **onnx_fuse_instance_norm**:

   -  **fuse_instance_norm**: (bool). Detect and replace multi-node InstanceNormalization patterns (composed of Sub, Mul, GlobalAveragePool, Reciprocal, Sqrt, and Add nodes) with a single InstanceNormalization operator. This simplification reduces graph complexity and can improve runtime performance on backends with native InstanceNormalization support.

*  **onnx_fuse_l2_norm**:

   -  **fuse_l2_norm**: (bool). Detect and replace multi-node L2 normalization patterns (composed of ReduceSum, Sqrt, Max, Reciprocal, Unsqueeze, and Mul nodes) with a single LpNormalization operator with p=2. This fusion reduces graph complexity and makes the normalization intent explicit for backend optimizers.

*  **onnx_fuse_layer_norm**:

   -  **fuse_layer_norm**: (bool). Detect and replace multi-node LayerNormalization patterns with a single LayerNormalization operator using the ONNX Runtime graph transformer. This reduces graph complexity and enables hardware-specific optimizations for LayerNorm. Requires opset version >= 17; the pass is automatically skipped for models with lower opset versions.

**BatchNorm Folding**

*  **onnx_fold_batch_norm**:

   -  **fold_batch_norm**: (bool). Fold BatchNormalization parameters (gamma, beta, mean, variance) into the weights and bias of the preceding Conv, ConvTranspose (group=1), or Gemm (transB=1) node. The BN node is removed from the graph and its computation is absorbed into the parent operator, reducing the number of nodes and improving inference performance. This is one of the most common preprocessing optimizations for CNN models.

*  **onnx_fold_batch_norm_after_concat**:

   -  **fold_batch_norm_after_concat**: (bool). Fold BatchNormalization parameters into upstream Conv, ConvTranspose, or Gemm nodes when the BN follows a Concat operation. The pass slices the BN parameters (gamma, beta, mean, variance) by channel to match each Concat input's channel count, and folds each slice into the corresponding upstream operator. This handles the case where standard BN folding fails because a Concat sits between the Conv and BN.

**Model Format Conversion**

*  **onnx_convert_fp16_to_fp32**:

   This pass supports converting **floating-point models** and **quantized models** (models with Q/DQ nodes) both.
   -  **convert_fp16_to_fp32**: (bool). Convert all FP16 (float16) tensors in the model to FP32 (float32), including initializers, value_info, and graph I/O types. Cast nodes targeting FP16 are updated to target FP32. Certain operator types (e.g., NonMaxSuppression, TopK, Resize) are excluded from conversion by default and automatically wrapped with Cast nodes to preserve correctness.
   -  **subgraphs_to_include**: (list[list[list[str], list[str]]], optional). A list of subgraph definitions to selectively convert from FP16 to FP32. Each element is a pair ``[start_node_names, end_node_names]`` where both are lists of node names. Nodes on paths between start and end nodes are included in the conversion. If ``start_node_names`` is empty, all upstream nodes leading to the end nodes are included. If ``end_node_names`` is empty, all downstream nodes from the start nodes are included, for example: convert only the subgraph between node_1，node_2 and node_X, node_Y, node_Z you can configure it as [[["node_1", "node_2"], ["node_X", "node_Y", "node_Z"]]]. Multiple pairs result in the union of all specified subgraphs being converted, for example: convert the subgraph between node_1， node_2 and node_X, node_Y, node_Z and the subgraph between node_A and node_B, node_C, node_D you can configure it as [[["node_1", "node_2"], ["node_X", "node_Y", "node_Z"]], [["node_A", "node_B"], ["node_C", "node_D"]]]. If not provided, the entire model is converted.

*  **onnx_convert_nchw_to_nhwc**:

   -  **convert_nchw_to_nhwc**: (bool or list[str]). Convert the model's data layout from NCHW (channels-first) to NHWC (channels-last) by modifying input/output tensor shapes and inserting Transpose nodes at the boundaries. If set to True, all 4D model inputs and outputs are converted. If a list of input or output names is provided (e.g., ["input_1", "output_2"]), only the specified tensors are converted. Quantized model outputs (ending with DequantizeLinear → QuantizeLinear) are handled automatically with additional Q/DQ nodes after the Transpose.

*  **onnx_convert_opset_version**:

   -  **target_opset_version**: (int). Convert the model's ONNX opset version to the specified target version (e.g., 21) using the official ONNX version converter. This is useful when downstream tools or hardware backends require a specific opset version.

**Initializer & Shape Management**

*  **onnx_copy_bias_init**:

   -  **shared_bias_op_types**: (list[str]). Duplicate shared bias initializers for the specified operator types so that each node gets its own independent copy. This is required before per-node bias quantization: when multiple nodes (e.g., Conv, ConvTranspose, Gemm) share the same bias tensor, they must be separated to allow individual quantization parameters. For example: ["Conv", "ConvTranspose", "Gemm"].

*  **onnx_copy_shared_init**:

   -  **shared_init_op_types**: (list[str]). Duplicate all shared initializers (weights, bias, etc.) for the specified operator types so that each node gets its own independent copy. Similar to ``onnx_copy_bias_init`` but applies to all initializer inputs, not just bias. This is essential before per-node quantization when multiple operators share the same weight tensor. For example: ["Conv", "ConvTranspose", "Gemm"].

*  **onnx_fix_shapes**:

   -  **input_output_name_shapes**: (dict[str, list[int]]). Override model input and output shapes with the specified values, and infer all intermediate tensor shapes accordingly. The pass rewrites the specified input/output shapes, then runs ONNX Runtime inference with random data to determine and update all intermediate tensor shapes in the graph's value_info. This is useful when the model has dynamic shapes that need to be fixed to static shapes for deployment. For example: {'input_1': [1, 224, 224, 3], 'output_1': [1, 1000]}.

*  **onnx_remove_input_init**:

   -  **remove_input_init**: (bool). Remove initializer tensors from the model's graph input list. In older ONNX IR versions (< 4), initializers were required to also appear as graph inputs. This pass cleans up that legacy behavior by removing initializer entries from the input list while keeping the initializers themselves intact. The IR version is automatically upgraded to 7 if it is below 4.

**General Optimization**

*  **onnx_optimize_with_ort**:

   -  **optimize_with_ort**: (bool). Optimize the model using the official ONNX Runtime graph optimizer. The model is loaded into an ONNX Runtime inference session with basic graph optimization enabled, which performs standard optimizations such as constant folding, redundant node elimination, and operator fusion. The optimized model is then returned. This pass provides a convenient way to apply ONNX Runtime's built-in optimizations as a preprocessing step.

*  **onnx_simplify**:

   -  **simplify**: (bool). Simplify the model using the ``onnxslim`` library, which performs optimizations such as constant folding, elimination of Identity nodes, removal of redundant operators, and general graph cleanup. This reduces the model size and the number of nodes, making it more efficient for subsequent quantization or inference. Additional options can be provided via the optional **simplify_Options** parameter (dict), which passes keyword arguments directly to ``onnxslim.slim()``.

Postprocessing Passes
^^^^^^^^^^^^^^^^^^^^^

Postprocessing passes are applied to quantized ONNX models to optimize Q/DQ nodes and adapt them for specific hardware targets.

.. note::

    In addition to the passes listed below, the preprocessing passes **onnx_convert_nchw_to_nhwc** and **onnx_fix_shapes** can also be applied to quantized models as postprocessing steps. ``onnx_convert_nchw_to_nhwc`` automatically handles Q/DQ nodes at model outputs during layout conversion, and ``onnx_fix_shapes`` can fix dynamic shapes to static shapes for deployment on quantized models.

**Quantization Parameter Adjustment**

*  **onnx_adjust_bias_scale**:

   -  **adjust_bias_scale**: (bool). Adjust bias scales in QDQ quantized models to ensure correctness. For Conv, Gemm, and ConvTranspose nodes, the pass verifies that the bias scale equals activation scale × weights scale. When a mismatch is detected and the bias is quantized as int32, the pass rescales the bias values and updates the bias scale initializer to the correct product. This prevents quantization accuracy loss caused by inconsistent bias scaling.

*  **onnx_align_scale**:

   -  **align_scale**: (str or list[str]). Align scale and zero-point of Q/DQ nodes for selected operator types to satisfy compiler constraints for float-scale quantized models. Supported op types: "Concat", "MaxPool", "AveragePool", "GlobalAveragePool", "Pad", "Slice", "Transpose", and "Reshape". Can be a single op type name or a list (e.g., ["Concat", "MaxPool", "AveragePool", "Pad"]). For Concat, Pad, Transpose, and Reshape, the output Q/DQ parameters are copied to all inputs; for MaxPool/AveragePool/GlobalAveragePool and Slice, the input Q/DQ parameters are copied to the output. The pass iterates up to 5 rounds until no further changes occur.

*  **onnx_remove_qdq_between_op_types**:

   -  **remove_qdq_between_op_types**: (list[list[str]]). Remove redundant QuantizeLinear/DequantizeLinear node pairs between specified operator type pairs. Each entry is a list of two operator type names [upper_op, lower_op]. For each pair, the pass finds patterns where an upper_op output feeds through Q→DQ into a lower_op input, and removes the Q/DQ nodes to connect them directly. Only single-consumer DQ outputs are removed to avoid breaking other connections. For example: [["Conv", "Relu"], ["Conv", "LeakyRelu"], ["Mul", "Add"]].

**Bfloat16 Processing**

*  **onnx_insert_clip_before_bfloat16_qdq**:

   -  **insert_clip_before_bfloat16_qdq**: (bool). Insert Clip nodes before Bfloat16 activation QuantizeLinear nodes to clamp input values to the valid bfloat16 range (approximately ±3.39e38). This prevents overflow during bfloat16 quantization. The pass only targets ExtendedQuantizeLinear nodes with bfloat16 zero-point whose input is not an initializer (i.e., activation inputs).

*  **onnx_remove_bfloat16_cast**:

   -  **remove_bfloat16_cast**: (bool). Remove redundant bfloat16 Cast operations from the model graph. The pass performs three cleanups: (1) removes consecutive fp32→bf16→fp32 Cast pairs in the activation path and reconnects edges directly; (2) for weight initializers connected through fp32→bf16→fp32 Cast chains, applies the bfloat16 precision loss directly to the initializer data and removes the Cast nodes; (3) removes fp32→bf16→fp32 Cast pairs at model outputs. This simplifies models that were exported with explicit bfloat16 Cast operations.

*  **onnx_replace_bfloat16_qdq_with_cast**:

   -  **replace_bfloat16_qdq_with_cast**: (bool). Replace ExtendedQuantizeLinear/ExtendedDequantizeLinear nodes that use bfloat16 with zero-point 0 by equivalent Cast operations. For ExtendedQuantizeLinear, if the scale is not 1, a Mul node with the reciprocal of the scale is inserted before the Cast to BFLOAT16. For ExtendedDequantizeLinear, a Cast to FLOAT is created, with an optional Mul node after it if the scale is not 1. This converts custom Q/DQ nodes into standard ONNX Cast operations for broader backend compatibility.

**XINT8/NPU Adaptation**

*  **onnx_xint8_adjust**:

   -  **xint8_adjust**: (bool). Adjust XINT8 (power-of-two scale) quantize positions in the model to satisfy NPU hardware constraints. The pass runs up to 5 iterative rounds, adjusting: (1) Concat, Pool, Pad, and Slice input/output positions to be aligned; (2) shift_cut (wpos + ipos - opos) for Conv/Gemm to [0, 16]; (3) shift_bias for Conv/Gemm with bias to [min_sb, 15]; (4) shift_read for Add/Sub to [0, 7]; (5) shift_write for Add to [-7, 25] and for Mul to [0, 32]; (6) HardSigmoid input pos to [0, 15] and output pos >= 7; (7) shift_swish for Swish-pattern Mul to [0, 15]. Positions are updated by modifying Q/DQ scale initializers via scale-to-position conversion.

*  **onnx_xint8_simulate**:

   -  **xint8_simulate**: (bool). Convert selected operations into NPU-simulated equivalents for XINT8 quantized models, matching the behavior of the target NPU hardware. Specific conversions include: (1) LeakyRelu alpha rounded to NPU format (round(alpha×256)/256); (2) Sigmoid replaced with HardSigmoid; (3) HardSigmoid scaled by a NPU-specific factor; (4) AveragePool and GlobalAveragePool rescaled with kernel-dependent NPU factors; (5) ReduceMean rescaled with reduction-size-dependent factors; (6) Softmax replaced with a polynomial approximation subgraph using bfloat16; (7) InstanceNormalization replaced with a custom NPU operator; (8) Clip bounds clamped to [-128, 127]. This pass ensures the quantized model faithfully simulates NPU arithmetic during evaluation.

*  **onnx_set_node_attributes**:

   -  **node_attribute_updates**: (list[dict]). Each element must provide **node_name** (str, must match ``NodeProto.name``) and **attributes** (dict mapping attribute name to a new value). The pass updates **only attributes that already exist** on that node; it does not add new attributes or create nodes. Missing attribute names are skipped with a warning; if no node matches **node_name**, a warning is logged. For primitive ONNX attribute kinds (INT, FLOAT, STRING, INTS, FLOATS, STRINGS), the configured value must have a **compatible Python type** with the existing ONNX attribute (e.g. native ``int`` for INT, ``float`` for scalar FLOAT attributes—integer literals are not accepted—``str`` for STRING). For **INTS**, **FLOATS**, and **STRINGS**, the value must be a ``list`` or ``tuple`` (a bare scalar is not accepted), and every element must match the element type: ``int`` for INTS, ``float`` for FLOATS, ``str`` for STRINGS. If the user supplies a mismatched type (for example a scalar ``9`` where ``[9]`` is required for INTS), the pass logs a warning and **skips** updating that attribute, leaving the model unchanged for that entry. Attributes whose ONNX kind is not one of INT, FLOAT, STRING, INTS, FLOATS, or STRINGS (for example **TENSOR** or **GRAPH**) are **not** modified; matching keys in the configuration are skipped with a warning. Use this pass to adjust scales, axes, flags, or custom-op parameters on specific operators after export or quantization without round-tripping through the original framework.
