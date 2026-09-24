.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Quantizing Using CrossLayerEqualization (CLE)
=============================================

CrossLayerEqualization (CLE) can equalize the weights of consecutive convolution layers, making the model weights easier to perform per-tensor quantization. For more details, please refer the paper `link <https://arxiv.org/abs/1906.04721>`__. Experiments show that the CLE technique improves PTQ accuracy for many models, especially those with depthwise convolutional layers, such as MobileNet and ShuffleNet.

Here is a simple example showing how to apply the CLE algorithm on an A8W8 (Activation-8bit-Weight-8bit) quantization.

.. code-block:: python

    from quark.onnx import ModelQuantizer, QConfig, QLayerConfig, UInt8Spec, Int8Spec, CLEConfig

    quant_config = QLayerConfig(input_tensors=UInt8Spec(), weight=Int8Spec())

    cle_config = CLEConfig(cle_steps=1, cle_scale_append_bias=True)

    config = QConfig(
        global_config=quant_config,
        algo_config=[cle_config],
    )

    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(input_model_path, quantized_model_path, calib_data_reader)

Arguments
---------

Here we only list a few important and commonly used arguments, please refer to the documentation of full arguments list for more details.

  - **cle_steps**: (Int) Specifies the steps for CrossLayerEqualization execution when include_cle is set to true. The default is 1. When set to -1, adaptive CrossLayerEqualization steps are conducted. The default value is 1.

  - **cle_scale_append_bias**: (Boolean) Whether the bias is included when calculating the scale of the weights. The default value is True.

Stem Equalization
=================

Regular CLE only equalizes consecutive ``Conv -> ... -> Conv`` chains and structurally skips the *stem* (first) convolution in architectures where it is followed by ``BatchNorm -> ReLU -> MaxPool`` with a concat fan-out, such as DenseNet. Whenever the stem weights are quantized with a single per-tensor scale shared across all output channels, a large per-output-channel weight magnitude spread leaves the low-magnitude channels with very little of the representable range, so they lose most of their precision — an error that then propagates through the whole network. This is independent of the bit width or scale format used.

Stem equalization addresses this case. For each stem output channel ``i``, it scales the weight up by a factor ``s_i`` (capped to avoid numerical instability) and folds the inverse ``1 / s_i`` into the affine parameters of the downstream BatchNorm. Because the stem output only passes through positively-homogeneous operations (ReLU / MaxPool, where ``f(s*x) = s*f(x)`` for ``s > 0``) before reaching the BatchNorm, the per-channel scale propagates unchanged and is absorbed exactly by the BatchNorm. The FP32 output is therefore mathematically unchanged, while the stem weights quantize far more evenly.

The pass auto-detects the stem convolution and its downstream BatchNorm(s), verifies the pass-through chain is positively-homogeneous (handling concat fan-out and channel offsets), and is a safe no-op when the pattern does not match.

Stem equalization runs automatically as the first step of CLE when ``include_cle`` is enabled. Because it depends on no CLE state, it can also be applied on its own as a standalone pre-processing step, before quantizing the model with any configuration:

.. code-block:: python

    import onnx
    from quark.onnx.algorithm import stem_equalize_transforms

    model = onnx.load(input_model_path)

    # Equalize the stem convolution in place; safe no-op if no stem/BatchNorm
    # pattern is found. The FP32 output of the model is unchanged.
    model = stem_equalize_transforms(
        model,
        op_types_to_quantize=["Conv"],
        nodes_to_quantize=[],
        nodes_to_exclude=[],
    )

    onnx.save(model, equalized_model_path)

The equalized float model can then be quantized as usual.

Example
=======

This :doc:`example <../../tutorials/onnx/accuracy_improvement/cle/onnx_cle_tutorial>` demonstrates quantizing a resnet152 model using the AMD Quark ONNX quantizer.
