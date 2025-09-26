.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Quantization Using AdaQuant and AdaRound
========================================

.. note::

    In this documentation, **AMD Quark** is sometimes referred to simply as **"Quark"** for ease of reference. When you  encounter the term "Quark" without the "AMD" prefix, it specifically refers to the AMD Quark quantizer unless otherwise stated. Please do not confuse it with other products or technologies that share the name "Quark."

.. note::

   For information on accessing AMD Quark ONNX examples, refer to :doc:`Accessing ONNX Examples <../onnx_examples>`.
   These examples and the relevant files are available at ``/onnx/accuracy_improvement/adaquant`` and ``/onnx/accuracy_improvement/adaround``.

.. note::

   AdaRound and AdaQuant cannot be used simultaneously; you can only choose one of them.


Fast Finetune
-------------

Fast finetune improves the quantized model's accuracy by training the output of each layer as close as possible to the floating-point model. It includes two practical algorithms: "AdaRound" and "AdaQuant". Applying fast finetune might achieve better accuracy for some models but takes much longer time than normal PTQ. It is disabled by default to save quantization time but can be turned on if you encounter accuracy issues. If this feature is enabled, `quark.onnx` will require the PyTorch package.

Here is a simple example showing how to apply the AdaRound algorithm on an A8W8 (Activation-8bit-Weight-8bit) quantization.

.. code-block:: python

    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config import QConfig
    from quark.onnx.quantization.config.spec import QLayerConfig, UInt8Spec, Int8Spec
    from quark.onnx.quantization.config.algorithm import AdaRoundConfig

    quant_config = QLayerConfig(activation=UInt8Spec(), weight=Int8Spec())

    adaround_config = AdaRoundConfig(
                      batch_size=1,
                      num_iterations=1000,
                      learning_rate=0.1)

    config = QConfig(
        global_config=quant_config,
        algo_config=[adaround_config],
    )

    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(input_model_path, quantized_model_path, calib_data_reader)

Arguments
~~~~~~~~~

Here we only list a few important and commonly used arguments, please refer to the documentation of full arguments list for more details.

  - **data_size**: (Int) Specifies the size of the data used for finetuning. Its recommended setting the batch size of the data to 1 in the data reader to ensure counting the size accurately. It uses all the data from the data reader by default.

  - **batch_size**: (Int) Batch size for finetuning. A larger batch size might result in better accuracy but longer training time. The default value is 1.

  - **num_iterations**: (Int) The number of iterations for finetuning. More iterations can lead to better accuracy but also longer training time. The default value is 1000.

  - **learning_rate**: (Float) Learning rate for finetuning. It significantly impacts the improvement of fast finetune, and experimenting with different learning rates might yield better results for your model. The default value is 0.1.

  - **optim_device**: (String) Specifies the compute device used for PyTorch model training during fast finetuning. Optional values are "cpu" and "cuda:0" (The latter is appliable for AMD and NV GPUs both). The default value is "cpu".

  - **infer_device**: (String) Specifies the compute device used for ONNX model inference during fast finetuning. Optional values are "cpu" and "cuda:0" (The latter is appliable for AMD and NV GPUs both). The default value is "cpu".

  - **mem_opt_level**: (Int) Specifies the level of memory optimization. Options are 0, 1 and 2. Setting it to 0 disables optimization, making training faster but using more memory for caching. Setting it to 1 caches data one layer at a time, reducing memory usage at the cost of longer training times. Setting it to 2 saves layer data to a cache directory on disk and loads only one batch at a time, greatly lowering memory consumption but further increasing training time. The default value is 1.

AdaRound
~~~~~~~~

**AdaRound**, short for "Adaptive Rounding," is a post-training quantization technique that aims to minimize the accuracy drop typically associated with quantization. Unlike standard rounding methods, which can be too rigid and cause significant deviations from the original model's behavior, AdaRound uses an adaptive approach to determine the optimal rounding of weights. Here is the `link <https://arxiv.org/abs/2004.10568>`__ to the paper.

AdaQuant
~~~~~~~~

**AdaQuant**, short for "Adaptive Quantization," is an advanced quantization technique designed to minimize the accuracy loss typically associated with post-training quantization. Unlike traditional static quantization methods, which apply uniform quantization across all layers and weights, AdaQuant dynamically adapts the quantization parameters based on the characteristics of the model and its data. Here is the `link <https://arxiv.org/abs/1712.01048>`__ to the paper.

Benefits of AdaRound and AdaQuant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Improved Accuracy**: By minimizing the quantization error, AdaRound helps preserve the model's accuracy closer to its original state. By dynamically adjusting quantization parameters, AdaQuant helps retain a higher level of model accuracy compared to traditional quantization methods.
2. **Flexibility**: AdaRound and AdaQuant can be applied to various layers and types of neural networks, making it a versatile tool for different quantization needs.
3. **Post-Training Application**: AdaRound does not require retraining the model from scratch. It can be applied after the model has been trained, making it a convenient choice for deploying pre-trained models in resource-constrained environments.
4. **Efficiency**: AdaQuant enables the deployment of high-performance models in resource-constrained environments, such as mobile and edge devices, without the need for extensive retraining.

Upgrades of AdaRound / AdaQuant in AMD Quark for ONNX
-----------------------------------------------------

Comparing with the original algorithms, AdaRound and AdaQuant in AMD Quark for ONNX are modified and upgraded to be more flexible.

1. **Unified Framework**: These two algorithms were integrated into a unified framework named as "fast finetune".
2. **Quantization Aware Finetuning**: Only the weight and bias (optional) will be updated, the scales and zero points are fixed, which ensures that all the quantizing information and the structure of the quantized model keep unchanged after finetuning.
3. **Flexibility**: AdaRound in Quark for ONNX is compatible with many more graph patterns-matching.
4. **More Advanced Options**

   - **Early Stop**: If the average loss of the current batch iterations decreases compared to the previous batch of iterations, the training of the layer will stop early. It will accelerate the finetuning process.
   - **Selective Update**: If the end-to-end accuracy does not improve after training a certain layer, discard the finetuning result of that layer.
   - **Adjust Learning Rate**: Besides the overall learning rate, you could set up a scheme to adjust learning rate layer-wise. For example, apply a larger learning rate on the layer that has a bigger loss.

Examples
--------

AdaRound
~~~~~~~~

This :doc:`example <../example_quark_onnx_adaround>` demonstrates quantizing a mobilenetv2_050.lamb_in1k model using the AMD Quark ONNX quantizer.

AdaQuant
~~~~~~~~

This :doc:`example <../example_quark_onnx_adaquant>` demonstrates quantizing a mobilenetv2_050.lamb_in1k model using the AMD Quark ONNX quantizer.
