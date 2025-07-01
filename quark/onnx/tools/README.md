# Tools

- [Convert a float16 model to a float32 model](#convert-a-float16-model-to-a-float32-model)
- [Convert a NCHW input model to a NHWC model](#convert-a-nchw-input-model-to-a-nhwc-model)
- [Quantize a Float Model with Random Data](#quantize-a-float-model-with-random-data)
- [Convert a A8W8 NPU model to a A8W8 CPU model](#convert-a-a8w8-npu-model-to-a-a8w8-cpu-model)

## Convert a float16 model to a float32 model

Since the quark.onnx tool only supports float32 models quantization currently, converting a model from float16 to float32 is required when quantizing a float16 model.

Use the convert_fp16_to_fp32 tool to convert a float16 model to a float32 model:

```bash
python -m quark.onnx.tools.convert_fp16_to_fp32 --input $FLOAT_16_ONNX_MODEL_PATH --output $FLOAT_32_ONNX_MODEL_PATH
```

If the input model is larger than 2GB, please use this command instead.

```bash
python -m quark.onnx.tools.convert_fp16_to_fp32 --input $FLOAT_16_ONNX_MODEL_PATH --output $FLOAT_32_ONNX_MODEL_PATH --save_as_external_data
```

## Convert a float32 model to a float16 model

Since the quark.onnx tool supports both float32 and float16 models quantization currently, converting a model from float32 to float16 is required when quantizing a float32 model.

Use the convert_fp32_to_fp16 tool to convert a float32 model to a float16 model:

```bash
python -m quark.onnx.tools.convert_fp32_to_fp16 --input $FLOAT_16_ONNX_MODEL_PATH --output $FLOAT_16_ONNX_MODEL_PATH
```

If the input model is larger than 2GB, please use this command instead.

```bash
python -m quark.onnx.tools.convert_fp32_to_fp16 --input $FLOAT_16_ONNX_MODEL_PATH --output $FLOAT_16_ONNX_MODEL_PATH --save_as_external_data
```

Use the convert_fp32_to_fp16 tool to convert a float32 model to a float16 model with inputs and outputs types maintain the float32 data type:

```bash
python -m quark.onnx.tools.convert_fp32_to_fp16 --input $FLOAT_16_ONNX_MODEL_PATH --output $FLOAT_16_ONNX_MODEL_PATH --keep_io_types
```

## Convert a float32 model to a bfloat16 model

Since there are more and more bfloat16 deployment demands. We need a conversion tool to convert a float32 model to a bfloat16 model. We provide 4 bfloat16 implementation formats: vitisqdq, with_cast, simulate_bf16 and bf16. The vitisqdq means that the bfloat16 conversion is implemented by inserting VitisQDQ of bfloat16. The with_cast means that the bfloat16 conversion is implemented by inserting Cast operations to convert from float32 to bfloat16. The simulate_bf16 means that the bfloat16 conversion is implemented by that all bfloat16 weights are stored as float format. The bf16 means that the bfloat16 conversion is implemented by that the float model is directly converted to bfloat16 and only the input and output are remained as float. The default value is with_cast.

Use the convert_fp32_to_bf16 tool to convert a float32 model to a bfloat16 model:

```bash
python -m quark.onnx.tools.convert_fp32_to_bf16 --input $FLOAT_32_ONNX_MODEL_PATH --output $BFLOAT_16_ONNX_MODEL_PATH --format $BFLOAT_FORMAT
```

If the input model is larger than 2GB, please use this command instead.

```bash
python -m quark.onnx.tools.convert_fp32_to_bf16 --input $FLOAT_32_ONNX_MODEL_PATH --output $BFLOAT_16_ONNX_MODEL_PATH --format $BFLOAT_FORMAT --save_as_external_data
```

## Convert a float16 model to a bfloat16 model

Since there are more and more bfloat16 deployment demands. We need a conversion tool to convert a float16 model to a bfloat16 model. We provide 4 bfloat16 implementation formats: vitisqdq, with_cast, simulate_bf16 and bf16. The vitisqdq means that the bfloat16 conversion is implemented by inserting VitisQDQ of bfloat16. The with_cast means that the bfloat16 conversion is implemented by inserting Cast operations to convert from float16 to bfloat16. The simulate_bf16 means that the bfloat16 conversion is implemented by that all bfloat16 weights are stored as float format. The bf16 means that the bfloat16 conversion is implemented by that the float16 model is directly converted to bfloat16 and only the input and output are remained as float16. The default value is with_cast.

Use the convert_fp16_to_bf16 tool to convert a float16 model to a bfloat16 model:

```bash
python -m quark.onnx.tools.convert_fp16_to_bf16 --input $FLOAT_16_ONNX_MODEL_PATH --output $BFLOAT_16_ONNX_MODEL_PATH --format $BFLOAT_FORMAT
```

If the input model is larger than 2GB, please use this command instead.

```bash
python -m quark.onnx.tools.convert_fp16_to_bf16 --input $FLOAT_16_ONNX_MODEL_PATH --output $BFLOAT_16_ONNX_MODEL_PATH --format $BFLOAT_FORMAT --save_as_external_data
```

## Convert a float32 model to a bfp16 model

Since there are more and more bfp16 deployment demands. We need a conversion tool to directly convert a float32 model to a bfp16 model.

Use the convert_fp32_to_bfp16 tool to convert a float32 model to a bfp16 model:

```bash
python -m quark.onnx.tools.convert_fp32_to_bfp16 --input $FLOAT_32_ONNX_MODEL_PATH --output $BFP_16_ONNX_MODEL_PATH
```

If the input model is larger than 2GB, please use this command instead.

```bash
python -m quark.onnx.tools.convert_fp32_to_bfp16 --input $FLOAT_32_ONNX_MODEL_PATH --output $BFP_16_ONNX_MODEL_PATH --save_as_external_data
```

## Convert a float16 model to a bfp16 model

Since there are more and more bfp16 deployment demands. We need a conversion tool to directly convert a float16 model to a bfp16 model.

Use the convert_fp16_to_bfp16 tool to convert a float16 model to a bfp16 model:

```bash
python -m quark.onnx.tools.convert_fp16_to_bfp16 --input $FLOAT_16_ONNX_MODEL_PATH --output $BFP_16_ONNX_MODEL_PATH
```

If the input model is larger than 2GB, please use this command instead.

```bash
python -m quark.onnx.tools.convert_fp16_to_bfp16 --input $FLOAT_16_ONNX_MODEL_PATH --output $BFP_16_ONNX_MODEL_PATH --save_as_external_data
```

## Convert a NCHW input model to a NHWC model

Given that some models are designed with an input shape of NCHW instead of NHWC, it's recommended to convert an NCHW input model to NHWC before quantizing a float32 model.

**Note**: The data layout, whether NCHW or NHWC, does not influence the quantization process itself. However, deployment efficiency is affected by the kernel design, which is often optimized for NHWC. Consequently, when input data is in NCHW format, a conversion to NHWC is recommended. This conversion introduces a small computational overhead, though the overall performance benefits from the optimized layout. While a transpose operation is required for the format change, the total number of other operations remains constant.

Use the convert_nchw_to_nhwc tool to convert a NCHW model to a NHWC model:

```bash
python -m quark.onnx.tools.convert_nchw_to_nhwc --input $NCHW_ONNX_MODEL_PATH --output $NHWC_ONNX_MODEL_PATH
```

## Convert a A8W8 NPU model to a A8W8 CPU model

Given that some models are quantized by A8W8 NPU, it's convenient and efficient to convert them to A8W8 CPU models.

Use the convert_a8w8_npu_to_a8w8_cpu tool to convert a A8W8 NPU model to a A8W8 CPU model:

```bash
python -m quark.onnx.tools.convert_a8w8_npu_to_a8w8_cpu --input [INPUT_PATH] --output [OUTPUT_PATH]
```

## Print names and quantity of A16W8 and A8W8 Conv for mix-precision models

Given that some models are mixed precision such as A18W8 and A8W8 mixed.

Use the print_a16w8_a8w8_nodes tool to print names and quantity of A16W8 and A8W8 Conv, ConvTranspose and Gemm:

```bash
python -m quark.onnx.tools.print_a16w8_a8w8_nodes --input [INPUT_PATH]
```

## Remove initializers from input and upgrade ir versions if it is below 4

Models with ir_version below 4 requires to include initilizer in graph input that may prevent some of the graph optimizations, like const folding and conv-bn folding.

Use the remove_initializer_from_input tool to upgrade ir versions if it is below 4 and remove initializers from input.

```bash
python -m quark.onnx.tools.remove_initializer_from_input --input [INPUT_PATH] --output [OUTPUT_PATH]
```

## Convert a U16U8 quantized model to a U8U8 model

Convert a U16U8 (activations are quantized by UINT16 and weights by UINT8) to a U8U8 model without calibration.

Use the convert_u16u8_to_u8u8 tool to do the conversion:

```bash
python -m quark.onnx.tools.convert_u16u8_to_u8u8 --input [INPUT_PATH] --output [OUTPUT_PATH]
```

## Remove BFloat16 Casts for a BFloat16 quantized model

Remove Casts(float32->bfloat16 or bfloat16->float32) in a bfloat16 quantized model.

Use the remove_bf16_cast tool:

```bash
python -m quark.onnx.tools.remove_bf16_cast --input_model_path [INPUT_PATH] --output_model_path [OUTPUT_PATH]
```

## Convert the opset version of input model

Convert the opset version of input model to the target version.

Use the convert_opset_version tool:

```bash
python -m quark.onnx.tools.convert_opset_version --input [INPUT_PATH] --target_opset [TARGET_OPSET_VERSION] --output [OUTPUT_PATH]
```

## Evaluate accuracy between baseline and quantized results folders

We often need to compare the differences in output images before and after quantization. Currently, we support four metrics: cosine similarity, L2 loss, PSNR, and VMAF, as well as three formats: JPG, PNG and NPY.

Use the evaluate tool:

```bash
python -m quark.onnx.tools.evaluate --baseline_results_folder [BASELINE_RESULTS_FOLDER_PATH] --quantized_results_folder [QUANTIZED_RESULTS_FOLDER_PATH]
```

## Quantize a Float Model with Random Data

Customers often need to verify the performance of the quantized model regardless of quantization accuracy. So we support the quantization without calibration dataset using random data generated automatically.

Use the random_quantize tool:

```bash
python -m quark.onnx.tools.random_quantize --input_model_path [FLOAT_MODEL_PATH] --quantized_model_path [QUANTIZED_MODEL_PATH]
```

## Assign Shapes for All Tensors in A Given Model

An onnx model may be missing the shape of some tensors. So we provide a tool that automatically assigns the correct shape to all tensors, regardless of whether the input model is a float model or a QDQ model.

Use the fix_shapes tool:

```bash
python -m quark.onnx.tools.fix_shapes --input_model_path [INPUT_MODEL_PATH] --output_model_path [OUTPUT_MODEL_PATH]
```

## Convert the Int32 Bias of the Quantized Model to Int16

The bias in a quantized model may need to be int16 instead of int32 in some cases. So we provide a tool that converts the int32 bias of a quantized model to int16.
**Note**: 1. ONNXRuntime only supports Int16 Bias inference when the opset version is 21 or higher, so please ensure that the input model's opset version is 21 or higher. 2. It is recommended to use the parameter **Int16Bias** together with **ADAROUND** or **ADAQUANT**; otherwise, the quantized model with Int16 bias may suffer from poor accuracy.

Use the `convert_bias_int32_to_int16` tool:

```bash
python -m quark.onnx.tools.convert_bias_int32_to_int16 --input_model_path [INPUT_MODEL_PATH] --output_model_path [OUTPUT_MODEL_PATH]
```

<!--
## License
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
-->
