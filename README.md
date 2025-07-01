<div align="center">

# AMD Quark Model Optimizer

[![Documentation](https://img.shields.io/badge/Documentation-latest-brightgreen.svg?style=flat)](https://quark.docs.amd.com/latest/)
[![version](https://img.shields.io/pypi/v/amd-quark?label=Release)](https://pypi.org/project/amd-quark/)
[![license](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
[![license](https://img.shields.io/badge/python-3.12-green)](https://www.python.org/)

[PyTorch Examples](https://quark.docs.amd.com/latest/pytorch/pytorch_examples.html) |
[ONNX Examples](https://quark.docs.amd.com/latest/onnx/onnx_examples.html) |
[Documentation](https://quark.docs.amd.com/) |
[Release Notes](https://quark.docs.amd.com/latest/release_note.html)

</div>

**AMD Quark** is a comprehensive cross-platform toolkit designed to simplify and enhance the quantization of deep learning models. Supporting both PyTorch and ONNX models, AMD Quark empowers developers to optimize their models for deployment on a wide range of hardware backends, achieving significant performance gains without compromising accuracy.

![image](https://quark.docs.amd.com/latest/_images/quark_stack.png)

## Features

| Feature Set            | PyTorch backend                                                                                                                     | ONNX backend                                                                              |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Data Types             | int4, uint4, int8, uint8, float16, bfloat16, OCP FP8 E4M3/E5M2, OCP MX INT8, OCP MX FP4, OCP MX FP6 E3M2/E2M3, OCP MX FP8 E4M3/E5M2 | int4, uint4, int8, uint8, int16, uint16, int32, uint32, float16, bfloat16, BFP16, MX4/MX6/MX9, OCP MX INT8, OCP MX FP4, OCP MX FP6 E3M2/E2M3, OCP MX FP8 E4M3/E5M2                              |
| Quant Mode             | eager mode, FX graph mode                                                                                                           | ONNX graph mode                                                                           |
| Quant Strategy         | static quant, dynamic quant, weight-only                                                                                            | static quant, dynamic quant, weight-only                                                  |
| Quant Scheme           | per-tensor, per-channel, per-group                                                                                                  | per-tensor, per-channel                                                                   |
| Symmetric              | symmetric, asymmetric                                                                                                               | symmetric, asymmetric                                                                     |
| Calibration Method     | MinMax, Percentile, MSE                                                                                                             | MinMax, Percentile, MinMSE, Entropy, NonOverflow                                          |
| Scale Type             | float16, float32                                                                                                                    | float16, float32                                                                          |
| KV-Cache Quant         | FP8 KV-Cache Quant                                                                                                                  | N/A                                                                                       |
| Supported Ops.         | `nn.Linear`, `nn.Conv2d`, `nn.ConvTranspose2d`, `nn.Embedding`, `nn.EmbeddingBag`,                                                  | Almost all ONNX ops,                                                                            |
|                        | `nn.BatchNorm2d`, `nn.BatchNorm3d`, `nn.LeakyReLU`, `nn.AvgPool2d`, `nn.AdaptiveAvgPool2d`                                          | see [Full List](https://quark.docs.amd.com/latest/onnx/user_guide_supported_optype_datatype.html) |
| Pre-Quant Optimization | SmoothQuant                                                                                                                         | QuaRot, SmoothQuant, CLE                                |
| Quantization Algorithm | AWQ, GPTQ                                                                                                                           | AdaQuant, AdaRound, GPTQ, Bias Correction                                                                  |
| Export Format          | ONNX, JSON-Safetensors, GGUF(Q4_1)                                                                                                  | N/A                                                                                       |
| Operating  Systems     | Linux {ROCm, CUDA, CPU}, Windows {CPU}                                                                                              | Linux {ROCm, CUDA, CPU}, Windows {CUDA, CPU}                                                    |

## Model Support Table

| Quantization Technique                | Supported Models                                                                                  |
| ------------------------------------- | ------------------------------------------------------------------------------------------------- |
| LLM Pruning                           | [Model Support](examples/torch/language_modeling/llm_pruning/example_quark_torch_llm_pruning.rst) |
| LLM Post Training Quantization (PTQ)  | [Model Support](examples/torch/language_modeling/llm_ptq/example_quark_torch_llm_ptq.rst)         |
| LLM Quantization Aware Training (QAT) | [Model Support](examples/torch/language_modeling/llm_qat/example_quark_torch_llm_qat.rst)         |
| Vision Model Quantization             | [Model Support](examples/torch/vision/model_support.md)                                           |
| Quark for ONNX                        | [Model Support](examples/onnx/model_support.md)

## Installation

Official releases of AMD Quark are available on PyPI https://pypi.org/project/amd-quark/, and can be installed with pip:

```shell
pip install amd-quark
```

> [!NOTE]\
> For full instructions to install AMD Quark from Python wheels or ZIP files, refer to our [🛠️Installation Guide](https://quark.docs.amd.com/latest/install.html). The Installation Guide also contains verification steps that apply to building from source.

### Installing from Source

1. Clone or download this repository.
2. Follow the steps from the [PyTorch](https://pytorch.org/get-started/locally/) website to install the appropriate PyTorch package for your system.
3. You can then build and install AMD Quark, and its dependencies, which are detailed in [requirements.txt](requirements.txt), by running:

```shell
git clone --recursive https://github.com/AMD/Quark
cd Quark

# [Optional] run git submodule if you are updating an existing Quark repository
git submodule sync
git submodule update --init --recursive

pip install .
```

## Resources

AMD Quark's documentation site contains [Getting Started](https://quark.docs.amd.com/latest/basic_usage.html), _API documentation_ for both [PyTorch](https://quark.docs.amd.com/latest/autoapi/pytorch_apis.html) and [ONNX](https://quark.docs.amd.com/latest/autoapi/onnx_apis.html) backends, and other detailed information.
The Installation Guide includes our [Recommended First Time User Installation](https://quark.docs.amd.com/latest/install.html#recommended-first-time-user-installation) guide, to get set up with Quark quickly.
Check out our _Frequently Asked Questions_ for both [PyTorch](https://quark.docs.amd.com/latest/pytorch/pytorch_faq.html) and [ONNX](https://quark.docs.amd.com/latest/onnx/onnx_faq.html) for more details.

* [📖Documentation](https://quark.docs.amd.com/)
* [📄FAQ (PyTorch)](https://quark.docs.amd.com/latest/pytorch/pytorch_faq.html)
* [📄FAQ (ONNX)](https://quark.docs.amd.com/latest/onnx/onnx_faq.html)

AMD Quark provides examples of Language Model and Image Classification model quantization, which can be found under [examples/torch/](examples/torch/) and  [examples/onnx/](examples/onnx/).
These examples are documented here:

* [💡PyTorch Examples](https://quark.docs.amd.com/latest/pytorch/pytorch_examples.html)
* [💡ONNX Examples](https://quark.docs.amd.com/latest/onnx/onnx_examples.html)

The examples folder also contain integrations of other quantizers under [examples/torch/extensions/](examples/torch/extensions/). You can read about those here:

* [Brevitas Integration](examples/torch/extensions/brevitas/example_quark_torch_brevitas.rst)
* [Integration with AMD Pytorch-light (APL)](examples/torch/extensions/pytorch_light/example_quark_torch_pytorch_light.rst).

## Contributing

AMD Quark is not set up to accept community contributions (bug reports, feature requests, or Pull Requests) just yet.
Please watch this space!

## License and Copyright

Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT.
See [LICENSE](LICENSE) file for detail.
