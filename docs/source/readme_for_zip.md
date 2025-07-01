# Quark

## Installation

1. Prerequisites
    Python 3.10, 3.11 or 3.12 is required. Python 3.13 is not supported at the moment.
    Install PyTorch for the compute platform(CUDA, ROCM, CPU...). Version of torch >= 2.2.0.
    Install ONNX of version >= 1.16.0, ONNX Runtime of version ~= 1.22.0, onnxruntime-extensions of version >= 0.4.2

2. Install quark wheel package in current path by

    ```bash
    pip install amd_quark*.whl
    ```

3. (Optional) Verify the installation by running `python -c "import quark"`. If it does not report error, the installation is done.

4. (Optional) Compile the `fast quantization kernels`. When using Quark-PyTorch's quantization APIs for the first time, it will compile the `fast quantization kernels` using your installed Torch and CUDA if available. This process may take a few minutes but subsequent quantization calls will be much faster. To invoke this compilation now and check if it is successful, run the following command:

    ```bash
    python -c "import quark.torch.kernel"
    ```

5. (Optional) Compile the `custom operators library`. When using Quark-ONNX's custom operators for the first time, it will compile the `custom operators library` using your local environment. To invoke this compilation now and check if it is successful, run the following command:

    ```bash
    python -c "import quark.onnx.operators.custom_ops"
    ```

## Documentation

For more information about Quark, please refers to the HTML documentation at `docs/html/index.html`.

## Examples

For more examples of Quark, please refer to ``examples`` folder.

## License

Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
