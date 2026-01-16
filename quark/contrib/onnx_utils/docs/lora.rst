..
    Copyright (c) 2024 Advanced Micro Devices, Inc.

LoRA
====

LoRA refers to Low Rank Adapter.

Given an ONNX model, and an LoRA adapter:

1. We can apply the LoRA adapter to the model.
2. The lora application happened on each targeted matmul projection.
3. The LoRA formula is `W_{new} = W + \\scaling_factor \\cdot (A \\cdot B)`


This tool preprocesses LoRA adapter files for use with our NPU-compatible LLM models.
It converts LoRA weights from SafeTensors format to binary format with BFP16
quantization and specific tensor layouts optimized for hardware acceleration.

Key Features:
- Loads LoRA adapters from SafeTensors files
- Converts weights to BFP16 format for efficient inference
- Reshapes tensors for optimized memory layouts
- Generates header and binary files for runtime loading
- Supports verification of written files
- Performs QKV fusion plus SVD-based delta matrix factorization
- Pads lower-rank matrices to target rank for consistency

Usage
-----

Lora Processing requires:

1. *lora-file* - Path to lora adapter
2. *base-model-header* - Base model header file to use for lora adapter

Optional:

1. *scaling-factor* - Scaling factor of lora weights, default is 1.0
2. *output-dir* - Directory to save output files, default is the "current_directory/lora_bin".
3. *lora-name* - Name of the target lora adapter, default is "lora".
4. *verify* - Verify the written files with original data, default is False.
5. *verbose* - Enable verbose output for debugging, default is False.


See the :ref:`command-line arguments <cli:lora-process>` for the full list of arguments and options.

Example
-------

As an example, the :onnxUtilsTree:`lora-process  --lora-file adapter_model.safetensors  --lora-name gsm8k --base-model-header /path/to/model.pb.bin  --scaling-factor 0.25`
preprocesses the LoRA adapter file `adapter_model.safetensors` for the base model header file `/path/to/model.pb.bin` with a scaling factor of `0.25`. The output files will be saved in the current directory under `lora_bin` as `gsm8k.bin` and `gsm8k.pb.bin`.
