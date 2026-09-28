# Public Skills Catalog

The tables list every public top-level entry. “Form” describes body ownership;
“layer” describes responsibility as defined by
[architecture.md](architecture.md).

## Shared

| Name | Form / layer | Responsibility |
|---|---|---|
| [`quark-env-preflight`](../../skills/quark-env-preflight/SKILL.md) | Transitional / L0 | Collect and normalize OS, Python, accelerator, and container facts before installation or PTQ planning. |
| [`quark-install`](../../skills/quark-install/SKILL.md) | Self-contained / L1 | Install or verify AMD Quark and an accelerator-matched PyTorch in the selected Python environment. |
| [`quark-create-shapeshifter-pass`](../../skills/quark-create-shapeshifter-pass/SKILL.md) | Transitional / L1 | Author a registered ONNX or PyTorch ShapeShifter pass together with its test and documentation entry. |

## Torch

| Name | Form / layer | Responsibility |
|---|---|---|
| [`quark-torch-ptq`](../../skills/quark-torch-ptq/SKILL.md) | Self-contained / L2 | Run confirmed end-to-end PTQ for a PyTorch or Hugging Face LLM and verify the quantized output. |
| [`quark-torch-quant-perf`](../../skills/quark-torch-quant-perf/SKILL.md) | Self-contained / L2 | Run managed quantization with an accuracy gate, optional throughput and performance optimization, and final reports. |
| [`quark-torch-install`](../../skills/quark-torch-install/SKILL.md) | Transitional / L1 | Install or verify the PyTorch build that matches the user's accelerator. |
| [`quark-torch-model-intake`](../../skills/quark-torch-model-intake/SKILL.md) | Transitional / L1 | Inspect a Hugging Face Transformers or SafeTensors model and produce facts for PTQ planning. |
| [`quark-torch-result-validator`](../../skills/quark-torch-result-validator/SKILL.md) | Transitional / L1 | Validate exported SafeTensors, auxiliary files, and configuration without loading the full model. |
| [`quark-torch-llm-eval`](../../skills/quark-torch-llm-eval/SKILL.md) | Transitional / L1 | Evaluate LLM accuracy on AMD ROCm through the supported serving and benchmark frameworks. |
| [`quark-torch-file2file-quantization`](../../skills/quark-torch-file2file-quantization/SKILL.md) | Transitional / L1 | Quantize very large sharded SafeTensors checkpoints without loading the whole model. |
| [`quark-torch-shrink-model`](../../skills/quark-torch-shrink-model/SKILL.md) | Transitional / L1 | Create a small representative SafeTensors model for faster debugging and workflow tests. |

## ONNX

| Name | Form / layer | Responsibility |
|---|---|---|
| [`quark-onnx-autosearch-pro`](../../skills/quark-onnx-autosearch-pro/SKILL.md) | Transitional / L3 | Search ONNX quantization configurations with AutoSearchPro. |
| [`quark-onnx-debug`](../../skills/quark-onnx-debug/SKILL.md) | Transitional / L1 | Diagnose Quark ONNX installation, calibration, quantization, custom-op, provider, or export failures. |
| [`quark-onnx-install`](../../skills/quark-onnx-install/SKILL.md) | Transitional / L1 | Install or verify the ONNX Runtime and ONNX packages for the selected accelerator. |
| [`quark-onnx-model-intake`](../../skills/quark-onnx-model-intake/SKILL.md) | Transitional / L1 | Inspect an ONNX graph and report structural, compatibility, and quantization-planning facts. |
| [`quark-onnx-ptq`](../../skills/quark-onnx-ptq/SKILL.md) | Transitional / L2 | Run an end-to-end ONNX-to-ONNX PTQ workflow with planning and confirmed execution. |
| [`quark-onnx-result-validator`](../../skills/quark-onnx-result-validator/SKILL.md) | Transitional / L1 | Validate quantized ONNX graph structure, metadata, auxiliary data, and preserved initializers. |
| [`quark-onnx-shapeshifter-run`](../../skills/quark-onnx-shapeshifter-run/SKILL.md) | Transitional / L1 | Apply existing ShapeShifter graph passes to an ONNX model. |
| [`quark-onnx-subgraph-partitioner`](../../skills/quark-onnx-subgraph-partitioner/SKILL.md) | Transitional / L1 | Partition an ONNX graph into named functional subgraphs and produce `subgraph_partition.json`. |
