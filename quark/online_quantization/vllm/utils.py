"""Utility functions for vLLM online quantization."""

import torch

FP8_E4M3_MAX = 448.0


def quark_aligned_fp8_per_channel_quant(
    weight: torch.Tensor,
    *,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    fp8_max: float = FP8_E4M3_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel FP8 quant matching Quark Torch's offline path.

    Mirrors ``calculate_fp8_quant_parameters`` (``scale = amax / fp8_max``)
    + ``real_quantize_fp8_per_channel_with_scale`` (``scale.to(inputs.dtype)``
    then ``inputs / scale`` then clamp then cast). The scale is round-tripped
    through ``bfloat16`` so it matches what offline writes to disk.

    Returns ``(qweight, scale)`` where ``qweight`` has dtype ``fp8_dtype`` and
    ``scale`` is shape ``(N,)`` ``float32`` (with bf16-precision values).
    """
    inputs_dtype = weight.dtype
    weight_fp32 = weight.to(torch.float32)
    amax = weight_fp32.abs().amax(dim=1, keepdim=True)
    scale_fp32 = amax / fp8_max
    # Round-trip through bf16 to match offline on-disk storage precision; the
    # subsequent divide then happens in bf16 (per Quark's
    # ``real_quantize_fp8_per_channel_with_scale``).
    scale_inputs = scale_fp32.to(inputs_dtype)
    safe_scale = torch.where(
        scale_inputs == 0,
        torch.ones_like(scale_inputs),
        scale_inputs,
    )
    qweight = (weight.to(inputs_dtype) / safe_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return qweight, scale_inputs.squeeze(-1).to(torch.float32)


def materialize_meta_weight_(layer: torch.nn.Module) -> None:
    """Replace ``layer.weight`` (allocated on meta) with a real, uninitialized
    tensor on ``layer._load_device``. Used by the dummy-weights path, where
    vLLM bypasses the layerwise pipeline and calls
    ``process_weights_after_loading`` directly without first streaming
    weights through ``online_process_loader``.
    """
    from vllm.model_executor.model_loader.weight_utils import (
        initialize_single_dummy_weight,
    )
    from vllm.model_executor.parameter import ModelWeightParameter

    if layer.weight.device != torch.device("meta"):
        return

    load_device = getattr(layer, "_load_device", torch.get_default_device())
    new = ModelWeightParameter(
        data=torch.empty_like(layer.weight, device=load_device),
        input_dim=1,
        output_dim=0,
        weight_loader=layer.weight.weight_loader,
    )
    layer.register_parameter("weight", new)
    initialize_single_dummy_weight(layer.weight)
