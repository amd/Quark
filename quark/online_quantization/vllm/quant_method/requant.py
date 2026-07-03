"""Online re-quantization wrapper for ``LinearBase`` layers (scenario B).

The shape mirrors the MoE version. ``offline`` handles loading the on-disk
offline-quantized weights; this class dequants to bf16, drops the offline
params, stages bf16 in ``layer.weight``, and delegates to
``online.process_weights_after_loading`` — which performs the actual
bf16->target quant op and finalization exactly as in scenario A.
"""

import gc
from typing import Any

import torch
from torch.nn import Parameter
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)

from ..dequant import dequant_layer


class OnlineRequantMethod(LinearMethodBase):
    """Layer-local dequant→requant for offline-quantized ``LinearBase``."""

    uses_meta_device: bool = True

    def __init__(
        self,
        offline: LinearMethodBase,
        online: LinearMethodBase,
        offline_cfg: dict[str, Any],
    ):
        self.offline = offline
        self.online = online
        self.offline_cfg = offline_cfg

    def create_weights(self, layer: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
        # Snapshot pre-existing params (notably ``bias``, registered by the
        # LinearBase subclass). They must survive the requant cleanup.
        pre_existing = {name for name, p in layer.named_parameters(recurse=False) if p is not None}

        self.offline.create_weights(layer, *args, **kwargs)
        layer.orig_dtype = kwargs.get("params_dtype", layer.weight.dtype)

        layer._offline_param_names = [
            name for name, p in layer.named_parameters(recurse=False) if p is not None and name not in pre_existing
        ]

        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._process_weights_after_loading(layer)

    def _process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        bf16_weight = dequant_layer(layer, self.offline_cfg)

        # Drop offline params (weight + scales). Bias is pre-existing and
        # not in _offline_param_names.
        for name in layer._offline_param_names:
            if hasattr(layer, name):
                delattr(layer, name)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Stage bf16 weight in the slot the online method reads from.
        layer.weight = Parameter(bf16_weight, requires_grad=False)

        # Make sure online's own pwal runs (it guards on this flag).
        if hasattr(layer, "_already_called_process_weights_after_loading"):
            delattr(layer, "_already_called_process_weights_after_loading")
        self.online.process_weights_after_loading(layer)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.online.apply(layer, x, bias)
