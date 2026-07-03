"""Quantization configuration for vLLM online quantization.

``QuarkVllmOnlineConfig`` is a thin **proxy** over two fully-formed inner
configs:

* ``_online_quant_config`` (``QuarkConfig``): the target online scheme. All
  per-layer config lookup (``layer_quant_config`` fnmatch, fused-shard
  consistency, ``layer_type_quant_config``, ``global_quant_config``
  fallback, ``packed_modules_mapping``, ``export.kv_cache_group``
  validation) is delegated to this object's existing machinery.
* ``_offline_quant_config`` (any ``QuantizationConfig``, may be ``None``):
  the input checkpoint's existing offline quant config. When present, we
  load its native parameters via its ``get_quant_method`` and wrap
  the result with our requant composers.

We only ``get_quant_method`` (the dispatch) and small public-surface
forwarders (``apply_vllm_mapper``, ``get_cache_scale``,
``packed_modules_mapping``) — everything else delegates.
"""

import copy
from typing import Any, cast

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import (
    QuantizationMethods,
    get_quantization_config,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.quark.utils import should_ignore_layer

from .quant_method.linear import (
    QuarkVllmOnlineFp8Method,
    QuarkVllmOnlineMxfp4Method,
)
from .quant_method.moe import (
    OnlineRequantMoeMethod,
    QuarkVllmOnlineFp8MoEMethod,
    QuarkVllmOnlineMxfp4MoEMethod,
)
from .quant_method.requant import OnlineRequantMethod


@register_quantization_config("quark_online")
class QuarkVllmOnlineConfig(QuantizationConfig):
    """See module docstring."""

    def __init__(
        self,
        online_quant_config: QuarkConfig,
        offline_quant_config: QuantizationConfig | None,
    ):
        # Set the inner configs BEFORE super().__init__() — the base
        # ``QuantizationConfig.__init__`` does
        # ``self.packed_modules_mapping = dict()``, which would otherwise
        # hit our setter (below) before we have a place to forward to.
        self._online_quant_config = online_quant_config
        self._offline_quant_config = offline_quant_config
        super().__init__()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QuarkVllmOnlineConfig":
        cfg = copy.deepcopy(config)
        online_dict = cfg.get("online_quant")
        if online_dict is None:
            raise ValueError(
                "quark_online quantization requires the HF "
                "quantization_config to contain an 'online_quant' sub-key. "
                "Use quark.online_quantization.vllm.HF_QUANTIZATION_CONFIGS "
                "or quark.online_quantization.vllm.online_quant_overrides "
                "to construct hf_overrides."
            )
        online_quant_config = QuarkConfig.from_config(online_dict)

        offline_quant_config: QuantizationConfig | None = None
        offline_dict = cfg.get("offline_quant")
        if offline_dict is not None:
            offline_method_name = offline_dict.get("quant_method")
            if not offline_method_name:
                raise ValueError("offline_quant sub-key is present but has no quant_method")
            offline_cls = get_quantization_config(offline_method_name)
            offline_quant_config = offline_cls.from_config(offline_dict)

        return cls(online_quant_config, offline_quant_config)

    # ------------------------------------------------------------------
    # Class-level metadata (vLLM reads these via the class)
    # ------------------------------------------------------------------

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "quark_online"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        # Same as QuarkConfig.get_supported_act_dtypes (which isn't a real
        # @classmethod in vLLM, so we can't call it unbound).
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return QuarkConfig.get_min_capability()

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    # ------------------------------------------------------------------
    # Instance surface vLLM and model code consume
    # ------------------------------------------------------------------

    @property
    def packed_modules_mapping(self) -> dict[str, list[str]]:
        """vLLM model code reads this to resolve fused-layer shard names
        (``qkv_proj`` → ``[q_proj, k_proj, v_proj]`` etc.). It also *writes*
        to it (``utils.py:291``: ``quant_config.packed_modules_mapping =
        packed_mapping``), so the setter must propagate too. We treat the
        online config as the source of truth; the offline config is kept in
        sync so its own dispatch still works."""
        return self._online_quant_config.packed_modules_mapping

    @packed_modules_mapping.setter
    def packed_modules_mapping(self, value: dict[str, list[str]]) -> None:
        self._online_quant_config.packed_modules_mapping = value
        if self._offline_quant_config is not None:
            self._offline_quant_config.packed_modules_mapping = value

    def apply_vllm_mapper(self, hf_to_vllm_mapper: Any) -> None:
        """Forward to both inner configs so their stored quant dicts get
        the same name remapping vLLM applies for non-HF model structures."""
        self._online_quant_config.apply_vllm_mapper(hf_to_vllm_mapper)
        if self._offline_quant_config is not None:
            self._offline_quant_config.apply_vllm_mapper(hf_to_vllm_mapper)

    def get_cache_scale(self, name: str) -> str | None:
        """KV-cache scale name remapping. Offline checkpoints are the ones
        that ship KV-cache scales on disk, so route through offline first;
        fall back to online (rare but possible if a user wires KV-cache
        quant via the online config)."""
        if self._offline_quant_config is not None:
            scale = self._offline_quant_config.get_cache_scale(name)
            if scale is not None:
                return scale
        return self._online_quant_config.get_cache_scale(name)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        ret = self._get_quant_method(layer, prefix)
        return ret

    def _get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        # Offline method (None in scenario A, or when offline ignores layer).
        offline_method: QuantizeMethodBase | None = None
        if self._offline_quant_config is not None:
            offline_method = self._offline_quant_config.get_quant_method(layer, prefix)

        online_dict = self._online_quant_config.quant_config
        online_exclude = cast(list[str], online_dict.get("exclude") or [])
        if should_ignore_layer(
            prefix,
            ignore=online_exclude,
            fused_mapping=self.packed_modules_mapping,
        ):
            if offline_method is not None:
                return offline_method
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            return None

        # Resolve the per-layer matched config via QuarkConfig's existing
        # ``_find_matched_config`` (handles fused-shard consistency,
        # ``layer_quant_config`` fnmatch lookup, ``layer_type_quant_config``
        # fallback, ``global_quant_config`` default).
        matched_cfg = self._online_quant_config._find_matched_config(prefix, layer)

        if isinstance(layer, LinearBase):
            online_method = self._build_online_method(matched_cfg)
            if offline_method is None or isinstance(offline_method, UnquantizedLinearMethod):
                return online_method
            return OnlineRequantMethod(
                offline=offline_method,
                online=online_method,
                offline_cfg=self._offline_dict(),
            )

        if isinstance(layer, FusedMoE):
            try:
                online_method = self._build_online_moe_method(layer, matched_cfg)
            except NotImplementedError as e:
                # Only warn when there's an offline method to fall back to.
                if offline_method is not None:
                    from vllm.logger import init_logger

                    init_logger(__name__).warning_once(
                        "Online MoE re-quant unavailable for %s; keeping the offline MoE method (%s). Reason: %s",
                        type(layer).__name__,
                        type(offline_method).__name__,
                        e,
                    )
                return offline_method

            # Scenario A (no offline ckpt): return online_method directly,
            # else vLLM falls back to bf16 and the MoE skips quant.
            if offline_method is None:
                return online_method
            return OnlineRequantMoeMethod(
                offline_moe=offline_method,
                online=online_method,
                offline_cfg=self._offline_dict(),
            )

        return offline_method

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _offline_dict(self) -> dict[str, Any] | None:
        """The raw offline config dict (needed by the requant wrappers to
        know how to dequant, e.g. ``weight_block_size``). Offline
        ``QuantizationConfig`` subclasses don't have a uniform attribute
        name for the original dict, so try the conventional ones."""
        cfg = self._offline_quant_config
        if cfg is None:
            return None
        for attr in ("quant_config", "_quant_config", "config"):
            d = getattr(cfg, attr, None)
            if isinstance(d, dict):
                return d
        # Last resort: reconstruct the minimal fields the requant wrappers
        # actually read (``quant_method``, ``weight_block_size``).
        return {
            "quant_method": cfg.get_name(),
            "weight_block_size": getattr(cfg, "weight_block_size", None),
        }

    def _build_online_method(self, matched_cfg: dict[str, Any]) -> QuantizeMethodBase:
        weight_dtype = (matched_cfg.get("weight") or {}).get("dtype", "")
        if weight_dtype == "fp4":
            return QuarkVllmOnlineMxfp4Method()
        return QuarkVllmOnlineFp8Method()

    def _build_online_moe_method(self, layer: Any, matched_cfg: dict[str, Any]) -> Any:
        weight_cfg = matched_cfg.get("weight") or {}
        input_cfg = matched_cfg.get("input_tensors")
        weight_dtype = weight_cfg.get("dtype", "")

        if weight_dtype == "fp4":
            return QuarkVllmOnlineMxfp4MoEMethod(weight_cfg, input_cfg, layer.moe_config)

        if input_cfg is None:
            raise ValueError("online_quant FP8 MoE requires an input_tensors config (activation quant spec).")
        method = QuarkVllmOnlineFp8MoEMethod(weight_cfg, input_cfg, layer.moe_config)

        # See _build_online_moe_method docstring in the prior version: on
        # ROCm without AITER the inherited apply path crashes on fp8e4b8.
        # Raise so the dispatcher can fall back to offline.
        from vllm.platforms import current_platform

        if current_platform.is_rocm() and not method.rocm_aiter_moe_enabled and not method.use_marlin:
            raise NotImplementedError(
                "FP8 per-channel MoE forward on ROCm needs AITER (install "
                "the `aiter` package and set VLLM_ROCM_USE_AITER=1) or "
                "Marlin; neither is available, and vLLM's generic Triton "
                "FP8 MoE kernel does not support AMD's fp8e4b8 dtype."
            )
        return method
