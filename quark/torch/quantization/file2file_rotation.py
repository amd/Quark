#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Model-free Hadamard rotation for the file-to-file quantization flow.

The standard rotation path (:class:`quark.torch.algorithm.rotation.rotation.RotationProcessor`)
walks a live ``nn.Module``: it fuses normalization weights, inserts wrapper modules, and
relies on cross-layer structure. The file-to-file flow never materializes a model — it
processes safetensors shards one tensor at a time — so that path cannot run here.

This module reimplements the subset of rotation that is expressible as a **per-tensor
weight transform** (plus a persisted ``input_rotation`` buffer that inference uses to
reconstruct the matching activation transform):

- **Online R1** (``r1=True, online_r1_rotation=True``): rotate each target linear's input
  channels by a fixed Hadamard, and emit an ``input_rotation`` buffer. At load time,
  :class:`quark.torch.export.nn.modules.qparamslinear.QParamsLinearWithRotation` rebuilds
  the activation-side Hadamard transform from that buffer and the rotation size.
- **R2** (``r2=True``): rotate ``v_proj`` output channels and ``o_proj`` input channels by
  a per-head Hadamard. This is fully fused into the weights (no buffer, no inference-time
  reconstruction), matching the non-trainable ``RotationProcessor.r2`` behavior.
- **R4** (``r4=True``): rotate ``down_proj`` input channels (online, same buffer mechanism
  as online R1).

Unsupported configurations raise with an actionable message:

- ``trainable=True`` — learned rotation (SpinQuant) requires gradient-based training,
  which the file-to-file flow cannot do.
- ``r1=True, online_r1_rotation=False`` (offline R1) — needs cross-shard normalization
  fusion and a residual-basis rotation shared across many tensors.
- ``r3=True`` — implemented by monkeypatching the model forward and not reconstructable on
  reload (see :meth:`RotationProcessor.prepare_model_for_reloading_fake`).
- ``random_r1=True`` / ``random_r2=True`` — a random basis cannot be deterministically
  reconstructed at inference; only fixed Hadamard rotations persist.
- an online rotation target that also matches an ``exclude`` pattern — excluded layers are
  not wrapped at load time, so an online rotation on them would break correctness.

Supported architectures and assumptions
----------------------------------------

The model-free target resolution is written for **Llama/Qwen-style dense and MoE decoder
transformers** (the families this flow is validated against). It assumes:

- **Decoder layers** are integer-indexed under ``RotationConfig.model_decoder_layers``,
  i.e. tensor names of the form ``"<model_decoder_layers>.<int>.<...>"`` (see
  :func:`iter_layer_indices`). Non-integer or non-nested layer containers are not found.
- **R1 rotation size** comes from ``RotationConfig.rotation_size`` when set, else falls back
  to the config's ``hidden_size``.
- **R2 rotation size** is the attention ``head_dim``: taken from the config's ``head_dim``
  when present, else derived as ``hidden_size // num_attention_heads``. The derived fallback
  is only correct when ``head_dim == hidden_size / num_attention_heads`` (not true for some
  GQA/MLA variants that set an explicit ``head_dim``); such models must provide ``head_dim``
  in the config.
- **R4 rotation size** comes from ``RotationConfig.rotation_size`` when set, else falls back
  to ``moe_intermediate_size`` / ``intermediate_size`` (i.e. it assumes
  ``down_proj.in_features`` equals that MLP intermediate dim).

These derivations are the primary source of out-of-scope failures. Rather than guess for
unfamiliar architectures, the flow fails loudly: every resolved rotation size is checked to
divide the corresponding tensor dimension (see :func:`_rotate_input_channels_online` for the
online path and :func:`apply_rotation_to_tensor` for R2), so a mismatched size raises a
clear, module-named error instead of silently producing wrong weights. To support a model
outside these assumptions, set ``rotation_size`` / ``head_dim`` explicitly in the config.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any

import torch

from quark.common.utils.log import ScreenLogger
from quark.torch.algorithm.rotation.hadamard import _get_hadamard_K
from quark.torch.algorithm.rotation.rotation_utils import (
    build_input_rotation_int8,
    expand_scaling_layer_targets,
    get_rotation_matrix,
    rotate_in_channels,
    rotate_input_channels_hadamard,
    rotate_out_channels,
)
from quark.torch.quantization.config.config import OnlineRotationConfig, QConfig, RotationConfig
from quark.torch.quantization.file2file_utils import iter_layer_indices, match_modules

logger = ScreenLogger(__name__)

__all__ = ["RotationPlan", "build_rotation_plan", "apply_rotation_to_tensor"]


def _module_name_of_weight(weight_tensor_name: str) -> str:
    """``"...q_proj.weight"`` -> ``"...q_proj"``."""
    return weight_tensor_name.removesuffix(".weight")


def _text_config_view(hf_model_config: dict[str, object] | None) -> dict[str, object]:
    """Return the sub-config carrying the transformer dims.

    Mirrors :func:`file2file_quantization._get_model_dtype_from_hf_model_config`:
    VLM/multimodal checkpoints nest the language-model dims under ``text_config``.
    """
    if not hf_model_config:
        return {}
    text_config = hf_model_config.get("text_config")
    if isinstance(text_config, dict):
        # Prefer nested values, but fall back to the root for anything missing.
        merged = dict(hf_model_config)
        merged.update(text_config)
        return merged
    return dict(hf_model_config)


def _get_dim(cfg: dict[str, object], *keys: str) -> int | None:
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, int):
            return value
    return None


@dataclass
class RotationPlan:
    """Per-tensor rotation instructions resolved from a :class:`RotationConfig`.

    Keys are full weight-tensor names (ending in ``.weight``). Values are the rotation
    size to apply.

    - ``online_in``: input-channel online rotations (R1 targets, R4 down_proj). These
      rotate the weight AND emit an ``input_rotation`` buffer for inference reconstruction.
    - ``r2_out``: output-channel rotations (v_proj), fully fused, no buffer.
    - ``r2_in``: input-channel rotations (o_proj), fully fused, no buffer.
    """

    online_in: dict[str, int] = field(default_factory=dict)
    r2_out: dict[str, int] = field(default_factory=dict)
    r2_in: dict[str, int] = field(default_factory=dict)

    # Lazily-populated caches keyed by rotation_size, filled during apply_rotation_to_tensor.
    # All online targets in a run share one rotation_size, so building the Hadamard matrix
    # (scipy.linalg.hadamard / kron) once and reusing it avoids repeating that work for every
    # target. Not part of the plan's identity, so excluded from init/repr/compare.
    _hadamard_cache: dict[int, tuple[torch.Tensor, int]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _input_rotation_cache: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False, compare=False)

    def is_empty(self) -> bool:
        return not (self.online_in or self.r2_out or self.r2_in)

    def hadamard_K(self, rotation_size: int) -> tuple[torch.Tensor, int]:
        """Cached ``_get_hadamard_K(rotation_size)`` -> ``(hadamard_K_cpu, K)``.

        Returns the CPU base matrix; callers move it to the target device (a fresh copy)
        and never mutate the cached tensor in place.
        """
        cached = self._hadamard_cache.get(rotation_size)
        if cached is None:
            cached = _get_hadamard_K(rotation_size)
            self._hadamard_cache[rotation_size] = cached
        return cached

    def input_rotation_int8(self, rotation_size: int) -> torch.Tensor:
        """Cached ``input_rotation`` int8 buffer, returned as a fresh clone per call.

        The clone is required because each target's buffer is written as a distinct
        tensor; ``safetensors.save_file`` rejects tensors that share storage.
        """
        cached = self._input_rotation_cache.get(rotation_size)
        if cached is None:
            cached = _build_input_rotation_int8(rotation_size, self)
            self._input_rotation_cache[rotation_size] = cached
        return cached.clone()


def build_rotation_plan(
    quant_config: QConfig,
    hf_model_config: dict[str, object] | None,
    linear_weight_names: set[str],
    bias_tensor_names: set[str] | None = None,
) -> RotationPlan | None:
    """Resolve a :class:`RotationConfig` into a model-free :class:`RotationPlan`.

    :param QConfig quant_config: The file-to-file quantization config. Its
        :meth:`QConfig.get_rotation_config` and ``exclude`` drive the plan.
    :param dict | None hf_model_config: The source HuggingFace ``config.json`` (used for
        ``hidden_size`` / ``head_dim`` / ``intermediate_size``).
    :param set[str] linear_weight_names: All linear weight-tensor names present across the
        checkpoint shards (names ending in ``.weight``).
    :param set[str] | None bias_tensor_names: All bias tensor names present across the
        checkpoint shards (names ending in ``.bias``). Used to reject R2 output-channel
        rotation on a biased ``v_proj``. Defaults to ``None`` (treated as no biases);
        callers whose checkpoints may carry attention biases must pass this.

    :return: A :class:`RotationPlan`, or ``None`` if there is no rotation config at all.
    :raises NotImplementedError: For rotation modes the file-to-file flow cannot support,
        including an R2 output-channel (``v_proj``) target that has a bias.
    :raises ValueError: When an online rotation target is also excluded from quantization,
        or when a rotation config is present but resolves to zero target layers.
    """
    rotation_config = quant_config.get_rotation_config()
    if rotation_config is None:
        return None

    rotation_config.validate_file_to_file()

    cfg = _text_config_view(hf_model_config)
    exclude_patterns = list(quant_config.exclude)
    linear_module_names = {_module_name_of_weight(name) for name in linear_weight_names}
    bias_module_names = {name.removesuffix(".bias") for name in (bias_tensor_names or set())}
    layer_indices = iter_layer_indices(linear_weight_names, rotation_config.model_decoder_layers)

    plan = RotationPlan()

    if rotation_config.r1:
        _plan_online_r1(rotation_config, cfg, layer_indices, linear_module_names, exclude_patterns, plan)

    if rotation_config.r2:
        _plan_r2(rotation_config, cfg, layer_indices, linear_module_names, bias_module_names, plan)

    if rotation_config.r4:
        _plan_r4(rotation_config, cfg, layer_indices, linear_module_names, exclude_patterns, plan)

    if plan.is_empty():
        # A RotationConfig is present, so config.json will record it regardless of the plan.
        # If we silently applied nothing, the reload path (get_online_rotation_layers) would
        # still resolve targets and expect input_rotation buffers we never wrote -> broken
        # checkpoint. Fail loudly instead, pointing at the likely culprit.
        raise ValueError(
            "A RotationConfig was provided but resolved to zero target layers in the checkpoint. "
            f"Searched decoder layers under model_decoder_layers='{rotation_config.model_decoder_layers}' "
            f"and found {len(layer_indices)} layer indices. This usually means model_decoder_layers, "
            "scaling_layers.target_modules/next_modules, or v_proj/o_proj/mlp do not match the "
            "checkpoint's tensor names. Fix the RotationConfig so it targets real layers, or remove it."
        )

    # Persist online-rotation targets so the reload path (sglang / QParamsLinearWithRotation)
    # knows which layers get the runtime ``x @ H`` rotation. Without this, config.json ships
    # ``online_rotation_layers=null`` and the rotation is silently dropped at serving time.
    # Skip when there are no online targets: an R2-only plan has none, and an online_config
    # there would trip RotationConfig.__post_init__'s "no effect" validation.
    online_layers = sorted(_module_name_of_weight(name) for name in plan.online_in)
    if online_layers:
        if rotation_config.online_config is None:
            rotation_config.online_config = OnlineRotationConfig(
                shared_parallel=None, online_rotation_layers=online_layers
            )
        else:
            rotation_config.online_config.online_rotation_layers = online_layers

    logger.info(
        "File-to-file rotation plan: %d online input-channel target(s), %d R2 v_proj, %d R2 o_proj.",
        len(plan.online_in),
        len(plan.r2_out),
        len(plan.r2_in),
    )
    return plan


def _resolved_targets_for_layer(
    scaling_layers: dict[str, list[dict[str, Any]]],
    layer_index: int,
    is_first: bool,
) -> list[str]:
    """Expand the online-R1 target templates for one decoder layer (``layer_id`` -> index).

    A scaling entry names its targets via ``target_modules``. When that key is absent, the
    entry falls back to ``next_modules`` -- this mirrors
    :meth:`RotationProcessor.get_scaling_layers` (rotation.py) and the ``RotationConfig``
    docstring. The reload path (``get_online_rotation_layers``) uses the same fallback to
    decide which layers to wrap in ``QParamsLinearWithRotation``, so file-to-file MUST match
    it: otherwise a ``next_modules``-only config would write no ``input_rotation`` buffer
    while reload still expects one, producing a broken checkpoint.

    ``last_layer`` is intentionally ignored: it is only used for offline R1 (skipped when
    ``online_r1_rotation=True``), which the file-to-file flow does not support.
    """
    patterns = scaling_layers["first_layer"] if is_first else scaling_layers["middle_layers"]
    return [name for pattern in patterns for name in expand_scaling_layer_targets(pattern, layer_index)]


def _plan_online_r1(
    rotation_config: RotationConfig,
    cfg: dict[str, object],
    layer_indices: list[int],
    linear_module_names: set[str],
    exclude_patterns: list[str],
    plan: RotationPlan,
) -> None:
    rotation_size = rotation_config.rotation_size
    if rotation_size is None:
        rotation_size = _get_dim(cfg, "hidden_size")
        if rotation_size is None:
            raise ValueError(
                "R1 rotation size could not be determined: RotationConfig.rotation_size is None and "
                "'hidden_size' is missing from the model config."
            )

    for layer_index in layer_indices:
        is_first = layer_index == layer_indices[0]
        for module_pattern in _resolved_targets_for_layer(rotation_config.scaling_layers, layer_index, is_first):
            for module_name in match_modules(module_pattern, linear_module_names):
                _guard_not_excluded(module_name, exclude_patterns, "R1")
                plan.online_in[f"{module_name}.weight"] = rotation_size


def _plan_r2(
    rotation_config: RotationConfig,
    cfg: dict[str, object],
    layer_indices: list[int],
    linear_module_names: set[str],
    bias_module_names: set[str],
    plan: RotationPlan,
) -> None:
    head_dim = _get_dim(cfg, "head_dim")
    if head_dim is None:
        hidden_size = _get_dim(cfg, "hidden_size")
        num_heads = _get_dim(cfg, "num_attention_heads")
        if hidden_size is None or num_heads is None:
            raise ValueError(
                "R2 rotation size (head_dim) could not be determined: provide 'head_dim', or both "
                "'hidden_size' and 'num_attention_heads', in the model config."
            )
        head_dim = hidden_size // num_heads

    prefix = rotation_config.model_decoder_layers
    for layer_index in layer_indices:
        v_pattern = f"{prefix}.{layer_index}.{rotation_config.v_proj}"
        o_pattern = f"{prefix}.{layer_index}.{rotation_config.o_proj}"
        for module_name in match_modules(v_pattern, linear_module_names):
            if module_name in bias_module_names:
                # Rotating output channels changes the output basis, so the bias must be rotated in
                # lockstep (see rotate_out_channels_). The file-to-file flow processes tensors
                # independently and cannot, so this would silently serve wrong outputs.
                raise NotImplementedError(
                    f"R2 rotates the output channels of '{module_name}', which requires rotating "
                    f"'{module_name}.bias' in lockstep — the per-tensor file-to-file flow cannot. "
                    "Exclude this layer from R2, or use `ModelQuantizer.quantize_model` (graph flow)."
                )
            plan.r2_out[f"{module_name}.weight"] = head_dim
        for module_name in match_modules(o_pattern, linear_module_names):
            plan.r2_in[f"{module_name}.weight"] = head_dim
    # No _guard_not_excluded here (unlike R1/R4): R2 is fully fused into the weights with no
    # input_rotation buffer, so an excluded layer needs no inference-time wrapper to stay correct.


def _plan_r4(
    rotation_config: RotationConfig,
    cfg: dict[str, object],
    layer_indices: list[int],
    linear_module_names: set[str],
    exclude_patterns: list[str],
    plan: RotationPlan,
) -> None:
    rotation_size = rotation_config.rotation_size
    if rotation_size is None:
        rotation_size = _get_dim(cfg, "moe_intermediate_size", "intermediate_size")
        if rotation_size is None:
            raise ValueError(
                "R4 rotation size could not be determined: RotationConfig.rotation_size is None and neither "
                "'moe_intermediate_size' nor 'intermediate_size' is present in the model config."
            )

    prefix = rotation_config.model_decoder_layers
    for layer_index in layer_indices:
        down_pattern = f"{prefix}.{layer_index}.{rotation_config.mlp}.down_proj"
        for module_name in match_modules(down_pattern, linear_module_names):
            _guard_not_excluded(module_name, exclude_patterns, "R4")
            plan.online_in[f"{module_name}.weight"] = rotation_size


def _guard_not_excluded(module_name: str, exclude_patterns: list[str], rotation_name: str) -> None:
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(module_name, pattern):
            raise ValueError(
                f"{rotation_name} online rotation targets module '{module_name}', but it also matches the "
                f"quantization exclude pattern '{pattern}'. Excluded layers are not wrapped at inference, so an "
                "online rotation on them would silently break correctness. Remove the layer from `exclude`, or "
                "remove it from the rotation targets."
            )


def _build_input_rotation_int8(rotation_size: int, plan: RotationPlan) -> torch.Tensor:
    """Build the persisted ``input_rotation`` buffer for an online rotation.

    Thin wrapper over the shared :func:`build_input_rotation_int8`, which is also used by
    :class:`InputRotationWrapperHadamard` in the graph flow — inference reconstructs the
    activation transform from this buffer, so both producers must emit identical bytes.
    Base matrices come from ``plan.hadamard_K`` so they are built once per rotation_size.
    """
    rotation_matrix, K = plan.hadamard_K(rotation_size)
    return build_input_rotation_int8(rotation_matrix, rotation_size, K, hadamard_K_fn=plan.hadamard_K)


def _rotate_input_channels_online(
    weight: torch.Tensor, rotation_size: int, plan: RotationPlan
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``weight``'s input channels by a fixed Hadamard (online R1 / R4).

    Mirrors :meth:`RotationProcessor.apply_online_r1`. Returns the rotated weight (same
    dtype as input) and the ``int8`` ``input_rotation`` buffer to persist alongside it. The
    Hadamard base matrix and int8 buffer are cached on ``plan`` (keyed by rotation_size), so
    they are built once and reused across all online targets in the run.
    """
    rotated, _, _ = rotate_input_channels_hadamard(weight, rotation_size, hadamard_K_fn=plan.hadamard_K)
    return rotated, plan.input_rotation_int8(rotation_size)


def apply_rotation_to_tensor(
    tensor_name: str,
    tensor: torch.Tensor,
    plan: RotationPlan,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the planned rotation (if any) to a single weight tensor.

    :param str tensor_name: Full tensor name (e.g. ``"model.layers.0.self_attn.q_proj.weight"``).
    :param torch.Tensor tensor: The (recovered, high-precision) weight tensor.
    :param RotationPlan plan: The plan from :func:`build_rotation_plan`.

    :return: ``(rotated_or_original_tensor, extra_buffers)`` where ``extra_buffers`` maps
        additional tensor names (e.g. ``"<module>.input_rotation"``) to tensors that must
        be written into the shard. Empty when the tensor is not a rotation target.

    A tensor can match more than one rotation, so every match is applied in sequence rather
    than returning after the first: ``v_proj`` takes R1 on its input channels and R2 on its
    output channels, and dropping the latter would leave ``o_proj`` rotated against an
    unrotated ``v_proj``. Order follows the graph flow (R1, then R2).
    """
    extra: dict[str, torch.Tensor] = {}
    rotated = tensor

    rotation_size = plan.online_in.get(tensor_name)
    if rotation_size is not None:
        rotated, input_rotation = _rotate_input_channels_online(rotated, rotation_size, plan)
        module_name = _module_name_of_weight(tensor_name)
        extra[f"{module_name}.input_rotation"] = input_rotation

    rotation_size = plan.r2_out.get(tensor_name)
    if rotation_size is not None:
        # R2 rotates output channels (dim 0). Guard divisibility explicitly with a
        # module-named message, mirroring the online path (_rotate_input_channels_online);
        # otherwise a wrong head_dim only surfaces as rotate_with_size's generic
        # "incompatible" error. out_features is tensor.shape[0].
        _guard_r2_divides(tensor_name, rotated.shape[0], rotation_size, "output (v_proj)")
        rotation_matrix = get_rotation_matrix(rotation_size, device=rotated.device, random=False)
        # Bias is not rotated here: the file-to-file flow never holds a weight and its
        # `.bias` sibling together, which is why a biased v_proj is rejected up front.
        rotated = rotate_out_channels(rotated, rotation_matrix)

    rotation_size = plan.r2_in.get(tensor_name)
    if rotation_size is not None:
        # R2 input rotation acts on input channels (dim 1). in_features is tensor.shape[1].
        _guard_r2_divides(tensor_name, rotated.shape[1], rotation_size, "input (o_proj)")
        rotation_matrix = get_rotation_matrix(rotation_size, device=rotated.device, random=False)
        rotated = rotate_in_channels(rotated, rotation_matrix)

    return rotated, extra


def _guard_r2_divides(tensor_name: str, dim_size: int, rotation_size: int, which: str) -> None:
    """Raise a clear, module-named error if the R2 head_dim does not divide the rotated dim.

    Mirrors the online-path check in :func:`_rotate_input_channels_online`. The head_dim
    comes from the model config (explicit ``head_dim`` or the ``hidden // num_heads``
    fallback); if that derivation is wrong for an out-of-scope architecture, this fails
    legibly instead of surfacing ``rotate_with_size``'s generic "incompatible" message.
    """
    if dim_size % rotation_size != 0:
        raise ValueError(
            f"R2 {which} rotation size (head_dim={rotation_size}) does not divide the {which} dimension "
            f"{dim_size} of '{tensor_name}'. This usually means the head_dim derived from the model "
            "config is wrong for this architecture. Set 'head_dim' explicitly in the model config."
        )
