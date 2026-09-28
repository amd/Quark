#
# Modifications copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copyright [2024] Yujun Lin, Haotian Tang, Shang Yang, Song Han

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import math
from collections.abc import Callable, Iterable
from typing import Any, cast

import torch
import torch.nn as nn
from scipy.linalg import hadamard

from quark.common.utils.log import ScreenLogger
from quark.torch.algorithm.rotation.hadamard import (
    _get_hadamard_K,
    get_hadamard_matrices,
    matmul_hadU,
    random_hadamard_matrix,
)
from quark.torch.algorithm.rotation.monkeypatch import add_wrapper_after_function_call_in_method
from quark.torch.algorithm.utils.utils import get_model_type_norm_constant

logger = ScreenLogger(__name__)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Initialize RMSNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm normalization to hidden states."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_with_size(
    x: torch.Tensor, rotation_size: int | None = None, rotation_matrix: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Rotates the input tensor `x` on its last dimension, per group of `rotation_size`.

    Denoting `k = rotation_size` and R_k the rotation of shape (k, k), and applying this rotation on x of shape (..., num_groups * k), the inverse transform is the block diagonal:

    [  R_k 0_k  ...  0_k ]
    [  0_k R_k           ]
    [     .    .         ]
    [     .      .       ]
    [     .              ]
    [     0_k   ...  R_k ]

    of shape (num_groups * k, num_groups * k).
    """
    if rotation_matrix is None:
        assert rotation_size is not None

        rotation_matrix = torch.tensor(
            hadamard(rotation_size, dtype=float), dtype=x.dtype, device=x.device
        ) / math.sqrt(rotation_size)
    else:
        assert rotation_size is None
        rotation_size = rotation_matrix.shape[0]

        if x.shape[-1] % rotation_size != 0:
            raise ValueError(
                f"The function rotate_with_size got the input x with x.shape[-1]={x.shape[-1]} and rotation_matrix of shape {rotation_size}, which are incompatible."
            )

    dtype = x.dtype

    needs_reshape = False
    if x.shape[-1] != rotation_size:
        needs_reshape = True
        x = x.reshape(*x.shape[:-1], -1, rotation_size)

    if rotation_matrix.device != x.device:
        logger.warning(
            f"Device mismatch! rotation_matrix device: {rotation_matrix.device}, x device: {x.device}. This is likely causing unnecessary slowness. please open an issue."
        )
        rotation_matrix = rotation_matrix.to(x.device)

    # TODO: fix type mismatch?
    x = x.to(torch.float64) @ rotation_matrix.to(dtype=torch.float64)
    x = x.to(dtype)
    if needs_reshape:
        x = x.reshape(*x.shape[:-2], -1)

    return x


def rotate_in_channels(weight: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Return ``weight`` with its input channels (last dim) rotated.

    Non-mutating counterpart of :func:`rotate_in_channels_`, so the graph flow and the
    file-to-file flow share one definition of R2 weight orientation and block rotation.
    ``rotate_with_size`` reshapes when the rotation is narrower than the dimension, and
    restores the input dtype.
    """
    return rotate_with_size(weight, rotation_matrix=rotation)


def rotate_out_channels(weight: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Return ``weight`` with its output channels (dim 0) rotated.

    Non-mutating counterpart of :func:`rotate_out_channels_`. Transposes so the output
    channels land on the last dim, rotates, and transposes back. Bias is deliberately not
    handled here — this operates on a bare tensor; callers holding a module (and therefore
    its ``.bias``) should use :func:`rotate_out_channels_`, which rotates both.
    """
    return rotate_with_size(weight.T.contiguous(), rotation_matrix=rotation).T.contiguous()


def rotate_in_channels_(module: nn.Module, rotation: torch.Tensor) -> None:
    """Rotate the input channels of a linear layer.
    If weight and rotation's sizes don't match, it reshapes weight in order to multiply them."""
    module.weight.data = rotate_in_channels(module.weight.data, rotation)


def rotate_out_channels_(module: nn.Module, rotation: torch.Tensor) -> None:
    """Rotate the output channels of a linear layer.
    If weight/bias and rotation's sizes don't match
    it reshapes weight/bias in order to multiply them."""
    module.weight.data = rotate_out_channels(module.weight.data, rotation)

    # The bias lives in the output basis too, so it rotates on its own last dim.
    if module.bias is not None:
        module.bias.data = rotate_with_size(module.bias.data, rotation_matrix=rotation)


def build_input_rotation_int8(
    rotation_matrix: torch.Tensor,
    rotation_size: int,
    K: int | None,
    hadamard_K_fn: Callable[[int], tuple[torch.Tensor, int]] = _get_hadamard_K,
) -> torch.Tensor:
    """Build the persisted ``input_rotation`` buffer for an online Hadamard rotation.

    Returns a ``±1`` ``int8`` matrix of shape ``(rotation_size, rotation_size)``,
    kron-expanded from ``rotation_matrix`` when the base Hadamard is smaller than
    ``rotation_size``.

    This buffer is what inference reads back to reconstruct the activation-side transform,
    so every producer must emit byte-identical values. Shared by
    :class:`InputRotationWrapperHadamard` (graph flow) and the file-to-file rotation path;
    do not reimplement it.

    :param torch.Tensor rotation_matrix: Base Hadamard matrix.
    :param int rotation_size: Target buffer size (per-block rotation width).
    :param int | None K: Hadamard block factor from :func:`_get_hadamard_K`.
    :param hadamard_K_fn: Override for the base-matrix lookup. Defaults to
        :func:`_get_hadamard_K`; callers with a cache may pass their own memoized version.

    :return: ``int8`` ``±1`` matrix of shape ``(rotation_size, rotation_size)``.
    :rtype: torch.Tensor
    """
    input_rotation = rotation_matrix.clone()

    if input_rotation.shape[0] != rotation_size:
        assert K is not None
        hadamard_1, _ = hadamard_K_fn(rotation_size // K)
        hadamard_1 = hadamard_1.to(input_rotation.device)

        input_rotation = input_rotation.to(dtype=torch.float64)
        input_rotation = torch.kron(input_rotation, hadamard_1)

    assert input_rotation.shape[0] == rotation_size, (
        f"input_rotation size {input_rotation.shape[0]} != rotation_size {rotation_size}"
    )
    assert (
        input_rotation[input_rotation == 1].numel() + input_rotation[input_rotation == -1].numel()
        == input_rotation.numel()
    ), "input_rotation must contain only +1/-1 entries"

    return input_rotation.to(torch.int8)


def substitute_layer_id(name: str, layer_index: int) -> str:
    """Substitute the decoder-layer placeholders in a ``scaling_layers`` template name.

    ``pre_layer_id`` -> ``layer_index - 1``, then ``layer_id`` -> ``layer_index``. The order
    matters: substituting ``layer_id`` first would corrupt ``pre_layer_id`` into ``pre_<N>``.
    """
    return name.replace("pre_layer_id", str(layer_index - 1)).replace("layer_id", str(layer_index))


def scaling_layer_target_templates(layers_pattern: dict[str, Any]) -> list[str]:
    """The raw (un-substituted) target templates one ``scaling_layers`` entry designates.

    ``target_modules`` names the targets; when that key is absent the entry falls back to
    ``next_modules``. That fallback is the part of the ``scaling_layers`` contract both flows
    must agree on: ``RotationProcessor.get_online_rotation_layers`` uses it to decide which
    layers get an inference-time wrapper, and the file-to-file flow uses it to decide which
    get an ``input_rotation`` buffer. If the two disagree, the checkpoint's buffers and
    wrappers do not line up and the reloaded model is silently wrong.

    The fallback keys on ``None``, not falsiness: an empty ``target_modules`` deliberately opts
    the entry out of online rotation.

    :param dict[str, Any] layers_pattern: One entry from ``scaling_layers``'s
        ``first_layer`` / ``middle_layers`` / ``last_layer`` list.

    :return: Target module name templates, with ``layer_id`` / ``pre_layer_id`` still in place.
    :rtype: list[str]
    """
    templates = layers_pattern.get("target_modules")
    if templates is None:
        templates = layers_pattern.get("next_modules", [])
    return cast(list[str], templates)


def expand_scaling_layer_targets(layers_pattern: dict[str, Any], layer_index: int) -> list[str]:
    """The target module names one ``scaling_layers`` entry designates for one decoder layer.

    Substitutes the decoder-layer placeholders in :func:`scaling_layer_target_templates`'s result;
    see there for the ``target_modules`` -> ``next_modules`` fallback both flows share.

    Placeholders are substituted but wildcards are left intact — each flow resolves those its
    own way: the graph flow against a live module tree (``resolve_star``), the file-to-file
    flow against checkpoint tensor names (``match_modules``).

    :param dict[str, Any] layers_pattern: One entry from ``scaling_layers``'s
        ``first_layer`` / ``middle_layers`` / ``last_layer`` list.
    :param int layer_index: Index of the decoder layer being expanded.

    :return: Target module names with ``layer_id`` / ``pre_layer_id`` substituted.
    :rtype: list[str]
    """
    return [substitute_layer_id(name, layer_index) for name in scaling_layer_target_templates(layers_pattern)]


def rotate_input_channels_hadamard(
    weight: torch.Tensor,
    rotation_size: int,
    hadamard_K_fn: Callable[[int], tuple[torch.Tensor, int]] = _get_hadamard_K,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Rotate ``weight``'s input channels by a fixed Hadamard (online R1 / R4).

    The weight-side half of an online rotation: inference re-applies the matching activation
    transform, rebuilt from the ``input_rotation`` buffer :func:`build_input_rotation_int8`
    emits. The two must stay in step, so this is shared by
    ``RotationProcessor.apply_online_r1`` (graph flow) and the file-to-file rotation path;
    do not reimplement it.

    :param torch.Tensor weight: ``(out_features, in_features)`` weight to rotate.
    :param int rotation_size: Per-block rotation width; must divide ``in_features``.
    :param hadamard_K_fn: Override for the base-matrix lookup. Defaults to
        :func:`_get_hadamard_K`; callers with a cache may pass their own memoized version.

    :return: ``(rotated_weight, hadamard_K, K)``. ``hadamard_K`` and ``K`` describe the matrix
        actually applied — kron-expanded when the base is narrower than ``rotation_size`` —
        which the graph flow hands to :class:`InputRotationWrapperHadamard`.
    :rtype: tuple[torch.Tensor, torch.Tensor, int]
    """
    dtype = weight.dtype
    in_features = weight.shape[1]
    if in_features % rotation_size != 0:
        raise ValueError(
            f"rotation_size={rotation_size} does not divide in_features={in_features} for a weight of shape "
            f"{tuple(weight.shape)}. Choose a rotation_size that divides the input dimension."
        )

    hadamard_K, K = hadamard_K_fn(rotation_size)
    # Move a copy to the weight's device: matmul_hadU / rotate_with_size otherwise emit a
    # device-mismatch warning and pay a host<->device copy per tensor. `.to` returns a new
    # tensor when the device differs, so a cached base matrix is never mutated in place.
    hadamard_K = hadamard_K.to(weight.device)

    weight = weight.contiguous()
    if rotation_size == in_features:
        # `inverse=True` is not required here as nn.Linear already transposes the weight.
        return matmul_hadU(weight, hadamard_K=hadamard_K, K=K).to(dtype), hadamard_K, K

    # Block-diagonal Hadamard: kron-expand the base up to rotation_size, then rotate each
    # contiguous block of `rotation_size` input channels.
    if hadamard_K.shape[0] != rotation_size:
        hadamard_1, _ = hadamard_K_fn(rotation_size // K)
        hadamard_1 = hadamard_1.to(weight.device)
        hadamard_K = torch.kron(hadamard_K.to(torch.float64), hadamard_1.to(torch.float64))
        K = rotation_size

    assert hadamard_K.shape[0] == rotation_size
    rotation_matrix = hadamard_K.to(torch.float64) / math.sqrt(rotation_size)
    return rotate_with_size(weight, rotation_matrix=rotation_matrix).to(dtype), hadamard_K, K


def get_rotation_matrix(num_channels: int, device: torch.device | str, random: bool = True) -> torch.Tensor:
    """Get a random rotation matrix for the given number of channels."""
    if random:
        rotation = random_hadamard_matrix(num_channels)
    else:
        hadamard_1, hadamard_K, K = get_hadamard_matrices(num_channels)
        hadamard_1 = hadamard_1.to(dtype=torch.float64)
        if K == 1:
            rotation = hadamard_1
        else:
            assert hadamard_K is not None
            hadamard_K = hadamard_K.to(dtype=torch.float64)
            rotation = torch.kron(hadamard_K, hadamard_1)
        rotation = rotation.mul_(1.0 / torch.tensor(num_channels, dtype=torch.float64).sqrt())

    return rotation.to(device)


def transform_norm_and_linear(
    prev_modules: Iterable[nn.Module],
    norm_module: nn.Module,
    next_modules: Iterable[nn.Module],
    prev_out_channels_dims: list[int],
) -> None:
    transform_rms_norm_and_linear(norm_module, next_modules)
    if isinstance(norm_module, nn.LayerNorm):
        assert prev_modules is not None
        prev_modules_linear = [mod for mod in prev_modules if isinstance(mod, nn.Linear)]
        transform_layer_norm_to_rms_norm(norm_module, prev_modules_linear, prev_out_channels_dims)


def transform_rms_norm_and_linear(norm: nn.Module, next_modules: Iterable[nn.Module]) -> None:
    next_modules_linear = [mod for mod in next_modules if isinstance(mod, nn.Linear)]
    ln_w = norm.weight.data.to(dtype=torch.float64)
    # The norm applies `weight + c`, so that is what must be folded into the next layers, and the
    # identity it is reset to is `1 - c`: zeros for the centered norms (`c == 1`), ones otherwise.
    c = get_model_type_norm_constant(norm)
    ln_w = ln_w + c
    norm.weight.data = torch.full_like(norm.weight.data, 1.0 - c)
    if hasattr(norm, "bias") and norm.bias is not None:
        ln_b = norm.bias.data.to(dtype=torch.float64)
        norm.bias = None  # type: ignore
    else:
        ln_b = None
    for linear in next_modules_linear:
        dtype = linear.weight.dtype
        fc_w = linear.weight.data.to(dtype=torch.float64)
        linear.weight.data = (fc_w * ln_w).to(dtype=dtype)
        if ln_b is not None:
            if linear.bias is None:
                linear.bias = nn.Parameter(torch.zeros(linear.out_features, dtype=dtype, device=linear.weight.device))
            linear.bias.data = (linear.bias.data.to(dtype=torch.float64) + torch.matmul(fc_w, ln_b)).to(dtype=dtype)


def transform_layer_norm_to_rms_norm(
    norm: nn.Module,
    prev_modules: Iterable[nn.Linear],
    prev_out_channels_dims: list[int],
) -> None:
    assert isinstance(norm, nn.LayerNorm)
    assert len(norm.normalized_shape) == 1, f"LayerNorm's #dims must be 1, got {len(norm.normalized_shape)}"
    assert norm.bias is None, "LayerNorm's bias must be None"
    # region move substract mean to the previous linear modules
    assert len(prev_modules) > 0, "No previous modules found"
    if isinstance(prev_out_channels_dims, int):
        prev_out_channels_dims = [prev_out_channels_dims] * len(prev_modules)
    for module, dim in zip(prev_modules, prev_out_channels_dims, strict=False):
        if isinstance(module, nn.LayerNorm):
            module.bias = None
        else:
            if isinstance(module, nn.Linear):
                assert dim == 0, "Linear module's output channels dimension is 0"
            elif isinstance(module, nn.Embedding):
                assert dim == 1, "Embedding module's output channels dimension is 1"
            dtype = module.weight.dtype
            W = module.weight.data.to(dtype=torch.float64)
            module.weight.data = W.sub_(W.mean(dim=dim, keepdim=True)).to(dtype=dtype)
            if hasattr(module, "bias") and module.bias is not None:
                B = module.bias.data.to(dtype=torch.float64)
                module.bias.data = B.sub_(B.mean()).to(dtype=dtype)
    # region replace LayerNorm with RMSNorm
    rms = RMSNorm(hidden_size=norm.normalized_shape[0], eps=norm.eps)
    rms.weight.data = norm.weight.data


class QKRotation(nn.Module):
    """Performs R3 rotation after RoPE of both Q and K, but does not do K quantization"""

    def __init__(self, func: Callable[..., Any]):
        super().__init__()
        self.func = func

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        q, k = self.func(*args, **kwargs)

        q = matmul_hadU(q)

        # k is transposed later on in the attention Q @ K.T, so no need to use `inverse=True` here.
        k = matmul_hadU(k)

        return q, k


def add_qk_rotation_after_function_call_in_forward(module: nn.Module, function_name: str) -> None:
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.

    This function used to insert the R3 rotation after the output of the call of the RoPE operation.
    Implementating it like this is not ideal, since we need to modify the forward function's globals. However, this is the
    trick used by both QuaRot and SpinQuant to insert a rotation after the RoPE operation. Ultimately it would better to
    find a way to implement this feature without touching globals.
    """

    attr_name = f"{function_name}_qk_rotation"
    assert not hasattr(module, attr_name)
    wrapper = add_wrapper_after_function_call_in_method(module, "forward", function_name, QKRotation)
    setattr(module, attr_name, wrapper)


class InputRotationWrapper(nn.Module):
    """
    Wrapper around a nn.Module that applies a Hadamard rotation before the module.
    If the module is an nn.Linear or nn.Conv, then Quark will replace it by a quantized linear layer
    If there is activation quantization, it is applied in between, i.e. after the rotation
    but before the forward pass of the module
    """

    def __getattr__(self, name: str) -> Any:
        # TODO: try to do a check on `self.original_module` attributes here
        if name in {"weight", "bias", "in_features", "out_features"}:
            return getattr(self.original_module, name)
        else:
            return super().__getattr__(name)

    def __setattr__(self, key: str, value: Any) -> None:
        # TODO: try to do a check on `self.original_module` attributes here
        if key in {"weight", "bias", "in_features", "out_features"}:
            setattr(self.original_module, key, value)
        else:
            super().__setattr__(key, value)

    def state_dict(
        self, *args: tuple[Any], destination: dict[str, Any] | None = None, prefix: str = "", keep_vars: bool = False
    ) -> dict[str, Any]:
        destination_local = super().state_dict(*args, prefix=prefix, keep_vars=keep_vars)

        for param_name in list(destination_local):
            if ".original_module." in param_name:
                new_name = param_name.replace(".original_module.", ".")
                destination_local[new_name] = destination_local.pop(param_name)

        if destination is not None:
            destination.update(destination_local)
        else:
            destination = destination_local

        return destination

    def forward(self, x: torch.Tensor) -> Any:
        x = self.transform(x)

        # quantization will happen here, in between (since it happens before a nn.Linear layer)
        x = self.original_module(x)
        return x


class InputRotationWrapperHadamard(InputRotationWrapper):
    def __init__(
        self,
        original_module: nn.Linear,
        rotation_size: int | None = None,
        hadamard_K: torch.Tensor | None = None,
        K: int | None = None,
    ):
        super().__init__()

        if not isinstance(original_module, nn.Linear):
            raise ValueError(
                f"InputRotationWrapper only supports module instance of torch.nn.Linear, got {original_module.__class__.__name__}"
            )

        self.original_module = original_module

        in_features = original_module.in_features

        if rotation_size is not None and in_features != rotation_size:
            if in_features % rotation_size == 0:
                self.rotation_size = rotation_size
                self.use_matmul_hadU = False
            else:
                raise ValueError(f"rotation_size={rotation_size} is not compatible with in_features={in_features}.")
        else:
            self.rotation_size = in_features
            self.use_matmul_hadU = True

        if hadamard_K is None or K is None:
            rotation_matrix, K = _get_hadamard_K(self.rotation_size)
        else:
            rotation_matrix = hadamard_K

        self.K = K

        rotation_matrix = rotation_matrix.to(self.original_module.weight.dtype)
        rotation_matrix = rotation_matrix.to(self.original_module.weight.device)

        input_rotation = build_input_rotation_int8(rotation_matrix, self.rotation_size, self.K)

        self.register_buffer("input_rotation", input_rotation)

        self.transform = HadamardTransform(
            use_matmul_hadU=self.use_matmul_hadU, rotation_size=self.rotation_size, hadamard_K=rotation_matrix, K=K
        )


class InputRotationWrapperOrthogonal(InputRotationWrapper):
    def __init__(
        self,
        original_module: nn.Linear,
        rotation_matrix: torch.Tensor,
    ):
        super().__init__()

        if not isinstance(original_module, nn.Linear):
            raise ValueError(
                f"InputRotationWrapper only supports module instance of torch.nn.Linear, got {original_module.__class__.__name__}"
            )

        self.original_module = original_module

        assert rotation_matrix is not None
        # Accept float64 (trained rotations) as well as fp16/bf16/fp32 stored
        # rotations. `rotate_with_size` upcasts to float64 at apply time, so a
        # lower-precision stored matrix does not change compute accuracy while
        # cutting the on-disk `input_rotation` size (4x vs float64).
        assert rotation_matrix.dtype in (torch.float64, torch.float32, torch.bfloat16, torch.float16), (
            f"unsupported input_rotation dtype {rotation_matrix.dtype}"
        )

        rotation_matrix = rotation_matrix.to(self.original_module.weight.device)

        self.transform = OrthogonalTransform(rotation_matrix)

        input_rotation = rotation_matrix.clone()
        self.register_buffer("input_rotation", input_rotation)


class OrthogonalTransform(nn.Module):
    def __init__(
        self,
        rotation_matrix: torch.Tensor,
    ):
        super().__init__()

        assert rotation_matrix is not None

        # Registered as a non-persistent buffer so it follows `module.to(device)`
        # (avoids a per-forward host->device copy of the rotation matrix). Non-
        # persistent => not added to state_dict, so serialization is unchanged.
        self.register_buffer("rotation_matrix", rotation_matrix, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rotate_with_size(x, rotation_matrix=self.rotation_matrix)
        return x


class HadamardTransform(nn.Module):
    def __init__(
        self,
        rotation_size: int,
        use_matmul_hadU: bool,
        hadamard_K: torch.Tensor | None = None,
        K: int | None = None,
    ):
        super().__init__()

        self.use_matmul_hadU = use_matmul_hadU
        self.rotation_size = rotation_size

        if not use_matmul_hadU:
            if hadamard_K is None:
                rotation_matrix, _ = _get_hadamard_K(self.rotation_size)
            else:
                rotation_matrix = hadamard_K

            # Ensure rotation matrix stays in float32 for numerical match.
            rotation_matrix = rotation_matrix.to(torch.float32)
            rotation_matrix = rotation_matrix / math.sqrt(self.rotation_size)
        else:
            assert K is not None
            assert hadamard_K is not None
            rotation_matrix = hadamard_K

        self.K = K
        self.rotation_matrix = rotation_matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_matmul_hadU:
            # TODO: ideally, rotate_with_size should handle this case well.
            x = matmul_hadU(x, hadamard_K=self.rotation_matrix, K=self.K)
        else:
            x = rotate_with_size(x, rotation_matrix=self.rotation_matrix)

        return x
