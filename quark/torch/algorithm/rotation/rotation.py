#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence

import torch
import torch.nn as nn
from tqdm import tqdm

from quark.shares.utils.log import ScreenLogger
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.algorithm.rotation.hadamard import _get_hadamard_K, matmul_hadU
from quark.torch.algorithm.rotation.rotation_utils import (
    InputRotationWrapperHadamard,
    InputRotationWrapperOrthogonal,
    add_qk_rotation_after_function_call_in_forward,
    get_rotation_matrix,
    rotate_in_channels_,
    rotate_out_channels_,
    rotate_with_size,
    transform_norm_and_linear,
)
from quark.torch.algorithm.utils.prepare import get_model_layers
from quark.torch.algorithm.utils.utils import clear_memory
from quark.torch.quantization.config.config import OnlineRotationConfig
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase
from quark.torch.utils import getattr_recursive, resolve_star, setattr_recursive
from quark.torch.utils.accelerate_helper import untie_parameters

if TYPE_CHECKING:
    from quark.torch.quantization.config.config import QConfig, RotationConfig


__all__ = ["RotationProcessor"]

logger = ScreenLogger(__name__)

VALIDATED_ARCHITECTURES = {"gpt_oss", "llama", "qwen3_moe"}


class OutputRotationWrapper(nn.Module):
    def __init__(
        self,
        original_module: nn.Module,
        rotation_out: nn.Parameter,
        hint_out: str | None = None,
    ) -> None:
        super().__init__()

        self.original_module = original_module
        self.hint_out = hint_out
        self.rotation_out = rotation_out

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        x = self.original_module(*args, **kwargs)

        x = rotate_with_size(x, rotation_matrix=self.rotation_out)

        return x


class TrainableRMSNorm(nn.Module):
    """
    Adapted from Transformers' LlamaRMSNorm.
    https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py#L53

    This class adds a trainable `smooth_values` parameter.

    Beware that other models architectures (e.g. gpt-oss) may have a slightly different normalization layer implementation.
    """

    def __init__(
        self,
        normalization: nn.Module,
        smooth_values: nn.Parameter,
        norm_layer_name: str | None = None,
    ) -> None:
        super().__init__()

        self.original_normalization = normalization
        self.smooth_values = smooth_values
        self.norm_layer_name = norm_layer_name

        if not isinstance(self.smooth_values, nn.Parameter):
            raise ValueError(f"Expected smooth_values to be nn.Parameter, got {type(smooth_values)}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.original_normalization.weight
        weight = weight * self.smooth_values

        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.original_normalization.variance_epsilon)
        return weight * x.to(input_dtype)


class RotationLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        hint_in: str | None = None,
        hint_out: str | None = None,
        rotation_in: nn.Parameter | None = None,
        rotation_out: nn.Parameter | None = None,
        rotate_activation: bool | None = None,
        smooth_values_in: nn.Parameter | None = None,
        smooth_values_out: nn.Parameter | None = None,
        smooth_first: bool | None = None,
    ) -> None:
        """
        Wrapper around a `nn.Linear` handling activation input and weight rotation for training.

        :param torch.nn.Linear linear: The original nn.Linear to wrap rotations around.
        :param Optional[str] hint_in: The class of rotations the input rotation belongs to. Example: ``"r1"``. Defaults to ``None``.
        :param Optional[str] hint_out: The class of rotations the weight rotation on output feature dimension belongs to. Example: ``"r4"``. Defaults to ``None``.
        :param Optional[torch.nn.Parameter] rotation_in: The optional learnable input rotation. Example: ``"r4"``. Defaults to ``None``.
        :param Optional[torch.nn.Parameter] rotation_out: The optional learnable weight rotation on its output feature dimension. Example: ``"r4"``. Defaults to ``None``.
        :param Optional[bool] rotate_activation: Whether activations should be rotated (on top of weight) when using ``rotation_in``. Defaults to ``None``.
        :param Optional[torch.nn.Parameter] smooth_values_in: The optional learnable normalization (SmoothQuant) scales, applied on the weight input features dimension. Defaults to ``None``.
        :param Optional[torch.nn.Parameter] smooth_values_out: The optional learnable normalization (SmoothQuant) scales, applied on the weight output features dimension. Defaults to ``None``.
        :param Optional[bool] smooth_first: When using learnable normalization SmoothQuant scales, whether to apply ``T = D @ R`` or ``T = R @ D``, with the rotation ``R`` and the SmoothQuant scales ``D`` (seen as a 1D vector, or a diagonal matrix). This parameter is similar to the approach in `OSTQuant <https://arxiv.org/abs/2501.13987>`__ paper, and is set automatically. Defaults to ``None``.
        """
        super().__init__()

        # Quantization is applied AFTER.
        assert isinstance(linear, nn.Linear)

        if rotation_in is not None:
            if not isinstance(rotation_in, nn.Parameter):
                raise ValueError(
                    f"rotation_in is not None, expected a torch.nn.Parameter, but got {type(rotation_in)}."
                )

            if rotate_activation is None:
                raise ValueError(
                    "`rotation_in` is not None, but `rotate_activation` is None. This is not expected. Please open an issue."
                )

        if rotation_out is not None and not isinstance(rotation_out, nn.Parameter):
            raise ValueError(f"rotation_out is not None, expected a torch.nn.Parameter, but got {type(rotation_out)}.")

        if smooth_values_in is not None:
            assert smooth_values_in.ndim == 1

            # Not equal for GQA.
            assert linear.in_features % smooth_values_in.shape[0] == 0

            if not isinstance(smooth_values_in, nn.Parameter):
                raise ValueError(
                    f"smooth_values_in is not None, expected a torch.nn.Parameter, but got {type(smooth_values_in)}."
                )

            if smooth_first is None:
                raise ValueError(
                    "`smooth_values_in` is not None, but `smooth_first` is None. This is not expected. Please open an issue."
                )

        if smooth_values_out is not None:
            assert isinstance(smooth_values_out, nn.Parameter)
            assert smooth_values_out.ndim == 1
            assert linear.weight.shape[0] == smooth_values_out.shape[0]

        self.linear = linear

        self.in_features = self.linear.in_features
        self.out_features = self.linear.out_features

        self.rotation_in = rotation_in
        self.rotation_out = rotation_out

        self.smooth_values_in = smooth_values_in
        self.smooth_first = smooth_first

        self.smooth_values_out = smooth_values_out

        self.hint_in = hint_in
        self.hint_out = hint_out

        self.rotate_activation = rotate_activation

    def apply_in_normalization_weight(self, weight: torch.Tensor) -> torch.Tensor:
        smooth_values_in = self.smooth_values_in  # type: ignore[union-attr]

        if smooth_values_in.shape[0] != self.in_features:  # type: ignore[union-attr]
            # Grouped-Query-Attention (GQA) case for v_proj/o_proj.
            weight = weight.view(weight.shape[0], -1, smooth_values_in.shape[0])  # type: ignore[union-attr]
            weight = weight * (1 / smooth_values_in[None, None])
            weight = weight.view(weight.shape[0], -1)
        else:
            weight = weight * (1 / smooth_values_in[None])

        return weight

    def apply_in_transform(self, x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.smooth_values_in is not None and self.smooth_first:
            weight = self.apply_in_normalization_weight(weight)

        if self.rotation_in is not None:
            weight = rotate_with_size(weight, rotation_matrix=self.rotation_in)

            if self.rotate_activation:
                x = rotate_with_size(x, rotation_matrix=self.rotation_in)

        if self.smooth_values_in is not None and not self.smooth_first:
            weight = self.apply_in_normalization_weight(weight)

        return x, weight

    def apply_out_transform(
        self, weight: torch.Tensor, bias: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.smooth_values_out is not None and self.smooth_first:
            weight = weight * self.smooth_values_out[:, None]

            if bias is not None:
                bias = bias * self.smooth_values_out

        if self.rotation_out is not None:
            # Assume the inverse rotation is applied on activations elsewhere.
            weight = rotate_with_size(weight.T, rotation_matrix=self.rotation_out)
            weight = weight.T.contiguous()

            if bias is not None:
                bias = rotate_with_size(bias, rotation_matrix=self.rotation_out)

        if self.smooth_values_out is not None and not self.smooth_first:
            weight = weight * self.smooth_values_out[:, None]

            if bias is not None:
                bias = bias * self.smooth_values_out

        return weight, bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: self.linear.weight is never modified in the training.
        weight = self.linear.weight
        bias = self.linear.bias

        # Apply transformations on weights, activations. For the weight, transformations may be applied on both
        # its in_features and out_features dimensions.

        # y = x @ R_i @ R_i.T @ W.T + b
        # So:
        # W <- W @ R_i
        # x <- x @ R_i
        # bias unchanged.
        x, weight = self.apply_in_transform(x, weight)

        # z = y @ W_2.T
        # z = y @ R_o @ R_o.T @ W_2.T + b_2
        # z = [x @ R_i @ [W @ R_i].T + b] @ R_o   @   [W_2 @ R_o].T  + b_2
        # So:
        # W <- R_o.T @ W @ R_i
        # b <- b @ R_o
        weight, bias = self.apply_out_transform(weight, bias)

        dtype = x.dtype
        device = x.device
        weight = weight.to(device=device, dtype=dtype)

        if isinstance(self.linear, QuantLinear):
            x = self.linear.forward_with_weight(x, weight=weight, bias=bias)
        elif isinstance(self.linear, nn.Linear):
            # Corresponds to lm_head.
            x = torch.nn.functional.linear(x, weight=weight, bias=bias)
        else:
            raise ValueError(
                f"Expected self.linear to be a QuantLinear or nn.Linear, got self.linear {self.linear.__class__}"
            )

        return x


class RotationProcessor(BaseAlgoProcessor):
    def __init__(self, model: nn.Module, rotation_config: RotationConfig, _data_loader: Any) -> None:
        self.model = model
        self.rotation_config = rotation_config

        self.layers = get_model_layers(self.model, rotation_config.model_decoder_layers)

        self.scaling_modules = rotation_config.scaling_layers
        self.rotation_size = rotation_config.rotation_size
        self.random_r1 = rotation_config.random_r1
        self.random_r2 = rotation_config.random_r2
        self.online_r1_rotation = rotation_config.online_r1_rotation

        self.scaling_layers = RotationProcessor.get_scaling_layers(
            self.model,
            rotation_config.scaling_layers,
            online_r1_rotation=self.online_r1_rotation,
            r1=rotation_config.r1,
            smooth_positions=rotation_config.smooth_positions,
            model_decoder_layers=rotation_config.model_decoder_layers,
        )
        assert self.scaling_layers is not None

        self.rotation_buffer = {}  # type: ignore

        self.trainable = rotation_config.trainable

        self.shared_parallel = None
        if rotation_config.online_config:
            self.shared_parallel = rotation_config.online_config.shared_parallel  # type: ignore[union-attr]

        self.train_smooth = rotation_config.train_smooth

        # List of "r1", "r2", "r4" where to apply normalization training.
        self.smooth_positions = rotation_config.smooth_positions

        # Chooses between `T = R @ D` or `T = D @ R` transform.
        # This has no effect in case `train_smooth=False`.
        # We use:
        # - In case of online rotation: `T = D @ R`
        # - In case of offline rotation: `T = R @ D`
        if not rotation_config.r1:
            self.smooth_first = True
        else:
            self.smooth_first = self.online_r1_rotation  # type: ignore[assignment]

        logger.debug(f"Using smooth_first={self.smooth_first}")

        online_rotation_layers = self.get_online_rotation_layers(rotation_config, model)
        if rotation_config.online_config is None:
            rotation_config.online_config = OnlineRotationConfig(
                shared_parallel=None, online_rotation_layers=online_rotation_layers
            )
        else:
            rotation_config.online_config.online_rotation_layers = online_rotation_layers

    def apply(self) -> None:
        # R1 needs to be applied on embed_tokens as:
        # W_e' = W_e @ R1
        # R1^(-1) needs to be applied on lm_head as:
        # W_lm' = R1^(-1) @ W_lm.
        # With tied weights we have W_e = W_lm in memory,
        # and end up with `R1^(-1) @ W_e @ R1` which is wrong.
        self.model = untie_parameters(self.model)

        # R1 can be disabled. In Quarot/SpinQuant, it is always offline (can be fully fused into other layers).
        if self.rotation_config.r1:
            self.r1()

        # R2 can be disabled. In Quarot/SpinQuant, it is always offline (can be fully fused into other layers).
        if self.rotation_config.r2:
            self.r2()

        # R3 is useful only in case KV cache is quantized. It is online (can not be fully fused into other layers).
        if self.rotation_config.r3:
            self.r3()

        # R4 is online, can not be fully fused into other layers.
        if self.rotation_config.r4:
            self.r4()

        if self.trainable and self.train_smooth:
            self.apply_train_smooth()

        if self.trainable:
            for name, param in self.model.named_parameters():
                if any(
                    [
                        substring in name
                        for substring in [
                            "shared_r1_rotation",
                            "rotation_out",
                            "rotation_in",
                            "smooth_values_in",
                            "smooth_values_out",
                        ]
                    ]
                ):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    @staticmethod
    def initialize_smooth_values(
        in_features: int, dtype: torch.dtype, device: torch.device, layers_pattern: dict[str, Any] | None
    ) -> tuple[torch.nn.Parameter, str | None]:
        # NOTE: It is useful to add
        # `+ torch.rand(in_features, dtype=dtype, device=device) * 0.5 + 0.5`
        # to verify the transform equivalence. Although transforms are mathematically equivalent,
        # there can be significant numerical differences, depending on the model.
        smooth_values = torch.ones(in_features, dtype=dtype, device=device)
        smooth_values = torch.nn.Parameter(smooth_values)

        norm_layer_name = None
        if layers_pattern is not None:
            norm_layer_name = layers_pattern["norm_module"]

        return smooth_values, norm_layer_name

    def apply_train_smooth(self) -> None:
        if "r1" in self.smooth_positions:  # type: ignore[operator]
            for layers_pattern in tqdm(self.scaling_layers, desc="Applying trainable smoothquant on R1"):
                next_modules = [
                    getattr_recursive(self.model, layer_name) for layer_name in layers_pattern["next_modules"]
                ]

                if isinstance(next_modules[0], RotationLinear):
                    dtype = next_modules[0].linear.weight.dtype
                    device = next_modules[0].linear.weight.device
                else:
                    dtype = next_modules[0].weight.dtype
                    device = next_modules[0].weight.device

                smooth_values, norm_layer_name = RotationProcessor.initialize_smooth_values(
                    in_features=next_modules[0].in_features,
                    dtype=dtype,
                    device=device,
                    layers_pattern=layers_pattern,
                )

                # TODO: cleanup hard-coded model.norm
                if "model.norm" not in layers_pattern["norm_module"]:
                    norm_layer = getattr_recursive(self.model, layers_pattern["norm_module"])
                    trainable_norm_layer = TrainableRMSNorm(norm_layer, smooth_values, norm_layer_name)
                    setattr_recursive(self.model, layers_pattern["norm_module"], trainable_norm_layer)

                for i, layer in enumerate(next_modules):
                    full_layer_name = layers_pattern["next_modules"][i]

                    # TODO: cleanup
                    if "lm_head" in full_layer_name:
                        logger.info("skip lm_head for trainable normalization!")
                        continue

                    # e.g. `"model.layers.10.mlp"`.
                    next_module_parent_name = ".".join(full_layer_name.split(".")[:-1])

                    if next_module_parent_name == "":
                        next_module_parent = self.model
                    else:
                        next_module_parent = getattr_recursive(self.model, next_module_parent_name)
                    relative_layer_name = full_layer_name.split(".")[-1]

                    if isinstance(layer, RotationLinear):
                        assert layer.smooth_values_in is None
                        layer.smooth_values_in = smooth_values
                        layer.smooth_first = self.smooth_first
                    else:
                        rotation_linear = RotationLinear(
                            layer, smooth_values_in=smooth_values, smooth_first=self.smooth_first
                        )

                        setattr(next_module_parent, relative_layer_name, rotation_linear)

        if "r2" in self.smooth_positions:  # type: ignore[operator]
            for i, layer in tqdm(
                enumerate(self.layers),
                desc="Applying trainable smoothquant on R2",
                total=self.model.config.num_hidden_layers,
            ):
                layer_proj_v = get_model_layers(layer, self.rotation_config.v_proj)
                layer_proj_o = get_model_layers(layer, self.rotation_config.o_proj)

                if isinstance(layer_proj_v, RotationLinear):
                    dtype = layer_proj_v.linear.weight.dtype
                    device = layer_proj_v.linear.weight.device
                else:
                    dtype = layer_proj_o.weight.dtype
                    device = layer_proj_o.weight.device

                # layer_proj_v.weight.shape: [out, in] = [num_key_value_heads * head_dim, hidden_size]
                # layer_proj_o.weight.shape: [out, in] = [hidden_size, num_attention_heads * head_dim]
                # with potentially num_key_value_heads < num_attention_heads.
                smooth_values, _ = RotationProcessor.initialize_smooth_values(
                    in_features=layer_proj_v.out_features,
                    dtype=dtype,
                    device=device,
                    layers_pattern=None,
                )

                if isinstance(layer_proj_v, RotationLinear):
                    layer_proj_v.smooth_values_out = smooth_values
                    layer_proj_v.smooth_first = self.smooth_first
                else:
                    rotation_linear_v_proj = RotationLinear(
                        layer_proj_v, smooth_values_out=smooth_values, smooth_first=self.smooth_first
                    )
                    setattr_recursive(layer, self.rotation_config.v_proj, rotation_linear_v_proj)

                if isinstance(layer_proj_o, RotationLinear):
                    layer_proj_o.smooth_values_in = smooth_values
                    layer_proj_o.smooth_first = self.smooth_first
                else:
                    rotation_linear_o_proj = RotationLinear(
                        layer_proj_o, smooth_values_in=smooth_values, smooth_first=self.smooth_first
                    )
                    setattr_recursive(layer, self.rotation_config.o_proj, rotation_linear_o_proj)

        if "r4" in self.smooth_positions:  # type: ignore[operator]
            for layer in tqdm(self.layers, desc="Applying trainable smoothquant on down_proj (R4)"):
                # We allow `rotation_config.mlp="mlp.experts.*"` in the case of MOE models.
                mlp_names = resolve_star([self.rotation_config.mlp], layer)

                mlp = getattr_recursive(layer, mlp_names[0])
                if isinstance(layer_proj_v, RotationLinear):
                    dtype = mlp.up_proj.linear.weight.dtype
                    device = mlp.up_proj.linear.weight.device
                else:
                    dtype = mlp.up_proj.weight.dtype
                    device = mlp.up_proj.weight.device

                if self.shared_parallel:
                    smooth_values, _ = RotationProcessor.initialize_smooth_values(
                        in_features=mlp.down_proj.in_features,
                        dtype=dtype,
                        device=device,
                        layers_pattern=None,
                    )

                for mlp_name in mlp_names:
                    mlp = getattr_recursive(layer, mlp_name)

                    if not self.shared_parallel:
                        smooth_values, _ = RotationProcessor.initialize_smooth_values(
                            in_features=mlp.down_proj.in_features,
                            dtype=dtype,
                            device=device,
                            layers_pattern=None,
                        )

                    # Fuse normalization into up_proj weight.
                    if isinstance(mlp.up_proj, RotationLinear):
                        assert mlp.up_proj.smooth_values_out is None
                        mlp.up_proj.smooth_values_out = smooth_values
                        mlp.up_proj.smooth_first = self.smooth_first
                    else:
                        rotation_linear_up_proj = RotationLinear(
                            mlp.up_proj, smooth_values_out=smooth_values, smooth_first=self.smooth_first
                        )

                        up_proj_name = f"{self.rotation_config.mlp}.up_proj"

                        setattr_recursive(layer, up_proj_name, rotation_linear_up_proj)

                    # Fuse normalization inverse into down_proj weight.
                    if isinstance(mlp.down_proj, RotationLinear):
                        assert mlp.down_proj.smooth_values_in is None
                        mlp.down_proj.smooth_values_in = smooth_values
                        mlp.down_proj.smooth_first = self.smooth_first
                    else:
                        rotation_linear_down_proj = RotationLinear(
                            mlp.down_proj, smooth_values_in=smooth_values, smooth_first=self.smooth_first
                        )

                        down_proj_name = f"{self.rotation_config.mlp}.down_proj"

                        setattr_recursive(layer, down_proj_name, rotation_linear_down_proj)

    def get_prev_out_channels_dims(self, prev_modules: list[nn.Module]) -> list[int]:
        prev_out_channels_dims = []
        for module in prev_modules:
            if isinstance(module, nn.Embedding):
                prev_out_channels_dims.append(1)
            elif isinstance(module, nn.Linear):
                prev_out_channels_dims.append(0)
            else:
                raise ValueError("prev_modules is wrong")
        return prev_out_channels_dims

    @classmethod
    def get_scaling_layers(
        cls,
        model: nn.Module,
        scaling_modules: dict[str, list[dict[str, Any]]],
        model_decoder_layers: str,
        online_r1_rotation: bool,
        r1: bool,
        smooth_positions: list[str] | None,
    ) -> list[dict[str, Sequence[str]]]:
        """
        Get the layers where R1 rotation is applied, and preceding normalization layer (used to fuse its weight), preceding layers into which the activation rotation may be merged.

        The argument `skip_last` is used e.g. in case of online R1 rotation, where rotation is not applied on the last layer (lm_head).
        """
        layers = get_model_layers(model, model_decoder_layers)

        if smooth_positions is None:
            smooth_positions = []

        def get_layer_config(layers_pattern: dict[str, Any], layer_index: int) -> dict[str, Any]:
            prev_modules = []
            norm_module = []
            next_modules = []

            if "prev_modules" in layers_pattern:
                prev_modules = [
                    layer_name.replace("pre_layer_id", str(layer_index - 1)).replace("layer_id", str(layer_index))
                    for layer_name in layers_pattern["prev_modules"]
                ]
                prev_modules = resolve_star(prev_modules, model)
            elif r1 and not online_r1_rotation:
                raise ValueError(
                    f"Expected layers_pattern={layers_pattern} to contain a key `'prev_modules'` when using online_r1_rotation=False. Make sure the provided configuration is correct."
                )

            if "norm_module" in layers_pattern:
                norm_module = layers_pattern["norm_module"].replace("layer_id", str(layer_index))
            elif (r1 and not online_r1_rotation) or "r1" in smooth_positions:
                raise ValueError(
                    f"Expected layers_pattern={layers_pattern} to contain a key `'norm_module'` when using online_r1_rotation=False, smooth_positions={smooth_positions}. Make sure the provided configuration is correct."
                )

            if "next_modules" in layers_pattern:
                next_modules = [
                    layer_name.replace("layer_id", str(layer_index)) for layer_name in layers_pattern["next_modules"]
                ]
                next_modules = resolve_star(next_modules, model)
            elif (r1 and not online_r1_rotation) or "r1" in smooth_positions:
                raise ValueError(
                    f"Expected layers_pattern={layers_pattern} to contain a key `'next_modules'` when using online_r1_rotation=False, smooth_positions={smooth_positions}. Make sure the provided configuration is correct."
                )

            if "target_modules" in layers_pattern:
                target_modules = [
                    layer_name.replace("layer_id", str(layer_index)) for layer_name in layers_pattern["target_modules"]
                ]
                target_modules = resolve_star(target_modules, model)
            else:
                target_modules = next_modules  # type: ignore[assignment]

            result = {
                "prev_modules": prev_modules,
                "norm_module": norm_module,
                "next_modules": next_modules,
                "target_modules": target_modules,
            }

            return result

        scaling_layers = []
        for layer_index in range(len(layers)):
            if layer_index == 0:
                for layers_pattern in scaling_modules["first_layer"]:
                    scaling_layers_dict = get_layer_config(layers_pattern, layer_index=layer_index)
                    scaling_layers.append(scaling_layers_dict)
            else:
                for layers_pattern in scaling_modules["middle_layers"]:
                    scaling_layers_dict = get_layer_config(layers_pattern, layer_index=layer_index)

                    scaling_layers.append(scaling_layers_dict)

                if layer_index == len(layers) - 1 and not online_r1_rotation:
                    for layers_pattern in scaling_modules["last_layer"]:
                        scaling_layers_dict = get_layer_config(layers_pattern, layer_index=layer_index)
                        scaling_layers.append(scaling_layers_dict)

        return scaling_layers

    def r1(self) -> None:
        if self.rotation_size is not None:
            r1_rotation_size = self.rotation_size
        else:
            r1_rotation_size = self.model.config.hidden_size

        device = next(self.model.parameters()).device

        r1_rotation = get_rotation_matrix(r1_rotation_size, random=self.random_r1, device=device)

        if self.trainable and not self.online_r1_rotation:
            r1_rotation = nn.Parameter(r1_rotation)
            self.model.shared_r1_rotation = r1_rotation

        for layers_pattern in tqdm(self.scaling_layers, desc="R1 Rotation"):
            logger.debug(f"layers_pattern: {layers_pattern}")

            next_modules = [getattr_recursive(self.model, layer_name) for layer_name in layers_pattern["next_modules"]]

            target_modules = [
                getattr_recursive(self.model, layer_name) for layer_name in layers_pattern["target_modules"]
            ]

            if self.online_r1_rotation and len(target_modules) == 0:
                continue

            prev_modules = [getattr_recursive(self.model, layer_name) for layer_name in layers_pattern["prev_modules"]]

            if not self.trainable:
                if self.online_r1_rotation:
                    self.apply_online_r1(
                        layers_pattern=layers_pattern, target_modules=target_modules, rotation_size=r1_rotation_size
                    )
                else:
                    self.apply_and_fuse_r1(
                        r1_rotation=r1_rotation,
                        layers_pattern=layers_pattern,
                        prev_modules=prev_modules,
                        next_modules=next_modules,
                    )
            else:
                if self.online_r1_rotation:
                    self.apply_r1_for_online_training(
                        layers_pattern=layers_pattern,
                        target_modules=target_modules,
                        rotation_size=r1_rotation_size,
                    )
                else:
                    self.apply_r1_for_offline_training(
                        r1_rotation=r1_rotation,
                        layers_pattern=layers_pattern,
                        prev_modules=prev_modules,
                        next_modules=next_modules,
                    )

        clear_memory()

    def fuse_normalization(
        self,
        layers_pattern: dict[str, Any],
        prev_modules: list[nn.Module],
        next_modules: list[nn.Module],
    ) -> None:
        """
        Fuse RMSNorm or LayerNorm into the following layers.
        """
        norm_module = getattr_recursive(self.model, layers_pattern["norm_module"])

        prev_out_channels_dims = self.get_prev_out_channels_dims(prev_modules)

        transform_norm_and_linear(
            prev_modules=prev_modules,
            norm_module=norm_module,
            next_modules=next_modules,
            prev_out_channels_dims=prev_out_channels_dims,
        )

    def apply_r1_for_online_training(
        self,
        layers_pattern: dict[str, Any],
        target_modules: list[nn.Module],
        rotation_size: int,
    ) -> None:
        if self.shared_parallel:
            r1_rotation = get_rotation_matrix(
                rotation_size, random=self.random_r1, device=target_modules[0].weight.device
            )
            r1_rotation = nn.Parameter(r1_rotation)
        else:
            r1_rotation = None

        for i, layer in enumerate(target_modules):
            full_layer_name = layers_pattern["target_modules"][i]

            # e.g. `"model.layers.10.mlp"`.
            next_module_parent_name = ".".join(full_layer_name.split(".")[:-1])

            if next_module_parent_name == "":
                next_module_parent = self.model
            else:
                next_module_parent = getattr_recursive(self.model, next_module_parent_name)
            relative_layer_name = full_layer_name.split(".")[-1]

            if not self.shared_parallel:
                r1_rotation = get_rotation_matrix(rotation_size, random=self.random_r1, device=layer.weight.device)
                r1_rotation = nn.Parameter(r1_rotation)
            else:
                assert r1_rotation is not None

            assert not isinstance(layer, RotationLinear)
            rotation_linear = RotationLinear(layer, rotation_in=r1_rotation, hint_in="r1", rotate_activation=True)

            setattr(next_module_parent, relative_layer_name, rotation_linear)

    def apply_r1_for_offline_training(
        self,
        r1_rotation: torch.nn.Parameter,
        layers_pattern: dict[str, Any],
        prev_modules: list[nn.Module],
        next_modules: list[nn.Module],
    ) -> None:
        assert isinstance(r1_rotation, torch.nn.Parameter)

        self.fuse_normalization(layers_pattern=layers_pattern, prev_modules=prev_modules, next_modules=next_modules)

        rotated_list = []
        prev_out_channels_dims = self.get_prev_out_channels_dims(prev_modules)

        for index in range(len(prev_out_channels_dims)):
            full_layer_name = layers_pattern["prev_modules"][index]

            # e.g. `"model.layers.10.mlp"`.
            prev_module_parent_name = ".".join(full_layer_name.split(".")[:-1])

            if prev_module_parent_name == "":
                prev_module_parent = self.model
            else:
                prev_module_parent = getattr_recursive(self.model, prev_module_parent_name)
            relative_layer_name = full_layer_name.split(".")[-1]

            prev_module = prev_modules[index]
            if prev_module in rotated_list:
                continue

            if isinstance(prev_module, nn.Embedding):
                wrapped_embedding = OutputRotationWrapper(prev_module, hint_out="r1", rotation_out=r1_rotation)

                setattr(prev_module_parent, relative_layer_name, wrapped_embedding)
            elif isinstance(prev_module, nn.Linear):
                rotation_linear = RotationLinear(
                    prev_module, rotation_out=r1_rotation, hint_out="r1", rotate_activation=False
                )

                setattr(prev_module_parent, relative_layer_name, rotation_linear)
            else:
                raise ValueError("unsupported!")

            rotated_list.append(prev_module)

        # Apply R1 on `next_modules` weights in the input feature, i.e. QKV projections, gate/up projections, lm_head.
        # We pass `rotate_activation=False` as activation rotations were fused into preceding layers (embed_tokens, o_proj and down_proj).
        for i, layer in enumerate(next_modules):
            full_layer_name = layers_pattern["next_modules"][i]

            # e.g. `"model.layers.10.mlp"`.
            next_module_parent_name = ".".join(full_layer_name.split(".")[:-1])

            if next_module_parent_name == "":
                next_module_parent = self.model
            else:
                next_module_parent = getattr_recursive(self.model, next_module_parent_name)
            relative_layer_name = full_layer_name.split(".")[-1]

            assert not isinstance(layer, RotationLinear)
            rotation_linear = RotationLinear(layer, rotation_in=r1_rotation, hint_in="r1", rotate_activation=False)

            setattr(next_module_parent, relative_layer_name, rotation_linear)

    def fuse_r1(
        self,
        r1_rotation: torch.Tensor,
        prev_modules: list[nn.Module],
        next_modules: list[nn.Module],
    ) -> None:
        """
        Fuse R1 in a decoder layer, assuming RMSNorm/LayerNorm have already been fused.
        """
        rotated_list = []
        prev_out_channels_dims = self.get_prev_out_channels_dims(prev_modules)

        for index in range(len(prev_out_channels_dims)):
            prev_module = prev_modules[index]
            if prev_module in rotated_list:
                continue

            if prev_out_channels_dims[index] == 0:
                rotate_fn = rotate_out_channels_
            else:
                rotate_fn = rotate_in_channels_

            rotate_fn(prev_module, rotation=r1_rotation)

            rotated_list.append(prev_module)

        for _, fc in enumerate(next_modules):
            if fc not in rotated_list:
                rotate_in_channels_(fc, rotation=r1_rotation)
                rotated_list.append(fc)

    def apply_and_fuse_r1(
        self,
        r1_rotation: torch.Tensor,
        layers_pattern: dict[str, Any],
        prev_modules: list[nn.Module],
        next_modules: list[nn.Module],
    ) -> None:
        self.fuse_normalization(layers_pattern=layers_pattern, prev_modules=prev_modules, next_modules=next_modules)

        self.fuse_r1(r1_rotation=r1_rotation, prev_modules=prev_modules, next_modules=next_modules)

    def apply_online_r1(
        self,
        layers_pattern: dict[str, Any],
        target_modules: list[nn.Module],
        rotation_size: int,
    ) -> None:
        """
        Inserts InputRotationWrapper modules in a decoder layer, running activation rotations online.
        """
        for i, layer in enumerate(target_modules):
            full_layer_name = layers_pattern["target_modules"][i]

            # e.g. `"model.layers.10.mlp"`.
            next_module_parent_name = ".".join(full_layer_name.split(".")[:-1])

            if next_module_parent_name == "":
                next_module_parent = self.model
            else:
                next_module_parent = getattr_recursive(self.model, next_module_parent_name)
            relative_layer_name = full_layer_name.split(".")[-1]

            dtype = layer.weight.data.dtype
            in_features = layer.weight.shape[-1]

            # The buffer `rotation_buffer` avoids recomputing the same hadamard matrices multiple times.
            if in_features not in self.rotation_buffer:
                hadamard_K, K = _get_hadamard_K(rotation_size)
                hadamard_K = hadamard_K.to(layer.weight.device)
                self.rotation_buffer[in_features] = (hadamard_K, K)
            else:
                hadamard_K, K = self.rotation_buffer[in_features]

            if rotation_size == layer.weight.data.shape[1]:
                # `inverse=True` is not required here as nn.Linear already transpose the weight.
                layer.weight.data = matmul_hadU(layer.weight.data, hadamard_K=hadamard_K, K=K).to(dtype)
            else:
                if hadamard_K.shape[0] != rotation_size:
                    hadamard_1, _ = _get_hadamard_K(rotation_size // K)

                    hadamard_1 = hadamard_1.to(layer.weight.device)
                    hadamard_K = hadamard_K.to(layer.weight.device)

                    hadamard_K = torch.kron(hadamard_K, hadamard_1)
                    K = rotation_size

                assert hadamard_K.shape[0] == rotation_size
                rotate_in_channels_(layer, rotation=hadamard_K.to(torch.float64) / math.sqrt(rotation_size))

            layer_with_input_rotation = InputRotationWrapperHadamard(
                layer, hadamard_K=hadamard_K, K=K, rotation_size=rotation_size
            )

            setattr(next_module_parent, relative_layer_name, layer_with_input_rotation)

    def r2(self) -> None:
        r2_rotation_size = getattr(
            self.model.config, "head_dim", self.model.config.hidden_size // self.model.config.num_attention_heads
        )

        if not self.trainable:
            device = next(self.model.parameters()).device
            rotation2 = get_rotation_matrix(r2_rotation_size, random=self.random_r2, device=device)

        for i, layer in tqdm(enumerate(self.layers), desc="R2 Rotation", total=self.model.config.num_hidden_layers):
            layer_proj_v = get_model_layers(layer, self.rotation_config.v_proj)
            layer_proj_o = get_model_layers(layer, self.rotation_config.o_proj)

            if not self.trainable:
                rotation2 = rotation2.to(layer_proj_v.weight.device)
                rotate_out_channels_(layer_proj_v, rotation=rotation2)

                rotation2 = rotation2.to(layer_proj_o.weight.device)
                rotate_in_channels_(layer_proj_o, rotation=rotation2)

                clear_memory()
            else:
                if isinstance(layer_proj_v, RotationLinear):
                    device = layer_proj_v.linear.weight.device
                else:
                    device = layer_proj_v.weight.device

                rotation2 = get_rotation_matrix(r2_rotation_size, random=self.random_r2, device=device)
                rotation2 = torch.nn.Parameter(rotation2)

                if self.rotation_config.r1:
                    assert isinstance(layer_proj_v, RotationLinear)
                    layer_proj_v.rotation_out = rotation2
                    layer_proj_v.hint_out = "r2"
                else:
                    rotation_linear_v_proj = RotationLinear(
                        layer_proj_v, rotation_out=rotation2, hint_out="r2", rotate_activation=False
                    )

                    setattr_recursive(layer, self.rotation_config.v_proj, rotation_linear_v_proj)

                if self.rotation_config.r1 and not self.online_r1_rotation:
                    assert isinstance(layer_proj_o, RotationLinear)
                    assert layer_proj_o.rotation_in is None

                    layer_proj_o.rotation_in = rotation2
                    layer_proj_o.hint_in = "r2"
                    layer_proj_o.rotate_activation = False
                else:
                    rotation_linear_o_proj = RotationLinear(
                        layer_proj_o, rotation_in=rotation2, hint_in="r2", rotate_activation=False
                    )
                    setattr_recursive(layer, self.rotation_config.o_proj, rotation_linear_o_proj)

    def r3(self) -> None:
        if self.rotation_size is not None:
            raise NotImplementedError("R3 does not support custom rotation size at the moment. Please open an issue.")

        for layer in tqdm(self.layers, desc="R3 Rotation"):
            add_qk_rotation_after_function_call_in_forward(
                get_model_layers(layer, self.rotation_config.self_attn),
                "apply_rotary_pos_emb",  # this is the name of the function called in Llama that actually does RoPE
            )
            clear_memory()

    def r4(self) -> None:
        if self.rotation_size is not None:
            custom_rotation_size = True
            r4_rotation_size = self.rotation_size
        else:
            custom_rotation_size = False
            r4_rotation_size = getattr(self.model.config, "moe_intermediate_size", self.model.config.intermediate_size)

        for layer in tqdm(self.layers, desc="R4 Rotation"):
            # We allow `rotation_config.mlp="mlp.experts.*"` in the case of MOE models.
            mlp_names = resolve_star([self.rotation_config.mlp], layer)

            mlp = getattr_recursive(layer, mlp_names[0])
            if isinstance(mlp.down_proj, RotationLinear):
                dtype = mlp.down_proj.linear.weight.dtype
                device = mlp.down_proj.linear.weight.device
            else:
                dtype = mlp.down_proj.weight.dtype
                device = mlp.down_proj.weight.device

            if self.trainable and self.shared_parallel:
                rotation4 = get_rotation_matrix(r4_rotation_size, random=False, device=device)
                rotation4 = torch.nn.Parameter(rotation4)

            # MOE layers may have several MLP.
            for mlp_name in mlp_names:
                mlp = getattr_recursive(layer, mlp_name)

                if not self.trainable:
                    if custom_rotation_size:
                        rotation4 = get_rotation_matrix(r4_rotation_size, random=False, device=device)  # type: ignore[arg-type]

                        rotate_in_channels_(mlp.down_proj, rotation=rotation4)
                    else:
                        # `inverse=True` is not required here as nn.Linear already transpose the weight.
                        mlp.down_proj.weight.data = matmul_hadU(mlp.down_proj.weight.data).to(dtype)

                    mlp.down_proj = InputRotationWrapperHadamard(mlp.down_proj, r4_rotation_size)
                else:
                    if not self.shared_parallel:
                        rotation4 = get_rotation_matrix(r4_rotation_size, random=False, device=device)
                        rotation4 = torch.nn.Parameter(rotation4)

                    if isinstance(mlp.down_proj, RotationLinear):
                        assert mlp.down_proj.rotation_in is None
                        mlp.down_proj.rotation_in = rotation4
                        mlp.down_proj.hint_in = "r4"
                        mlp.down_proj.rotate_activation = True
                    else:
                        rotation_linear_down_proj = RotationLinear(
                            mlp.down_proj, rotation_in=rotation4, hint_in="r4", rotate_activation=True
                        )

                        down_proj_name = f"{mlp_name}.down_proj"

                        setattr_recursive(layer, down_proj_name, rotation_linear_down_proj)

            clear_memory()

    @staticmethod
    def get_online_rotation_layers(rotation_config: RotationConfig, model: nn.Module) -> list[str]:
        """
        Get the submodule names that are using online rotations.
        """
        layers_online_rotation = []  # type: ignore

        online_r1_rotation = rotation_config.online_r1_rotation

        scaling_layers = rotation_config.scaling_layers

        if rotation_config.r1 and online_r1_rotation:
            if scaling_layers is None:
                raise ValueError("Expected `scaling_layers` to be not None with r1=True, online_r1_rotation=True.")

            scaling_layers = RotationProcessor.get_scaling_layers(
                model,
                scaling_layers,
                r1=rotation_config.r1,
                online_r1_rotation=online_r1_rotation,
                smooth_positions=rotation_config.smooth_positions,
                model_decoder_layers=rotation_config.model_decoder_layers,
            )  # type: ignore

            for scaling_dict in scaling_layers:
                layers_online_rotation = layers_online_rotation + list(scaling_dict["target_modules"])

        # R4 is always online.
        if rotation_config.r4:
            for name, _ in model.named_modules():
                if name.endswith("down_proj"):
                    layers_online_rotation.append(name)

        return layers_online_rotation

    @staticmethod
    def post_process_trained_rotation(model: nn.Module, quantization_config: QConfig) -> nn.Module:
        """
        This method removes trainable rotations, fuse the weight rotations, and if necessary inserts ``InputRotationWrapperOrthogonal`` for online activation rotation.
        """
        model = model.eval()

        named_modules_dict = dict(model.named_modules())

        # We do not need to explore the `layer.linear` layers, which were wrapped by `RotationLinear`.
        skip_processing = set()

        for name, submodule in named_modules_dict.items():
            if isinstance(submodule, TrainableRMSNorm):
                logger.debug(f"Converting TrainableRMSNorm: {name}")
                weight = submodule.original_normalization.weight.data * submodule.smooth_values.data
                submodule.original_normalization.weight.data = weight
                setattr_recursive(model, submodule.norm_layer_name, submodule.original_normalization)

            if isinstance(submodule, RotationLinear):
                logger.debug(f"Converting RotationLinear: {name}")
                original_linear = submodule.linear
                skip_processing.add(name + ".linear")

                # Handle weight transformations on the input dimension.
                if submodule.smooth_values_in is not None and submodule.smooth_first:
                    logger.debug("    Fusing smooth_values_in with smooth_first=True")
                    weight = submodule.apply_in_normalization_weight(original_linear.weight.data)
                    original_linear.weight.data = weight

                if submodule.rotation_in is not None:
                    logger.debug("    Fusing rotation_in")
                    rotate_in_channels_(original_linear, submodule.rotation_in)

                if submodule.smooth_values_in is not None and not submodule.smooth_first:
                    logger.debug("    Fusing smooth_values_in with smooth_first=False")
                    weight = submodule.apply_in_normalization_weight(original_linear.weight.data)
                    original_linear.weight.data = weight

                # Handle weight transformations on the output dimension.
                if submodule.smooth_values_out is not None and submodule.smooth_first:
                    logger.debug("    Fusing smooth_values_out with smooth_first=True")
                    original_linear.weight.data = original_linear.weight.data * submodule.smooth_values_out[:, None]

                if submodule.rotation_out is not None:
                    logger.debug("    Fusing rotation_out")
                    rotate_out_channels_(original_linear, submodule.rotation_out)

                    assert original_linear.bias is None

                if submodule.smooth_values_out is not None and not submodule.smooth_first:
                    logger.debug("    Fusing smooth_values_out with smooth_first=False")
                    original_linear.weight.data = original_linear.weight.data * submodule.smooth_values_out[:, None]

                # TODO: Support layer_quant_config here.
                # We need to re-initialize quantizers, as `weight.is_dynamic` is set to
                # `True` during rotation training, and thus its `scale`, `zero_point`, etc.
                # are set as non-persistent buffers.
                # Moreover, we allow to train with QDQ only for activations, and may want
                # to re-insert QDQ for weights here (e.g. SpinQuant + GPTQ).
                if isinstance(original_linear, QuantLinear):
                    trained_weight_spec = original_linear._weight_qspec

                    logger.debug(f"    Re-initializing quantizers for {name}")
                    original_linear.init_quantizer(
                        quantization_config.global_quant_config, device=original_linear.device
                    )

                    # In case of rotation training, weight quantization must be handled in
                    # a later `quantizer.model_quantize` call.
                    # Weight quantization may be disabled during rotation training (e.g. SpinQuant),
                    # and `post_process_trained_rotation` does NOT change the model output,
                    # only properly re-initializes quantizers for later quantization.
                    if trained_weight_spec is None and isinstance(original_linear._weight_quantizer, FakeQuantizeBase):
                        original_linear._weight_quantizer.disable_fake_quant()

                if not submodule.rotate_activation:
                    # Offline activation rotation.
                    logger.debug("    Setting offline rotation")
                    setattr_recursive(model, name, original_linear)
                else:
                    # Online activation rotation.
                    assert submodule.rotation_in is not None

                    logger.debug("    Setting online rotation with InputRotationWrapperOrthogonal")
                    layer_with_input_rotation = InputRotationWrapperOrthogonal(
                        original_linear,
                        rotation_matrix=submodule.rotation_in.data,
                    )

                    setattr_recursive(model, name, layer_with_input_rotation)
            elif isinstance(submodule, QuantLinear) and name not in skip_processing:
                trained_weight_spec = submodule._weight_qspec

                # See the comment above.
                submodule.init_quantizer(quantization_config.global_quant_config, device=submodule.device)

                # See the comment above.
                if trained_weight_spec is None and isinstance(submodule._weight_quantizer, FakeQuantizeBase):
                    logger.debug("    Call submodule._weight_quantizer.disable_fake_quant")
                    submodule._weight_quantizer.disable_fake_quant()

        for name, submodule in model.named_modules():
            if isinstance(submodule, OutputRotationWrapper):
                logger.debug(f"Removing OutputRotationWrapper used for {name}")
                original_module = submodule.original_module

                original_module.weight.data = rotate_with_size(
                    original_module.weight.data,
                    rotation_matrix=submodule.rotation_out.data,
                )

                setattr_recursive(model, name, original_module)

        if hasattr(model, "shared_r1_rotation"):
            delattr(model, "shared_r1_rotation")
            torch.cuda.empty_cache()

        return model

    @staticmethod
    def get_trainable_parameters(
        model: nn.Module, rotation_config: RotationConfig
    ) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """
        Gets a list of trainable parameters to train orthogonal transforms.

        This function assumes that the model has run through `model = quantizer.quantize_model` already.
        """
        if not rotation_config.trainable:
            raise ValueError(
                "RotationProcessor.get_trainable_parameters was called but rotation_config.trainable is False. Something is likely wrong."
            )
        trainable_parameters = []
        trainable_parameters_pointers = set()

        trainable_parameters_adam = []

        for name, param in model.named_parameters():
            if param.data_ptr() not in trainable_parameters_pointers and (
                "rotation_in" in name or "rotation_out" in name or "shared_r1_rotation" in name
            ):
                trainable_parameters.append(param)
                trainable_parameters_pointers.add(param.data_ptr())

        if rotation_config.train_smooth:
            for name, submodule in model.named_modules():
                if isinstance(submodule, RotationLinear):
                    if submodule.smooth_values_in is not None:
                        assert submodule.smooth_values_in.requires_grad

                        # Avoids /root/miniforge3/lib/python3.12/site-packages/torch/_compile.py:53: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
                        if submodule.smooth_values_in.data_ptr() not in trainable_parameters_pointers:
                            trainable_parameters_adam.append(submodule.smooth_values_in)
                            trainable_parameters_pointers.add(submodule.smooth_values_in.data_ptr())

                    if submodule.smooth_values_out is not None:
                        assert submodule.smooth_values_out.requires_grad

                        # Avoids /root/miniforge3/lib/python3.12/site-packages/torch/_compile.py:53: UserWarning: optimizer contains a parameter group with duplicate parameters; in future, this will cause an error; see github.com/pytorch/pytorch/issues/40967 for more information
                        if submodule.smooth_values_out.data_ptr() not in trainable_parameters_pointers:
                            trainable_parameters_adam.append(submodule.smooth_values_out)
                            trainable_parameters_pointers.add(submodule.smooth_values_out.data_ptr())

        return trainable_parameters, trainable_parameters_adam

    @staticmethod
    def prepare_model_for_reloading_fake(model: nn.Module, quantization_config: QConfig) -> None:
        """
        Prepares a model using ``weight_format="fake_quantized"`` to reload online rotations.

        The case ``weight_format="real_quantized"`` is handled directly in ``QParamsLinearWithRotation``.
        """
        rotation_config = quantization_config.get_rotation_config()

        if rotation_config is None:
            raise RuntimeError(
                "rotation_config is None in `prepare_model_for_reloading`, which is unexpected. Please open an issue."
            )

        layers_online_rotation = RotationProcessor.get_online_rotation_layers(rotation_config, model)

        if rotation_config.r3:
            raise NotImplementedError(
                "Reloading a model quantization using rotation algorithm with r3=True is not supported at the moment. Please open an issue."
            )

        for name, module in model.named_modules():
            if isinstance(module, QuantLinear) and name in layers_online_rotation:
                rotation_size = rotation_config.rotation_size  # type: ignore[union-attr]
                trainable = rotation_config.trainable  # type: ignore[union-attr]

                if rotation_size is None:
                    rotation_size = module.in_features

                if trainable:
                    rotation_dtype = torch.float64
                else:
                    rotation_dtype = torch.int8

                input_rotation = torch.zeros(
                    (rotation_size, rotation_size), device=module.weight.device, dtype=rotation_dtype
                )
                module.register_buffer("input_rotation", input_rotation)
