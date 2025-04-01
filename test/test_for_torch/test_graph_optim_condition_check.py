#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys

sys.path.append("..")

import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch.jit import Final
from torch.nn import functional as F
from torch._export import capture_pre_autograd_graph
# -------- init config -----
from quark.torch.quantization.config.config import QuantizationSpec, QuantizationConfig, Config
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType, QuantizationMode
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver
from quark.torch import ModelQuantizer
from quark.torch.quantization.graph.processor.processor import prepare_quant_model
from quark.shares.utils.testing_utils import torch_device

INT8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                        qscheme=QSchemeType.per_tensor,
                                        observer_cls=PerTensorMinMaxObserver,
                                        symmetric=True,
                                        scale_type=ScaleType.float,
                                        round_method=RoundType.half_even,
                                        is_dynamic=False)
quant_config = QuantizationConfig(input_tensors=INT8_PER_TENSOR_SPEC,
                                  output_tensors=INT8_PER_TENSOR_SPEC,
                                  weight=INT8_PER_TENSOR_SPEC,
                                  bias=INT8_PER_TENSOR_SPEC)
quant_config = Config(global_quant_config=quant_config, quant_mode=QuantizationMode.fx_graph_mode)


# ================== following aims to test conv's weight is not a pure attr node that save parameter
class CondConv2d(nn.Module):
    """ Conditionally Parameterized Convolution
    """
    __constants__ = ['in_channels', 'out_channels', 'dynamic_padding']

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 num_experts=4):
        super(CondConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.dynamic_padding = False
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.num_experts = num_experts

        self.weight_shape = (self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = torch.nn.Parameter(torch.Tensor(self.num_experts, weight_num_param))

        if bias:
            self.bias_shape = (self.out_channels, )
            self.bias = torch.nn.Parameter(torch.Tensor(self.num_experts, self.out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, routing_weights):
        B, C, H, W = x.shape
        weight = torch.matmul(routing_weights, self.weight)
        new_weight_shape = (B * self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight = weight.view(new_weight_shape)
        bias = None
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(B * self.out_channels)
        # move batch elements with channels so each batch element can be efficiently convolved with separate kernel
        # reshape instead of view to work with channels_last input
        x = x.reshape(1, B * C, H, W)

        out = F.conv2d(x,
                       weight,
                       bias,
                       stride=self.stride,
                       padding=self.padding,
                       dilation=self.dilation,
                       groups=self.groups * B)
        out = out.permute([1, 0, 2, 3]).view(B, self.out_channels, out.shape[-2], out.shape[-1])
        return out


class TinyModel(nn.Module):
    '''
    This model is particularly designed to test graph optimization function:
    If conv's weight is not a attr Node that save the Parameter,
    skip replace torch.ops.aten.conv2d -> QuantConv2d
    '''

    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 32, 3, bias=False, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.routing_fn = nn.Linear(32, 4)
        self.conv2d_1 = CondConv2d(32, 64, 3, bias=False)
        self.relu_1 = nn.ReLU(inplace=True)
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # CondConv routing
        routing_weights = torch.sigmoid(self.routing_fn(pooled_inputs))
        x = self.conv2d_1(x, routing_weights)
        x = self.relu_1(x)
        x = self.adaptive_avg_pool2d(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


# NOTE: in this model conv2d_1's weight is not a ordinary node that save a single Parameters
# in this case we do not perform torch.ops.aten.conv2d -> QuantConv2d
'''
sigmoid = torch.ops.aten.sigmoid.default(linear)
_param_constant3 = self._param_constant3
matmul = torch.ops.aten.matmul.default(sigmoid, _param_constant3)
view = torch.ops.aten.view.default(matmul, [64, 32, 3, 3])
reshape = torch.ops.aten.reshape.default(relu_, [1, 32, 54, 54])
conv2d_1 = torch.ops.aten.conv2d.default(reshape, view)
'''


def test_graph_conv_weight_replace_condition():
    torch.cuda.empty_cache()
    float_model = TinyModel().to(torch_device).eval()
    example_inputs = (torch.ones(1, 3, 14, 14).to(torch_device), )
    # prepare the torch.fx.GraphModule
    float_model(example_inputs[0])
    graph_model = capture_pre_autograd_graph(float_model, example_inputs)
    prepared_quant_model = prepare_quant_model(graph_model, quant_config)
    print("Finish test: test_graph_conv_weight_replace_condition")
    torch.cuda.empty_cache()

# ======== following test when some operation is in CPU, however some operation may in GPU.
# If not in same device, we will not (1) insert quantizer and (2) convert scalar to attr
def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def get_decomposed_rel_pos_bias(
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py
    Args:
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        bias (Tensor): attention bias to add to attention map
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn_bias = rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    return attn_bias.reshape(-1, q_h * q_w, k_h * k_w)


def rot(x):
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def apply_rot_embed_cat(x: torch.Tensor, emb):
    sin_emb, cos_emb = emb.tensor_split(2, -1)
    if sin_emb.ndim == 3:
        return x * cos_emb.unsqueeze(1).expand_as(x) + rot(x) * sin_emb.unsqueeze(1).expand_as(x)
    return x * cos_emb + rot(x) * sin_emb


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm=False,
        attn_drop=0.,
        proj_drop=0.,
        norm_layer=nn.LayerNorm,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert rope is None
            assert (input_size is not None), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim))
        self.rope = rope

    def forward(self, x):
        B, H, W, _ = x.shape
        N = H * W
        x = x.reshape(B, N, -1)
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # qkv with shape (3, B, nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, N, -1).unbind(0)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_rel_pos:
            attn_bias = get_decomposed_rel_pos_bias(q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        else:
            attn_bias = None
            if self.rope is not None:
                rope = self.rope.get_embed()
                q = apply_rot_embed_cat(q, rope).type_as(v)
                k = apply_rot_embed_cat(k, rope).type_as(v)

        if self.fused_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_bias is not None:
                attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.view(B, self.num_heads, N, -1).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = x.view(B, H, W, -1)
        return x


class TinyAttentionModel(nn.Module):
    '''
    This model is particularly designed to test graph optimization function,
    If some operation is forced to run in the CPU,
    perform the device check to determine whether to convert scalars to tensors.
    e.g:
        1. Model is in GPU.
        2. if torch.ops.aten.mul.Tensor(tensor1, 2.0) # tensor1 is in CPU
            then: 2 will not convert to torch.Tensor([2])
        2. if torch.ops.aten.mul.Tensor(tensor1, 2.0) # tensor1 is in GPU
            then: 2 will not convert to torch.Tensor([2])
    '''

    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 32, 3, bias=False, padding=1)
        # Batch 14, 14, 768
        self.conv2d_0 = nn.Conv2d(32, 768, 3, bias=False, padding=1)
        self.attention = Attention(dim=768, use_rel_pos=True, input_size=(14, 14))
        self.conv2d_1 = nn.Conv2d(768, 32, 3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.routing_fn = nn.Linear(32, 64)
        self.relu_1 = nn.ReLU(inplace=True)
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.conv2d_0(x)
        # input torch.Size([1, 14, 14, 768])
        x = x.permute((0, 2, 3, 1))
        # NOTE in this case, scalars in attention will not convert to attrs
        x = self.attention(x)
        # output torch.Size([1, 14, 14, 768])
        x = x.permute((0, 3, 1, 2))
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.adaptive_avg_pool2d(x)
        x = torch.flatten(x, 1)
        x = self.routing_fn(x)
        # NOTE test convert the scalar to attrs
        x = x + int(1)
        x = x * int(2)
        x = x / float(2.0)
        x = self.linear(x)
        return x


def test_graph_scalar_convert_insert_quantizer_condition():
    torch.cuda.empty_cache()
    float_model = TinyAttentionModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 14, 14).to(torch_device), )
    # prepare the torch.fx.GraphModule
    graph_model = capture_pre_autograd_graph(float_model, example_inputs)
    quantizer = ModelQuantizer(quant_config)
    quantized_model = quantizer.quantize_model(graph_model, [torch.rand(1, 3, 14, 14).to(torch_device) for _ in range(3)])
    quantized_model(example_inputs[0])
    print("Finish test: test_graph_scalar_convert_insert_quantizer_condition")
    torch.cuda.empty_cache()


# ================ Test if conv + bn, but another layer use conv's output, then skip fold
class TinyConvModel(nn.Module):
    '''
    This model is particularly designed to test graph optimization function,
    if conv -> bn and conv -> another_layer
    then will not perform fold conv + bn.
    '''

    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 32, 3, bias=False, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.conv2d_1 = nn.Conv2d(32, 32, 3, bias=False, padding=1)
        self.bn_1 = nn.BatchNorm2d(32)
        self.conv2d_2 = nn.Conv2d(32, 64, 3, bias=False, padding=1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv2d(x)
        x_bn = self.bn(x)
        x_path = self.conv2d_1(x_bn)
        x_path = self.bn_1(x_bn)
        x = x + x_path
        x = self.conv2d_2(x)
        x = self.bn_2(x)
        x = self.adaptive_avg_pool2d(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


def test_graph_skip_fold_conv_bn_condition():
    torch.cuda.empty_cache()
    float_model = TinyConvModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 14, 14).to(torch_device), )
    # prepare the torch.fx.GraphModule
    float_model(example_inputs[0])
    graph_model = capture_pre_autograd_graph(float_model, example_inputs)
    quantizer = ModelQuantizer(quant_config)
    quantized_model = quantizer.quantize_model(graph_model, [torch.rand(1, 3, 14, 14).to(torch_device) for _ in range(3)])
    quantized_model(example_inputs[0])
    print("Finish test: test_graph_skip_fold_conv_bn_condition")
    torch.cuda.empty_cache()


# =============== Test reshape param change ============
class TinyReshapeParamChangeModel(nn.Module):
    '''
    This model is particularly designed to test graph optimization function,
    If reshape params are all set, we can let the the first reshape param to -1.
    e.g reshape(tensor, [10, 64, 64]) to reshape(tensor, [-1, 64, 64])
    '''

    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 32, 3, bias=True, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(32, 32)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.adaptive_avg_pool2d(x)
        # x.shape = [b, 32, 1, 1]
        x = torch.reshape(x, [x.shape[0], x.shape[1]])  # reshape [b, 32]
        x = self.linear(x)
        return x


def test_graph_reshape_param_change():
    torch.cuda.empty_cache()
    float_model = TinyReshapeParamChangeModel().to(torch_device).eval()
    example_inputs = (torch.rand(4, 3, 112, 112).to(torch_device), )
    # prepare the torch.fx.GraphModule
    float_model(example_inputs[0])
    graph_model = capture_pre_autograd_graph(float_model, example_inputs)
    quantizer = ModelQuantizer(quant_config)
    quantized_model = quantizer.quantize_model(graph_model, [torch.rand(4, 3, 112, 112).to(torch_device) for _ in range(3)])
    quantized_model(example_inputs[0])
    print("Finish test: test_graph_reshape_param_change")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    test_graph_conv_weight_replace_condition()
    test_graph_scalar_convert_insert_quantizer_condition()
    test_graph_skip_fold_conv_bn_condition()
    test_graph_skip_fold_conv_bn_condition()
    torch.cuda.empty_cache()
