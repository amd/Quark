#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn

from quark.torch.utils.torch_utils import infer_decoder_layers_path


class SimpleBlock(nn.Module):
    """A simple transformer-like block for testing."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class MoEBlock(nn.Module):
    """A MoE block with nested ModuleList (experts)."""

    def __init__(self, dim: int = 64, num_experts: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.experts = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified: just use first expert
        return self.experts[0](self.norm(x))


# =============================================================================
# Test: Standard LLM structure (model.layers)
# =============================================================================
def test_standard_llm_structure():
    """Test detection of standard 'model.layers' structure."""

    class LlamaLikeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 64)
            self.layers = nn.ModuleList([SimpleBlock() for _ in range(4)])
            self.norm = nn.LayerNorm(64)

    model = LlamaLikeModel()
    result = infer_decoder_layers_path(model)
    assert result == "layers", f"Expected 'layers', got '{result}'"


def test_nested_model_layers():
    """Test detection of 'model.layers' when wrapped in model attribute."""

    class WrappedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 64)
            self.layers = nn.ModuleList([SimpleBlock() for _ in range(4)])

    class OuterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = WrappedModel()
            self.lm_head = nn.Linear(64, 1000)

    model = OuterModel()
    result = infer_decoder_layers_path(model)
    assert result == "model.layers", f"Expected 'model.layers', got '{result}'"


# =============================================================================
# Test: Nested ModuleLists (MoE models)
# =============================================================================
def test_moe_model_filters_nested_experts():
    """Test that nested expert ModuleLists are filtered out."""

    class MoEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 64)
            self.layers = nn.ModuleList([MoEBlock() for _ in range(4)])
            self.norm = nn.LayerNorm(64)

    model = MoEModel()
    result = infer_decoder_layers_path(model)
    # Should return 'layers', not 'layers.0.experts' or similar
    assert result == "layers", f"Expected 'layers', got '{result}'"


def test_deeply_nested_modulelists():
    """Test with multiple levels of nested ModuleLists."""

    class DeeplyNestedBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub_layers = nn.ModuleList([nn.ModuleList([nn.Linear(64, 64) for _ in range(2)]) for _ in range(2)])

    class DeepModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([DeeplyNestedBlock() for _ in range(4)])

    model = DeepModel()
    result = infer_decoder_layers_path(model)
    assert result == "layers", f"Expected 'layers', got '{result}'"


# =============================================================================
# Test: Edge cases
# =============================================================================
def test_empty_model_no_modulelist():
    """Test model with no ModuleList returns empty string."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(64, 64)
            self.linear2 = nn.Linear(64, 64)

    model = SimpleModel()
    result = infer_decoder_layers_path(model)
    assert result == "", f"Expected '', got '{result}'"


def test_single_modulelist():
    """Test model with exactly one ModuleList."""

    class SingleListModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(64, 64) for _ in range(3)])

    model = SingleListModel()
    result = infer_decoder_layers_path(model)
    assert result == "blocks", f"Expected 'blocks', got '{result}'"


def test_multiple_independent_modulelists():
    """Test model with multiple independent ModuleLists at same depth."""

    class MultiListModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_layers = nn.ModuleList([SimpleBlock() for _ in range(4)])
            self.decoder_layers = nn.ModuleList([SimpleBlock() for _ in range(4)])

    model = MultiListModel()
    result = infer_decoder_layers_path(model)
    # Both are at same depth, should pick first alphabetically
    assert result == "decoder_layers", f"Expected 'decoder_layers', got '{result}'"


def test_different_depth_modulelists():
    """Test that shallower ModuleList is preferred over deeper one."""

    class InnerModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.nested_layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(2)])

    class OuterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([SimpleBlock() for _ in range(4)])
            self.inner = InnerModule()

    model = OuterModel()
    result = infer_decoder_layers_path(model)
    # 'layers' is shallower (1 level) vs 'inner.nested_layers' (2 levels)
    assert result == "layers", f"Expected 'layers', got '{result}'"


# =============================================================================
# Test: Real-world-like architectures
# =============================================================================
def test_mixtral_like_moe_llm():
    """Test Mixtral/DeepSeek-like MoE LLM structure with realistic components."""

    class RMSNorm(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.weight

    class Attention(nn.Module):
        def __init__(self, dim: int = 256, num_heads: int = 4):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.o_proj = nn.Linear(dim, dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.o_proj(x)

    class Expert(nn.Module):
        """Single expert in MoE layer."""

        def __init__(self, dim: int = 256, intermediate_dim: int = 512):
            super().__init__()
            self.gate_proj = nn.Linear(dim, intermediate_dim, bias=False)
            self.up_proj = nn.Linear(dim, intermediate_dim, bias=False)
            self.down_proj = nn.Linear(intermediate_dim, dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    class MoEMLP(nn.Module):
        """Mixture of Experts MLP layer."""

        def __init__(self, dim: int = 256, num_experts: int = 8, num_experts_per_tok: int = 2):
            super().__init__()
            self.num_experts = num_experts
            self.num_experts_per_tok = num_experts_per_tok
            self.gate = nn.Linear(dim, num_experts, bias=False)
            self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Simplified: just use first expert
            return self.experts[0](x)

    class MoEDecoderLayer(nn.Module):
        """Single decoder layer with attention and MoE MLP."""

        def __init__(self, dim: int = 256, num_heads: int = 4, num_experts: int = 8):
            super().__init__()
            self.input_layernorm = RMSNorm(dim)
            self.self_attn = Attention(dim, num_heads)
            self.post_attention_layernorm = RMSNorm(dim)
            self.mlp = MoEMLP(dim, num_experts)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = x + self.self_attn(self.input_layernorm(x))
            out = h + self.mlp(self.post_attention_layernorm(h))
            return out

    class MoEModel(nn.Module):
        """Inner model with embedding and layers."""

        def __init__(self, vocab_size: int = 32000, dim: int = 256, num_layers: int = 4, num_experts: int = 8):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, dim)
            self.layers = nn.ModuleList([MoEDecoderLayer(dim, num_experts=num_experts) for _ in range(num_layers)])
            self.norm = RMSNorm(dim)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            h = self.embed_tokens(input_ids)
            for layer in self.layers:
                h = layer(h)
            return self.norm(h)

    class MixtralForCausalLM(nn.Module):
        """Full Mixtral-like model for causal LM."""

        def __init__(self, vocab_size: int = 32000, dim: int = 256, num_layers: int = 4, num_experts: int = 8):
            super().__init__()
            self.model = MoEModel(vocab_size, dim, num_layers, num_experts)
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            h = self.model(input_ids)
            return self.lm_head(h)

    # Create the model
    model = MixtralForCausalLM(vocab_size=1000, dim=64, num_layers=4, num_experts=8)

    # Test inference
    result = infer_decoder_layers_path(model)

    # Should return 'model.layers', NOT any of the nested expert paths like:
    # - 'model.layers.0.mlp.experts'
    # - 'model.layers.1.mlp.experts'
    # etc.
    assert result == "model.layers", f"Expected 'model.layers', got '{result}'"

    # Verify the model structure has the expected nested ModuleLists
    modulelist_paths = [name for name, module in model.named_modules() if isinstance(module, nn.ModuleList)]
    assert "model.layers" in modulelist_paths
    assert any("experts" in path for path in modulelist_paths), "Model should have expert ModuleLists"

    # Count expert ModuleLists (should be num_layers * 1 = 4 expert lists)
    expert_paths = [p for p in modulelist_paths if "experts" in p]
    assert len(expert_paths) == 4, f"Expected 4 expert ModuleLists, got {len(expert_paths)}"


def test_gpt_like_architecture():
    """Test GPT-like architecture with transformer.h.* structure."""

    class GPTBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_1 = nn.LayerNorm(64)
            self.attn = nn.Linear(64, 64)
            self.ln_2 = nn.LayerNorm(64)
            self.mlp = nn.Linear(64, 64)

    class GPTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(1000, 64)
            self.h = nn.ModuleList([GPTBlock() for _ in range(4)])
            self.ln_f = nn.LayerNorm(64)

    class GPTLMHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = GPTModel()
            self.lm_head = nn.Linear(64, 1000)

    model = GPTLMHead()
    result = infer_decoder_layers_path(model)
    assert result == "transformer.h", f"Expected 'transformer.h', got '{result}'"


def test_encoder_decoder_architecture():
    """Test encoder-decoder model picks encoder (alphabetically first at same depth)."""

    class EncoderDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.ModuleDict({"layers": nn.ModuleList([SimpleBlock() for _ in range(4)])})
            self.decoder = nn.ModuleDict({"layers": nn.ModuleList([SimpleBlock() for _ in range(4)])})

    model = EncoderDecoder()
    result = infer_decoder_layers_path(model)
    # Both 'encoder.layers' and 'decoder.layers' are at depth 2
    # 'decoder.layers' comes first alphabetically
    assert result == "decoder.layers", f"Expected 'decoder.layers', got '{result}'"
