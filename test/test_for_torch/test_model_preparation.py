"""
Simple test for model loading using get_model() function.
"""

import os
import sys

import pytest
import torch

# Add the examples directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../examples/torch/language_modeling")))

from quark.torch.utils.llm import get_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_model_qwen35_0_8b():
    """Test loading Qwen/Qwen3.5-0.8B model using get_model()."""
    model_id = "Qwen/Qwen3.5-0.8B"

    # Load model with default parameters
    model, model_dtype = get_model(
        ckpt_path=model_id,
        data_type="auto",
        device="cuda",
        multi_gpu=False,
        multi_device=False,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    # Verify model is loaded correctly
    assert model is not None, "Model should not be None"
    assert isinstance(model, torch.nn.Module), "Model should be a torch.nn.Module"

    # Verify model is in eval mode
    assert not model.training, "Model should be in eval mode"

    # Verify model dtype
    assert model_dtype in [torch.float16, torch.bfloat16, torch.float32], f"Unexpected dtype: {model_dtype}"

    # Verify model config
    assert hasattr(model, "config"), "Model should have a config attribute"
    assert model.config._name_or_path == model_id, "Model config should have correct _name_or_path"

    # Verify model has parameters
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0, "Model should have parameters"

    print(f"✓ Successfully loaded {model_id}")
    print(f"  - Model type: {model.config.model_type}")
    print(f"  - Dtype: {model_dtype}")
    print(f"  - Parameters: {num_params:,}")
    print(f"  - Device: {next(model.parameters()).device}")


if __name__ == "__main__":
    print("Running Qwen/Qwen3.5-0.8B model loading tests...\n")

    if torch.cuda.is_available():
        try:
            test_get_model_qwen35_0_8b()
            print("\n✓ All tests passed!")
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            raise
    else:
        print("⚠ CUDA not available, skipping tests")
