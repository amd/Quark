#!/bin/bash

# ============================================================================
# xDiT Inference with AMD Quark Quantization - Usage Examples
# ============================================================================

# This file contains example commands for running xDiT inference with
# AMD Quark quantization support using the xdit_quark_inference.py script.

# ============================================================================
# Basic Examples
# ============================================================================

# Example 1: Single GPU inference with FP8 quantization
echo "Example 1: Single GPU with FP8 quantization"
python xdit_quark_inference.py \
    --model "black-forest-labs/FLUX.1-dev" \
    --prompt "A beautiful sunset over mountains" \
    --height 1024 \
    --width 1024 \
    --num_inference_steps 50 \
    --use_quark_quantize \
    --quark_quantization_mode fp8 \
    --output_directory "./outputs"

# Example 2: Single GPU inference with MXFP4 quantization
echo "Example 2: Single GPU with MXFP4 quantization"
python xdit_quark_inference.py \
    --model "black-forest-labs/FLUX.1-dev" \
    --prompt "A futuristic city at night" \
    --height 1024 \
    --width 1024 \
    --num_inference_steps 50 \
    --use_quark_quantize \
    --quark_quantization_mode mxfp4 \
    --output_directory "./outputs"

# ============================================================================
# Distributed Execution with torchrun
# ============================================================================

# Example 3: 2 GPUs with FP8 quantization (sequence parallel)
echo "Example 3: 2 GPUs with sequence parallel and FP8 quantization"
torchrun --nproc_per_node 2 xdit_quark_inference.py \
    --model "black-forest-labs/FLUX.1-dev" \
    --prompt "A serene lake surrounded by forests" \
    --height 1024 \
    --width 1024 \
    --num_inference_steps 50 \
    --ulysses_degree 2 \
    --use_quark_quantize \
    --quark_quantization_mode fp8 \
    --output_directory "./outputs"

# Example 4: 4 GPUs with sequence parallel (Ulysses 2x Ring 2x)
echo "Example 4: 4 GPUs with Ulysses and Ring sequence parallel"
torchrun --nproc_per_node 4 xdit_quark_inference.py \
    --model "black-forest-labs/FLUX.1-dev" \
    --prompt "An astronaut riding a horse on Mars" \
    --height 1024 \
    --width 1024 \
    --num_inference_steps 50 \
    --ulysses_degree 2 \
    --ring_degree 2 \
    --use_quark_quantize \
    --quark_quantization_mode fp8 \
    --output_directory "./outputs"

# Example 5: 8 GPUs with advanced parallel configuration
echo "Example 5: 8 GPUs with advanced parallel configuration"
torchrun --nproc_per_node 8 xdit_quark_inference.py \
    --model "black-forest-labs/FLUX.1-dev" \
    --prompt "A magical forest with glowing mushrooms" \
    --height 2048 \
    --width 2048 \
    --num_inference_steps 50 \
    --ulysses_degree 4 \
    --ring_degree 2 \
    --use_quark_quantize \
    --quark_quantization_mode fp8 \
    --output_directory "./outputs"

# ============================================================================
# PipeFusion Parallel Examples
# ============================================================================

# Example 6: 4 GPUs with PipeFusion parallel
echo "Example 6: 4 GPUs with PipeFusion parallel"
torchrun --nproc_per_node 4 xdit_quark_inference.py \
    --model "black-forest-labs/FLUX.1-dev" \
    --prompt "A detailed portrait of a wise old wizard" \
    --height 1024 \
    --width 1024 \
    --num_inference_steps 50 \
    --pipefusion_parallel_degree 2 \
    --ulysses_degree 2 \
    --use_quark_quantize \
    --output_directory "./outputs"

# ============================================================================
# Batch Processing with Prompt Files
# ============================================================================

# Example 7: Batch processing from JSON file
# First create a prompts.json file with this format:
# [
#   {"id": 1, "prompt": "A cat sitting on a windowsill"},
#   {"id": 2, "prompt": "A dog playing in a park"},
#   {"id": 3, "prompt": "A bird flying over the ocean"}
# ]

echo "Example 7: Batch processing from JSON file"
cat > prompts.json << 'EOF'
[
  {"id": 1, "prompt": "A majestic lion in the savanna"},
  {"id": 2, "prompt": "A colorful parrot in the rainforest"},
  {"id": 3, "prompt": "A dolphin jumping out of the water"}
]
EOF

torchrun --nproc_per_node 2 xdit_quark_inference.py \
    --model "black-forest-labs/FLUX.1-dev" \
    --prompt_file prompts.json \
    --height 1024 \
    --width 1024 \
    --num_inference_steps 50 \
    --ulysses_degree 2 \
    --use_quark_quantize \
    --output_directory "./outputs"

# ============================================================================
# Without Quantization (Baseline)
# ============================================================================

# Example 8: 2 GPUs without quantization (for comparison)
echo "Example 8: 2 GPUs without quantization"
torchrun --nproc_per_node 2 xdit_quark_inference.py \
    --model "black-forest-labs/FLUX.1-dev" \
    --prompt "A comparison test image" \
    --height 1024 \
    --width 1024 \
    --num_inference_steps 50 \
    --ulysses_degree 2 \
    --output_directory "./outputs"

# ============================================================================
# Performance Profiling
# ============================================================================

# Example 9: Run with profiling enabled
echo "Example 9: Run with profiling"
torchrun --nproc_per_node 2 xdit_quark_inference.py \
    --model "black-forest-labs/FLUX.1-dev" \
    --prompt "A profiling test image" \
    --height 1024 \
    --width 1024 \
    --num_inference_steps 50 \
    --ulysses_degree 2 \
    --use_quark_quantize \
    --profile \
    --profile_wait 1 \
    --profile_warmup 1 \
    --profile_active 3 \
    --output_directory "./outputs"

# ============================================================================
# FLUX.2-dev Model Examples
# ============================================================================

# Example 10: FLUX.2-dev with 4 GPUs
echo "Example 10: FLUX.2-dev with 4 GPUs and quantization"
torchrun --nproc_per_node 4 xdit_quark_inference.py \
    --model "black-forest-labs/FLUX.2-dev" \
    --prompt "A cyberpunk street scene at night" \
    --height 1024 \
    --width 1024 \
    --num_inference_steps 28 \
    --ulysses_degree 2 \
    --ring_degree 2 \
    --use_quark_quantize \
    --quark_quantization_mode fp8 \
    --output_directory "./outputs"

# ============================================================================
# Additional Configuration Options
# ============================================================================

# Example 11: With VAE tiling for memory efficiency
echo "Example 11: With VAE tiling and slicing"
torchrun --nproc_per_node 2 xdit_quark_inference.py \
    --model "black-forest-labs/FLUX.1-dev" \
    --prompt "A high resolution landscape" \
    --height 2048 \
    --width 2048 \
    --num_inference_steps 50 \
    --ulysses_degree 2 \
    --use_quark_quantize \
    --enable_tiling \
    --enable_slicing \
    --output_directory "./outputs"

# Example 12: With custom seed for reproducibility
echo "Example 12: With custom seed"
torchrun --nproc_per_node 2 xdit_quark_inference.py \
    --model "black-forest-labs/FLUX.1-dev" \
    --prompt "A reproducible test image" \
    --height 1024 \
    --width 1024 \
    --num_inference_steps 50 \
    --seed 12345 \
    --ulysses_degree 2 \
    --use_quark_quantize \
    --output_directory "./outputs"

# ============================================================================
# Notes:
# ============================================================================
# - Ensure AMD Quark is installed: pip install amd-quark
# - Adjust --nproc_per_node based on your available GPUs
# - Parallel degree products should equal number of GPUs:
#   (ulysses_degree × ring_degree × pipefusion_parallel_degree = nproc_per_node)
# - FP8 quantization typically offers better speed with minimal quality loss
# - MXFP4 offers higher compression but may affect quality more
# - Use --profile for performance analysis
# ============================================================================
