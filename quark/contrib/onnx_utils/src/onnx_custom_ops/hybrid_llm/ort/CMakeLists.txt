# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

set(target onnx_custom_ops_hybrid_llm_ort)
set(
  srcs
  "ort_operator.cpp"
  "cast.cpp"
  "concat.cpp"
  "conv.cpp"
  "gqa.cpp"
  "matmul.cpp"
  "rotary_embedding.cpp"
  "simplified_layer_norm.cpp"
  "skip_simplified_layer_norm.cpp"
  "slice.cpp"
  "split.cpp"
  "transpose.cpp"
)

source_group(
  TREE "${CMAKE_CURRENT_SOURCE_DIR}"
  PREFIX "Source Files"
  FILES ${srcs}
)

add_library(${target} INTERFACE)
target_sources(${target} INTERFACE ${srcs})
