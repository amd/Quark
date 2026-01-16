# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

set(target onnx_custom_ops_hybrid_llm_operators)
set(
  srcs
  "cast.cpp"
  "conv_split_mul.cpp"
  "conv.cpp"
  "gather.cpp"
  "gqo.cpp"
  "hybrid_kernel.cpp"
  "lut.cpp"
  "matmul.cpp"
  "matmulnbits.cpp"
  "qmoe.cpp"
  "reducesum.cpp"
  "rope.cpp"
  "slrn.cpp"
  "ssgmlp.cpp"
  "sslrn.cpp"
  "ssmlp.cpp"
  "sub.cpp"
)

if(ONNX_UTILS_ENABLE_CORELIB)
  list(APPEND srcs "corelib_kernel.cpp")
endif()

source_group(
  TREE "${CMAKE_CURRENT_SOURCE_DIR}"
  PREFIX "Source Files"
  FILES ${srcs}
)

add_library(${target} INTERFACE)
target_sources(${target} INTERFACE ${srcs})
