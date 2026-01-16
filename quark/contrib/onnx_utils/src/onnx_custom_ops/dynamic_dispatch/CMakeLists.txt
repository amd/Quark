# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

set(target onnx_custom_ops_dynamic_dispatch)

add_library(
  ${target}
  OBJECT
  dynamic_dispatch.cpp
  dynamic_pad.cpp
  external_buffers.cpp
  main.cpp
  onnx.cpp
  pad_input_ids.cpp
)
target_include_directories(
  ${target}
  PRIVATE
    ${CUSTOM_OPS_INCLUDE_DIRS}
    $<TARGET_PROPERTY:${PROJECT_NAME},INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(
  ${target}
  PUBLIC DynamicDispatch::dyn_dispatch_core onnxruntime::onnxruntime
  PRIVATE ryzen_mm::ryzen_mm XRT::xrt_coreutil
)
target_compile_options(
  ${target}
  PRIVATE
    ${ONNX_UTILS_DEFAULT_COMPILE_OPTIONS}
    ${ONNX_UTILS_USER_COMPILE_OPTIONS}
)
target_compile_definitions(${target} PRIVATE DD_USE_AVX512)
if(DD_ENABLE_FUSION_PROFILING)
  target_compile_definitions(${target} PRIVATE FUSION_PROFILING)
endif()

if(MSVC)
  target_compile_options(${target} PUBLIC "/Zc:__cplusplus")
endif()

target_compile_definitions(${target} PRIVATE ORT_API_MANUAL_INIT)
