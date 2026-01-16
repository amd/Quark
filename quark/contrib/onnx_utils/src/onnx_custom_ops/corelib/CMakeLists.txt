# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

set(target onnx_custom_ops_corelib)

include(protobuf_utils)

ryzenai_onnx_utils_resolve_protobuf()

add_library(${target} OBJECT corelib.cpp)

find_package(ryzenai_corelib)

target_include_directories(
  ${target}
  PRIVATE
    ${CUSTOM_OPS_INCLUDE_DIRS}
    $<TARGET_PROPERTY:${PROJECT_NAME},INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(
  ${target}
  PUBLIC ryzenai_corelib::corelib onnxruntime::onnxruntime external_data
)

target_compile_options(
  ${target}
  PRIVATE
    ${ONNX_UTILS_DEFAULT_COMPILE_OPTIONS}
    ${ONNX_UTILS_USER_COMPILE_OPTIONS}
)

if(MSVC)
  target_compile_options(${target} PUBLIC "/Zc:__cplusplus")
endif()
