# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

find_package(Dml REQUIRED)

set(target onnx_custom_ops_dml_gpu)
file(
  GLOB_RECURSE SRC
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
)
source_group(
  TREE "${CMAKE_CURRENT_SOURCE_DIR}"
  PREFIX "Source Files"
  FILES ${SRC}
)

add_library(${target} STATIC ${SRC} $<TARGET_OBJECTS:external_data>)
target_compile_options(
  ${target}
  PRIVATE
    ${ONNX_UTILS_DEFAULT_COMPILE_OPTIONS}
    ${ONNX_UTILS_USER_COMPILE_OPTIONS}
)

target_include_directories(
  ${target}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/onnx_custom_ops
    ${HYBRID_LLM_INCLUDE_DIRS}
)
target_link_libraries(
  ${target}
  PUBLIC dml external_data
  PRIVATE ryzen_mm::ryzen_mm
)
