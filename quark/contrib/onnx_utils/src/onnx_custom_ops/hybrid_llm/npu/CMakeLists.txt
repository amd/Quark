# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

set(target onnx_custom_ops_npu)
find_package(Eigen3 REQUIRED)

file(
  GLOB_RECURSE SRC
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
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

if(MSVC)
  target_compile_options(
    ${target}
    PRIVATE /we4456 /we4457 /we4458 /we4459 /WX
  )
endif()

target_compile_definitions(${target} PRIVATE DD_USE_AVX512)
if(DD_ENABLE_EAGER_PROFILING)
  target_compile_definitions(${target} PRIVATE EAGER_PROFILING)
endif()

target_include_directories(
  ${target}
  PRIVATE
    $<TARGET_PROPERTY:${PROJECT_NAME},INTERFACE_INCLUDE_DIRECTORIES>
    ${HYBRID_LLM_INCLUDE_DIRS}
)
target_link_libraries(
  ${target}
  PUBLIC
    DynamicDispatch::dyn_dispatch_core
    Eigen3::Eigen
    onnxruntime::onnxruntime
    external_data
  PRIVATE ryzen_mm::ryzen_mm XRT::xrt_coreutil
)
if(MSVC)
  target_compile_options(${target} PUBLIC "/Zc:__cplusplus")
endif()

target_compile_definitions(${target} PRIVATE ORT_API_MANUAL_INIT)
