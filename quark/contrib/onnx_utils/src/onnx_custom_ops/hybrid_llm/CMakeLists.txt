# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

set(target onnx_custom_ops_hybrid_llm)

include(protobuf_utils)
ryzenai_onnx_utils_resolve_protobuf()

list(
  APPEND
  HYBRID_LLM_INCLUDE_DIRS
  ${CUSTOM_OPS_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
)

set(LINK_LIBS "")

add_subdirectory(operators)
add_subdirectory(ort)
list(
  APPEND
  LINK_LIBS
  onnx_custom_ops_hybrid_llm_operators
  onnx_custom_ops_hybrid_llm_ort
)

add_library(${target} OBJECT main.cpp)
target_compile_options(
  ${target}
  PRIVATE
    ${ONNX_UTILS_DEFAULT_COMPILE_OPTIONS}
    ${ONNX_UTILS_USER_COMPILE_OPTIONS}
)

if(ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU)
  add_subdirectory(gpu)
  list(APPEND LINK_LIBS onnx_custom_ops_dml_gpu)
endif()

if(ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU)
  add_subdirectory(npu)
  list(APPEND LINK_LIBS onnx_custom_ops_npu)
endif()

if(ONNX_UTILS_ENABLE_CORELIB)
  list(APPEND LINK_LIBS ryzenai_corelib::corelib)
endif()

if(ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU)
  target_compile_definitions(${target} PRIVATE DML_TARGET_VERSION_USE_LATEST)
  target_include_directories(
    ${target}
    PRIVATE
      ${PROJECT_SOURCE_DIR}/external/packages/Microsoft.AI.DirectML.1.15.4/include
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/src/onnx_custom_ops/shared_memory
  )
endif()

target_include_directories(
  ${target}
  PRIVATE
    ${HYBRID_LLM_INCLUDE_DIRS}
    $<TARGET_PROPERTY:${PROJECT_NAME},INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(
  ${target}
  PRIVATE ${LINK_LIBS} onnxruntime::onnxruntime external_data ryzen_mm::ryzen_mm
)

target_compile_definitions(${target} PRIVATE ORT_API_MANUAL_INIT)
