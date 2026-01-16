# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

include(protobuf_utils)
ryzenai_onnx_utils_resolve_protobuf()

set(target onnx_custom_ops)
set(ep_target onnxruntime_providers_ryzenai)
configure_file(
  custom_ops_options.in.hpp
  ryzenai/onnx_utils/custom_ops_options.hpp
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/ryzenai/onnx_utils/custom_ops_options.hpp
  DESTINATION include/ryzenai/onnx_utils
)
set(CUSTOM_OPS_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

set(LINK_LIBS "")

find_package(XRT REQUIRED)

if(ONNX_UTILS_ENABLE_PROJECT_DD)
  add_subdirectory(dynamic_dispatch)
  list(APPEND LINK_LIBS onnx_custom_ops_dynamic_dispatch)
endif()

if(ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM)
  add_subdirectory(hybrid_llm)
  list(APPEND LINK_LIBS onnx_custom_ops_hybrid_llm)
endif()

if(ONNX_UTILS_ENABLE_CORELIB)
  add_subdirectory(corelib)
  list(APPEND LINK_LIBS onnx_custom_ops_corelib)
endif()

set(OBJ_LIBS "")
foreach(LIB ${LINK_LIBS})
  list(APPEND OBJ_LIBS $<TARGET_OBJECTS:${LIB}>)
endforeach()

# Used to suppress error C1189 (The C++ Standard Library forbids macroizing the
# keyword "bool".)
# TODO(chiz): This seems to be related to DD main branch only. We need to fix it
# in the DD
add_compile_definitions(${target} PRIVATE _XKEYCHECK_H)

add_library(
  ${target}
  SHARED
  custom_ops.cpp
  lora.cpp
  external_data.cpp
  execution_provider_fake.cpp
  "string.cpp"
  ${OBJ_LIBS}
)
set(target_libs ${target})

target_compile_definitions(${target} PRIVATE ORT_API_MANUAL_INIT)

if(ONNX_UTILS_USE_RYZENAI_EP)
  add_library(
    ${ep_target}
    SHARED
    custom_ops.cpp
    lora.cpp
    external_data.cpp
    execution_provider.cpp
    execution_provider_cpugate.cpp
    "string.cpp"
    ${OBJ_LIBS}
  )
  list(APPEND target_libs ${ep_target})

  target_compile_definitions(${ep_target} PRIVATE ORT_API_MANUAL_INIT)
endif()

ryzenai_onnx_utils_add_protobuf_target(external_data external_data.proto)

foreach(target_lib ${target_libs})
  target_link_libraries(
    ${target_lib}
    PRIVATE
      ${LINK_LIBS}
      onnxruntime::onnxruntime
      ryzen_mm::ryzen_mm
      external_data
      DynamicDispatch::dyn_dispatch_core
  )

  target_include_directories(
    ${target_lib}
    PRIVATE $<TARGET_PROPERTY:${PROJECT_NAME},INTERFACE_INCLUDE_DIRECTORIES>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  )

  target_compile_options(
    ${target_lib}
    PRIVATE
      ${ONNX_UTILS_DEFAULT_COMPILE_OPTIONS}
      ${ONNX_UTILS_USER_COMPILE_OPTIONS}
  )
  # because the custom ops now use the custom allocator, it brings in these
  # dependencies into the custom op DLL. To avoid linking against it at runtime,
  # include those files as souces for the custom op library as well
  if(ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY)
    target_sources(${target_lib} PRIVATE "custom_allocator.cpp")
  endif()
  install(TARGETS ${target_lib} EXPORT RyzenAiOnnxUtilsConfig)
endforeach()

if(ONNX_UTILS_USE_RYZENAI_EP)
  target_link_libraries(
    ${ep_target}
    PRIVATE onnxruntime::onnxruntime_providers_shared
  )

  target_include_directories(
    ${ep_target}
    PRIVATE
      $<TARGET_PROPERTY:SafeInt::SafeInt,INTERFACE_INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:Microsoft::GSL,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif()

target_sources(${PROJECT_NAME} PRIVATE "string.cpp")
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:./include>
)
if(ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY)
  target_sources(${PROJECT_NAME} PRIVATE "custom_allocator.cpp")
  # onnxruntime is private because including this project to build OGA fails
  # since this target doesn't exist yet when the link happens
  target_link_libraries(${PROJECT_NAME} PRIVATE onnxruntime::onnxruntime)
endif()
