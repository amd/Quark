# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.28)
project(ryzenai_onnx_utils VERSION 0.5.0)

if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
  set(CMAKE_C_STANDARD 99)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED True)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(CMAKE_CXX_EXTENSION OFF)
  # if using before DD #2620, turn on to export all symbols from DD
  # set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

option(ONNX_UTILS_ENABLE_PROJECT_DD "Enable DynamicDispatch custom ops" OFF)
option(ONNX_UTILS_ENABLE_CORELIB "Enable CoreLib custom ops" OFF)
option(
  ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM
  "Enable hybrid GPU/NPU custom ops"
  OFF
)
option(
  ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_CPU
  "Enable the CPU implementations for hybrid GPU/NPU custom ops"
  OFF
)
option(
  ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_GPU
  "Enable the GPU implementations for hybrid GPU/NPU custom ops"
  OFF
)
option(
  ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU
  "Enable the NPU implementations for hybrid GPU/NPU custom ops"
  OFF
)
option(
  ONNX_UTILS_ENABLE_PROJECT_SHARED_MEMORY
  "Enable the implementation for shared memory"
  OFF
)
option(ONNX_UTILS_BUILD_EXAMPLES "Build examples" OFF)
option(ONNX_UTILS_BUILD_TESTS "Build tests" OFF)
option(ONNX_UTILS_USE_STATIC_PROTOBUF "Use static protobuf" ON)
option(ONNX_UTILS_USE_STATIC_DD "Use static DynamicDispatch" ON)
option(ONNX_UTILS_USE_RYZENAI_EP "Use RyzenAI EP" OFF)
option(
  ONNX_UTILS_INSTALL_PROTOBUF
  "Install protobuf files (only valid if building Protobuf from source)"
  OFF
)
option(DD_ENABLE_FUSION_PROFILING "Enable DD fusion mode profiling" OFF)
option(DD_ENABLE_EAGER_PROFILING "Enable DD eager mode profiling" OFF)

if(ONNX_UTILS_ENABLE_PROJECT_DD OR ONNX_UTILS_ENABLE_PROJECT_HYBRID_LLM_NPU)
  set(ONNX_UTILS_ENABLE_DD ON)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
    message(
      FATAL_ERROR
      "GCC 14 or newer is required. Found: ${CMAKE_CXX_COMPILER_VERSION}. Try adding: -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14"
    )
  endif()
endif()

if(WIN32)
  set(
    ONNX_UTILS_DEFAULT_COMPILE_OPTIONS
    "/guard:cf"
    CACHE STRING
    "Flags to add to all compilation targets by default"
  )
else()
  set(
    ONNX_UTILS_DEFAULT_COMPILE_OPTIONS
    -fPIC
    -march=native
    -mavx512f
    -mavx512fp16
    -Wpsabi
    -flax-vector-conversions
    CACHE STRING
    "Flags to add to all compilation targets by default"
  )
endif()

set(
  ONNX_UTILS_USER_COMPILE_OPTIONS
  ""
  CACHE STRING
  "Flags to add to all compilation targets from the user"
)

if(NOT DEFINED SKBUILD)
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
  # disable clang-tidy checks on any projects brought in with FetchContent
  set(empty_clang_tidy_file "Checks: -*")
  file(
    WRITE
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/.clang-tidy
    ${empty_clang_tidy_file}
  )

  find_package(onnxruntime REQUIRED)
  find_package(ryzen_mm REQUIRED)

  if(NOT DEFINED ONNX_UTILS_EP)
    set(ONNX_UTILS_EP "CPUExecutionProvider")
  endif()

  if(ONNX_UTILS_USE_RYZENAI_EP AND NOT DEFINED ONNX_UTILS_RYZENAI_EP_NAME)
    set(ONNX_UTILS_RYZENAI_EP_NAME "RyzenAILightExecutionProvider")
  endif()

  if(ONNX_UTILS_ENABLE_DD)
    message(STATUS "Using Dynamic Dispatch")
    find_package(DynamicDispatch REQUIRED)
  endif()

  add_library(${PROJECT_NAME} SHARED)
  target_link_libraries(${PROJECT_NAME} PRIVATE ryzen_mm::ryzen_mm)
  target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:./include>
  )
  target_compile_options(
    ${PROJECT_NAME}
    PRIVATE
      ${ONNX_UTILS_DEFAULT_COMPILE_OPTIONS}
      ${ONNX_UTILS_USER_COMPILE_OPTIONS}
  )
  set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON
  )

  install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets)

  # DD doesn't install its DLL if it's not the top-level project
  if(ONNX_UTILS_ENABLE_DD)
    if(WIN32)
      set(SHARED_LIB_PATH bin)
    else()
      set(SHARED_LIB_PATH lib)
    endif()
    if(NOT ONNX_UTILS_USE_STATIC_DD)
      install(
        FILES $<TARGET_FILE:DynamicDispatch::dyn_dispatch_core>
        DESTINATION ${SHARED_LIB_PATH}
      )
    endif()
    # even without building as a shared lib, if ZLIB is in the env, it may be
    # dynamic, which adds a dynamic dependency from DD to the custom ops lib.
    if(WIN32)
      # with zlib in conda on windows, ZLIB::ZLIB doesn't exist as a target here
      # assume it's dynamic in this case
      if(NOT TARGET ZLIB::ZLIB)
        set(ZLIB_TYPE "")
      else()
        get_target_property(ZLIB_TYPE ZLIB::ZLIB TYPE)
      endif()
      if(NOT ZLIB_TYPE STREQUAL "STATIC_LIBRARY")
        find_file(ZLIB_DLL NAMES zlib.dll REQUIRED)
        install(FILES "${ZLIB_DLL}" DESTINATION ${SHARED_LIB_PATH})
      endif()
    endif()
  endif()

  export(
    TARGETS ${target}
    NAMESPACE ryzenai_onnx_utils::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/RyzenAiOnnxUtilsConfig.cmake"
  )
  install(
    EXPORT ${PROJECT_NAME}-targets
    NAMESPACE ${PROJECT_NAME}::
    COMPONENT runtime
    DESTINATION lib/cmake/${PROJECT_NAME}
  )
  include(CMakePackageConfigHelpers)
  configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/config.in.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
    @ONLY
  )
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
  )
  install(
    FILES
      ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
    COMPONENT runtime
    DESTINATION lib/cmake/${PROJECT_NAME}
  )
  install(DIRECTORY include/ DESTINATION include)

  add_subdirectory(src/onnx_custom_ops)

  if(ONNX_UTILS_BUILD_EXAMPLES)
    add_subdirectory(examples)
  endif()
  if(ONNX_UTILS_BUILD_TESTS)
    add_subdirectory(tests)
  endif()
endif()
