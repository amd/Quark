# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

include(FetchContent)

include(protobuf_utils)
ryzenai_onnx_utils_resolve_protobuf()

find_package(XRT REQUIRED)
find_package(DynamicDispatch CONFIG QUIET)

if(DynamicDispatch_FOUND)
  message(STATUS "Using pre-built DynamicDispatch")
  return()
endif()

if(NOT DEFINED DYNAMIC_DISPATCH_SRC)
  message(STATUS "Using DynamicDispatch from FetchContent")
  FetchContent_Declare(
    DynamicDispatch
    # Do not commit forks of DD to main
    GIT_REPOSITORY
      "https://gitenterprise.xilinx.com/VitisAI/DynamicDispatch.git"
    # current latest tag of main
    GIT_TAG "87a1abaaa65e5466a0eeb56c2aaf22d80be1ef86"
    SYSTEM
  )
  set(
    DYNAMIC_DISPATCH_SRC
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/dynamicdispatch-src
  )
else()
  cmake_path(SET DYNAMIC_DISPATCH_SRC ${DYNAMIC_DISPATCH_SRC})
  message(STATUS "Using DynamicDispatch from local: " ${DYNAMIC_DISPATCH_SRC})
  FetchContent_Declare(
    DynamicDispatch
    SOURCE_DIR
    ${DYNAMIC_DISPATCH_SRC}
    SYSTEM
  )
endif()

set(DD_USE_AVX512 ON)
# if we found DD via find_package above, then assume it's built as shared
# otherwise, configure this based on the CMake option configured.
if(NOT DynamicDispatch_FOUND)
  if(ONNX_UTILS_USE_STATIC_DD)
    set(BUILD_SHARED_LIBS OFF)
  else()
    set(BUILD_SHARED_LIBS ON)
  endif()
else()
  set(ONNX_UTILS_USE_STATIC_DD OFF)
endif()

FetchContent_MakeAvailable(DynamicDispatch)

if(NOT DynamicDispatch_FOUND)
  message(STATUS "Building DynamicDispatch from source")
  set(DynamicDispatch_FOUND TRUE)

  set_property(
    TARGET dyn_dispatch_core
    APPEND
    PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES
        ${DYNAMIC_DISPATCH_SRC}/src
        ${DYNAMIC_DISPATCH_SRC}/include
        ${DYNAMIC_DISPATCH_SRC}/xclbin/include
        ${XRT_INCLUDE_DIRS}
  )

  # DD uses VS props for static code analysis but this propagates these rules
  # to Protobuf/Abseil if they're being compiled statically with the build.
  # This option clears the DD static checks so they don't apply here
  set_target_properties(dyn_dispatch_core PROPERTIES VS_USER_PROPS "")
  # set(DYNAMIC_DISPATCH_LIBS dyn_dispatch_core)
  add_library(DynamicDispatch::dyn_dispatch_core ALIAS dyn_dispatch_core)
else()
  message(STATUS "Using existing DynamicDispatch installation")
  set_property(
    TARGET DynamicDispatch::dyn_dispatch_core
    APPEND
    PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES
        ${DYNAMIC_DISPATCH_SRC}/src
        ${DYNAMIC_DISPATCH_SRC}/include
        ${DYNAMIC_DISPATCH_SRC}/xclbin/include
        ${XRT_INCLUDE_DIRS}
  )
  set_target_properties(
    DynamicDispatch::dyn_dispatch_core
    PROPERTIES VS_USER_PROPS ""
  )
endif()
