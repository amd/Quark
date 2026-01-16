# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

function(ryzenai_onnx_utils_add_protobuf_target target file)
  add_library(${target} OBJECT ${file})
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

  protobuf_generate(
    TARGET ${target}
    LANGUAGE cpp
    "$<LIST:TRANSFORM,$<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
  )
  target_include_directories(
    ${target}
    SYSTEM
    PRIVATE
      $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES>
  )
  target_link_libraries(${target} INTERFACE protobuf::libprotobuf)
  target_compile_options(
    ${target}
    PRIVATE
      ${ONNX_UTILS_DEFAULT_COMPILE_OPTIONS}
      ${ONNX_UTILS_USER_COMPILE_OPTIONS}
  )
endfunction()

function(ryzenai_onnx_utils_resolve_protobuf)
  if(ONNX_UTILS_USE_STATIC_PROTOBUF)
    set(Protobuf_USE_STATIC_LIBS ON)
  endif()

  find_package(Protobuf CONFIG)
  if(NOT Protobuf_FOUND)
    message(STATUS "Using Protobuf from FetchContent")
    include(FetchContent)
    FetchContent_Declare(
      protobuf
      GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
      GIT_TAG v6.31.1
      OVERRIDE_FIND_PACKAGE
      SYSTEM
    )
    if(ONNX_UTILS_USE_STATIC_PROTOBUF)
      message(STATUS "Build Static Protobuf")
      set(ABSL_MSVC_STATIC_RUNTIME OFF)
      set(ABSL_ENABLE_INSTALL ${ONNX_UTILS_INSTALL_PROTOBUF})
      set(utf8_range_ENABLE_TESTS OFF)
      set(utf8_range_ENABLE_INSTALL ${ONNX_UTILS_INSTALL_PROTOBUF})
      set(protobuf_MSVC_STATIC_RUNTIME OFF)
      set(protobuf_INSTALL ${ONNX_UTILS_INSTALL_PROTOBUF})
      set(protobuf_BUILD_TESTS OFF)
      set(protobuf_BUILD_SHARED_LIBS OFF)
      set(BUILD_SHARED_LIBS OFF)
    else()
      set(
        protobuf_MSVC_STATIC_RUNTIME
        OFF
        CACHE INTERNAL
        "Use dynamic libraries for runtime"
      )
      set(protobuf_BUILD_TESTS OFF CACHE INTERNAL "Build tests OFF")
    endif()
    FetchContent_MakeAvailable(protobuf)

    # if protobuf is being built with the library, we don't have the
    # protobuf_generate command yet
    find_package(Protobuf CONFIG REQUIRED)
    if(EXISTS ${protobuf_SOURCE_DIR}/cmake/protobuf-generate.cmake)
      include(${protobuf_SOURCE_DIR}/cmake/protobuf-generate.cmake)
    endif()

    set(Protobuf_VERSION "6.31.1")
  endif()

  # newer versions of protobuf have extra dependencies that we should install
  # for improving packaging downstream
  if(WIN32 AND Protobuf_VERSION VERSION_GREATER_EQUAL 4.22.0)
    if(TARGET absl::abseil_dll)
      install(IMPORTED_RUNTIME_ARTIFACTS absl::abseil_dll DESTINATION bin)

      find_file(
        UTF8_VALIDITY_DLL
        NAMES libutf8_validity.dll utf8_validity.dll
        REQUIRED
      )

      install(FILES "${UTF8_VALIDITY_DLL}" DESTINATION bin)
      message(STATUS "Installing abseil DLLs")
    endif()
  endif()
endfunction()
