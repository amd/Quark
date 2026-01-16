# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

find_package(onnxruntime-genai REQUIRED)

set(targets test_single_matmul test_e2e)
foreach(target ${targets})
  add_executable(${target} ${target}.cpp)
  target_compile_options(
    ${target}
    PRIVATE
      ${ONNX_UTILS_DEFAULT_COMPILE_OPTIONS}
      ${ONNX_UTILS_USER_COMPILE_OPTIONS}
  )
  # target_include_directories(${target} PRIVATE ${ONNX_CUSTOM_OPS_DIR})
  target_link_libraries(
    ${target}
    PRIVATE
      onnxruntime::onnxruntime
      ryzenai_onnx_utils
      onnxruntime-genai::onnxruntime-genai
      GTest::gtest
  )
  add_custom_command(
    TARGET ${target}
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:${target}>
      $<TARGET_RUNTIME_DLLS:${target}>
    COMMAND_EXPAND_LISTS
  )
endforeach()
