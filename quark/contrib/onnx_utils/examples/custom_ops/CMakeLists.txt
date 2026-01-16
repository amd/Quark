# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

set(target example_custom_ops)
add_executable(${target} example.cc)
set_target_properties(${target} PROPERTIES OUTPUT_NAME "custom_ops")
target_compile_options(
  ${target}
  PRIVATE
    ${ONNX_UTILS_DEFAULT_COMPILE_OPTIONS}
    ${ONNX_UTILS_USER_COMPILE_OPTIONS}
)
# target_include_directories(${target} PRIVATE ${ONNX_CUSTOM_OPS_DIR})
target_link_libraries(
  ${target}
  PRIVATE onnxruntime::onnxruntime ${PROJECT_NAME}
)
# rebuild this target when ${PROJECT_NAME} changes so the post-build command
# runs when the DLL changes
set_source_files_properties(
  example.cc
  PROPERTIES OBJECT_DEPENDS ${PROJECT_NAME}
)

add_custom_command(
  TARGET ${target}
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:${target}>
    $<TARGET_RUNTIME_DLLS:${target}>
  COMMAND_EXPAND_LISTS
)
install(
  TARGETS ${target}
  EXPORT RyzenAIOnnxUtilsConfig
  DESTINATION bin/examples
)
