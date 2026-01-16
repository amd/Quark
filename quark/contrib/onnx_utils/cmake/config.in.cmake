# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

set(@PROJECT_NAME@_FOUND True)

include(CMakeFindDependencyMacro)

include("${CMAKE_CURRENT_LIST_DIR}/@PROJECT_NAME@-targets.cmake")
get_target_property(TARGET_LOCATION @PROJECT_NAME@::@PROJECT_NAME@ LOCATION)
message(STATUS "Found @PROJECT_NAME@::@PROJECT_NAME@: ${TARGET_LOCATION}")

get_filename_component(
  @PROJECT_NAME@_CMAKE_DIR
  "${CMAKE_CURRENT_LIST_FILE}"
  PATH
)
