# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

find_package(ryzenai_corelib CONFIG QUIET)

if(NOT TARGET ryzenai_corelib::corelib)
  include(FetchContent)

  if(DEFINED CORELIB_SRC AND EXISTS "${CORELIB_SRC}/CMakeLists.txt")
    message(STATUS "Using local CoreLib: " ${CORELIB_SRC})
    FetchContent_Declare(ryzenai_corelib SOURCE_DIR "${CORELIB_SRC}" SYSTEM)
  else()
    message(STATUS "Using CoreLib from FetchContent")
    FetchContent_Declare(
      ryzenai_corelib
      GIT_REPOSITORY
        https://gitenterprise.xilinx.com/VitisAI/ryzenai-corelib.git
      GIT_TAG release/1.0-rc3
      GIT_SHALLOW TRUE
      SYSTEM
    )
  endif()

  FetchContent_MakeAvailable(ryzenai_corelib)
endif()

if(TARGET ryzenai_corelib::corelib)
  set(ryzenai_corelib_FOUND TRUE)
else()
  set(ryzenai_corelib_FOUND FALSE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  ryzenai_corelib
  DEFAULT_MSG
  ryzenai_corelib_FOUND
)
