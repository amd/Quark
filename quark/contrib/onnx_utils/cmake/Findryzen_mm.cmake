# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

find_package(ryzen_mm CONFIG QUIET)

if(NOT TARGET ryzen_mm::ryzen_mm)
  include(FetchContent)

  if(DEFINED RMM_SRC AND EXISTS "${RMM_SRC}/CMakeLists.txt")
    message(STATUS "Using local RMM: " ${RMM_SRC})
    FetchContent_Declare(ryzen_mm SOURCE_DIR "${RMM_SRC}" SYSTEM)
  else()
    message(STATUS "Using RMM from FetchContent")
    FetchContent_Declare(
      ryzen_mm
      GIT_REPOSITORY https://gitenterprise.xilinx.com/VitisAI/ryzenai-mm.git
      GIT_TAG 62340ff64793be729a6a2601f068314e3d36d33c
      GIT_SHALLOW TRUE
      SYSTEM
    )
  endif()

  FetchContent_MakeAvailable(ryzen_mm)
endif()

if(TARGET ryzen_mm::ryzen_mm)
  set(ryzen_mm_FOUND TRUE)
else()
  set(ryzen_mm_FOUND FALSE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ryzen_mm DEFAULT_MSG ryzen_mm_FOUND)
