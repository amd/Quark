# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

if(NOT TARGET dml)
  set(target dml)
  # download nuget to restore the dependency packages
  set(NUGET_EXE ${PROJECT_BINARY_DIR}/nuget/nuget.exe)
  set(NUGET_PACKAGES_DIR ${PROJECT_SOURCE_DIR}/external/packages)
  if(NOT EXISTS ${NUGET_EXE})
    message(STATUS "Downloading NuGet...")
    file(
      DOWNLOAD
        https://dist.nuget.org/win-x86-commandline/latest/nuget.exe
        ${NUGET_EXE}
    )
  endif()

  # run nuget command to restore from packages.config which lists the packages.
  # the packages are saved in external folder in the root directory
  message(STATUS "Restoring nuget packages...")
  execute_process(
    COMMAND
      ${NUGET_EXE} restore ${PROJECT_SOURCE_DIR}/packages.config
      -PackagesDirectory ${NUGET_PACKAGES_DIR}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/nuget
  )

  add_library(${target} INTERFACE)

  # add preprocessor to use the latest DML version
  target_compile_definitions(${target} INTERFACE DML_TARGET_VERSION_USE_LATEST)

  target_include_directories(
    ${target}
    SYSTEM
    INTERFACE
      ${PROJECT_SOURCE_DIR}/external/packages/Microsoft.AI.DirectML.1.15.4/include
      ${PROJECT_SOURCE_DIR}/external/packages/Microsoft.Direct3D.D3D12.1.614.1/build/native/include
  )

  target_link_libraries(
    ${target}
    INTERFACE "d3d12.lib" "dxgi.lib" "directml.lib" "D3DCompiler.lib"
  )
endif()
