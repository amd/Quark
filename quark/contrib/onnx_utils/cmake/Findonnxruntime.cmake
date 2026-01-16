# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

# if this project is included in ORT with FetchContent, it fails to build due to
# incomplete CMake files existing in the build tree
if(CMAKE_PROJECT_NAME STREQUAL "onnxruntime")
  add_library(ryzenai_onnx_utils_onnxruntime INTERFACE)
  target_link_libraries(ryzenai_onnx_utils_onnxruntime INTERFACE onnxruntime)
  add_library(onnxruntime::onnxruntime ALIAS ryzenai_onnx_utils_onnxruntime)
  # since no DLL is being set, if this project is included in ORT, you cannot
  # build any executables that depend on the DLL
  set(ORT_SHARED_LIB "")
else()
  find_package(onnxruntime CONFIG QUIET)

  if(TARGET onnxruntime::onnxruntime)
    get_target_property(
      ORT_SHARED_LIB
      onnxruntime::onnxruntime
      IMPORTED_LOCATION_RELEASE
    )
  else()
    message(STATUS "Manually configuring OnnxRuntime")
    # in this case, no CMake config files were found so have to it manually
    if(NOT DEFINED ORT_INSTALL_DIRS)
      set(ORT_INSTALL_DIRS ${PROJECT_SOURCE_DIR}/external/onnxruntime)
    endif()

    if(NOT DEFINED ORT_INCLUDE_DIRS)
      set(ORT_INCLUDE_DIRS ${ORT_INSTALL_DIRS}/include)
    else()
      cmake_path(SET ORT_INCLUDE_DIRS ${ORT_INCLUDE_DIRS})
    endif()

    if(NOT DEFINED ORT_LIB_DIRS)
      set(ORT_LIB_DIRS ${ORT_INSTALL_DIRS}/lib)
    else()
      cmake_path(SET ORT_LIB_DIRS ${ORT_LIB_DIRS})
    endif()

    if(NOT DEFINED ORT_BIN_DIRS)
      set(ORT_BIN_DIRS ${ORT_INSTALL_DIRS}/bin)
    else()
      cmake_path(SET ORT_BIN_DIRS ${ORT_BIN_DIRS})
    endif()

    if(WIN32)
      find_path(
        ORT_DLL_DIRS
        NAMES "onnxruntime.dll"
        HINTS ${ORT_BIN_DIRS} ${ORT_LIB_DIRS}
        NO_DEFAULT_PATH
      )
      set(ORT_SHARED_LIB ${ORT_DLL_DIRS}/onnxruntime.dll)
      set(ORT_LIB ${ORT_LIB_DIRS}/onnxruntime.lib)
    else()
      find_library(
        ORT_SHARED_LIB
        NAMES "onnxruntime"
        PATHS ${ORT_BIN_DIRS} ${ORT_LIB_DIRS}
      )
      set(ORT_LIB ${ORT_SHARED_LIB})
      set(ORT_SHARED_LIB ${ORT_SHARED_LIB})
    endif()

    add_library(onnxruntime::onnxruntime SHARED IMPORTED)

    set_target_properties(
      onnxruntime::onnxruntime
      PROPERTIES IMPORTED_IMPLIB ${ORT_LIB} IMPORTED_LOCATION ${ORT_SHARED_LIB}
    )
  endif(TARGET onnxruntime::onnxruntime)

  if(ONNX_UTILS_USE_RYZENAI_EP)
    include(FetchContent)

    if(DEFINED ORT_BIN_DIRS AND DEFINED ORT_LIB_DIRS)
      if(WIN32)
        set(ORT_PS_DLL ${ORT_BIN_DIRS}/onnxruntime_providers_shared.dll)
        set(ORT_PS_LIB ${ORT_LIB_DIRS}/onnxruntime_providers_shared.lib)
      else()
        find_library(
          ORT_PROVIDERS_SHARED
          NAMES "onnxruntime_providers_shared"
          PATHS ${ORT_BIN_DIRS} ${ORT_LIB_DIRS}
        )
        set(ORT_PS_DLL ${ORT_PROVIDERS_SHARED})
        set(ORT_PS_LIB ${ORT_PROVIDERS_SHARED})
      endif()
    else()
      find_library(ORT_PROVIDERS_SHARED NAMES "onnxruntime_providers_shared")
      set(ORT_PS_DLL ${ORT_PROVIDERS_SHARED})
      set(ORT_PS_LIB ${ORT_PROVIDERS_SHARED})
    endif()

    add_library(onnxruntime::onnxruntime_providers_shared SHARED IMPORTED)

    if(NOT (DEFINED ORT_SRC AND EXISTS "${ORT_SRC}/cmake/CMakeLists.txt"))
      FetchContent_Declare(
        onnxruntime
        GIT_REPOSITORY https://gitenterprise.xilinx.com/VitisAI/onnxruntime.git
        GIT_TAG 378625893f2d847339d62fc0834441657882b2ac
        SOURCE_SUBDIR
        _noop_
        GIT_SHALLOW TRUE
        GIT_SUBMODULES ""
        SYSTEM
      )

      FetchContent_MakeAvailable(onnxruntime)

      set(
        ORT_SRC
        "${onnxruntime_SOURCE_DIR}"
        CACHE PATH
        "Path to ONNX Runtime sources"
        FORCE
      )
    endif()

    if(NOT (DEFINED ORT_DEPS_SRC))
      # hashes came from https://github.com/microsoft/onnxruntime/blob/v1.22.1/cmake/deps.txt

      FetchContent_Declare(
        SafeInt
        GIT_REPOSITORY https://github.com/dcleblanc/SafeInt.git
        # tag of v3.0.28, it's a git hash because tags are mutable
        GIT_TAG 4cafc9196c4da9c817992b20f5253ef967685bf8
        SYSTEM
        # the patch command gets applied after every update. But if there's no
        # change, the patch fails to apply if it's already been applied. This
        # is a hacky workaround to avoid this problem to allow rebuilding
        # without needing to delete the safeint directory. After the initial
        # download, the version of safeint will not change until you delete
        # the safeint directory and it redownloads the new version.
        UPDATE_DISCONNECTED true
        PATCH_COMMAND
          git apply "${CMAKE_CURRENT_LIST_DIR}/disable_safeint_tests.patch"
      )

      FetchContent_Declare(
        GSL
        URL https://github.com/microsoft/GSL/archive/refs/tags/v4.0.0.zip
        URL_HASH SHA1=cf368104cd22a87b4dd0c80228919bb2df3e2a14
        SYSTEM
      )

      FetchContent_MakeAvailable(SafeInt)
      FetchContent_MakeAvailable(GSL)

      # SafeInt_SOURCE_DIR might point to "Test\MsvcTest" subdir
      # instead of the actual root. Lets remove the suffix.
      #
      if(SafeInt_SOURCE_DIR MATCHES [[[/\\]Test[/\\]MsvcTest$]])
        get_filename_component(
          _safeint_root
          "${SafeInt_SOURCE_DIR}/../.."
          ABSOLUTE
        )

        message(
          STATUS
          "Fixing SafeInt_SOURCE_DIR from "
          "${SafeInt_SOURCE_DIR}  -->  ${_safeint_root}"
        )

        set(SafeInt_SOURCE_DIR "${_safeint_root}")
      endif()
    else()
      set(SafeInt_SOURCE_DIR "${ORT_DEPS_SRC}/safeint-src")
      set(GSL_SOURCE_DIR "${ORT_DEPS_SRC}/gsl-src")
    endif()

    add_library(SafeInt::SafeInt INTERFACE IMPORTED)
    add_library(Microsoft::GSL INTERFACE IMPORTED)

    set_target_properties(
      SafeInt::SafeInt
      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${SafeInt_SOURCE_DIR}"
    )
    set_target_properties(
      Microsoft::GSL
      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${GSL_SOURCE_DIR}/include"
    )

    set(
      ORT_INCLUDE_DIRS
      "${ORT_SRC}"
      "${ORT_SRC}/onnxruntime"
      "${ORT_SRC}/include"
      "${ORT_SRC}/include/onnxruntime"
      "${ORT_SRC}/include/onnxruntime/core/session"
    )

    set_target_properties(
      onnxruntime::onnxruntime_providers_shared
      PROPERTIES IMPORTED_IMPLIB ${ORT_PS_LIB} IMPORTED_LOCATION ${ORT_PS_DLL}
    )
  endif(ONNX_UTILS_USE_RYZENAI_EP)

  set_target_properties(
    onnxruntime::onnxruntime
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIRS}"
  )
endif(CMAKE_PROJECT_NAME STREQUAL "onnxruntime")
