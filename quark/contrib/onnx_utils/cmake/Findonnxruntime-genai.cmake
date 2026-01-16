# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

find_path(oga_INCLUDE_DIR ort_genai.h)

find_library(oga_LIB onnxruntime-genai $ENV{onnxruntime-genai_DIR})
string(REPLACE ".lib" ".dll" oga_DLL ${oga_LIB})

mark_as_advanced(oga_INCLUDE_DIR oga_LIB oga_DLL)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  onnxruntime-genai
  REQUIRED_VARS oga_INCLUDE_DIR oga_LIB oga_DLL
)

if(onnxruntime-genai_FOUND AND NOT TARGET onnxruntime-genai::onnxruntime-genai)
  add_library(onnxruntime-genai::onnxruntime-genai SHARED IMPORTED)
  set_target_properties(
    onnxruntime-genai::onnxruntime-genai
    PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_IMPLIB "${oga_LIB}"
      IMPORTED_LOCATION "${oga_DLL}"
      INTERFACE_INCLUDE_DIRECTORIES "${oga_INCLUDE_DIR}"
  )
endif()
