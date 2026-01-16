# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

find_package(GTest CONFIG QUIET REQUIRED)

add_subdirectory(hybrid_llm)
# uncomment to build a DD test
# add_subdirectory(dynamic_dispatch)
