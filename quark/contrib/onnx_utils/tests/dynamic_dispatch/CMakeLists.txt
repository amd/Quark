# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

# assuming a new file named main.cpp will be added
add_executable(test_dd main.cpp)
target_link_libraries(test_dd PRIVATE DynamicDispatch::dyn_dispatch_core)
