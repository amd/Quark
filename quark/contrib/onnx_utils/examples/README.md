<!--
Copyright (c) 2024 Advanced Micro Devices, Inc.
-->

# Examples

To build examples, add `-DONNX_UTILS_BUILD_EXAMPLES="ON"` to your CMake build command.
The built examples are in `build/examples/<example>/<config>/*.exe.

CMake will copy in any DLLs the examples may need so there's no conflict with any libraries in your host system.
