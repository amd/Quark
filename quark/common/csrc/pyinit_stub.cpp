//
// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stable-ABI / ORT extensions register ops via static constructors and need no
// Python module init, but Windows setuptools links with /EXPORT:PyInit_<name>
// and fails without the symbol — so we provide one. QUARK_PYINIT_MODULE_NAME
// retargets the exported name (e.g. _C_cpu) so one stub serves every extension.
//

#include <Python.h>

#ifndef QUARK_PYINIT_MODULE_NAME
#define QUARK_PYINIT_MODULE_NAME _C
#endif

#define QUARK_STR2(x) #x
#define QUARK_STR(x) QUARK_STR2(x)
#define QUARK_CAT2(a, b) a##b
#define QUARK_CAT(a, b) QUARK_CAT2(a, b)
#define QUARK_PYINIT_FN QUARK_CAT(PyInit_, QUARK_PYINIT_MODULE_NAME)

static struct PyModuleDef _pyinit_module_def = {
  PyModuleDef_HEAD_INIT, QUARK_STR(QUARK_PYINIT_MODULE_NAME), nullptr, -1,
  nullptr
};

extern "C"
#ifdef _WIN32
  __declspec(dllexport)
#endif
  PyObject* QUARK_PYINIT_FN(void) {
  return PyModule_Create(&_pyinit_module_def);
}
