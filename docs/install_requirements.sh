#!/bin/bash

#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Usage: ./install_requirements.sh [--include-torch] [--include-onnxruntime] [--include-transformers]
#
# Without flags, only the Sphinx toolchain is installed (Quark and its runtime deps are expected to be
# installed already). The --include-* flags additionally install the CPU builds of the packages autodoc
# needs to import the full public quark.torch / quark.onnx API, e.g. on Read the Docs. CPU builds avoid
# pulling the CUDA torch wheel and its nvidia-* dependencies, which only the GPU runtime needs.

# Keep in sync with the defaults in .github/workflows/ci_doc.yml
TORCH_VERSION="2.13.0"
ONNXRUNTIME_VERSION="1.28.0"
TRANSFORMERS_VERSION="5.15.0"

include_torch=0
include_onnxruntime=0
include_transformers=0
for arg in "$@"; do
    case "${arg}" in
        --include-torch) include_torch=1 ;;
        --include-onnxruntime) include_onnxruntime=1 ;;
        --include-transformers) include_transformers=1 ;;
        *)
            echo "ERROR: Unknown argument '${arg}'. Supported: --include-torch, --include-onnxruntime, --include-transformers"
            exit 1
            ;;
    esac
done

# Upgrade pip before any install: the pip 23.1 seeded into fresh virtualenvs (e.g. on Read the Docs) rejects
# the typing_extensions wheel on the PyTorch CPU index and falls back to an sdist it cannot build.
# TODO: Install latest `pip` when https://github.com/jazzband/pip-tools/issues/2252 is fixed
pip install --upgrade pip==25.2 pip-tools

set -e

if [[ ${include_torch} -eq 1 ]]; then
    pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu
fi
if [[ ${include_onnxruntime} -eq 1 ]]; then
    pip install onnxruntime==${ONNXRUNTIME_VERSION}
fi
if [[ ${include_transformers} -eq 1 ]]; then
    pip install transformers==${TRANSFORMERS_VERSION}
fi

set +e

# Update source/sphinx/requirements.txt from source/sphinx/requirements.in
rm -f source/sphinx/requirements.txt
LC_ALL=C pip-compile source/sphinx/requirements.in

pip install quark-dashboard --no-cache --trusted-host xcoartifactory.xilinx.com -i https://xcoartifactory.xilinx.com/artifactory/api/pypi/uai-pip-local/simple/

# Install updated requirements.txt
pip install -r source/sphinx/requirements.txt
LC_ALL=C sphinx-build --version
