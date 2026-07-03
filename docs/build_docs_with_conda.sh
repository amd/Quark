#!/bin/bash

#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

set -e
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${THIS_DIR}/../tools/ci/utils.sh"
configure_ci_verbose

python_version=${1}
if [[ -z "${python_version}" ]]; then
    echo "ERROR: The input python_version must be set, but is not set."
    exit 1
fi

workspace_root_dir=${2}
if [[ -z "${workspace_root_dir}" ]]; then
    echo "ERROR: The input workspace_root_dir must be set, but is not set."
    exit 1
fi

onnxruntime_version=${3}
if [[ -z "${onnxruntime_version}" ]]; then
    echo "ERROR: The input onnxruntime_version must be set, but is not set."
    exit 1
fi

torch_version=${4}
if [[ -z "${torch_version}" ]]; then
    echo "ERROR: The input torch_version must be set, but is not set."
    exit 1
fi

transformers_version=${5}
if [[ -z "${transformers_version}" ]]; then
    echo "ERROR: The input transformers_version must be set, but is not set."
    exit 1
fi

accelerator_version=${6,,}
if [[ -z "${accelerator_version}" ]]; then
    echo "ERROR: The input accelerator_version must be set, but is not set."
    exit 1
fi

# Optional: path to a prebuilt quark wheel to install for docs instead of
# rebuilding from source (see the install step below). Empty for standalone
# callers (upload_whl docs job), which rebuild as before.
reuse_wheel=${7:-""}

# Common functions needed by the unit test script
cd ${workspace_root_dir}
source ./tools/ci/install_quark.sh ${python_version} ${workspace_root_dir} ${accelerator_version}
conda_env_name="quark-env"
set_conda ${python_version} ${conda_env_name} ${workspace_root_dir} ${onnxruntime_version} ${torch_version} ${accelerator_version} ${transformers_version} "activate"
cd ${workspace_root_dir}/docs
./install_requirements.sh
pip uninstall -y amd-quark
# When the caller already built a wheel (reuse_wheel), install that via the
# shared install_quark entry point instead of rebuilding from source: a rebuild
# here duplicates work and, for the universal bare wheel, would emit a second
# +torch-tagged wheel in output_whl that collides with the publishable one. With
# torch active, importing the wheel JIT-compiles _C so autoapi resolves the full
# API. Empty reuse_wheel falls through to a from-source build.
if [[ -n "${reuse_wheel}" ]]; then
    echo "Reusing prebuilt quark wheel for docs: ${reuse_wheel}"
    install_quark "${reuse_wheel}"
else
    install_quark_from_current_src "false" ${workspace_root_dir}
fi

cd ${workspace_root_dir}
source ./docs/build_docs.sh
