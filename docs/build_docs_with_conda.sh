#!/bin/bash

#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

set -e
set -x
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

hardware_type=${6,,}
if [[ -z "${hardware_type}" ]]; then
    echo "ERROR: The input hardware_type must be set, but is not set."
    exit 1
fi

# Common functions needed by the unit test script
cd ${workspace_root_dir}
source ./tools/ci/install_quark.sh ${python_version} ${workspace_root_dir} ${hardware_type}
conda_env_name="quark-env"
set_conda ${python_version} ${conda_env_name} ${workspace_root_dir} ${onnxruntime_version} ${torch_version} ${hardware_type} ${transformers_version} "activate"
cd ${workspace_root_dir}/docs
./install_requirements.sh
pip uninstall -y amd-quark
install_quark_from_current_src "false" ${workspace_root_dir}

cd ${workspace_root_dir}
source ./docs/build_docs.sh
