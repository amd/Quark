#!/bin/bash

#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

set -e
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${THIS_DIR}/../tools/ci/utils.sh"
configure_ci_verbose

run_code_coverage=${1,,:-false}

# Unit tests must be run from `test` folder
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${THIS_DIR}"

# Add Contrib tests to pytest check
CONTRIB_TESTS_FOLDER=""
for subfolder in ${THIS_DIR}/../quark/contrib/*/; do
    if [[ -d "$subfolder" && -d "$subfolder/test" ]]; then
        CONTRIB_TESTS_FOLDER+="$(realpath $subfolder/test) "
    fi
    if [[ -d "$subfolder" && -d "$subfolder/tests" ]]; then
        CONTRIB_TESTS_FOLDER+="$(realpath $subfolder/tests) "
    fi
done

# Install tests requirements
pip install -r requirements.txt
pip install -r ../quark/experimental/cli/requirements.txt

# Run tests with or without code coverage.
if [[ "${run_code_coverage}" == "1" || "${run_code_coverage,,}" == "true" || "${run_code_coverage,,}" == "yes" ]]; then
    echo "Running pytest with coverage test..."
    pytest -s --cov=../quark --cov-report=html ${THIS_DIR} ${CONTRIB_TESTS_FOLDER}
else
    echo "Running pytest without coverage test..."
    pytest -s ${THIS_DIR} ${CONTRIB_TESTS_FOLDER}
fi
