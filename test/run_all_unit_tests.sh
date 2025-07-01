#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
#!/bin/bash
set -e
set -x

run_code_coverage=${1,,:-false}

# Unit tests must be run from `test` folder
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${THIS_DIR}"

# Install tests requirements
pip install -r requirements.txt

# Run tests with or without code coverage.
if [[ "${run_code_coverage}" == "1" || "${run_code_coverage,,}" == "true" || "${run_code_coverage,,}" == "yes" ]]; then
    echo "Running pytest with coverage test..."
    pytest -s --cov=../quark --cov-report=html
else
    echo "Running pytest without coverage test..."
    pytest -s
fi
