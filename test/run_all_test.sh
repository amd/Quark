#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
if [ -n "$COVERAGE" ]; then
    echo "Running pytest with coverage test..."
    pytest -s --cov=../quark --cov-report=html
    pytest_exit_code=$?
else
    echo "Running pytest without coverage test..."
    pytest -s
    pytest_exit_code=$?
fi

echo "exit run_all_test.sh with the status of pytest: $pytest_exit_code"
exit $pytest_exit_code
