#!/bin/bash

#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

set -e
set -x
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

install_jupyter_notebooks_dependencies() {
    pip install -r ${THIS_DIR}/source/tutorials/requirements.txt
}

build_docs() {
    echo "[QUARK-INFO] Delete cache files..."

    if [ -e "./_docs" ]; then
        rm -rf ./_docs
    fi

    echo `pwd`
    cp -r ./source _docs
    # Delete unused files
    rm -f ./_docs/readme_for_zip.md

    # Check if QUARK_SPHINX_BUILD_SKIP_TUTORIALS env var and skip the Jupyter notebook build when set
    # The `tutorials` subfolder is deleted from `./_docs/` to prevent warnings from unused files from sphinx-build
    QUARK_SPHINX_BUILD_SKIP_TUTORIALS=${QUARK_SPHINX_BUILD_SKIP_TUTORIALS:-""}
    echo "[QUARK-INFO] QUARK_SPHINX_BUILD_SKIP_TUTORIALS=${QUARK_SPHINX_BUILD_SKIP_TUTORIALS}"
    if [[ "${QUARK_SPHINX_BUILD_SKIP_TUTORIALS}" == "1" || "${QUARK_SPHINX_BUILD_SKIP_TUTORIALS,,}" == "true" || "${QUARK_SPHINX_BUILD_SKIP_TUTORIALS,,}" == "yes" ]]
    then
        echo "[QUARK-INFO] Converting Jupyter Notebooks into ReStructuredText files from tutorials/* subfolder..."
        # Install pandoc binary required by nbconvert
        python -c "from pypandoc.pandoc_download import download_pandoc; download_pandoc()"
        # Pandoc is installed by default on $HOME/bin
        export PATH=~/bin:${PATH}

        find "./_docs/tutorials/" -type f -name "*.ipynb" -print0 | while IFS= read -r -d $'\0' notebook_file; do
            echo "Converting Jupyter Notebook into ReStructuredText file: $notebook_file"
            jupyter nbconvert --to rst "${notebook_file}"
            rm -v ${notebook_file}
        done
    else
        find "./_docs/tutorials/" -type f -name "*.ipynb" -print0 | while IFS= read -r -d $'\0' notebook_file; do
            echo "Clearing outputs from Jupyter Notebook cells: ${notebook_file}"
            jupyter nbconvert --clear-output --inplace "${notebook_file}"
        done
        unset QUARK_SPHINX_BUILD_SKIP_TUTORIALS
        install_jupyter_notebooks_dependencies
    fi

    if [ -e "../_docs_build" ]; then
        rm -rf ../_docs_build
    fi

    echo "[QUARK-INFO] Copy version file and example documentation..."
    cp -f ../quark/version.txt ./_docs/version.txt
    # Examples: copy examples documentation (*.rst) from ../examples/{torch,onnx} to ./_docs/{onnx,pytorch}
    mkdir -p ./_docs/{onnx,pytorch}
    find ../examples/torch -type f -name "*.rst" | while IFS= read -r mdfile; do
        cp ${mdfile} ./_docs/pytorch/
    done
    find ../examples/onnx -type f -name "*.rst" | while IFS= read -r mdfile; do
        cp ${mdfile} ./_docs/onnx/
    done

    echo "[QUARK-INFO] Building Quark documentation..."

    mkdir -p ./_docs/output/

    if [[ -n "${QUARK_DOC_FAIL_ON_WARNING}" && ( "${QUARK_DOC_FAIL_ON_WARNING}" == "1" || "${QUARK_DOC_FAIL_ON_WARNING,,}" == "true" || "${QUARK_DOC_FAIL_ON_WARNING,,}" == "yes" ) ]]; then
        echo "[QUARK-INFO] Quark documentation build will fail on warnings..."
        sphinx_build_fail_on_warning_args=" --keep-going --fail-on-warning --nitpicky"
    else
        echo "[QUARK-INFO] Quark documentation build WILL NOT fail on warnings..."
        sphinx_build_fail_on_warning_args=""
    fi

    # Using LC_ALL=C to avoid https://stackoverflow.com/questions/14547631/python-locale-error-unsupported-locale-setting
    SPHINX_BUILD_CMD="LC_ALL=C sphinx-build -M html ./_docs/ ../_docs_build/ -v --show-traceback ${sphinx_build_fail_on_warning_args}"
    echo "${SPHINX_BUILD_CMD}"
    bash -c "${SPHINX_BUILD_CMD}"

    echo "[QUARK-INFO] Uploading results to dashboard"
    for file in ./_docs/output/*; do
        if [[ "$file" == *.json ]]; then
            echo "$file"
            python -m quark_dashboard.api --path "$file" --api_url "http://xcomx250-1.xilinx.com:8000/"
        fi
    done
}

cd ${THIS_DIR}
./install_requirements.sh
build_docs
