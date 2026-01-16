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

enforce_jupyter_notebook_are_stored_on_tutorials_folder() {
    # Define the allowed subfolder path.
    ALLOWED_DIR=${1:-"*/tutorials/*"}

    # Check if the allowed directory exists (optional, but good practice)
    if [ ! -d "$ALLOWED_DIR" ]; then
        echo "Warning: The allowed directory '$ALLOWED_DIR' does not exist."
    fi

    # Find all violating files and store them in a variable.
    # -not -path "$ALLOWED_DIR/*" excludes files within the allowed directory.
    # We use 'local' here to ensure the VIOLATIONS variable is scoped to the function.
    local VIOLATIONS=$(find . -type f -name "*.ipynb" -not -path "$ALLOWED_DIR" 2>/dev/null)

    # Check if the VIOLATIONS variable is non-empty (-n)
    if [ -n "$VIOLATIONS" ]; then
        echo "--- VIOLATION FOUND ---"
        echo "Error: Found *.ipynb files outside the allowed subfolder '$ALLOWED_DIR/':"
        echo ""
        echo "$VIOLATIONS"
        echo ""
        echo "Returning status 1 (Failure)."
        # In a function, use return to set the exit status
        exit 1
    else
        echo "Success: All *.ipynb files are either in '$ALLOWED_DIR/' or not present."
        echo "Returning status 0 (Success)."
    fi
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

    enforce_jupyter_notebook_are_stored_on_tutorials_folder "*/tutorials/*"

    # Install pandoc binary required by nbconvert
    python -c "from pypandoc.pandoc_download import download_pandoc; download_pandoc()"
    # Pandoc is installed by default on $HOME/bin
    export PATH=~/bin:${PATH}

    # When env var QUARK_SPHINX_BUILD_SKIP_TUTORIALS=1, skip ALL Jupyter notebook build when set
    # When env var QUARK_SPHINX_BUILD_SKIP_TUTORIALS=0, skip only Jupyter notebook present in env var QUARK_DOC_MODIFIED_TUTORIALS
    # The skipped `tutorials` is deleted from `./_docs/` to prevent warnings from unused files from sphinx-build
    QUARK_SPHINX_BUILD_SKIP_TUTORIALS=${QUARK_SPHINX_BUILD_SKIP_TUTORIALS:-""}
    echo "[QUARK-INFO] QUARK_SPHINX_BUILD_SKIP_TUTORIALS=${QUARK_SPHINX_BUILD_SKIP_TUTORIALS}"
    if [[ "${QUARK_SPHINX_BUILD_SKIP_TUTORIALS}" == "1" || "${QUARK_SPHINX_BUILD_SKIP_TUTORIALS,,}" == "true" || "${QUARK_SPHINX_BUILD_SKIP_TUTORIALS,,}" == "yes" ]]
    then
        echo "[QUARK-INFO] Converting Jupyter Notebooks into ReStructuredText files from tutorials/* subfolder..."
        find "./_docs/tutorials/" -type f -name "*.ipynb" -print0 | while IFS= read -r -d $'\0' notebook_file; do
            echo "Converting Jupyter Notebook into ReStructuredText file: $notebook_file"
            jupyter nbconvert --to rst "${notebook_file}"
            rm -v ${notebook_file}
        done
    else
        echo "[QUARK-INFO] QUARK_DOC_MODIFIED_TUTORIALS=${QUARK_DOC_MODIFIED_TUTORIALS}"
        find "./_docs/tutorials/" -type f -name "*.ipynb" -print0 | while IFS= read -r -d $'\0' notebook_file; do
            echo "Clearing outputs from Jupyter Notebook cells: ${notebook_file}"
            dir=$(dirname "${notebook_file}")
            requirements_txt_file="${dir}/requirements.txt"
            if [ -f "${requirements_txt_file}" ]; then
                echo "[QUARK-INFO] Installing requirements for ${dir}: ${requirements_txt_file}"
                pip install -r "${requirements_txt_file}"
            else
                echo "[QUARK-INFO] No local requirements.txt found under the directory ${dir}. Installation is skipped."
            fi
            jupyter nbconvert --clear-output --inplace "${notebook_file}"

            if [[ -n "${QUARK_DOC_MODIFIED_TUTORIALS}" ]]; then
                # The notebook path in the find command is relative to the current directory (e.g., ./_docs/tutorials/...)
                # QUARK_DOC_MODIFIED_TUTORIALS contains paths relative to the repo root (e.g., docs/source/tutorials/...)
                # We need to check if the notebook file path is present in the list of modified tutorials.
                relative_notebook_file=${notebook_file#./_docs/} # remove ./_docs/ prefix
                relative_notebook_file="docs/source/${relative_notebook_file}" # prepend docs/
                echo "[QUARK-INFO] relative_notebook_file=${relative_notebook_file}"
                modified_by_pr=0
                for modified_tutorial in ${QUARK_DOC_MODIFIED_TUTORIALS}; do
                    if [[ " ${modified_tutorial} " == *" ${relative_notebook_file} "* ]]; then
                        echo "Jupyter Notebook ${notebook_file} was modified and needs to be compiled!"
                        modified_by_pr=1
                        break
                    fi
                done
                if [[ ${modified_by_pr} -eq 0 ]]; then
                    echo "Jupyter Notebook ${notebook_file} is unmodified and will be converted into ReStructuredText because QUARK_DOC_MODIFIED_TUTORIALS is not empty"
                    jupyter nbconvert --to rst "${notebook_file}"
                    rm -v ${notebook_file}
                fi

            fi
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
    find ../examples/contrib -type f -name "*.rst" | while IFS= read -r mdfile; do
        cp ${mdfile} ./_docs/pytorch/
    done
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
            python -m quark_dashboard.api --path "$file" --api_url "http://quark.amd.com/dashboard/"
        fi
    done
}

cd ${THIS_DIR}
./install_requirements.sh
build_docs
