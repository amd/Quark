#!/bin/bash

#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

set -e
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defined inline (not sourced from tools/ci/utils.sh) because tools/ is not shipped to
# the public repo, so utils.sh is absent when Read the Docs builds from github.com/amd/quark.
configure_ci_verbose() {
    if [[ "${QUARK_CI_VERBOSE:-0}" == "1" ]]; then
        set -x
        unset TQDM_DISABLE
        unset HF_HUB_DISABLE_PROGRESS_BARS
    else
        set +x
        export TQDM_DISABLE=1
        export HF_HUB_DISABLE_PROGRESS_BARS=1
    fi
}
configure_ci_verbose

install_jupyter_notebooks_dependencies() {
    pip install -r ${THIS_DIR}/source/tutorials/requirements.txt
}

enforce_jupyter_notebook_are_stored_on_tutorials_folder() {
    # Define the allowed subfolder path patterns (comma-separated, e.g., "./_docs/tutorials/*,./source/tutorials/*")
    ALLOWED_DIRS=${1}

    # Find all notebooks first
    local ALL_NOTEBOOKS=$(find . -type f -name "*.ipynb" 2>/dev/null)

    # If no notebooks found, exit successfully
    if [ -z "${ALL_NOTEBOOKS}" ]; then
        echo "Success: No Jupyter notebooks found."
        return 0
    fi

    # Convert comma-separated directories to array
    IFS=',' read -ra DIR_PATTERNS <<< "${ALLOWED_DIRS}"
    local VIOLATIONS=""

    # Check each notebook file against allowed directories
    while IFS= read -r notebook_file; do
        if [ -z "${notebook_file}" ]; then
            continue
        fi

        local is_allowed=0
        # Check if notebook is in any of the allowed directories
        for dir_pattern in "${DIR_PATTERNS[@]}"; do
            # Trim leading and trailing whitespace from directory pattern using native Bash
            dir_pattern="${dir_pattern#"${dir_pattern%%[![:space:]]*}"}"  # Remove leading whitespace
            dir_pattern="${dir_pattern%"${dir_pattern##*[![:space:]]}"}"  # Remove trailing whitespace
            # Check if notebook path matches the pattern (using glob matching)
            case "${notebook_file}" in
                ${dir_pattern})
                    is_allowed=1
                    break
                    ;;
            esac
        done

        # If notebook is not in any allowed directory, add to violations
        if [ ${is_allowed} -eq 0 ]; then
            if [ -z "${VIOLATIONS}" ]; then
                VIOLATIONS="${notebook_file}"
            else
                VIOLATIONS="${VIOLATIONS}"$'\n'"${notebook_file}"
            fi
        fi
    done <<< "${ALL_NOTEBOOKS}"

    # Check if the VIOLATIONS variable is non-empty (-n)
    if [ -n "$VIOLATIONS" ]; then
        echo "--- VIOLATION FOUND ---"
        echo "Error: Found *.ipynb files outside the allowed directory(ies): ${ALLOWED_DIRS}"
        echo ""
        echo "$VIOLATIONS"
        echo ""
        echo "All Jupyter notebooks must be in one of the allowed directories (which get copied from root 'tutorials/' folder)."
        echo "Returning status 1 (Failure)."
        # In a function, use return to set the exit status
        exit 1
    else
        echo "Success: All *.ipynb files are in the allowed directory(ies): ${ALLOWED_DIRS}"
        echo "Returning status 0 (Success)."
    fi
}

build_docs() {
    echo "[QUARK-INFO] Delete cache files..."

    if [ -e "./_docs" ]; then
        rm -rf ./_docs
    fi

    echo `pwd`

    # Copy tutorials from root to docs/source/tutorials before copying source to _docs
    echo "[QUARK-INFO] Copying tutorials from root to docs/source/tutorials..."
    if [ -d "../tutorials" ]; then
        # Remove existing tutorials folder to avoid mixing old and new files
        rm -rf ./source/tutorials
        cp -r ../tutorials ./source/tutorials
    else
        echo "Warning: tutorials folder not found at root. Expected: ../tutorials"
    fi

    cp -r ./source _docs
    # Delete unused files
    rm -f ./_docs/readme_for_zip.md

    # Enforce that all Jupyter notebooks are stored in the allowed Sphinx directories
    enforce_jupyter_notebook_are_stored_on_tutorials_folder "./source/tutorials/*,./_docs/tutorials/*"

    # Install pandoc binary required by nbconvert
    source ${THIS_DIR}/install_pandoc.sh
    # Pandoc may be installed on $HOME/bin when using pypandoc
    export PATH=~/bin:${PATH}

    # When env var QUARK_SPHINX_FORCE_BUILD_TUTORIALS=1, force build ALL Jupyter notebooks regardless of modification status
    # When env var QUARK_SPHINX_SKIP_BUILD_TUTORIALS=1, skip ALL Jupyter notebook build when set
    # When env var QUARK_SPHINX_SKIP_BUILD_TUTORIALS=0, build only Jupyter notebook present in env var QUARK_DOC_MODIFIED_TUTORIALS and convert the rest into ReStructuredText files
    # The skipped `tutorials` is deleted from `./_docs/` to prevent warnings from unused files from sphinx-build
    QUARK_SPHINX_FORCE_BUILD_TUTORIALS=${QUARK_SPHINX_FORCE_BUILD_TUTORIALS:-""}
    QUARK_SPHINX_SKIP_BUILD_TUTORIALS=${QUARK_SPHINX_SKIP_BUILD_TUTORIALS:-""}
    echo "[QUARK-INFO] QUARK_SPHINX_FORCE_BUILD_TUTORIALS=${QUARK_SPHINX_FORCE_BUILD_TUTORIALS}"
    echo "[QUARK-INFO] QUARK_SPHINX_SKIP_BUILD_TUTORIALS=${QUARK_SPHINX_SKIP_BUILD_TUTORIALS}"

    # Priority: Force build > Skip build > Default behavior
    if [[ "${QUARK_SPHINX_FORCE_BUILD_TUTORIALS}" == "1" || "${QUARK_SPHINX_FORCE_BUILD_TUTORIALS,,}" == "true" || "${QUARK_SPHINX_FORCE_BUILD_TUTORIALS,,}" == "yes" ]]
    then
        # Although QUARK_SPHINX_MODIFIED_TUTORIALS is set by CI/CD, we still need to build all notebooks
        # to allow this script to use outside of CI/CD.
        echo "[QUARK-INFO] Force building ALL Jupyter Notebooks (QUARK_SPHINX_FORCE_BUILD_TUTORIALS is set)..."
        find "./_docs/tutorials/" -type f -name "*.ipynb" -print0 | while IFS= read -r -d $'\0' notebook_file; do
            echo "Force building Jupyter Notebook: ${notebook_file}"
            dir=$(dirname "${notebook_file}")
            requirements_txt_file="${dir}/requirements.txt"
            if [ -f "${requirements_txt_file}" ]; then
                echo "[QUARK-INFO] Installing requirements for ${dir}: ${requirements_txt_file}"
                pip install -r "${requirements_txt_file}"
            else
                echo "[QUARK-INFO] No local requirements.txt found under the directory ${dir}. Installation is skipped."
            fi
            jupyter nbconvert --clear-output --inplace "${notebook_file}"
        done
        unset QUARK_SPHINX_FORCE_BUILD_TUTORIALS
        install_jupyter_notebooks_dependencies
    elif [[ "${QUARK_SPHINX_SKIP_BUILD_TUTORIALS}" == "1" || "${QUARK_SPHINX_SKIP_BUILD_TUTORIALS,,}" == "true" || "${QUARK_SPHINX_SKIP_BUILD_TUTORIALS,,}" == "yes" ]]
    then
        echo "[QUARK-INFO] Converting Jupyter Notebooks into ReStructuredText files from tutorials/* subfolder..."
        find "./_docs/tutorials/" -type f -name "*.ipynb" -print0 | while IFS= read -r -d $'\0' notebook_file; do
            echo "Converting Jupyter Notebook into ReStructuredText file: $notebook_file"
            jupyter nbconvert --to rst "${notebook_file}"
            rm -v ${notebook_file}
        done
    else
        echo "[QUARK-INFO] QUARK_SPHINX_MODIFIED_TUTORIALS=${QUARK_SPHINX_MODIFIED_TUTORIALS}"
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

            if [[ -n "${QUARK_SPHINX_MODIFIED_TUTORIALS}" ]]; then
                # The notebook path in the find command is relative to the current directory (e.g., ./_docs/tutorials/...)
                # QUARK_SPHINX_MODIFIED_TUTORIALS contains paths relative to the repo root (e.g., tutorials/...)
                # We need to check if the notebook file path is present in the list of modified tutorials.
                relative_notebook_file=${notebook_file#./_docs/} # remove ./_docs/ prefix
                # relative_notebook_file="tutorials/${relative_notebook_file}" # prepend tutorials/
                echo "[QUARK-INFO] relative_notebook_file=${relative_notebook_file}"
                modified_by_pr=0
                for modified_tutorial in ${QUARK_SPHINX_MODIFIED_TUTORIALS}; do
                    if [[ " ${modified_tutorial} " == *" ${relative_notebook_file} "* ]]; then
                        echo "Jupyter Notebook ${notebook_file} was modified and needs to be compiled!"
                        modified_by_pr=1
                        break
                    fi
                done
                if [[ ${modified_by_pr} -eq 0 ]]; then
                    echo "Jupyter Notebook ${notebook_file} is unmodified and will be converted into ReStructuredText because QUARK_SPHINX_MODIFIED_TUTORIALS is not empty"
                    jupyter nbconvert --to rst "${notebook_file}"
                    rm -v ${notebook_file}
                fi

            fi
        done
        unset QUARK_SPHINX_SKIP_BUILD_TUTORIALS
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
