#!/bin/bash
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
    if [[ "${QUARK_SPHINX_BUILD_SKIP_TUTORIALS}" == "1" || "${QUARK_SPHINX_BUILD_SKIP_TUTORIALS,,}" == "true" || "${QUARK_SPHINX_BUILD_SKIP_TUTORIALS,,}" == "yes" ]]
    then
        echo "[QUARK-INFO] QUARK_SPHINX_BUILD_SKIP_TUTORIALS=${QUARK_SPHINX_BUILD_SKIP_TUTORIALS} was set."
        echo "[QUARK-INFO] Deleteing tutorials/* subfolder from build..."
        rm -rfv ./_docs/tutorials/
    else
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

    if [[ -n "${QUARK_DOC_FAIL_ON_WARNING}" && ( "${QUARK_DOC_FAIL_ON_WARNING}" == "1" || "${QUARK_DOC_FAIL_ON_WARNING,,}" == "true" || "${QUARK_DOC_FAIL_ON_WARNING,,}" == "yes" ) ]]; then
        echo "[QUARK-INFO] Quark documentation build will fail on warnings..."
        sphinx_build_fail_on_warning_args=" --keep-going --fail-on-warning --nitpicky"
    else
        echo "[QUARK-INFO] Quark documentation build WILL NOT fail on warnings..."
        sphinx_build_fail_on_warning_args=""
    fi

    # Using LC_ALL=C to avoid https://stackoverflow.com/questions/14547631/python-locale-error-unsupported-locale-setting
    SPHINX_BUILD_CMD="LC_ALL=C sphinx-build -M html ./_docs/ ../_docs_build/ --show-traceback ${sphinx_build_fail_on_warning_args}"
    echo "${SPHINX_BUILD_CMD}"
    bash -c "${SPHINX_BUILD_CMD}"
}

cd ${THIS_DIR}
./install_requirements.sh
build_docs
