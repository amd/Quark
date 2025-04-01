#!/bin/bash
set -e

echo "[QUARK-INFO] Delete cache files..."

if [ -e "./_docs" ]; then
  rm -rf ./_docs
fi

cp -r ./source _docs
# Delete unused files
rm -f ./_docs/readme_for_zip.md

if [ -e "../_docs_build" ]; then
  rm -rf ../_docs_build
fi

echo "[QUARK-INFO] Copy version file and example documentation..."
cp -f ../quark/version.txt ./_docs/version.txt
# Examples: copy examples documentation (*.rst) from ../examples/{torch,onnx} to ./_docs/{onnx,pytorch}
mkdir -p ./_docs/{onnx,pytorch}
find ../examples/torch -type f -name "*.rst" | while IFS= read -r mdfile; do
  cp $mdfile ./_docs/pytorch/
done
find ../examples/onnx -type f -name "*.rst" | while IFS= read -r mdfile; do
  cp $mdfile ./_docs/onnx/
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
SPHINX_BUILD_CMD="LC_ALL=C sphinx-build -M html ./_docs/ ../_docs_build/ ${sphinx_build_fail_on_warning_args}"
echo "${SPHINX_BUILD_CMD}"
bash -c "${SPHINX_BUILD_CMD}"
