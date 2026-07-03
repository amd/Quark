#!/usr/bin/env bash
#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# One-shot DeepSeek-V4-Pro -> NVFP4 pipeline (Stage 1 quantize -> 2 calibrate ->
# 3 merge -> 4 PPL eval). Aborts on the first failure.
#
#   SRC=/path/to/DeepSeek-V4-Pro OUT=/path/to/output-nvfp4 bash run_pipeline.sh
#   N_BLOCKS=5 SRC=... OUT=... bash run_pipeline.sh   # quick smoke test on N layers
#
# Env vars: SRC, OUT (required); QUARK (auto-detected), GPU (0), N_BLOCKS (all),
# CALIB_BATCH, N_CALIB_SAMPLES, CALIB_SEQLEN, PPL_BATCH, PPL_MAX_CHUNKS. See README.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUARK="${QUARK:-$(cd "$HERE/../../../../../.." && pwd)}"  # repo root, 6 levels up
SRC="${SRC:?Set SRC to the source DeepSeek-V4-Pro checkpoint directory}"
OUT="${OUT:?Set OUT to the desired NVFP4 output directory}"
GPU="${GPU:-0}"
N_BLOCKS="${N_BLOCKS:-}"
INPUT_SCALE="$OUT/input_scale.safetensors"

export PYTHONPATH="$QUARK${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"

# Smaller calibration/eval defaults in smoke mode; overridable.
if [[ -n "$N_BLOCKS" ]]; then
    CALIB_BATCH="${CALIB_BATCH:-4}"; N_CALIB_SAMPLES="${N_CALIB_SAMPLES:-8}"
    CALIB_SEQLEN="${CALIB_SEQLEN:-256}"; PPL_BATCH="${PPL_BATCH:-2}"
else
    CALIB_BATCH="${CALIB_BATCH:-48}"; N_CALIB_SAMPLES="${N_CALIB_SAMPLES:-64}"
    CALIB_SEQLEN="${CALIB_SEQLEN:-512}"; PPL_BATCH="${PPL_BATCH:-4}"
fi
PPL_MAX_CHUNKS="${PPL_MAX_CHUNKS:-}"

# Prerequisite: the source checkpoint must keep its inference/ model code.
[[ -d "$SRC/inference" ]] || { echo "ERROR: $SRC/inference not found (need model.py + config.json)"; exit 1; }

# Smoke mode: build a source dir with only layers 0..N-1 (symlinked shards).
QUANT_SRC="$SRC"
NBLK_ARG=()
if [[ -n "$N_BLOCKS" ]]; then
    QUANT_SRC="$OUT.smoke_in"
    python - "$SRC" "$N_BLOCKS" "$QUANT_SRC" <<'PY'
import json, re, os, sys
src, n, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
wm = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
pat = re.compile(r"(?:^|\.)layers\.(\d+)\.")
keep = {s for k, s in wm.items() if (m := pat.search(k)) is None or int(m.group(1)) < n}
os.makedirs(dst, exist_ok=True)
for e in os.listdir(src):
    if e.endswith(".safetensors") and e not in keep:
        continue
    d = os.path.join(dst, e)
    if not os.path.lexists(d):
        os.symlink(os.path.join(src, e), d)
print(f"smoke input dir: {dst} ({len(keep)} shards)")
PY
    NBLK_ARG=(--n-blocks "$N_BLOCKS")
fi

echo "### [1/4] Stage 1 — weight quantization"
python "$HERE/stage1_quantize_weight.py" --input-model-path "$QUANT_SRC" --output-path "$OUT" --device cuda

echo "### [2/4] Stage 2 — input_scale calibration"
python "$HERE/stage2_calibrate_input_scale.py" \
    --model-dir "$SRC" --batch-size "$CALIB_BATCH" --n-calib-samples "$N_CALIB_SAMPLES" \
    --calib-seqlen "$CALIB_SEQLEN" "${NBLK_ARG[@]}" --output "$INPUT_SCALE"

echo "### [3/4] Stage 3 — merge input_scale"
python "$HERE/stage3_merge.py" --checkpoint-path "$OUT" --input-scale-path "$INPUT_SCALE"

echo "### [4/4] Stage 4 — weight-only PPL eval"
[[ -e "$OUT/inference" ]] || ln -s "$SRC/inference" "$OUT/inference"  # Stage 1 doesn't copy model code
PPL_ARGS=(--model-dir "$OUT" --batch-size "$PPL_BATCH")
[[ -n "$N_BLOCKS" ]] && PPL_ARGS+=(--n-blocks "$N_BLOCKS" --seqlen "$CALIB_SEQLEN")
[[ -n "$PPL_MAX_CHUNKS" ]] && PPL_ARGS+=(--max-chunks "$PPL_MAX_CHUNKS")
python "$HERE/stage4_ppl.py" "${PPL_ARGS[@]}"

echo "DONE — final NVFP4 checkpoint at: $OUT"
