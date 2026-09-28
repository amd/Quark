#!/usr/bin/env bash
# Convert an FSDP checkpoint to a vLLM-loadable EAGLE-3 draft.
set -euo pipefail
ASSET_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export ASSET_DIR
source "$ASSET_DIR/scripts/_common.sh"
ITER="${1:?checkpoint iter dir (relative to run root)}"; OUT="${2:?output dir}"
DRAFT_CONFIG="${DRAFT_CONFIG:-/workspace/runtime/draft_config.json}"
EMBEDDING_PREFIX="${TARGET_EMBEDDING_KEY:-model.embed_tokens.weight}"
EMBEDDING_PREFIX="${EMBEDDING_PREFIX%.weight}"

docker run --rm --network=host -v "$MODELS":/models -v "$EX_DIR":/workspace \
  -v "$TORCHSPEC_DIR":/workspace/TorchSpec:ro -w /workspace \
  -e PYTHONPATH=/workspace/TorchSpec --entrypoint python3 "$IMG" \
  TorchSpec/tools/convert_to_hf.py \
    --input-dir "/workspace/$ITER/model" \
    --output-dir "/workspace/$OUT" \
    --config "$DRAFT_CONFIG" \
    --target-model-path "$MODEL" \
    --embedding-key "$EMBEDDING_PREFIX" \
    --trust-remote-code --vllm --force
echo "converted -> $OUT"
