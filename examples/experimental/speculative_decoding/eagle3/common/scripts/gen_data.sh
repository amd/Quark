#!/usr/bin/env bash
# Generate on-policy responses from one or more target-model endpoints.
set -euo pipefail
ASSET_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export ASSET_DIR
source "$ASSET_DIR/scripts/_common.sh"
IN="${1:?prompts jsonl (relative to run root)}"; OUT="${2:?out jsonl}"
SRV="${3:?comma-separated host:port list}"; CONC="${4:-512}"; MAXTOK="${5:-1024}"
[[ "$IN" = /* ]] || IN="/workspace/$IN"
[[ "$OUT" = /* ]] || OUT="/workspace/$OUT"
IFS=',' read -ra S <<< "$SRV"; SRV_ARGS=(--server-address "${S[@]}")

docker rm -f qsd-gen >/dev/null 2>&1 || true
docker run -d --name qsd-gen --network=host \
  -v "$EX_DIR":/workspace -v "$DATA_DIR":/data -v "$TORCHSPEC_DIR":/workspace/TorchSpec:ro -w /workspace \
  --ulimit nofile=1048576:1048576 -e PYTHONPATH=/workspace/TorchSpec \
  --entrypoint python3 "$IMG" /workspace/TorchSpec/tools/generate_data.py \
    --model "$SERVED_NAME" --temperature 0 --max-tokens "$MAXTOK" --concurrency "$CONC" \
    "${SRV_ARGS[@]}" \
    --input-file-path "$IN" --output-file-path "$OUT"
echo "launched qsd-gen: $IN -> $OUT. logs: docker logs -f qsd-gen"
