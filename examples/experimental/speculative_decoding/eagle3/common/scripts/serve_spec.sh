#!/usr/bin/env bash
# Serve the target model with a trained EAGLE-3 draft.
set -euo pipefail
ASSET_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export ASSET_DIR
source "$ASSET_DIR/scripts/_common.sh"
NAME="${1:?name}"; PORT="${2:?port}"; DRAFT="${3:?draft dir (host path under run root)}"
TP="${4:-$TARGET_TP_SIZE}"; NST="${5:-3}"
DRAFT_CONTAINER="/workspace/${DRAFT#"$EX_DIR/"}"
read -r -a PROFILE_SERVE_ARGS <<< "$SERVE_EXTRA_ARGS"
if [ "${ENABLE_RESPONSE_PARSERS:-0}" = "1" ] && [ -n "$SERVE_PARSER_ARGS" ]; then
  read -r -a PROFILE_PARSER_ARGS <<< "$SERVE_PARSER_ARGS"
  PROFILE_SERVE_ARGS+=("${PROFILE_PARSER_ARGS[@]}")
fi
RUNTIME_ENV_ARGS=(-e "VLLM_ROCM_USE_AITER=$VLLM_ROCM_USE_AITER")
if [ "$RUNNER_PROFILE" != "qwen3_8b_quick_start" ]; then
  RUNTIME_ENV_ARGS+=(
    -e "VLLM_ROCM_USE_AITER_MOE=$VLLM_ROCM_USE_AITER_MOE"
    -e "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=$VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS"
    -e "VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"
  )
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --network=host --ipc=host --shm-size 16G \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$MODELS":/models -v "$EX_DIR":/workspace \
  ${GPUS:+-e CUDA_VISIBLE_DEVICES=$GPUS} \
  "${RUNTIME_ENV_ARGS[@]}" \
  --entrypoint vllm "$IMG" \
  serve "$MODEL" --served-model-name "$SERVED_NAME" --trust-remote-code \
    --tensor-parallel-size "$TP" --max-model-len "$TARGET_MAX_MODEL_LEN" \
    --host 0.0.0.0 --port "$PORT" "${PROFILE_SERVE_ARGS[@]}" \
    --speculative-config "{\"method\":\"eagle3\",\"model\":\"${DRAFT_CONTAINER}\",\"num_speculative_tokens\":${NST}}"
echo "launched $NAME (spec, tp=$TP port=$PORT, nst=$NST, draft=$DRAFT_CONTAINER). logs: docker logs -f $NAME"
