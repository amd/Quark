#!/usr/bin/env bash
# Cold-start EAGLE-3 training via TorchSpec (streaming extraction).
set -euo pipefail
ASSET_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export ASSET_DIR
source "$ASSET_DIR/scripts/_common.sh"
CFG="${1:-/runner-config/$(basename "$TRAIN_CONFIG")}"; NAME="${2:-qsd-train}"
TRAIN_GPUS="${TRAIN_GPUS:-$(gpu_csv 0 "$WORLD_SIZE")}"
# Ray pre-starts one idle Python worker per detected CPU, and every interpreter
# start auto-imports the sitecustomize.py below, which pulls in torch (~8.5s
# wall each). On a many-core host that storm makes workers miss the raylet
# registration timeout and ray.init() never returns. Bounding Ray's advertised
# CPU count is enough: RAY_OVERRIDE_RESOURCES merges by key, so GPU
# autodetection is untouched and the container keeps all real cores for
# everything else. min(nproc, 64) makes smaller hosts a true no-op.
TRAIN_RAY_CPUS="${TRAIN_RAY_CPUS:-$(( $(nproc) < 64 ? $(nproc) : 64 ))}"
RUNTIME_ENV_ARGS=(
  -e "VLLM_ROCM_USE_AITER=$VLLM_ROCM_USE_AITER"
  -e "VLLM_USE_BREAKABLE_CUDAGRAPH=$VLLM_USE_BREAKABLE_CUDAGRAPH"
  -e "VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"
  -e "MC_STORE_MEMCPY=$MC_STORE_MEMCPY"
)
if [ "$RUNNER_PROFILE" != "qwen3_8b_quick_start" ]; then
  RUNTIME_ENV_ARGS+=(
    -e "VLLM_ROCM_USE_AITER_MOE=$VLLM_ROCM_USE_AITER_MOE"
    -e "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=$VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS"
  )
fi

# sitecustomize.py is loaded by `site.py` because the directory below lands on
# PYTHONPATH -- nothing imports it by name. If it is missing, site.py skips it
# without a word and training runs with delimiters that do not match how the
# target is served, which shows up only as a lower acceptance length. Confirm
# the file is where PYTHONPATH will look before starting the container.
[ -f "$ASSET_DIR/python/sitecustomize.py" ] || \
  die "missing $ASSET_DIR/python/sitecustomize.py; chat templates would go unregistered."

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --network=host --ipc=host --shm-size 64G \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$MODELS":/models -v "$EX_DIR":/workspace -v "$DATA_DIR":/data \
  -v "$TORCHSPEC_DIR":/workspace/TorchSpec:ro -v "$ASSET_DIR":/runner:ro \
  -v "$CONFIG_DIR":/runner-config:ro -w /workspace \
  -e "CUDA_VISIBLE_DEVICES=$TRAIN_GPUS" \
  -e "TORCHSPEC_CHAT_TEMPLATE_PROFILE=$CHAT_TEMPLATE" \
  -e RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
  -e RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 \
  -e RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 \
  -e "RAY_OVERRIDE_RESOURCES={\"CPU\": ${TRAIN_RAY_CPUS}}" \
  "${RUNTIME_ENV_ARGS[@]}" \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/cache/compiled_kernels \
  -e PYTHONPATH=/runner/python:/workspace/TorchSpec \
  --entrypoint python3 "$IMG" -m torchspec.train_entry --config "$CFG"
echo "launched $NAME (cfg=$CFG). logs: docker logs -f $NAME"
