#!/usr/bin/env bash
# One-time setup shared by every EAGLE-3 runner profile.
set -euo pipefail
ASSET_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export ASSET_DIR

while [ $# -gt 0 ]; do
  case "$1" in
    --base_model) BASE_MODEL="${2:?--base_model needs a value}"; shift 2;;
    --base_model=*) BASE_MODEL="${1#*=}"; shift;;
    -h|--help) echo "usage: bash scripts/00_setup.sh [--base_model <hf-id-or-local-dir>]"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 1;;
  esac
done
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
source "$ASSET_DIR/scripts/_common.sh"

info "Step 0/3: checking prerequisites"
preflight_setup
mkdir -p "$CACHE_DIR" "$MODELS"

TORCHSPEC_REPO="${TORCHSPEC_REPO:-https://github.com/lightseekorg/TorchSpec.git}"
TORCHSPEC_COMMIT="${TORCHSPEC_COMMIT:-d230a3f13212e8a3eb35e04f99fbdd8fa45241f0}"
if [ -d "$TORCHSPEC_DIR/.git" ]; then
  [ -z "$(git -C "$TORCHSPEC_DIR" status --porcelain)" ] || \
    die "TorchSpec checkout has local changes: $TORCHSPEC_DIR"
else
  info "Step 1/3: cloning TorchSpec -> $TORCHSPEC_DIR"
  git clone --filter=blob:none "$TORCHSPEC_REPO" "$TORCHSPEC_DIR"
fi
if [ "$(git -C "$TORCHSPEC_DIR" rev-parse HEAD)" = "$TORCHSPEC_COMMIT" ]; then
  info "Step 1/3: TorchSpec pinned at $TORCHSPEC_COMMIT — skipping checkout"
else
  info "Step 1/3: checking out validated TorchSpec revision $TORCHSPEC_COMMIT"
  git -C "$TORCHSPEC_DIR" fetch --depth 1 origin "$TORCHSPEC_COMMIT"
  git -C "$TORCHSPEC_DIR" checkout --detach "$TORCHSPEC_COMMIT"
fi

if [ "${FORCE_BUILD:-0}" != "1" ] && docker image inspect "$IMG" >/dev/null 2>&1; then
  info "Step 2/3: image $IMG already exists — skipping"
else
  info "Step 2/3: building image $IMG (cached layers make repeat runs fast)"
  # Empty means "use the Dockerfile's own pin"; an empty --build-arg would
  # override it with nothing. (An `&&` list here would trip `set -e`.)
  if [ -n "${VLLM_ROCM_IMAGE:-}" ]; then
    info "base image override: $VLLM_ROCM_IMAGE"
  fi
  docker build -f "$ASSET_DIR/docker/Dockerfile.rocm" -t "$IMG" \
    ${VLLM_ROCM_IMAGE:+--build-arg VLLM_ROCM_IMAGE="$VLLM_ROCM_IMAGE"} \
    "$ASSET_DIR/docker"
fi

mkdir -p "$MODELS"
# Check container write access before spending tens of GB of network on it.
assert_container_writable "$MODELS" "MODELS ($CACHE_DIR/models)"
if [ -f "$MODELS/$MODEL_NAME/config.json" ]; then
  info "Step 3/3: $MODEL_NAME already present under $MODELS — skipping"
elif [ -f "$BASE_MODEL/config.json" ]; then
  info "Step 3/3: copying local target $BASE_MODEL -> $MODELS/$MODEL_NAME"
  mkdir -p "$MODELS/$MODEL_NAME"
  cp -a --reflink=auto "$BASE_MODEL/." "$MODELS/$MODEL_NAME/"
else
  info "Step 3/3: downloading $BASE_MODEL -> $MODELS/$MODEL_NAME (one time)"
  # Both names arrive as argv rather than being interpolated into the source:
  # a repository id carrying a quote would otherwise close the string literal
  # and run whatever follows it inside the container.
  docker run --rm -v "$MODELS":/models -e HF_HUB_DISABLE_XET=1 --entrypoint python3 "$IMG" -c \
    "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir='/models/' + sys.argv[2], ignore_patterns=['*.pth','original/*','*.gguf'])" \
    "$BASE_MODEL" "$MODEL_NAME"
fi

echo
info "Setup complete. BASE_MODEL=$BASE_MODEL IMG=$IMG CACHE_DIR=$CACHE_DIR"
info "Next (pass the same --base_model you used here):"
echo "    python3 -m quark.experimental.speculative_decoding.run --base_model $BASE_MODEL"
