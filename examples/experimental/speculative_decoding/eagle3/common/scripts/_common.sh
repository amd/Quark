#!/usr/bin/env bash
# Shared helpers for the profile-aware EAGLE-3 runner.
#
# ASSET_DIR is the read-only common runner. EX_DIR is the writable run root,
# CACHE_DIR holds reusable dependencies, and CONFIG_DIR selects one profile's
# training assets.

: "${ASSET_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${EAGLE3_DIR:=$(cd "$ASSET_DIR/.." && pwd)}"
: "${RUNNER_PROFILE:=qwen3_8b_quick_start}"
# Neither default may land inside the asset tree: these hold multi-GB
# checkpoints, drafts and model snapshots, and the tree is packaged into the
# sdist and wheel. Matches what the Python runner passes in, so both entry
# points reuse one cache and keep run outputs separate from it.
: "${EX_DIR:=$PWD/ckpts/eagle3-$RUNNER_PROFILE}"
: "${CACHE_DIR:=${XDG_CACHE_HOME:-$HOME/.cache}/amd-quark/eagle3}"
: "${TORCHSPEC_DIR:=$CACHE_DIR/TorchSpec}"
: "${IMG:=quark-specdec-rocm:latest}"
: "${MODELS:=$CACHE_DIR/models}"
: "${DATA_DIR:=$EX_DIR/data}"
: "${BASE_MODEL:=Qwen/Qwen3-8B}"

case "$RUNNER_PROFILE" in
  qwen|qwen3_8b|qwen3_8b_quick_start)
    RUNNER_PROFILE="qwen3_8b_quick_start"
    : "${CONFIG_DIR:=$EAGLE3_DIR/qwen3_8b_quick_start/configs}"
    : "${TRAIN_CONFIG:=$CONFIG_DIR/qwen3_8b_eagle3.yaml}"
    : "${DRAFT_CONFIG_SOURCE:=$CONFIG_DIR/qwen3_8b_eagle3_draft.json}"
    : "${TARGET_TP_SIZE:=${target_tp_size:-1}}"
    : "${SERVE_EXTRA_ARGS:=}"
    : "${SERVE_PARSER_ARGS:=}"
    : "${TARGET_EMBEDDING_KEY:=model.embed_tokens.weight}"
    : "${TARGET_LM_HEAD_KEY:=lm_head.weight}"
    : "${TARGET_NORM_KEY:=model.norm.weight}"
    : "${REQUIRED_FREE_GB:=120}"
    ;;
  minimax|minimax_m3|minimax_m3_best_recipe|large_model)
    RUNNER_PROFILE="minimax_m3_best_recipe"
    : "${CONFIG_DIR:=$EAGLE3_DIR/minimax_m3_best_recipe/configs}"
    : "${TRAIN_CONFIG:=$CONFIG_DIR/large_model_eagle3.yaml}"
    : "${DRAFT_CONFIG_SOURCE:=}"
    : "${TARGET_TP_SIZE:=${target_tp_size:-4}}"
    : "${SERVE_EXTRA_ARGS:=--block-size 128 --moe-backend aiter --attention-backend TRITON_ATTN --language-model-only --no-enable-prefix-caching}"
    # Response-shaping parsers move reasoning and tool calls out of `content`.
    # They mirror deployment but must stay off while capturing on-policy data,
    # otherwise the recorded assistant text is not the token sequence the
    # target actually generates.
    : "${SERVE_PARSER_ARGS:=--tool-call-parser minimax_m3 --reasoning-parser minimax_m3 --enable-auto-tool-choice}"
    : "${TARGET_EMBEDDING_KEY:=language_model.model.embed_tokens.weight}"
    : "${TARGET_LM_HEAD_KEY:=language_model.lm_head.weight}"
    : "${TARGET_NORM_KEY:=language_model.model.norm.weight}"
    : "${REQUIRED_FREE_GB:=600}"
    ;;
  *)
    : "${CONFIG_DIR:?CONFIG_DIR is required for custom RUNNER_PROFILE=$RUNNER_PROFILE}"
    : "${TRAIN_CONFIG:?TRAIN_CONFIG is required for custom RUNNER_PROFILE=$RUNNER_PROFILE}"
    : "${DRAFT_CONFIG_SOURCE:=}"
    : "${TARGET_TP_SIZE:=${target_tp_size:-1}}"
    : "${SERVE_EXTRA_ARGS:=}"
    : "${SERVE_PARSER_ARGS:=}"
    : "${TARGET_EMBEDDING_KEY:=model.embed_tokens.weight}"
    : "${TARGET_LM_HEAD_KEY:=lm_head.weight}"
    : "${TARGET_NORM_KEY:=model.norm.weight}"
    : "${REQUIRED_FREE_GB:=120}"
    ;;
esac

: "${WORLD_SIZE:=8}"
: "${INFERENCE_NUM_GPUS:=4}"
: "${TRAINING_NUM_GPUS:=4}"
: "${TARGET_MAX_MODEL_LEN:=8192}"
: "${VLLM_ROCM_USE_AITER:=1}"
: "${VLLM_ROCM_USE_AITER_MOE:=0}"
: "${VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS:=0}"
: "${VLLM_ATTENTION_BACKEND:=TRITON_ATTN}"
: "${VLLM_USE_BREAKABLE_CUDAGRAPH:=0}"
: "${MC_STORE_MEMCPY:=0}"

MODEL_NAME="$(basename "$BASE_MODEL")"
: "${MODEL:=/models/$MODEL_NAME}"
: "${SERVED_NAME:=target}"
MODEL_SLUG="$(printf '%s' "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '-' | sed -E 's/-+/-/g; s/^-//; s/-$//')"

export ASSET_DIR EAGLE3_DIR RUNNER_PROFILE CONFIG_DIR TRAIN_CONFIG DRAFT_CONFIG_SOURCE
export EX_DIR CACHE_DIR TORCHSPEC_DIR IMG MODELS DATA_DIR
export BASE_MODEL MODEL_NAME MODEL SERVED_NAME MODEL_SLUG
export WORLD_SIZE TARGET_TP_SIZE INFERENCE_NUM_GPUS TRAINING_NUM_GPUS TARGET_MAX_MODEL_LEN
export SERVE_EXTRA_ARGS SERVE_PARSER_ARGS
export TARGET_EMBEDDING_KEY TARGET_LM_HEAD_KEY TARGET_NORM_KEY
export REQUIRED_FREE_GB
export VLLM_ROCM_USE_AITER VLLM_ROCM_USE_AITER_MOE VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS
export VLLM_ATTENTION_BACKEND VLLM_USE_BREAKABLE_CUDAGRAPH MC_STORE_MEMCPY

_c() {
  if [ -t 2 ]; then printf '\033[%sm%s\033[0m' "$1" "$2"; else printf '%s' "$2"; fi
}
info() { echo "$(_c '1;36' '==>') $*" >&2; }
warn() { echo "$(_c '1;33' 'WARN:') $*" >&2; }
die()  { echo "$(_c '1;31' 'ERROR:') $*" >&2; exit 1; }
have_cmd() { command -v "$1" >/dev/null 2>&1; }

gpu_csv() {
  local start="${1:?start GPU}" count="${2:?GPU count}" value="" i
  for ((i=0; i<count; i++)); do
    value="${value:+$value,}$((start + i))"
  done
  printf '%s\n' "$value"
}

gpu_count() {
  local n=""
  if have_cmd rocm-smi; then
    n="$(rocm-smi --showid 2>/dev/null | grep -oE 'GPU\[[0-9]+\]' | sort -u | wc -l | tr -d ' ')"
  fi
  if [ -z "$n" ] || [ "$n" = 0 ]; then
    n="$(ls /dev/dri/renderD* 2>/dev/null | wc -l | tr -d ' ')"
  fi
  echo "${n:-0}"
}

free_gb() { df -PB1 "$1" 2>/dev/null | awk 'NR==2{printf "%d", $4/1024/1024/1024}'; }

# Phases write as root inside the container, which NFS root_squash maps to
# nobody -- a directory you can write is still refused there, and it surfaces
# hours later as a bare EPERM. Probe up front with the image the phases use.
assert_container_writable() {
  local dir="${1:?directory}" label="${2:-directory}"
  mkdir -p "$dir" || die "cannot create $label: $dir"
  docker run --rm -v "$dir":/probe --entrypoint sh "$IMG" \
    -c 'touch /probe/.quark_write_probe && rm -f /probe/.quark_write_probe' >/dev/null 2>&1 || \
    die "the container cannot write to $label: $dir
       Use a local-disk path (NFS root_squash refuses container root):
         QUARK_EAGLE3_CACHE=/scratch/\$USER/amd-quark/eagle3
         training.output_dir=/scratch/\$USER/eagle3-runs/<run-name>"
}

preflight_setup() {
  have_cmd docker || die "docker not found. Install Docker with ROCm GPU access."
  docker info >/dev/null 2>&1 || die "cannot reach the docker daemon (is it running? are you in the 'docker' group?)."
  have_cmd git || die "git not found (needed to clone TorchSpec)."
  { [ -e /dev/kfd ] && [ -e /dev/dri ]; } || \
    warn "/dev/kfd or /dev/dri is missing — ROCm GPUs may not be visible inside containers."
  local g; g="$(gpu_count)"
  mkdir -p "$CACHE_DIR"
  info "Detected $g GPU render node(s); $(free_gb "$CACHE_DIR")GB free at $CACHE_DIR."
  [ "${g:-0}" -ge "$WORLD_SIZE" ] || \
    die "this runner profile requires $WORLD_SIZE visible AMD GPUs (detected ${g:-0})."
  [ "$(free_gb "$CACHE_DIR")" -ge "$REQUIRED_FREE_GB" ] 2>/dev/null || \
    warn "less than ${REQUIRED_FREE_GB}GB free at $CACHE_DIR — model + data + checkpoints may not fit."
}

preflight_run() {
  docker image inspect "$IMG" >/dev/null 2>&1 || \
    die "image '$IMG' not found. Run the one-time Python setup command for $BASE_MODEL."
  [ -f "$MODELS/$MODEL_NAME/config.json" ] || \
    die "$MODEL_NAME weights not found under $MODELS. Run the one-time Python setup command."
  [ -d "$TORCHSPEC_DIR/.git" ] || \
    die "TorchSpec not found under $TORCHSPEC_DIR. Run the one-time Python setup command."
  [ -f "$TRAIN_CONFIG" ] || die "training config not found: $TRAIN_CONFIG"
  # Both are bind-mounted into every phase container.
  assert_container_writable "$CACHE_DIR" "CACHE_DIR"
  assert_container_writable "$EX_DIR" "the run output directory (training.output_dir)"
  local g; g="$(gpu_count)"
  [ "${g:-0}" -ge "$WORLD_SIZE" ] || \
    die "this runner profile requires $WORLD_SIZE visible AMD GPUs (detected ${g:-0})."
}

wait_ready() {
  local tries="${2:-120}" hp
  IFS=',' read -ra _S <<< "$1"
  for hp in "${_S[@]}"; do
    local ok=0 i
    for ((i=0; i<tries; i++)); do
      if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://$hp/v1/models" 2>/dev/null)" = "200" ]; then
        ok=1
        break
      fi
      sleep 5
    done
    [ "$ok" = 1 ] || die "server $hp did not become ready in time. Check: docker logs <name>"
  done
}

follow_until_exit() {
  local name="$1" hb="${2:-60}" t0 el last
  t0="$(date +%s)"
  info "$name is running — live logs:  docker logs -f $name"
  while [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" = "true" ]; do
    sleep "$hb"
    el=$(( ($(date +%s) - t0) / 60 ))
    last="$(
      docker logs --since "${hb}s" "$name" 2>&1 \
        | tr '\r' '\n' \
        | awk 'NF { line=$0 } END { print substr(line, 1, 500) }'
    )"
    info "[$name] ~${el}m elapsed — ${last:-（still warming up）}"
  done
  docker inspect -f '{{.State.ExitCode}}' "$name" 2>/dev/null || echo 1
}

runner_input_sha256() {
  [ "$#" -gt 0 ] || die "runner_input_sha256 requires at least one file"
  local path
  {
    for path in "$@"; do
      [ -f "$path" ] || die "run identity input is missing: $path"
      sha256sum "$path"
    done
  } | sha256sum | awk '{print $1}'
}

resolve_export_checkpoint() {
  local ckpt_dir="${1:?checkpoint directory}"
  local best_file="$ckpt_dir/best_checkpointed_iteration.txt"
  local iter_dir=""

  if [ -f "$best_file" ]; then
    local iter_id
    iter_id="$(tr -d '[:space:]' < "$best_file")"
    if [[ "$iter_id" =~ ^[0-9]+$ ]]; then
      iter_dir="$ckpt_dir/iter_$(printf '%07d' "$iter_id")"
      if [ -d "$iter_dir/model" ]; then
        info "export checkpoint: best iter_$(printf '%07d' "$iter_id") (TorchSpec best_checkpointed_iteration.txt)"
        printf '%s\n' "$iter_dir"
        return 0
      fi
      warn "best_checkpointed_iteration.txt points to an incomplete iter_$(printf '%07d' "$iter_id"); using latest complete checkpoint"
    else
      warn "invalid best_checkpointed_iteration.txt contents; using latest checkpoint"
    fi
  fi

  local candidate
  iter_dir=""
  for candidate in "$ckpt_dir"/iter_*; do
    [ -d "$candidate/model" ] || continue
    iter_dir="$candidate"
  done
  if [ -n "$iter_dir" ]; then
    info "export checkpoint: latest $(basename "$iter_dir")"
    printf '%s\n' "$iter_dir"
  fi
}

QSD_CONTAINERS="qsd-gen qsd-gen-0 qsd-gen-1 qsd-gen-2 qsd-gen-3 qsd-gen-4 qsd-gen-5 qsd-gen-6 qsd-gen-7 qsd-train qsd-base qsd-spec"
cleanup_containers() { docker rm -f $QSD_CONTAINERS >/dev/null 2>&1 || true; }
