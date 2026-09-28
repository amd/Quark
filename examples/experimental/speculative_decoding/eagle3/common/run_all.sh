#!/usr/bin/env bash
# Profile-aware EAGLE-3 speculative decoding on one AMD Instinct node:
#   on-policy data -> cold-start training -> convert -> measured speedup.
#
# Profile selectors set RUNNER_PROFILE, CONFIG_DIR, TRAIN_CONFIG and
# TARGET_TP_SIZE, then delegate here. Qwen3-8B remains the default.
#
# Usage:
#   bash run_all.sh                                  # full run, default target Qwen/Qwen3-8B
#   bash run_all.sh --base_model Qwen/Qwen3-8B       # pick any HF causal-LM target
#   PROFILE=quick bash run_all.sh --base_model ...   # fast plumbing check
#
# Handy knobs (all optional, via env):
#   NUM_PROMPTS=  EPOCHS=  EVAL_N=  BENCH_N=          # override profile defaults
#   RUNNER_PROFILE=  CONFIG_DIR=  TARGET_TP_SIZE=     # model/profile assets
#   CHAT_TEMPLATE=qwen                                # serving template family
#   SKIP_DATA=1   SKIP_TRAIN=1  SKIP_BENCH=1          # reuse earlier phases
#   PROMPT_OPENS_THINK=1                              # template opens <think> in the prompt
#   TRAIN_MAX_SEQ_LEN=  TRAIN_MAX_BATCHED_TOKENS=     # training limits; the first
#   MOONCAKE_SEGMENT_SIZE=                            #   must exceed generation max tokens
#   AUTO_RESUME=0                                     # retrain from scratch

set -euo pipefail
ASSET_DIR="$(cd "$(dirname "$0")" && pwd)"
export ASSET_DIR

# Parse --base_model (optional) before sourcing _common so paths derive from it.
while [ $# -gt 0 ]; do
  case "$1" in
    --base_model) BASE_MODEL="${2:?--base_model needs a value}"; shift 2;;
    --base_model=*) BASE_MODEL="${1#*=}"; shift;;
    -h|--help) echo "usage: [PROFILE=quick] bash run_all.sh [--base_model <hf-id-or-local-dir>]"; exit 0;;
    *) echo "unknown argument: $1 (did you mean --base_model?)" >&2; exit 1;;
  esac
done
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
source "$ASSET_DIR/scripts/_common.sh"

# --- Workload profile: defaults remain the validated Qwen CLI defaults -------
PROFILE="${PROFILE:-full}"
case "$PROFILE" in
  quick) NUM_PROMPTS="${NUM_PROMPTS:-2000}";   EPOCHS="${EPOCHS:-1}"; EVAL_N="${EVAL_N:-64}";  EVAL_INTERVAL="${EVAL_INTERVAL:-100}"; BENCH_N="${BENCH_N:-16}";;
  full)  NUM_PROMPTS="${NUM_PROMPTS:-150000}"; EPOCHS="${EPOCHS:-2}"; EVAL_N="${EVAL_N:-256}"; EVAL_INTERVAL="${EVAL_INTERVAL:-500}"; BENCH_N="${BENCH_N:-40}";;
  *) die "unknown PROFILE='$PROFILE' (use 'full' or 'quick').";;
esac
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen}"

# Writable paths are rooted at EX_DIR; packaged ASSET_DIR remains read-only.
OUT_REL="${TRAIN_DIR_REL:-training}"
RELEASE_REL="${RELEASE_DIR_REL:-release/draft_hf}"
RUNTIME_REL="runtime"
DRAFT_CFG_REL="$RUNTIME_REL/draft_config.json"
ACTIVE_CFG_REL="$RUNTIME_REL/active_config.yaml"
RESUME_CFG_REL="$RUNTIME_REL/resume_config.yaml"
OUT_DIR="$EX_DIR/$OUT_REL"
RELEASE_DIR="$EX_DIR/$RELEASE_REL"
DRAFT_CFG="$EX_DIR/$DRAFT_CFG_REL"
ACTIVE_CFG="$EX_DIR/$ACTIVE_CFG_REL"
RESUME_CFG="$EX_DIR/$RESUME_CFG_REL"
DATA_ALL="$DATA_DIR/onpolicy_all.jsonl"
TRAIN="$DATA_DIR/onpolicy_train.jsonl"
EVAL="$DATA_DIR/onpolicy_eval.jsonl"
REPORT_PATH="${REPORT_PATH:-$EX_DIR/report.json}"
PROMPT_DATASET="${PROMPT_DATASET:-allenai/tulu-3-sft-mixture}"
GENERATION_MAX_TOKENS="${GENERATION_MAX_TOKENS:-4096}"
BENCH_ROUNDS="${BENCH_ROUNDS:-3}"
NST="${NST:-3}"
# The gate default depends on the profile and the workload: the Qwen gates are
# validated for that target at full size, a smoke run has too little data to
# mean anything, and other targets report observed metrics instead. Compute the
# default, then let `:-` honour a value the caller or recipe set explicitly --
# assigning first and overwriting after would discard it silently.
if [ "$PROFILE" = "quick" ] || [ "$RUNNER_PROFILE" != "qwen3_8b_quick_start" ]; then
  _default_min_speedup=0
  _default_min_served_al=0
else
  _default_min_speedup=1.25
  _default_min_served_al=2.35
fi
MIN_SPEEDUP="${MIN_SPEEDUP:-$_default_min_speedup}"
MIN_SERVED_AL="${MIN_SERVED_AL:-$_default_min_served_al}"

preflight_run
[ "$TARGET_TP_SIZE" -ge 1 ] 2>/dev/null || die "TARGET_TP_SIZE must be a positive integer."
[ $((WORLD_SIZE % TARGET_TP_SIZE)) -eq 0 ] || \
  die "WORLD_SIZE=$WORLD_SIZE must be divisible by TARGET_TP_SIZE=$TARGET_TP_SIZE."
trap cleanup_containers EXIT INT TERM
mkdir -p "$DATA_DIR" "$OUT_DIR" "$RELEASE_DIR" "$EX_DIR/$RUNTIME_REL" "$CACHE_DIR"

info "target=$BASE_MODEL runner_profile=$RUNNER_PROFILE workload=$PROFILE target_tp=$TARGET_TP_SIZE"
info "prompts=$NUM_PROMPTS epochs=$EPOCHS eval=$EVAL_N bench=$BENCH_N rounds=$BENCH_ROUNDS image=$IMG"
info "run_dir=$EX_DIR data_cache=$DATA_DIR"
[ "$PROFILE" = quick ] && warn "QUICK is a plumbing check on tiny data — expect a LOW acceptance length / speedup. Use the full run for real numbers."

# A profile may provide a static validated draft config. Other targets derive
# public architecture dimensions from their downloaded config.json.
if [ "$MODEL_NAME" = "Qwen3-8B" ] && [ -n "$DRAFT_CONFIG_SOURCE" ] && [ -f "$DRAFT_CONFIG_SOURCE" ]; then
  info "draft config: using profile asset $DRAFT_CONFIG_SOURCE"
  cp "$DRAFT_CONFIG_SOURCE" "$DRAFT_CFG"
else
  info "draft config: deriving from $MODEL_NAME/config.json -> $DRAFT_CFG"
  docker run --rm -v "$MODELS":/models -v "$EX_DIR":/workspace -v "$ASSET_DIR":/runner:ro \
    -w /workspace --entrypoint python3 "$IMG" \
    /runner/scripts/make_draft_config.py --target "$MODEL" --out "/workspace/$DRAFT_CFG_REL"
fi
AUX_HIDDEN_LAYERS="$(
  python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c.get("eagle_aux_hidden_state_layer_ids", ""))' \
    "$DRAFT_CFG"
)"

# --- Phase 1: prompts + on-policy data (WORLD_SIZE / target TP serves) -------
if [ "${SKIP_DATA:-0}" = "1" ] || { [ -s "$TRAIN" ] && [ -s "$EVAL" ]; }; then
  # Name the rows that will be trained on; the summary above still carries the
  # requested workload size, which a cached dataset need not match. SKIP_DATA
  # reaches this branch whether or not the files are there, so only count them
  # when they exist -- a bare redirect would print a shell error and an empty
  # count in the one case where the operator most needs a clear message.
  if [ -s "$TRAIN" ] && [ -s "$EVAL" ]; then
    info "Phase 1/4: reusing cached on-policy data in $DATA_DIR ($(wc -l < "$TRAIN") train / $(wc -l < "$EVAL") eval rows)"
  else
    warn "Phase 1/4: SKIP_DATA=1 but $DATA_DIR holds no finalized train/eval data; later phases will fail"
  fi
else
  info "Phase 1/4: on-policy data generation ($NUM_PROMPTS prompts)"
  PROMPTS="$DATA_DIR/prompts.jsonl"
  if [ "${DOMAIN_MANIFEST_ACTIVE:-0}" = "1" ]; then
    GENERATE_N="$(wc -l < "$PROMPTS")"
    NUM_PROMPTS="$GENERATE_N"
    info "using $GENERATE_N generic-domain prompts prepared from $DOMAIN_MANIFEST"
  else
    # Thinking-model generations that hit max_tokens are discarded below.
    # Generate a reserve so the Qwen path still yields exactly NUM_PROMPTS.
    GENERATE_N=$(( (NUM_PROMPTS * 110 + 99) / 100 ))
    if [ -s "$PROMPTS" ] && [ "$(wc -l < "$PROMPTS")" -ge "$GENERATE_N" ]; then
      info "reusing $GENERATE_N prepared prompts from $PROMPTS"
    else
      docker run --rm -v "$DATA_DIR":/data -v "$ASSET_DIR":/runner:ro -w /workspace \
        --entrypoint python3 "$IMG" /runner/scripts/prepare_prompts.py \
        --dataset "$PROMPT_DATASET" --n "$GENERATE_N" --out /data/prompts.jsonl
    fi
  fi
  TARGET_REPLICAS=$((WORLD_SIZE / TARGET_TP_SIZE))
  SRV=""
  for ((i=0; i<TARGET_REPLICAS; i++)); do
    GPUS="$(gpu_csv $((i * TARGET_TP_SIZE)) "$TARGET_TP_SIZE")" \
      bash "$ASSET_DIR/scripts/serve_target.sh" qsd-gen-$i $((8000+i)) "$TARGET_TP_SIZE" >/dev/null
    SRV="$SRV,localhost:$((8000+i))"
  done
  SRV="${SRV#,}"
  info "waiting for $TARGET_REPLICAS target serve(s) to load $MODEL_NAME..."
  wait_ready "$SRV"
  bash "$ASSET_DIR/scripts/gen_data.sh" /data/prompts.jsonl /data/onpolicy_all.jsonl "$SRV" 512 "$GENERATION_MAX_TOKENS"
  rc="$(follow_until_exit qsd-gen 30)"
  if [ "$rc" != 0 ]; then
    docker logs qsd-gen >&2 2>&1 || true
    die "data generation failed (exit $rc)."
  fi
  for ((i=0; i<TARGET_REPLICAS; i++)); do docker rm -f qsd-gen-$i >/dev/null 2>&1; done
  docker rm -f qsd-gen >/dev/null 2>&1 || true
  SHORTFALL_ARG=()
  [ "${DOMAIN_MANIFEST_ACTIVE:-0}" = "1" ] && SHORTFALL_ARG=(--allow-shortfall)
  # Set for targets whose chat template opens <think> in the assistant header;
  # without it the thinking-balance filter keeps only truncated generations.
  PROMPT_THINK_ARG=()
  [ "${PROMPT_OPENS_THINK:-0}" = "1" ] && PROMPT_THINK_ARG=(--prompt-opens-think)
  docker run --rm -v "$DATA_DIR":/data -v "$ASSET_DIR":/runner:ro \
    --entrypoint python3 "$IMG" /runner/scripts/finalize_onpolicy_data.py \
      --input /data/onpolicy_all.jsonl \
      --train /data/onpolicy_train.jsonl \
      --eval /data/onpolicy_eval.jsonl \
      --manifest-out /data/manifest.json \
      --requested "$NUM_PROMPTS" --eval-size "$EVAL_N" \
      --generation-max-tokens "$GENERATION_MAX_TOKENS" --seed 0 \
      "${SHORTFALL_ARG[@]}" "${PROMPT_THINK_ARG[@]}"
fi

# --- Phase 2: cold-start EAGLE-3 training (8 GPUs, streaming) ----------------
# Derive an active config first; its hash is the resume/reuse identity.
sed -e "s#^\(\s*target_model_path:\).*#\1 ${MODEL}#" \
    -e "s#^\(\s*draft_model_config:\).*#\1 /workspace/${DRAFT_CFG_REL}#" \
    -e "s#^\(\s*embedding_key:\).*#\1 ${TARGET_EMBEDDING_KEY:-model.embed_tokens.weight}#" \
    -e "s#^\(\s*lm_head_key:\).*#\1 ${TARGET_LM_HEAD_KEY:-lm_head.weight}#" \
    -e "s#^\(\s*norm_key:\).*#\1 ${TARGET_NORM_KEY:-model.norm.weight}#" \
    -e "s#^\(\s*train_data_path:\).*#\1 /data/onpolicy_train.jsonl#" \
    -e "s#^\(\s*eval_data_path:\).*#\1 /data/onpolicy_eval.jsonl#" \
    -e "s#^\(output_dir:\).*#\1 /workspace/${OUT_REL}#" \
    -e "s/^\(\s*chat_template:\).*/\1 ${CHAT_TEMPLATE}/" \
    -e "s/^\(\s*num_epochs:\).*/\1 ${EPOCHS}/" \
    -e "s/^\(\s*eval_interval:\).*/\1 ${EVAL_INTERVAL}/" \
    -e "s/^\(\s*training_num_gpus_per_node:\).*/\1 ${TRAINING_NUM_GPUS}/" \
    -e "s/^\(\s*inference_num_gpus:\).*/\1 ${INFERENCE_NUM_GPUS}/" \
    -e "s/^\(\s*inference_num_gpus_per_engine:\).*/\1 ${TARGET_TP_SIZE}/" \
    -e "s/^\(\s*tp_size:\).*/\1 ${TARGET_TP_SIZE}/" \
    -e "s/^\(\s*aux_hidden_states_layers:\).*/\1 ${AUX_HIDDEN_LAYERS}/" \
    -e "s/^\(\s*max_seq_length:\).*/\1 ${TRAIN_MAX_SEQ_LEN:-8192}/" \
    -e "s/^\(\s*max_seq_len:\).*/\1 ${TRAIN_MAX_SEQ_LEN:-8192}/" \
    -e "s/^\(\s*max_num_batched_tokens:\).*/\1 ${TRAIN_MAX_BATCHED_TOKENS:-8192}/" \
    -e "s/^\(\s*global_segment_size:\).*/\1 ${MOONCAKE_SEGMENT_SIZE:-32GB}/" \
    -e "s#^\(\s*load_path:\).*#\1 null#" \
    "$TRAIN_CONFIG" > "$ACTIVE_CFG"
# Reuse is valid only for the exact resolved config, draft architecture, target
# metadata, and train/eval bytes. Hashing just active_config.yaml would silently
# reuse a checkpoint after the content-addressed data cache changed.
ACTIVE_HASH="$(
  runner_input_sha256 \
    "$ACTIVE_CFG" \
    "$DRAFT_CFG" \
    "$MODELS/$MODEL_NAME/config.json" \
    "$TRAIN" \
    "$EVAL"
)"
TRAIN_STAMP="$OUT_DIR/.quark_complete_config_sha256"
START_STAMP="$OUT_DIR/.quark_started_config_sha256"
CKPT_TRACKER="$OUT_DIR/checkpoints/latest_checkpointed_iteration.txt"
AUTO_RESUME="${AUTO_RESUME:-1}"

# Newest checkpoint id, empty when there is none.
latest_iter() { [ -f "$CKPT_TRACKER" ] && tr -dc '0-9' < "$CKPT_TRACKER" || true; }

# Resume instead of redoing hours of training. Only this config's own
# checkpoints qualify: the start stamp identifies them, and BASELINE_ITER
# marks foreign ones already on disk.
resolve_train_cfg() {
  local iter; iter="$(latest_iter)"
  if [ "$AUTO_RESUME" = 0 ] || [ -z "$iter" ] || [ "$iter" = "$BASELINE_ITER" ] || \
     [ ! -d "$OUT_DIR/checkpoints/iter_$(printf '%07d' "$((10#$iter))")/model" ]; then
    printf '%s\n' "$ACTIVE_CFG_REL"
    return
  fi
  sed -e "s#^\(\s*load_path:\).*#\1 /workspace/${OUT_REL}/checkpoints#" "$ACTIVE_CFG" > "$RESUME_CFG"
  info "resuming from iter_$(printf '%07d' "$((10#$iter))") (set AUTO_RESUME=0 to train from scratch)"
  printf '%s\n' "$RESUME_CFG_REL"
}

if [ "${SKIP_TRAIN:-0}" = "1" ] || \
   { [ -f "$TRAIN_STAMP" ] && [ "$(<"$TRAIN_STAMP")" = "$ACTIVE_HASH" ] && \
     compgen -G "$OUT_DIR/checkpoints/iter_*" >/dev/null; }; then
  info "Phase 2/4: reusing completed checkpoints in $OUT_DIR"
else
  info "Phase 2/4: cold-start EAGLE-3 training ($EPOCHS epoch(s))"
  if [ -f "$START_STAMP" ] && [ "$(<"$START_STAMP")" = "$ACTIVE_HASH" ]; then
    BASELINE_ITER=""
  else
    BASELINE_ITER="$(latest_iter)"
    if [ -n "$BASELINE_ITER" ]; then
      warn "checkpoints under $OUT_DIR are from a different config; training from scratch"
    fi
  fi
  printf '%s\n' "$ACTIVE_HASH" > "$START_STAMP"
  TRAIN_MAX_ATTEMPTS="${TRAIN_MAX_ATTEMPTS:-3}"
  attempt=1
  while true; do
    info "training attempt $attempt/$TRAIN_MAX_ATTEMPTS"
    bash "$ASSET_DIR/scripts/train.sh" "/workspace/$(resolve_train_cfg)" qsd-train
    rc="$(follow_until_exit qsd-train 120)"
    if [ "$rc" = 0 ]; then
      break
    fi
    docker logs qsd-train >&2 2>&1 || true
    if [ "$attempt" -ge "$TRAIN_MAX_ATTEMPTS" ]; then
      die "training failed after $TRAIN_MAX_ATTEMPTS attempts (last exit $rc)."
    fi
    warn "training actor exited unexpectedly (exit $rc); retrying in 15s (from the newest checkpoint if one was saved)"
    docker rm -f qsd-train >/dev/null 2>&1 || true
    attempt=$((attempt + 1))
    sleep 15
  done
  printf '%s\n' "$ACTIVE_HASH" > "$TRAIN_STAMP"
fi

# --- Phase 3: convert best checkpoint to a vLLM draft -----------------------
info "Phase 3/4: convert checkpoint -> vLLM draft"
CKPT_DIR="$OUT_DIR/checkpoints"
ITER="$(resolve_export_checkpoint "$CKPT_DIR")"
[ -n "$ITER" ] || die "no checkpoint found under ${CKPT_DIR}/"
DRAFT_CONFIG="/workspace/$DRAFT_CFG_REL" \
  bash "$ASSET_DIR/scripts/convert.sh" "${ITER#"$EX_DIR/"}" "$RELEASE_REL"

# --- Phase 4: baseline + spec serves, measure speedup ----------------------
if [ "${SKIP_BENCH:-0}" != "1" ]; then
  info "Phase 4/4: baseline vs speculative speedup ($BENCH_N prompts)"
  # Benchmark serves mirror deployment, so response-shaping parsers belong here
  # rather than in Phase 1, where they would strip reasoning out of the captured
  # on-policy text.
  export ENABLE_RESPONSE_PARSERS=1
  BASELINE_GPUS="$(gpu_csv 0 "$TARGET_TP_SIZE")"
  if [ $((2 * TARGET_TP_SIZE)) -le "$WORLD_SIZE" ]; then
    SPEC_GPUS="$(gpu_csv "$TARGET_TP_SIZE" "$TARGET_TP_SIZE")"
    GPUS="$BASELINE_GPUS" \
      bash "$ASSET_DIR/scripts/serve_target.sh" qsd-base 8000 "$TARGET_TP_SIZE" >/dev/null
    GPUS="$SPEC_GPUS" \
      bash "$ASSET_DIR/scripts/serve_spec.sh" qsd-spec 8100 "$RELEASE_REL" "$TARGET_TP_SIZE" "$NST" >/dev/null
    info "waiting for baseline + spec serves..."; wait_ready "localhost:8000,localhost:8100"
    docker run --rm --network=host -v "$EX_DIR":/workspace -v "$DATA_DIR":/data \
      -v "$ASSET_DIR":/runner:ro -w /workspace --entrypoint python3 "$IMG" \
      /runner/scripts/bench.py --eval /data/onpolicy_eval.jsonl --model "$SERVED_NAME" \
        --baseline-port 8000 --spec-port 8100 --n "$BENCH_N" --rounds "$BENCH_ROUNDS" \
        --min-speedup "$MIN_SPEEDUP" --min-served-al "$MIN_SERVED_AL" \
        --draft-path "$RELEASE_DIR" --json-out /workspace/report.json
    docker rm -f qsd-base qsd-spec >/dev/null 2>&1
  else
    info "target TP consumes the node; benchmarking baseline and spec sequentially"
    GPUS="$BASELINE_GPUS" \
      bash "$ASSET_DIR/scripts/serve_target.sh" qsd-base 8000 "$TARGET_TP_SIZE" >/dev/null
    wait_ready "localhost:8000"
    docker run --rm --network=host -v "$EX_DIR":/workspace -v "$DATA_DIR":/data \
      -v "$ASSET_DIR":/runner:ro -w /workspace --entrypoint python3 "$IMG" \
      /runner/scripts/bench.py --mode baseline --eval /data/onpolicy_eval.jsonl \
        --model "$SERVED_NAME" --baseline-port 8000 --n "$BENCH_N" --rounds "$BENCH_ROUNDS" \
        --json-out /workspace/runtime/baseline_benchmark.json
    docker rm -f qsd-base >/dev/null 2>&1
    GPUS="$BASELINE_GPUS" \
      bash "$ASSET_DIR/scripts/serve_spec.sh" qsd-spec 8100 "$RELEASE_REL" "$TARGET_TP_SIZE" "$NST" >/dev/null
    wait_ready "localhost:8100"
    docker run --rm --network=host -v "$EX_DIR":/workspace -v "$DATA_DIR":/data \
      -v "$ASSET_DIR":/runner:ro -w /workspace --entrypoint python3 "$IMG" \
      /runner/scripts/bench.py --mode speculative \
        --baseline-json /workspace/runtime/baseline_benchmark.json \
        --eval /data/onpolicy_eval.jsonl --model "$SERVED_NAME" --spec-port 8100 \
        --n "$BENCH_N" --rounds "$BENCH_ROUNDS" \
        --min-speedup "$MIN_SPEEDUP" --min-served-al "$MIN_SERVED_AL" \
        --draft-path "$RELEASE_DIR" --json-out /workspace/report.json
    docker rm -f qsd-spec >/dev/null 2>&1
  fi
else
  info "Phase 4/4: skipped (SKIP_BENCH=1)"
fi

if [ "$REPORT_PATH" != "$EX_DIR/report.json" ] && [ -f "$EX_DIR/report.json" ]; then
  cp "$EX_DIR/report.json" "$REPORT_PATH"
fi
info "Done. vLLM-loadable draft: $(_c '1;32' "$RELEASE_DIR")"
# An `&&` list here would make the script exit non-zero on the full profile,
# because a false test as the final command becomes the script's exit status.
if [ "$PROFILE" = quick ]; then
  info "Plumbing verified. Now run the full pipeline for real numbers:  bash run_all.sh --base_model $BASE_MODEL"
fi
