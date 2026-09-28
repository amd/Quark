# Quant-Perf Quick Start (ROCm)

Requires Docker with ROCm GPU access, an available MI350X/MI355X GPU, a complete
local copy of `Qwen/Qwen3.5-35B-A3B`, and access to the source/package repositories.
These commands use GPU 0 and TP=1.

## 1. Start Docker (host)

```bash
export MODEL_DIR=/absolute/path/to/Qwen3.5-35B-A3B
export WORK_DIR="$HOME/quant-perf-work"
mkdir -p "$WORK_DIR/runs" "$WORK_DIR/tools"

docker pull vllm/vllm-openai-rocm:v0.25.0
docker run -it --name quark-quant-perf \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --network=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --mount "type=bind,src=$WORK_DIR,dst=/workspace" \
  --mount "type=bind,src=$MODEL_DIR,dst=/models/Qwen3.5-35B-A3B,readonly" \
  --workdir /workspace --entrypoint /bin/bash \
  vllm/vllm-openai-rocm:v0.25.0
```

## 2. Install a coding agent (container)

Install Claude Code or Codex inside the container and complete its authentication
setup. GEAK optimization requires Claude Code 2.1.177 or newer, even if Codex is
your primary agent.

## 3. Install Quark and dependencies (container)

Run subsequent commands inside the container. Keep the image's PyTorch, vLLM,
AITER, and FlyDSL versions. Use `main` or a release branch that includes Quant-Perf.
The constraints below list the validated runtime versions for this image.

```bash
export QUARK_REF=main  # Or the desired release branch name.
git clone --branch "$QUARK_REF" https://gitenterprise.xilinx.com/AMDNeuralOpt/Quark.git /workspace/Quark
cd /workspace/Quark
export PATH="$HOME/.local/bin:$PATH"
apt-get update
apt-get install -y libcairo2-dev pkg-config

python -c 'import torch; assert torch.version.hip and torch.cuda.is_available(); x=torch.ones(1,device="cuda"); torch.cuda.synchronize(); print(x.item())'
cat > /workspace/runtime-constraints.txt <<'EOF'
torch==2.11.0+gitd0c8b1f
vllm==0.25.0+rocm723
amd-aiter==0.1.16.post2
flydsl==0.2.0
grpcio==1.78.0
grpcio-status==1.78.0
EOF
export PIP_CONSTRAINT=/workspace/runtime-constraints.txt
python -m pip install pycairo
python -m pip install --no-build-isolation -e ".[quant_perf]"
python -m pip check
```

## 4. Prepare optimization tools (scenario 2 only)

Install the public [TraceLens](https://github.com/AMD-AGI/TraceLens#quick-start)
and [GEAK](https://github.com/AMD-AGI/GEAK#getting-started) packages below.

```bash
python -m pip install "git+https://github.com/AMD-AGI/TraceLens.git@v1.0.0"
git clone --depth 1 --branch v4.1.0 https://github.com/AMD-AGI/GEAK.git /workspace/tools/GEAK
export GEAK_ROOT=/workspace/tools/GEAK
export CLAUDE_BIN="$(command -v claude)"
test -f "$GEAK_ROOT/kernel_workflow/kernel_workflow.js"
test -f "$GEAK_ROOT/e2e_workflow/scripts/parse_profile.py"
claude --version  # Requires >= 2.1.177.
command -v rocprofv3
python - <<'PY'
from TraceLens import TreePerfAnalyzer
from TraceLens.EventReplay.event_replay import EventReplayer

print("TraceLens OK")
PY
python -m pip check
```

## 5. Configure AMD LLM Gateway (when needed)

Quark uses the AMD LLM Gateway for GEAK and LLM-assisted repair. Scenario 2
needs it if the workflow enters GEAK; scenario 1 needs it only for an LLM-assisted
repair. Claude Code authentication does not provide these variables. In the same
container shell that will launch your agent, run:

```bash
read -rsp "AMD LLM Gateway key: " AMD_LLM_API_KEY
export AMD_LLM_API_KEY
export AMD_LLM_USER="your.name@amd.com"
```

At the prompt, paste your gateway key and press Enter; `read -s` keeps it hidden.
Replace `your.name@amd.com` with your AMD email address. These exports apply to
the current shell and its child processes, so set them again after opening a new
shell. Do not save the key in the repository.

## 6. Start your agent and run a prompt

Launch your installed agent inside the container:

```bash
cd /workspace/Quark
claude
# Or: codex
```

Choose either prompt below. Replace MI350X with MI355X if applicable.

Normally omit `--search-moe-backend` (default `auto`). Quark selects the search
backend from the intersection of plugin QDQ adapters and vLLM support for each
actual MoE layer. Registered paths include generic Triton, unfused Triton for
packed MXFP4, and AITER W4A16 two-stage with matching source weights. Selection
does not depend on model names. A short execution probe checks the activation
hooks before candidate accuracy evaluation. Logs record the selected concrete
implementation and rejection reasons; inference has a separate backend policy.

### Scenario 1: accuracy only

#### Prompt

```text
Quantize /models/Qwen3.5-35B-A3B with vLLM on MI350X GPU 0 and tensor parallelism 1.
Search at most 5 configurations using MXFP4 and PTPC-FP8 with native KV cache.
Screen each configuration on 64 GSM8K samples; evaluate the original and exported
models on all 1319 samples and require no more than 5% relative accuracy loss.
Save the session to /workspace/runs/qwen35-accuracy.
```

#### Equivalent CLI command

```bash
cd /workspace/Quark
quark-quant-perf \
  --model /models/Qwen3.5-35B-A3B \
  --framework vllm --gpu-type mi350x --gpu-id 0 \
  --vllm-extra-arg "--tensor-parallel-size 1" \
  --layer-precision-candidates mxfp4 ptpc_fp8 \
  --kv-cache-precision-candidates native \
  --max-search-candidates 5 \
  --search-gsm8k-num-samples 64 --gsm8k-num-samples 1319 \
  --accuracy-gap 0.05 --performance-mode off \
  --workspace-source auto \
  --session-dir /workspace/runs/qwen35-accuracy
```

#### Outputs

Session directory: `/workspace/runs/qwen35-accuracy`
(host: `$WORK_DIR/runs/qwen35-accuracy`).

| Final artifact | Contents |
|---|---|
| `quant_ckpt/` | Selected exported checkpoint with quantized weights, configuration, and tokenizer files. |
| `reports/final.md` / `reports/final.json` | Terminal status, selected configuration, baseline/exported accuracy, and relative accuracy loss. |
| `session_report.md` / `session_breakdown.json` | Detailed human-readable report and structured run evidence. |

Success is `done` with the exported-checkpoint accuracy gate passed. No
throughput target is evaluated in this scenario.

### Scenario 2: accuracy and throughput

#### Prompt

```text
Quantize /models/Qwen3.5-35B-A3B with vLLM on MI350X GPU 0 and tensor parallelism 1.
Search at most 5 configurations using MXFP4 and PTPC-FP8 with native KV cache.
Screen each configuration on 64 GSM8K samples; evaluate the original and exported
models on all 1319 samples and require no more than 5% relative accuracy loss.
Target at least 1.5x throughput at input/output length 1024/1024 and concurrency 64.
Save the session to /workspace/runs/qwen35-accuracy-performance.
```

#### Equivalent CLI command

```bash
cd /workspace/Quark
quark-quant-perf \
  --model /models/Qwen3.5-35B-A3B \
  --framework vllm --gpu-type mi350x --gpu-id 0 \
  --vllm-extra-arg "--tensor-parallel-size 1" \
  --layer-precision-candidates mxfp4 ptpc_fp8 \
  --kv-cache-precision-candidates native \
  --max-search-candidates 5 \
  --search-gsm8k-num-samples 64 --gsm8k-num-samples 1319 \
  --accuracy-gap 0.05 --performance-mode optimize --target-gain 1.5 \
  --isl 1024 --osl 1024 --bench-concurrency 64 \
  --workspace-source auto \
  --session-dir /workspace/runs/qwen35-accuracy-performance
```

#### Outputs

Session directory: `/workspace/runs/qwen35-accuracy-performance`
(host: `$WORK_DIR/runs/qwen35-accuracy-performance`).

| Final artifact | Contents |
|---|---|
| `quant_ckpt/` | Selected exported checkpoint with quantized weights, configuration, and tokenizer files. |
| `reports/final.md` / `reports/final.json` | Accuracy, baseline/quantized/final throughput, achieved gain, target status, and retained changes. |
| `session_report.md` / `session_breakdown.json` | Detailed report and structured evidence for throughput and optimization attempts. |

If the final gain is below 1.5x after accuracy passes, the terminal status is
`perf_failed`; final reports and the checkpoint are still produced.

## 7. Inspect or resume with the agent

```text
Check /workspace/runs/qwen35-accuracy and report its current progress.
If interrupted and non-terminal, resume it with the same settings.
When finished, summarize reports/final.md and show the checkpoint path.
```

For scenario 2, use `/workspace/runs/qwen35-accuracy-performance` instead.
Use a new session directory for a new experiment.
