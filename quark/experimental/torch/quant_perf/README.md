# Quant-Perf Pipeline

`quark.experimental.torch.quant_perf` is an experimental, CLI-first pipeline
for quantizing PyTorch and Hugging Face LLM checkpoints, validating the
exported model's accuracy, and optionally measuring or optimizing inference
performance.

For a container-based walkthrough with accuracy-only and accuracy-plus-performance
examples, start with the [ROCm Quick Start](QUICKSTART.md).

The default quantization route uses
`quark.experimental.torch.mix_precision`. Supplying `--quant-strategy` selects
the direct PTQ route. vLLM is the supported default inference framework; Atom
is available as an experimental framework.

## Managed workflow

```text
input and runtime validation
  -> baseline framework health
  -> quantization or mixed-precision search
  -> exported-checkpoint accuracy gate
  -> [measure or optimize] baseline and quantized throughput
  -> [optimize, when the target is missed]
       TraceLens / vendor tuning / GEAK
       -> candidate validation and retention
  -> FINAL cleanup and reports
```

The accuracy path is always required. Performance stages are selected with
`--performance-mode`:

| Mode | Behavior |
|---|---|
| `off` | Stop after the exported-checkpoint accuracy gate. This is the default. |
| `measure` | Measure baseline and quantized throughput without running performance optimization. |
| `optimize` | Check the throughput target and, when needed, run the available performance-optimization stages. |

There is one FINAL stage in the managed run. It runs after the selected stages
finish, or after a terminal failure. The `report` subcommand may explicitly
regenerate those terminal artifacts later without rerunning GPU work.

## Installation

Install the PyTorch build that matches the target accelerator first, following
the repository-level [installation instructions](../../../../README.md#installation).
Then install Quark with the Quant-Perf dependencies.

From a released package:

```bash
python -m pip install "amd-quark[quant_perf]"
```

From a Quark source checkout:

```bash
python -m pip install --no-build-isolation ".[quant_perf]"
```

Verify that the CLI is available:

```bash
quark-quant-perf --help
```

The `quant_perf` extra installs the common Python dependencies used by the
managed workflow. It does not install or replace accelerator-specific vLLM or
AITER builds. Install a compatible vLLM build and, when the selected backend
requires it, a compatible AITER build for the target ROCm environment.

Automatic mixed-precision search runs from the installed package. The direct
PTQ route selected by `--quant-strategy` also requires a discoverable Quark
source checkout containing the `quark-torch-ptq` skill. Set `QUARK_ROOT` only
when automatic checkout discovery fails or when a specific checkout must be
selected.

The `cli` and `quant_perf` extras may be installed together. Their current
Transformers constraints overlap, and the combined environment uses the
narrower Quant-Perf-compatible range.

### Runtime components

The following versions provide the runtime reference stack:

| Component | Required when | Reference version |
|---|---|---|
| vLLM | The managed vLLM workflow | `v0.25.0` |
| AITER | A selected backend requires AITER | Minimum `0.1.15`; example pairing: `0.1.16` |
| FlyDSL | Required by the selected AITER build | `0.2.0` for AITER `0.1.16` |
| TraceLens | A missed optimization target enters PerfOpt | `v1.0.0` |
| GEAK | Source-level kernel optimization is required | `v4.1.0` |
| Forge | Optional vendor GEMM tuning is enabled | Repository default branch |

Reuse compatible vLLM and AITER packages already installed for the target ROCm
runtime. Check their package metadata without importing AITER or triggering JIT:

```bash
python - <<'PY'
from importlib.metadata import version

for package in ("vllm", "amd-aiter", "flydsl"):
    print(package, version(package))
PY
```

Upgrade AITER only when it is below `0.1.15`; older releases can suffer a long
FP8 block-scale dispatch-table compilation. For example, install an AITER
`0.1.16` wheel matching the ROCm, Python, and platform together with FlyDSL:

```bash
python -m pip install "/path/to/amd_aiter-0.1.16-<matching-build>.whl" "flydsl==0.2.0"
```

Use each AITER release's declared FlyDSL version rather than upgrading FlyDSL
independently. If first-use JIT exceeds vLLM's execution deadline, use
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 quark-quant-perf ...` for that
invocation without changing the accuracy or workload settings.

Install TraceLens before an `optimize` run that may enter PerfOpt after missing
its throughput target:

```bash
python -m pip install "git+https://github.com/AMD-AGI/TraceLens.git@v1.0.0"
```

Prepare a GEAK source checkout when source-level kernel optimization is needed.
Quant-Perf invokes its Workflow files directly, so the `geak` Python package
does not need to be installed:

```bash
git clone --depth 1 --branch v4.1.0 https://github.com/AMD-AGI/GEAK.git /path/to/GEAK
export GEAK_ROOT=/path/to/GEAK
export CLAUDE_BIN="$(command -v claude)"
test -f "$GEAK_ROOT/kernel_workflow/kernel_workflow.js"
test -f "$GEAK_ROOT/e2e_workflow/scripts/parse_profile.py"
claude --version  # Requires >= 2.1.177.
```

Forge is optional. If `forge-gemm-tune` is unavailable, vendor GEMM tuning is
skipped without failing the main workflow.

Advanced FlyDSL and ASM paths are capability-gated at runtime. A newer package
version alone does not guarantee compatibility with the selected GPU and
backend.

## Environment

Set an AMD LLM Gateway key when using direct PTQ, LLM-assisted runtime repair, or
GEAK. Evaluation-profile discovery may also use the gateway to extract settings
from unstructured evidence, but that optional call fails open to the packaged
evaluation policy when no usable key is available:

```bash
export AMD_LLM_API_KEY=<your-key>
```

`AMD_LLM_GATEWAY_KEY` and `LLM_GATEWAY_KEY` are also accepted. Common optional
variables are:

| Variable | Purpose |
|---|---|
| `AMD_LLM_USER` / `AMD_LLM_BASE_URL` | AMD Gateway identity and endpoint |
| `QUARK_ROOT` | Explicit Quark checkout for the direct PTQ skill |
| `ATOM_ROOT` | Atom checkout when `--framework atom` is selected |
| `GEAK_ROOT` | GEAK source checkout |
| `GEAK_MODEL` | GEAK model override |
| `CLAUDE_BIN` | Explicit Claude CLI used by the Agent SDK |
| `QUARK_QUANT_PERF_EXPERIENCE_STORE_PATH` | Cross-session experience database |
| `QUARK_QUANT_PERF_FORGE_GEMM_TUNE` | Explicit `forge-gemm-tune` executable |
| `QUARK_QUANT_PERF_MXFP4_MOE_BACKEND` | Default MXFP4 MoE backend override |
| `QUARK_QUANT_PERF_MXFP4_GEMM_BACKEND` | Default dense MXFP4 backend override |
| `QUARK_QUANT_PERF_W4A8_GEMM_BACKEND` | Default dense W4A8 backend override |
| `AITER_CONFIG_FMOE` | AITER tuned fused MoE configuration for FlyDSL |

The accepted gateway variables may instead be stored in the Quark checkout's
gitignored `.env` file. Never commit a real key.

## CLI usage

The examples below use:

- vLLM on one MI355X GPU;
- the GPU-supported default layer-precision search space;
- native KV cache;
- 100 GSM8K samples for search and exported-checkpoint validation;
- a maximum relative accuracy degradation of `0.03`.

`native` preserves the model's existing KV precision. On ROCm, DeepSeek V4
uses its native `fp8_ds_mla` storage automatically. Explicit `--kv-cache-scheme`
or `--vllm-extra-arg '--kv-cache-dtype=...'` settings apply to search, accuracy,
throughput and serving; conflicting settings are rejected before execution.

The current GPU-specific defaults are:

- MI300X / MI325X: `native`, `fp8`, `ptpc_fp8`
- MI350X / MI355X: `native`, `fp8`, `ptpc_fp8`, `mxfp4`,
  `mxfp4_fp8`

`mxfp6_e2m3` is excluded from the automatic defaults but may be requested
explicitly on supported hardware.

Adjust the model path, GPU selection, tensor parallelism, sequence lengths, and
concurrency for the target system.

### Accuracy only

Use `off` when the quantized checkpoint only needs to satisfy the accuracy
requirement:

```bash
quark-quant-perf \
  --model /models/Qwen3.5-35B-A3B \
  --framework vllm \
  --gpu-type mi355x \
  --gpu-id 0 \
  --kv-cache-precision-candidates native \
  --gsm8k-num-samples 100 \
  --accuracy-gap 0.03 \
  --performance-mode off
```

### Accuracy plus throughput measurement

Use `measure` to record baseline and quantized throughput without running
TraceLens, vendor tuning, or GEAK:

```bash
quark-quant-perf \
  --model /models/Qwen3.5-35B-A3B \
  --framework vllm \
  --gpu-type mi355x \
  --gpu-id 0 \
  --kv-cache-precision-candidates native \
  --gsm8k-num-samples 100 \
  --accuracy-gap 0.03 \
  --performance-mode measure \
  --isl 1024 \
  --osl 1024 \
  --bench-concurrency 64
```

### Accuracy and performance targets

Use `optimize` when both the accuracy requirement and a minimum end-to-end
throughput gain must be met. A target of `1.55` means 1.55 times the baseline
throughput, or a 55% gain:

```bash
quark-quant-perf \
  --model /models/Qwen3.5-35B-A3B \
  --framework vllm \
  --gpu-type mi355x \
  --gpu-id 0 \
  --kv-cache-precision-candidates native \
  --gsm8k-num-samples 100 \
  --accuracy-gap 0.03 \
  --performance-mode optimize \
  --target-gain 1.55 \
  --isl 1024 \
  --osl 1024 \
  --bench-concurrency 64
```

Supplying `--target-gain` without `--performance-mode` also selects
`optimize`. Using `--performance-mode optimize` without a target uses the
default target of `1.2`.

### Automatic source discovery

Use `auto` when framework and kernel repository paths are not supplied.
Quant-Perf discovers editable installations or creates session-local source
overlays when possible:

```bash
quark-quant-perf \
  --model /models/Qwen3.5-35B-A3B \
  --framework vllm \
  --gpu-type mi355x \
  --gpu-id 0 \
  --kv-cache-precision-candidates native \
  --gsm8k-num-samples 100 \
  --accuracy-gap 0.03 \
  --performance-mode optimize \
  --target-gain 1.55 \
  --isl 1024 \
  --osl 1024 \
  --bench-concurrency 64 \
  --workspace-source auto
```

`auto` is the default workspace-source policy. It is shown explicitly here to
make the source-selection behavior clear.

The workspace-source policy applies to every performance mode because it also
controls runtime source activation and eligible repair work. TraceLens, vendor
tuning, GEAK, and candidate retention remain specific to `optimize`.

### Explicit framework and kernel repositories

Use `explicit` when performance work must use specific writable source
checkouts:

```bash
quark-quant-perf \
  --model /models/Qwen3.5-35B-A3B \
  --framework vllm \
  --gpu-type mi355x \
  --gpu-id 0 \
  --kv-cache-precision-candidates native \
  --gsm8k-num-samples 100 \
  --accuracy-gap 0.03 \
  --performance-mode optimize \
  --target-gain 1.55 \
  --isl 1024 \
  --osl 1024 \
  --bench-concurrency 64 \
  --workspace-source explicit \
  --framework-repo /path/to/vllm \
  --kernel-repo /path/to/aiter
```

Full source repositories are normally needed for compiled C++/HIP changes,
rebuilds, formal patches, or a specific branch or tag. Use
`--workspace-source readonly` only when source modification must be disabled.

### Search-space notes

- Omit `--layer-precision-candidates` to use the complete supported default
  search space for the selected GPU.
- Set `--layer-precision-candidates` only when the search must be restricted or
  extended explicitly.
- `native` is always retained as the layer-precision fallback, including when
  an explicit list is supplied.
- KV-cache candidates do not add an implicit fallback. Include `native`
  explicitly when it should be searched.
- `--max-search-candidates 20` is the default. Use
  `--max-search-candidates 0` to evaluate the complete generated search space.
- Supplying `--quant-strategy` switches from mixed-precision search to the
  direct PTQ route.
- The resolved configuration is printed before execution and persisted in the
  session state.

### Custom precision search space

The following example explicitly searches FP8, MXFP4, and W4A8 layer
configurations, together with native and FP8 KV cache:

```bash
quark-quant-perf \
  --model /models/Qwen3.5-35B-A3B \
  --framework vllm \
  --gpu-type mi355x \
  --gpu-id 0 \
  --layer-precision-candidates fp8 mxfp4 mxfp4_fp8 \
  --kv-cache-precision-candidates native fp8 \
  --gsm8k-num-samples 100 \
  --accuracy-gap 0.03 \
  --performance-mode off
```

The explicit layer list replaces the automatically selected quantized modes;
`native` is still added as the fallback. Use a smaller list to narrow the
search or add another mode only when it is supported by the selected GPU.

### Optional backend overrides

`--search-moe-backend` and `--inference-moe-backend` independently control search
and exported-checkpoint inference (both default to `auto`). Search isolates
AITER MoE settings and negotiates each layer's implementation from the plugin
QDQ adapter registry and vLLM's compatibility checks. Users normally omit
`--search-moe-backend`. The filtered candidate set determines the weight and
activation requirements; selection runs during layer construction, before its
weights are created. Known paths include generic Triton, unfused Triton with
packed MXFP4 weights, and AITER W4A16 two-stage with matching source weights.
New models sharing these implementations need no model-name rule. Explicit
incompatible overrides fail with concrete rejection reasons. Inference keeps
its separate defaults.

The search log records the requested and selected backends with the reason.
`state.json` also retains `mix_precision_search.moe_backend_resolution` and
`resolved_runtime_args` when search returns or raises, including engine errors.
If engine startup fails before the worker report is available, the decision can
remain `pending`; consult the worker log for detailed rejection reasons.

The general examples intentionally omit MXFP4 backend flags. Without explicit
overrides, exported-checkpoint inference uses `aiter` for MXFP4 MoE, `triton` for dense MXFP4
GEMM, and `triton` for dense W4A8, unless the corresponding environment
variables select another backend. These defaults are the normal supported
path.

Select FlyDSL only when it is an explicit requirement and the installed AITER
runtime supports it:

```bash
quark-quant-perf \
  --model /models/Qwen3.5-35B-A3B \
  --framework vllm \
  --gpu-type mi355x \
  --gpu-id 0 \
  --layer-precision-candidates mxfp4 mxfp4_fp8 \
  --kv-cache-precision-candidates native \
  --gsm8k-num-samples 100 \
  --accuracy-gap 0.03 \
  --performance-mode optimize \
  --target-gain 1.55 \
  --isl 1024 \
  --osl 1024 \
  --bench-concurrency 64 \
  --mxfp4-moe-backend flydsl \
  --mxfp4-gemm-backend flydsl \
  --w4a8-gemm-backend flydsl
```

`--mxfp4-moe-backend` selects the MXFP4 MoE backend,
`--mxfp4-gemm-backend` selects the dense W4A4 backend, and
`--w4a8-gemm-backend` selects the dense W4A8 backend. Each flag affects only
candidates that use the corresponding scheme; backend selection does not add
or remove layer-precision candidates.

`--mxfp4-moe-backend triton` selects AITER Triton W4A4 kernels;
`--inference-moe-backend triton` selects native vLLM Triton and takes precedence.

## Natural-language requests

The following prompts are intended for a coding agent with the
`quark-torch-quant-perf` skill. They are not arguments passed directly to the
CLI.

### Accuracy only

```text
Run the managed Quant-Perf workflow for /models/Qwen3.5-35B-A3B.

Requirements:
- Objective: satisfy the accuracy requirement only; do not run throughput
  measurement or performance optimization.
- Framework: vLLM.
- Layer precision search space: use the defaults supported by the selected GPU.
- KV-cache precision candidates: native.
- GSM8K samples for search and exported-checkpoint validation: 100.
- Maximum relative accuracy degradation: 0.03.
```

### Accuracy plus throughput measurement

```text
Run the managed Quant-Perf workflow for /models/Qwen3.5-35B-A3B.

Requirements:
- Objective: satisfy the accuracy requirement and measure baseline versus
  quantized throughput; do not run performance optimization.
- Framework: vLLM.
- Layer precision search space: use the defaults supported by the selected GPU.
- KV-cache precision candidates: native.
- GSM8K samples for search and exported-checkpoint validation: 100.
- Maximum relative accuracy degradation: 0.03.
- Performance mode: measure.
- Benchmark workload: ISL 1024, OSL 1024, concurrency 64.
```

### Accuracy and performance targets

```text
Run the complete managed Quant-Perf workflow for
/models/Qwen3.5-35B-A3B.

Requirements:
- Objective: satisfy both the accuracy and end-to-end performance targets.
- Framework: vLLM.
- Layer precision search space: use the defaults supported by the selected GPU.
- KV-cache precision candidates: native.
- GSM8K samples for search and exported-checkpoint validation: 100.
- Maximum relative accuracy degradation: 0.03.
- Performance mode: optimize.
- Minimum end-to-end throughput multiplier: 1.55x, corresponding to a 55%
  gain over the baseline.
- Benchmark workload: ISL 1024, OSL 1024, concurrency 64.
```

### Automatic source discovery

```text
Run the complete managed Quant-Perf workflow for
/models/Qwen3.5-35B-A3B.

Requirements:
- Objective: satisfy a 0.03 maximum relative accuracy degradation and a 1.55x
  end-to-end throughput target.
- Framework: vLLM.
- Layer precision search space: use the defaults supported by the selected GPU.
- KV-cache precision candidates: native.
- GSM8K samples for search and exported-checkpoint validation: 100.
- Benchmark workload: ISL 1024, OSL 1024, concurrency 64.
- Workspace source policy: auto; discover or materialize the required sources
  without pre-supplied repository paths.
```

### Explicit source repositories

```text
Run the complete managed Quant-Perf workflow for
/models/Qwen3.5-35B-A3B.

Requirements:
- Objective: satisfy a 0.03 maximum relative accuracy degradation and a 1.55x
  end-to-end throughput target.
- Framework: vLLM.
- Layer precision search space: use the defaults supported by the selected GPU.
- KV-cache precision candidates: native.
- GSM8K samples for search and exported-checkpoint validation: 100.
- Benchmark workload: ISL 1024, OSL 1024, concurrency 64.
- Workspace source policy: explicit.
- Framework repository: /path/to/vllm.
- Kernel repository: /path/to/aiter.
```

### Custom precision and backend selection

```text
Run the complete managed Quant-Perf workflow for
/models/Qwen3.5-35B-A3B.

Requirements:
- Objective: satisfy a 0.03 maximum relative accuracy degradation and a 1.55x
  end-to-end throughput target.
- Framework: vLLM.
- Layer precision candidates: FP8, MXFP4, and MXFP4 weights with FP8
  activations; retain native as the fallback.
- KV-cache precision candidates: native and FP8.
- GSM8K samples for search and exported-checkpoint validation: 100.
- Benchmark workload: ISL 1024, OSL 1024, concurrency 64.
- MXFP4 MoE backend: AITER FlyDSL.
- Dense MXFP4 x MXFP4 backend: FlyDSL.
- Dense MXFP4 weights with FP8 activations backend: FlyDSL.
```

## Session status and outputs

Operational commands:

```bash
quark-quant-perf status --session runs/<session>
quark-quant-perf report --session runs/<session>
quark-quant-perf bottlenecks --session runs/<session>
quark-quant-perf eval --help
quark-quant-perf backend-probe --help
quark-quant-perf knowledge --help
quark-quant-perf gc --dry-run
```

Each run writes its outputs beneath `--session-dir`:

| Output | Purpose |
|---|---|
| `quant_ckpt/` | Selected quantized checkpoint and its `config.json`; this is the primary model artifact. |
| `reports/final.md` | Short human-readable terminal result. |
| `reports/final.json` | Machine-readable terminal result. |
| `session_report.md` | Detailed execution report, including stages, warnings, and optimization attempts. |
| `session_breakdown.json` | Complete structured evidence used to generate the reports. |
| `reports/patches/` | Patch manifest and retained framework or kernel diffs from managed workspace directories. |
| `state.json` / `progress.json` | Resume state and live progress, including repair rounds, generation/verification phases, timestamps, and final status. |
| `repair/` | Full repair diagnostics, per-round agent output, and rejected candidate patches. |

During repair, `status` shows the current round, phase, and elapsed times.
These times are calculated when read and are not a worker heartbeat. The Repair
Journey summarizes per-round and total timing; older sessions without timing
records show missing values.

Use `status` while a session is active. Use `report` to regenerate the final
artifacts after the session reaches a terminal state.

Common terminal states:

| State | Meaning |
|---|---|
| `done` | Accuracy passed and all requested performance work completed. |
| `perf_failed` | Accuracy passed, but the final gain remained below the requested target. |
| `failed` | Quantization, exported-checkpoint accuracy, or required throughput measurement failed. |
| `base_unhealthy` | The unquantized model was unhealthy in the selected runtime. |

Terminal run state and report-generation status are tracked separately. A
reporting failure does not replace the run result; use `report` to retry
terminal artifact generation.

For normal result handoff, keep `quant_ckpt/`, `reports/final.*`, and any files
under `reports/patches/`. Trace and intermediate evaluation directories are
mainly useful for diagnosis or further optimization.

Sessions created by earlier standalone versions are intentionally not
compatible with this in-tree module.

## Code layout

```text
orchestration/  run intake, specification construction, and orchestration
evaluation/     accuracy profiles, GSM8K, and throughput measurement
knowledge/      reusable runtime experience, review, and audit records
landing/        framework adapters and managed model serving
llm/            audited direct LLM calls
perfopt/        trace analysis, source resolution, GEAK, and vendor tuning
pipeline/       reusable accuracy, benchmark, landing, and retention stages
quantize/       mixed-precision search and direct PTQ delegation
repair/         failure classification and bounded repair workflows
reporting/      FINAL structured and Markdown report generation
runtime/        backend environment, recovery policy, and inventory
session/        persisted contracts, checkpoints, locks, and progress
workspace/      source discovery and managed Git worktrees
```

The console entry point is
`quark.experimental.torch.quant_perf.cli:main`.
