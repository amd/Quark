# Quant-Perf CLI Mapping

Always run `quark-quant-perf --help` before launch and treat the installed
CLI's options and defaults as authoritative.

## Core inputs

| User intent | CLI |
|---|---|
| model/checkpoint | `--model PATH_OR_ID` |
| separate baseline | `--base-model PATH_OR_ID` |
| framework | `--framework vllm` or experimental `atom` |
| GPU target | `--gpu-type mi300x\|mi325x\|mi350x\|mi355x` |
| input/output length | `--isl N --osl N` |
| GPU index | `--gpu-id N` |
| tensor parallelism | `--vllm-extra-arg "--tensor-parallel-size N"` |
| allow model remote code | `--vllm-extra-arg=--trust-remote-code` |
| vLLM scheduler sequence limit in baseline, search, accuracy, and throughput | `--vllm-extra-arg "--max-num-seqs 64"` |
| maximum accuracy drop | `--accuracy-gap FRACTION` |
| mixed-precision search sample budget | `--search-gsm8k-num-samples N` |
| baseline and exported-checkpoint accuracy samples | `--gsm8k-num-samples N` |
| no performance work | omit `--performance-mode` and `--target-gain` |
| throughput measurement only | `--performance-mode measure` |
| target throughput multiplier | `--target-gain MULTIPLIER` (implies optimize) |
| benchmark concurrency | `--bench-concurrency N` |

## Quantization route

Precision mode names describe complete weight/activation schemes:

| Mode | Meaning |
|---|---|
| `native` | Preserve the source precision |
| `fp8` | Static per-tensor FP8 weights and activations (W8A8) |
| `ptpc_fp8` | Per-channel FP8 weights with dynamic per-token FP8 activations |
| `mxfp4` | MXFP4 weights and dynamic MXFP4 activations (W4A4) |
| `mxfp4_fp8` | MXFP4 weights and static FP8 activations (W4A8) |
| `mxfp6_e2m3` | MXFP6 E2M3 weights and dynamic activations; explicit opt-in on supported hardware |

Treat explicitly named quantized modes as a closed search set, separate from
the implicit `native` fallback. For example, "target MXFP4, but some layers can
use FP8" maps to
`--layer-precision-candidates mxfp4 fp8`. Do not infer or add compound
precision modes from their components; add `mxfp4_fp8` only when the user
explicitly requests that mode, W4A8, or MXFP4 weights with FP8 activations.
Normalize aliases one-to-one instead of composing or decomposing modes based on
their names. When no modes are named and the user requests an unrestricted
search, preserve the CLI defaults.

Automatic GPU-default search:

```bash
quark-quant-perf \
  --model /absolute/model/path \
  --gpu-type mi355x \
  --accuracy-gap 0.03 \
  --workspace-source auto \
  --session-dir /absolute/session/path
```

Omitting `--layer-precision-candidates` selects the architecture-aware defaults:

- MI300X / MI325X: `native`, `fp8`, `ptpc_fp8`
- MI350X / MI355X: `native`, `fp8`, `ptpc_fp8`, `mxfp4`,
  `mxfp4_fp8`

`mxfp6_e2m3` is excluded from Quant-Perf's automatic defaults but may be
requested explicitly on supported hardware.

Explicit mixed-precision search space:

```bash
quark-quant-perf \
  --model /absolute/model/path \
  --gpu-type mi355x \
  --layer-precision-candidates mxfp4 fp8 \
  --kv-cache-precision-candidates native fp8 \
  --search-gsm8k-num-samples 64 \
  --gsm8k-num-samples 1319 \
  --accuracy-gap 0.03 \
  --workspace-source auto \
  --session-dir /absolute/session/path
```

Layer search always retains `native` as a fallback. KV-cache search does not,
so `native` must be listed there when desired.

When `--search-gsm8k-num-samples` is omitted, it inherits
`--gsm8k-num-samples` for backward compatibility. Use separate values when
screening many candidates cheaply but requiring a full exported-checkpoint
accuracy gate.

Fixed PTQ:

```bash
quark-quant-perf \
  --model /absolute/model/path \
  --gpu-type mi355x \
  --quant-strategy "FP8, exclude the embedding layer" \
  --accuracy-gap 0.02 \
  --session-dir /absolute/session/path
```

Do not add search-space flags to a fixed PTQ request.
`--quant-strategy` selects the direct PTQ route; it does not run a second
mixed-precision search. Clarify the request if the user asks for both routes.
The direct route also requires a discoverable Quark checkout containing
`quark-torch-ptq`; use `QUARK_ROOT` only for explicit checkout selection or
when automatic discovery fails.

## Source workspace and PerfOpt controls

Source workspace controls apply to every run because they determine runtime
package activation and whether baseline, load, accuracy, or throughput repair
may modify sources.

| User intent | CLI |
|---|---|
| source discovery or repair without supplied repositories | `--workspace-source auto` |
| supplied writable repositories | `--workspace-source explicit --framework-repo PATH --kernel-repo PATH` |
| runtime/evaluation without source modification | `--workspace-source readonly` |

PerfOpt controls matter only when `--performance-mode optimize` is effective.
Supplying `--target-gain` enables that mode.
TraceLens is required only if the run misses the quantization-only target and
enters PerfOpt.

| User intent | CLI |
|---|---|
| top N kernels | `--top-kernels N` |
| differential analysis | `--bottleneck-mode differential` |
| quantized-only analysis | `--bottleneck-mode absolute` |
| mixed-precision search timeout | `--search-timeout SECONDS` |
| GEAK direction budget | `--geak-direction-budget N` |
| GEAK timeout | `--geak-timeout SECONDS` |
| patch retention floor | `--keep-floor FRACTION` |
| explicit TraceLens architecture | `--tracelens-gpu-arch-json PATH` |

## Runtime backend controls

Search and inference have independent MoE backend controls. Inference includes
real accuracy evaluation, serving, throughput, and optimization validation.
Backend controls do not add precision candidates to the search space.

| Purpose | CLI | Default |
|---|---|---|
| Search MoE | `--search-moe-backend auto\|triton\|triton_unfused\|aiter\|aiter_mxfp4_bf16\|emulation` | `auto` (selected per layer by the plugin) |
| Inference MoE | `--inference-moe-backend auto\|triton\|aiter\|aiter_mxfp4_bf16\|emulation` | `auto` |
| MXFP4 MoE | `--mxfp4-moe-backend aiter\|triton\|flydsl` | `aiter` |
| Dense MXFP4 W4A4 | `--mxfp4-gemm-backend triton\|flydsl\|asm` | `triton` |
| Dense MXFP4/FP8 W4A8 | `--w4a8-gemm-backend triton\|flydsl` | `triton` |
| Tuned FlyDSL fused MoE CSV | `--aiter-config-fmoe PATH` | unset |

Omit backend flags unless the user explicitly requests an override. The
corresponding `QUARK_QUANT_PERF_*_BACKEND` environment variables may provide
the defaults.

Search `auto` chooses from the intersection of registered plugin QDQ adapters
and vLLM's compatibility checks for each actual MoE layer. Model names do not
select the backend. Generic Triton supports unquantized/FP8 sources; unfused
Triton and AITER W4A16 two-stage support compatible packed MXFP4 sources.
Adapters that preserve packed weights cannot perform a source-to-target weight
format conversion. Explicit backend choices must satisfy the same checks;
emulation is explicit-only. If no adapter is compatible, search reports the
rejection reasons instead of silently running without an a2 hook.

The isolated search worker clears inference's inner MoE kernel preferences.
Auto allows AITER capability discovery while preferring compatible Triton
adapters. A selected Triton layer retains Triton's runtime settings; explicit
`aiter` or `aiter_mxfp4_bf16` requires a supported prequantized MXFP4 source
with two-stage MoE. After loading, the plugin validates the concrete expert
implementation and runs an a1/a2 execution probe before candidate evaluation.
Models that directly construct experts without invoking a generic selector
must pass the same adapter, source-weight, and explicit-backend checks on the
loaded implementation; reports identify these selections as `model_native`.
Search reports record the selected implementations, rejected alternatives,
and execution evidence. Scoped environment changes are restored on exit.

Inference `auto` enables AITER for MXFP4 MoE checkpoints and architectures
requiring AITER. `--inference-moe-backend aiter` explicitly requests it for
other supported models. The default `--mxfp4-moe-backend aiter` selects AITER
CK kernels; its `triton` choice means AITER Triton W4A4 kernels, not native
vLLM Triton. Explicit inference backend choices take precedence over these
inner kernel preferences.

Legacy `--vllm-extra-arg=--moe-backend=...` still applies to both phases when
their dedicated options are omitted. Conflicting legacy and dedicated options
are rejected. Use the dedicated options to choose different phase backends.

## Resume and inspection

```bash
quark-quant-perf status --session /absolute/session/path
quark-quant-perf report --session /absolute/session/path
quark-quant-perf bottlenecks --session /absolute/session/path
quark-quant-perf eval --from-session /absolute/session/path --session-dir /absolute/new-eval-session
```

Use `status` while a session is running. `report` only regenerates artifacts
after the session reaches a terminal state.

Resume flags belong on the main command:

```text
--recheck-baseline
--retry-accuracy-gate
--retry-perfopt
```
