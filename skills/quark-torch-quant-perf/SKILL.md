---
name: quark-torch-quant-perf
description: >
  Run, resume, monitor, diagnose, and report Quark Quant-Perf workflows for
  PyTorch and HuggingFace transformers models. Use whenever the user asks for
  mixed-precision search, Quark MXFP4/FP8/PTPC-FP8 quantization with accuracy
  and performance validation, vLLM throughput, TraceLens bottleneck analysis,
  GEAK kernel optimization, PerfOpt retries, workspace source discovery, or a
  natural-language request that must become a quark-quant-perf command. Not for
  ONNX models.
layer: l2-workflows
backend: torch
primary_artifact: session_report.md
source_knowledge:
  - quark/experimental/torch/quant_perf/README.md
  - quark/experimental/torch/quant_perf/cli.py
  - quark/experimental/torch/quant_perf/orchestration/orchestrator.py
  - quark/experimental/torch/quant_perf/quantize/search.py
  - quark/experimental/torch/quant_perf/reporting/service.py
  - quark/experimental/torch/quant_perf/workspace/sources.py
---

# Quark Torch Quant-Perf

## Purpose

Translate a user's PyTorch or HuggingFace quantization request into the current
`quark-quant-perf` CLI, run or resume its managed Orchestrator, monitor the
session to a terminal state, and report the generated artifacts. Preserve the
user's accuracy, workload, source, and explicit performance requirements
throughout.

Use the managed pipeline:

```text
input and runtime validation
→ baseline health
→ quantize or mixed-precision search
→ real quantized accuracy gate
→ [measure or optimize] baseline and quantized throughput
→ [optimize, when the target is missed] conditional TraceLens / vendor tuning /
  GEAK → candidate validation and retention
→ FINAL reports
```

Performance modes skip inapplicable optional stages; they do not create
separate FINAL stages. The Orchestrator enters FINAL after the selected stages
complete or a terminal failure is recorded. The `report` subcommand may later
regenerate terminal artifacts without rerunning GPU work.

Do not replace a stage with a custom evaluation, benchmark, or proxy metric.

## Inputs

- Model path or HuggingFace model ID.
- Accuracy-gap requirement and optional performance intent.
- GPU type, GPU index, TP, ISL, OSL, and concurrency when specified.
- Fixed quantization intent, or a mixed-precision search space.
- Framework and kernel source policy: explicit, auto, or readonly.
- A discoverable Quark checkout containing `quark-torch-ptq` for fixed direct
  PTQ; `QUARK_ROOT` may select one explicitly.
- Existing session directory when resuming.
- Optional TraceLens architecture JSON and backend overrides.

Only `--model` is required by the CLI. Preserve CLI defaults for omitted
options rather than inventing values.

Before constructing a command:

1. Run `quark-quant-perf --help` for the installed CLI options and defaults.
2. Read [CLI mapping](references/cli-mapping.md) for intent-to-option mapping.
3. Read [session lifecycle](references/session-lifecycle.md) before resume,
   monitoring, diagnosis, or final reporting.
4. Inspect the session's `state.json` and `progress.json` when resuming or
   reporting.

Treat the installed CLI help as authoritative if a local reference differs.

## Outputs: session_report.md

The managed FINAL stage produces:

```text
<session>/reports/final.json
<session>/reports/final.md
<session>/session_breakdown.json
<session>/session_report.md
```

Treat `session_breakdown.json` as the complete structured fact source and
`session_report.md` as the detailed human-readable result. A terminal failure
may still have complete reports and useful quantized artifacts.

## Interaction Flow

1. **Intake**
   - Resolve local paths to absolute paths.
   - Preserve the requested model, accuracy gap, performance mode, target gain,
     workload, GPU, tensor parallelism, search modes, and repository policy.
   - Treat explicitly named quantized precision modes as a closed set, separate
     from any runtime-required `native` fallback. Do not infer compound modes
     from their components. Normalize aliases one-to-one; for example, use
     `mxfp4_fp8` only when the user explicitly requests that mode, W4A8, or
     MXFP4 weights with FP8 activations.
   - If the user does not name precision modes, omit
     `--layer-precision-candidates` and preserve the GPU-specific automatic
     search space.
   - If the user does not request throughput or performance optimization,
     leave performance at the default `off`.
   - Inspect existing session state before creating a duplicate run.

2. **Route**
   - Omit `--quant-strategy` for mixed-precision search.
   - Use `--quant-strategy "<normalized intent>"` for one fixed PTQ recipe.
   - For fixed PTQ, use the automatically discovered Quark checkout when
     available. Set `QUARK_ROOT` only when discovery fails or the user selects
     a specific checkout.
   - When writable framework or kernel repositories are supplied, use
     `--workspace-source explicit` with those repositories.
   - When repositories are not supplied, use `--workspace-source auto` so
     runtime repair and PerfOpt can discover or materialize writable sources.
   - Use `--workspace-source readonly` only when the user explicitly requests
     evaluation or inspection without source modification.
   - Treat fresh or no-history execution as an experience-store policy, not a
     source policy. It never implies `readonly`; keep `auto` unless the user
     separately prohibits source modification.
   - Before launch, if the command contains `--workspace-source readonly`,
     identify the user's explicit no-source-modification request. If there is
     none, remove the option and use the default `auto`.
   - Apply the source policy to every run. It governs runtime activation and
     eligible repair work even when performance mode is `off` or `measure`;
     PerfOpt source modification remains specific to `optimize`.

3. **Plan**
   - Maintain the native task plan for every multi-stage run. Use
     `update_plan` in Codex and `TaskCreate` / `TaskUpdate` in Claude Code.
   - Before launch, show every applicable top-level task derived from the
     resolved run intent. Always include validation, baseline health,
     quantization or search, the real accuracy gate, and FINAL reporting. Add
     throughput for `measure` or `optimize`, and add the PerfOpt decision plus
     conditional bottleneck, optimization, and retention tasks for `optimize`.
     Do not use one universal task list for every performance mode.
   - Keep exactly one plan step in progress.
   - Mark an applicable conditional task as completed with the skip reason when
     the Orchestrator does not need it. Add candidate, retry, repair, and kernel
     subtasks dynamically when command output, `state.json`, `progress.json`,
     or an artifact shows that they exist.
   - On resume or context compaction, reconstruct the complete run-specific
     plan from the persisted run specification and current session evidence
     rather than conversation memory.

4. **Execute**
   - Launch through `quark-quant-perf`; do not call internal stage functions as
     a replacement pipeline.
   - Keep long runs in a managed terminal or tool session that can be polled;
     do not detach them with `nohup` or shell `&`.
   - Keep the exact command, session directory, and process handle visible.
   - Do not pause, signal, or terminate an active process without explicit user
     approval unless immediate system safety or data loss is at risk.

5. **Monitor and summarize**
   - Poll process output while the run is active. When no new output arrives,
     inspect `state.json` and `progress.json` about every 30 seconds. Do not
     redraw an unchanged plan, but always refresh it before a status response
     or wait.
   - Use `progress.json` for the current `stage` and `stage_detail`, and use
     `state.json` for completed, failed, resumed, and terminal facts.
   - During mixed-precision search, show the persisted
     `total_configs_evaluated` / `total_configs_available` counts. Also account
     for `candidate_cursor`, `candidate_queue`, `partial_timeout`, and
     `termination_reason` when describing export fallback or a salvaged search.
   - Keep all applicable top-level tasks visible throughout the run. Enrich the
     active task and add retry, repair, candidate, and kernel subtasks when
     their corresponding attempts, measurements, journeys, or artifacts appear.
   - Use terminal output to enrich the current task, not as the sole evidence
     for completion.
   - Distinguish search accuracy from the authoritative real accuracy gate.
   - Distinguish local kernel speedup from final retained end-to-end gain.
   - Continue through FINAL even when accuracy or performance targets fail.
   - Report the terminal status, selected quantization, accuracy, throughput,
     retained patches, rejected attempts, and artifact paths.

## Command Rules

Use current canonical option names:

```text
--max-search-candidates
--search-timeout
--layer-precision-candidates
--kv-cache-precision-candidates
--search-gsm8k-num-samples
--geak-direction-budget
```

Do not emit removed historical names:

```text
--max-rounds
--layer-mode
--kv-cache-mode
--geak-budget
```

Convert percentages to ratios:

- 3% maximum accuracy drop → `--accuracy-gap 0.03`
- 35% target speedup → `--target-gain 1.35`
- 2x throughput target → `--target-gain 2.0`

Performance intent:

- No performance request → omit both options; the effective mode is `off`.
- Throughput measurement only → `--performance-mode measure`.
- Optimization target → `--target-gain MULTIPLIER`; this implies
  `--performance-mode optimize`.
- Explicit optimize without a target uses the CLI default target of `1.2`.

Precision and backend intent:

- No requested layer modes → omit `--layer-precision-candidates` and use the
  GPU-specific defaults.
- Explicit layer modes → pass only the named quantized modes; `native` remains
  an implicit fallback.
- No requested backend → omit the MXFP4/W4A8 backend flags and preserve their
  CLI or environment defaults.
- Explicit backend → pass only the corresponding backend option; backend
  selection does not add a precision mode to the search space.

Use `--tracelens-gpu-arch-json` when the installed TraceLens package lacks the
requested GPU architecture data. Do not copy architecture data into
site-packages.

Use `--retry-accuracy-gate` when the real accuracy gate must be reopened. Use
`--retry-perfopt` only when reusable accuracy and quant-only throughput
evidence still match the current runtime fingerprint.

## Monitoring Rules

- A status request, parameter correction, turn interruption, or new message is
  not permission to stop an active task.
- Do not infer current state from old warnings in logs.
- Use `state.json`, terminal command output, and FINAL reports for conclusions.
- A local GEAK result is not a final performance result.
- A patch is retained only after the managed correctness, accuracy, and
  end-to-end performance gates accept it.
- Preserve unrelated user changes and dirty worktrees.

## Recovery

- Reuse the same session when the user asks to continue or resume.
- Diagnose the failing boundary before changing code or configuration.
- Use `--recheck-baseline` only to bypass an exact cached baseline failure.
- Use `--retry-accuracy-gate` after a failed accuracy stage or after a runtime
  change invalidates the saved accuracy fingerprint.
- Use `--retry-perfopt` after `perf_failed` only when upstream evidence remains
  reusable.
- Do not silently change TP, model, ISL, OSL, concurrency, quantization modes,
  KV-cache modes, or benchmark implementation during recovery.
- If source paths or runtime identity change, expect fingerprints to invalidate
  cached measurements and rerun the required managed gates.
- Preserve failure logs and candidate evidence even when the candidate source
  change is reverted.
