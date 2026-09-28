# Quant-Perf Session Lifecycle

## Canonical phases

1. Validate inputs and environment.
2. Check baseline framework health.
3. Quantize or search configurations.
4. Load the exported checkpoint and run the real quantized accuracy gate.
5. If performance mode is `measure` or `optimize`, benchmark baseline and
   quantized throughput.
6. If performance mode is `optimize`, decide whether PerfOpt is required.
7. Run TraceLens bottleneck analysis when optimization is required.
8. Optimize eligible kernels with GEAK or vendor tuning.
9. Validate and retain accepted changes.
10. Run terminal cleanup and generate FINAL reports.

Steps 5-9 are optional according to the selected performance mode and whether
the target is already met. Every terminal path runs the same FINAL stage once.

## Session artifacts

Inspect these files in priority order:

```text
<session>/state.json
<session>/progress.json
<session>/reports/final.json
<session>/session_breakdown.json
<session>/session_report.md
<session>/kernel_source_resolution.json
<session>/geak/
```

`state.json` owns resumable state. `progress.json` is a live human-facing
projection and may retain historical warnings. FINAL reports own the terminal
summary.

## Common terminal states

| State | Meaning |
|---|---|
| `done` | Accuracy passed and all requested performance work completed |
| `perf_failed` | Accuracy passed, but final gain remained below target |
| `failed` | Quantization, quantized accuracy, or required throughput measurement failed |
| `base_unhealthy` | The unquantized model was unhealthy in the selected runtime |

Terminal stage and `report_status` are independent. A reporting failure
preserves the terminal run result; use `quark-quant-perf report` to retry
artifact generation.

## Resume decisions

- Reuse the session without a retry flag when it stopped at a resumable
  non-terminal stage.
- Use `status` for a running session. Use `report` only after a terminal state
  has been recorded.
- Use `--retry-accuracy-gate` to reopen a terminal accuracy failure or when the
  runtime/source fingerprint changed.
  A failed mixed-precision search without a usable candidate also reopens at
  quantization with this flag. Matching healthy baseline evidence is reused;
  the normal runtime fingerprint checks still apply.
- Use `--retry-perfopt` to reopen `perf_failed` only when accuracy and the
  baseline/quantized throughput pair remain reusable.
- Do not edit `state.json` manually to bypass fingerprint or stage checks.
- If source repositories change, rerun every gate invalidated by the new
  runtime fingerprint.

## Performance interpretation

```text
accuracy gap = (baseline score - quantized score) / baseline score
quant-only gain = quantized TPS / baseline TPS
final gain = retained-patch TPS / baseline TPS
```

Microbenchmark speedup is candidate evidence only. KEEP requires the managed
correctness checks and a real end-to-end gain above the effective retention
floor. Rejected candidates may still be useful diagnostics but are not final
artifacts.
