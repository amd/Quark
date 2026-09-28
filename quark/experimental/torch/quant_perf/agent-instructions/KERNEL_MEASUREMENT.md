# Workload kernel measurement

Apply this addendum during Benchmark preparation, before freezing the
COMMANDMENT and performance harness. GEAK owns the harness, optimization,
verification and final measurement. Keep the correctness oracle immutable.
Use the supplied workload cases and weights without changing their meaning.

## Build the implementation being measured

Compile the candidate from its current workspace and the baseline from GEAK's
frozen source. Store generated build metadata, source-link overlays and binary
caches under the workspace's `build/` directory, which GEAK excludes when
copying workspaces. Recreate them from the current paths after a copy; a
candidate source link must never retain a parent workspace's candidate path.
Check the resolved compiler inputs before testing each candidate, and ensure
source changes invalidate the corresponding compiled code.

## Reproduce inputs and data reuse

Call the supplied input fixture directly when available; otherwise use the
production input producer. Do not independently regenerate a fixture's routing
or padding. Preserve scalar
arguments, strides, aliasing, masks, routing and padding, not just tensor shapes.
Derive reported dimensions from the actual tensors. A kernel's indexing factor
does not determine how many routing records its producer creates. Keep these
inputs fixed across the pristine baseline and candidate. Record the producer
and its parameters in the COMMANDMENT alongside the measurement policy.

Freeze the data-reuse policy from the operator's workload before measuring;
model size alone does not establish whether its operands are reused. When
flushing is appropriate, call GEAK's existing `flush_cache()` before each
sample, outside the Event interval. Use its existing configuration and record
the size; do not write a separate flush implementation. For `inner > 1`, prepare
independent operand buffers for every call in the batch, preserving their
values, layout and alias relationships; have the callable rotate through them.
Flushing once and then repeating one buffer makes all but the first call warm.
Preserve shared buffers only when the production workload reuses that data.
Allocate and initialize everything outside measurements. Do not choose a
buffer pool or cache policy by how closely its latency fits the trace.

## Measure the device work

The trace's `baseline_latency_ms` measures device execution. Events around a
Python loop or a graph replay can include GPU idle time while the CPU submits
work. Graph replay alone does not prove that a measurement excludes this time.
Do not substitute trace values for measured values or apply correction factors.

For a generated PyTorch performance harness, provide a task-local timing helper
accepting a zero-argument callable. Warm up compilation, allocations and Event
initialization before measurement. The callable must complete its device work
on the measured stream; join any other streams with device dependencies.

Queue a GPU delay **outside** the Event interval so the CPU can submit the
entire batch before the GPU reaches its start. Use `torch.cuda._sleep(cycles)`
on the measured stream. For each sample, the ordering is:

```python
torch.cuda._sleep(cycles)
start.record()
for _ in range(inner):
    call()
end.record()
primed = not start.query()  # Query AFTER submitting the complete batch.
end.synchronize()
```

Only `primed=True` proves that the complete batch was queued before its start.
Measure `start.elapsed_time(end) / inner` only for such samples. During
preparation, calibrate the delay over at most seven sizes, from 1,000,000 cycles
through 64,000,000 by doubling. Check the condition on EVERY measured sample;
a failed check invalidates the measurement. Do not silently discard failed
samples, report their latency as device time, or fall back to unprimed Events.
Unavailable GPU delay support or an incompatible callable is a measurement
failure with an explicit reason, not permission to invent a timing result.

Obtain a valid single-call measurement of the pristine baseline first. For
cases below 0.1 ms use `inner=32` to amortize Event overhead; otherwise use
`inner=1`. Freeze that choice per case for BOTH baseline and candidate. Keep
warmup and cache conditions identical between them; cache preparation belongs
outside the Event interval. Use the measured single-call latency to choose
`inner`, never the trace latency. Do not select batch sizes to fit the trace.

## Verify before freezing

Use the same helper for baseline, candidate, BENCHMARK and FULL_BENCHMARK.
Check correctness first. In preparation, test an unchanged implementation
against itself, and inject a 2 ms CPU delay after `start.record()` before the
batch. With a sufficient calibrated GPU delay, each check must remain primed
and the median device time must stay within 5% of the undelayed measurement.
An insufficient delay must fail the check. This is a preparation check, not
extra work in every optimization round. Use the existing PROFILE command to
cross-check device timings when the workload baseline does not reproduce.
Profiling can perturb short kernels; do not replace an unprofiled benchmark
with a profiled number. Trace kernel durations do not contain CPU-submit gaps.

For trace-backed cases, use the arithmetic mean of valid samples to match the
trace's mean latency. Freeze this statistic for baseline and candidate in the
COMMANDMENT. If alignment fails, run GEAK's existing profiler entry and inspect
its actual GPU kernel names, launch geometry and durations; a harness flag
named `--profile` alone is not profiling evidence. Fix only confirmed input or
measurement errors during Benchmark preparation, then repeat validation before
freezing the harness. Preserve failed measurements and reasons. Do not tune
inputs, cache policy or the statistic to pass the alignment gate.

Publish `baseline_timing.json` atomically only after the complete measurement
passes the supplied workload alignment requirement. The recorded latencies
must be the actual benchmark outputs, with the same case identities and counts.
Set `workload_aligned` from the conjunction of the per-case alignment checks.
On any failure, report `workload_aligned=false` and the reason; do not optimize
against a baseline that has not passed.

Preserve the usual `GEAK_RESULT_LATENCY_MS` output and also emit
`GEAK_TIMING_RECEIPT` using GEAK's existing contract. Compute `primed` from the
checks above, never from a timer label. Retain per-case evidence for every
measured leg so Director validation can cover BOTH the frozen baseline and
the final candidate. Set `all_primed` only when all those checks passed;
missing evidence cannot become `True`. Record the measurement commands and
this receipt requirement in the COMMANDMENT so verification and Director use
the same instrument. Emit one aggregate receipt with the existing schema:

```text
GEAK_TIMING_RECEIPT: {"all_primed": <bool>, "timer_unprimed": <bool>, "cases": {"<case>": {"baseline": {"primed": <bool>}, "current": {"primed": <bool>}}}}
```

Populate the booleans from the measured checks for both legs. Keep the harness
out of the optimized source patch.
