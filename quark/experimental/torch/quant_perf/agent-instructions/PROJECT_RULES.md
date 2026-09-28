# Quark Quant-Perf Project Rules

## Ask Before Pausing or Stopping

Ask the user for explicit approval before pausing, interrupting, or terminating
any running Quant-Perf, GEAK, vLLM, evaluation, benchmark, or related task.

A status request, parameter correction, turn interruption, or new message is
not permission to stop an existing task. Stop without approval only to prevent
an immediate system-safety or data-loss risk, and report the reason promptly.

## Use the Managed Pipeline

Run quantization, mixed-precision search, accuracy evaluation, throughput
benchmarking, kernel optimization, and final reporting through the
`quark-quant-perf` CLI and Quant-Perf Orchestrator.

Do not replace pipeline stages with custom GSM8K or throughput scripts, and do
not substitute proxy metrics for the required measurements. Treat `cli.py` as
the source of truth for CLI options and defaults.

## Maintain Dynamic Run Status

For every multi-stage Quant-Perf workflow, create and continuously update the
available native task or plan list. Keep it through parameter corrections,
status requests, background waits, context compaction, and resume.

Keep the command in a managed terminal or tool session and retain its process
handle and session directory. Before launch, show all applicable top-level
tasks derived from the resolved run intent, including conditional tasks that
may later be marked skipped. Do not use one universal lifecycle checklist for
every performance mode.

Poll active work at bounded intervals and refresh the plan when the observed
stage, detail, attempt, candidate, or result changes. Also refresh before each
status response, wait, or yield, and rebuild it from session evidence after
context compaction or resume. Add runtime-specific candidate, retry, repair,
and kernel subtasks only when command output, `state.json`, `progress.json`, or
produced artifacts show that they exist. Keep exactly one task in progress and
do not replace the native list with only a Markdown checklist.

## Use Descriptive Names

Prefer function and variable names that make their purpose clear at the point
of use. Name functions for the action or result they provide, variables for the
value or state they represent, and files or modules for their primary
responsibility. Avoid opaque abbreviations and generic placeholders.

## Preserve User Work

Preserve dirty worktrees and unrelated user changes. Do not discard, overwrite,
or include unrelated changes in a task commit. Do not run destructive Git
cleanup or history-changing commands without explicit user approval.
