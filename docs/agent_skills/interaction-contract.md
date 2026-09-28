# Interaction And Safety Contract

## Authority

- The selected public entry and the user's concrete scope govern the run.
- For a self-contained entry, its top-level `SKILL.md` is authoritative for flow, checkpoints, execution, and recovery.
- For a transitional entry, the public stub is authoritative for routing and its delegated `_legacy_impl` body is authoritative for implementation, checkpoints, artifacts, and recovery.
- Skill-specific checkpoints control sequencing, but never waive the baseline safety gates.

## Baseline safety gates

1. Establish the goal, inputs, constraints, target environment, and whether the user wants planning, execution, or both.
2. Before cost or mutation, show material assumptions, risks, exact commands, affected paths or environments, and expected outputs.
3. Obtain explicit approval before package or environment changes, destructive or overwriting writes, heavy PTQ or evaluation, unapproved script or source generation, and process-control actions.
4. Treat approval as scoped to the displayed commands, parameters, paths, and workflow; reconfirm any deviation, added mutation, destructive cleanup, or changed retry.
5. Read-only inspection within the requested scope does not need confirmation. Never infer success from an exit code alone; report observed outputs and remaining risks.

## Current exceptions and checkpoints

### `quark-torch-ptq`

The workflow always stops at four checkpoints, in order:

1. Accept the model analysis.
2. Accept the quantization plan.
3. Approve the exact `quark-cli torch-llm-ptq` command.
4. Accept the verified result.

Plan approval is not execution approval, and direct invocation may not skip any checkpoint.

### `quark-install`

The first gate confirms compute mode: GPU remains the default, while CPU mode requires an explicit
choice after hardware and PyTorch evidence is shown. The second gate shows and approves the exact
environment-changing commands, interpreter, package source, dependency changes, and relevant
compilation risk. Neither CPU acceptance nor a prior install discussion authorizes execution.

### `quark-torch-quant-perf`

An explicit request to execute or resume a concrete Quant-Perf workflow, together with a visible
exact command, is the outer launch authorization for that command and scope. Fixed direct PTQ may
use internal non-interactive delegation under that authorization. This does not bypass the four
checkpoints for direct `quark-torch-ptq` calls. Process control still requires approval unless
immediate safety or data loss is at risk.

## Recovery

When execution cannot continue, return the blocking reason and preserved evidence, identify the
missing artifact or precondition, propose the smallest safe unblock action, and name the checkpoint
or step from which to resume.
