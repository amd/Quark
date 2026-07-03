# Artifact Contracts

These are the canonical handoff surface between Quark skills. **Each artifact has exactly one producer.** When a downstream skill needs combined facts, it reads multiple artifacts; producers never share write access to the same file.

## `session_context.json`

- producer: `quark-torch-router`
- consumers: all downstream skills
- purpose: capture user goal, selected workflow, constraints, and unresolved questions
- failure fallback: return a partial context with `open_questions` populated
- references: may include `env_context_ref`, `workspace_context_ref`, `pytorch_install_result_ref`, `quark_install_result_ref` pointing at the relevant artifact files when downstream decisions need them

## `env_context.json`

- producer: `quark-env-preflight`
- consumers: `quark-torch-router`, `quark-torch-install`, `quark-install`, `quark-torch-model-intake`, `quark-torch-quant-plan`, `quark-torch-llm-ptq-workflow`, `quark-torch-debug`
- purpose: capture machine-level facts — OS, Python, accelerator, GPU details — independent of any installation action
- failure fallback: leave unresolved fields as `null` or `"unknown"`; surface gaps to `quark-torch-router` for `session_context.json`'s `open_questions`

## `workspace_context.json`

- producer: `quark-workspace-validate`
- consumers: `quark-torch-model-intake`, `quark-torch-llm-ptq-workflow`, `quark-torch-export`
- purpose: record validated model paths, output directories, and repo locations, distinguished as local vs HuggingFace ID
- failure fallback: keep ambiguous references unresolved; surface to `quark-torch-router` for `open_questions`

## `pytorch_install_result.json`

- producer: `quark-torch-install`
- consumers: `quark-install`, `quark-torch-llm-ptq-workflow`, `quark-torch-debug`
- purpose: record what PyTorch build was installed and verified — version, accelerator backend tag, verification status
- failure fallback: emit `status: "failed"` with the exact failing verification command

## `quark_install_result.json`

- producer: `quark-install`
- consumers: `quark-torch-llm-ptq-workflow`, `quark-torch-quant-plan`, `quark-torch-debug`
- purpose: record installed Quark version, optional extras (ONNX runtime, LLM PTQ deps), and verification status
- failure fallback: emit `status: "failed"` with the exact failing verification command

## `model_analysis.json`

- producer: `quark-torch-model-intake`
- consumers: `quark-torch-quant-plan`, `quark-torch-llm-ptq-workflow`, `quark-torch-debug`
- purpose: store model family, structure cues, loading risks, and quantization-sensitive components
- failure fallback: emit `analysis_status: "partial"` and enumerate unresolved model risks

## `quant_plan.json`

- producer: `quark-torch-quant-plan`
- consumers: `quark-torch-llm-ptq-workflow`, `quark-torch-export`, `quark-torch-debug`
- purpose: record the proposed quantization scheme, exclusions, overrides, algorithm choice, and evaluation intent
- failure fallback: emit a draft plan with `requires_confirmation: true`

## `run_manifest.yaml`

- producer: `quark-torch-llm-ptq-workflow`
- consumers: `quark-torch-export`, `quark-torch-debug`, `quark-torch-eval-runner`
- purpose: define commands, inputs, outputs, checkpoints, and expected artifacts for an executable run
- failure fallback: emit a manual-only manifest with missing steps listed under `blocked_by`

## `validation_report.md`

- producer: `quark-torch-debug`, `quark-torch-eval-runner`, governance skills (each produces its own report file — not a shared mutable artifact)
- consumers: users, maintainers, regression review
- purpose: summarize execution results, failures, applied fixes, evidence, and next actions
- failure fallback: write a diagnostic report even when execution did not start
