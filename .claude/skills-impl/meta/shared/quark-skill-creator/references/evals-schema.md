# Per-Skill Evals Schema

Each skill in this repo can ship its own `evals/evals.json`. These evals are consumed by `quark-torch-eval-runner` during the iteration loop and at every governance pass. They are how a skill proves — empirically, not just structurally — that its description routes correctly, its plan is consistent, its artifact validates, and its recovery path actually unblocks the user.

This file defines the JSON schema, the four eval categories, and how the evals integrate with the governance loop.

## File location

```text
.claude/skills-impl/<layer>/<skill-name>/
└── evals/
    └── evals.json
```

`scaffold_evals.py` writes a starter file with one placeholder per category.

## Schema

```json
{
  "skill_name": "quark-foo",
  "skill_path": ".claude/skills-impl/l1-atomic/quark-foo",
  "evals": [
    {
      "id": 1,
      "category": "routing",
      "name": "fp8-llama-routing",
      "prompt": "I want to quantize Llama-3-8B to FP8.",
      "expected_skill": "quark-torch-ptq",
      "expectations": [
        "Claude invokes quark-torch-ptq (or its workflow) first",
        "The plan references quark-torch-quant-plan as the next step"
      ]
    },
    {
      "id": 2,
      "category": "planning",
      "name": "fp8-default-exclusions",
      "prompt": "Plan FP8 quantization for a Qwen3-8B at /models/qwen3-8b.",
      "input_artifacts": ["evals/inputs/qwen3-8b.model_analysis.json"],
      "expectations": [
        "quant_plan.global_scheme == 'fp8'",
        "quant_plan.exclude_layers includes 'lm_head'",
        "quant_plan.requires_confirmation is true when kv_cache_dtype is overridden"
      ]
    },
    {
      "id": 3,
      "category": "artifact",
      "name": "manifest-schema-conformance",
      "prompt": "Run the workflow against this plan and emit a run manifest.",
      "input_artifacts": ["evals/inputs/fp8.quant_plan.json"],
      "expectations": [
        "Output validates against shared/contracts/run_manifest.schema.json",
        "Output references the export.formats key"
      ]
    },
    {
      "id": 4,
      "category": "recovery",
      "name": "transformers-version-mismatch",
      "prompt": "PTQ failed with: AttributeError: 'PreTrainedTokenizerFast' object has no attribute 'get_max_length'.",
      "expectations": [
        "Diagnosis names the transformers version as the root cause",
        "Fix command pins a working transformers version"
      ]
    }
  ]
}
```

## Field semantics

| Field | Required | Notes |
|-------|----------|-------|
| `skill_name` | yes | Must match the skill's `frontmatter.name`. |
| `skill_path` | yes | Repo-root-relative path to the skill directory. Used by `quark-torch-eval-runner` to resolve the target. |
| `evals[].id` | yes | Unique integer within this file. |
| `evals[].category` | yes | One of `routing`, `planning`, `artifact`, `recovery` — matches the four MVP eval classes. |
| `evals[].name` | yes | Short kebab-case identifier; surfaces in the `quark-torch-eval-runner` report. |
| `evals[].prompt` | yes | Realistic user-style prompt — avoid abstract requests. Include concrete model names, paths, error messages. |
| `evals[].expected_skill` | routing only | The skill that should be invoked first. |
| `evals[].input_artifacts` | optional | List of repo-root-relative paths to artifact files the eval depends on. |
| `evals[].expectations` | yes | List of objectively verifiable statements. Each must be checkable from the run transcript or output files. |

## The four categories

These map 1:1 to `quark-torch-eval-runner`'s sections in `validation_report.md`. Every new skill should land **at least one** eval in the relevant categories:

- **routing** — does the description trigger the skill on realistic user phrasing? Required for any skill with a `.claude/skills/` entry stub.
- **planning** — given the inputs, does the proposed plan match the expected shape? Required for skills that emit a planning artifact (`quant_plan.json`, `model_analysis.json`).
- **artifact** — does the produced artifact validate against its schema in `shared/contracts/`? Required for any skill that emits a contract artifact.
- **recovery** — does the skill's recovery path correctly diagnose and unblock a known failure? Required for any skill referenced by `quark-torch-debug` or that has a non-trivial `## Recovery` section.

A pure `meta` skill (like `quark-skill-creator` itself) typically only needs routing and artifact evals.

## Integration with the iteration loop

After writing the skill draft and running `scaffold_evals.py`:

1. Fill in 1–4 evals per applicable category. Quality > quantity — three good evals beat ten weak ones.
2. Hand the skill + `evals/evals.json` to `quark-torch-eval-runner`. It walks each prompt manually (per the protocol in `meta/quark-torch-eval-runner/SKILL.md`) and records pass/fail per `expectations[]`.
3. Read the resulting `validation_report.md`. If any expectation fails, treat it like the official skill-creator's iteration loop: improve the skill body or sharpen the eval (vague expectations are a worse problem than failing skills), and re-run.
4. Stop when all relevant categories pass and at least one eval per category exercises a non-trivial branch.

## Writing good expectations

An expectation is *discriminating* when it passes only if the skill genuinely succeeds and fails when it does not.

Bad — surface compliance:

- "The output is a JSON file."
- "The skill produced a non-empty result."

Good — substance:

- "`quant_plan.global_scheme` equals `'fp8'`."
- "`run_manifest.export.formats` includes `'hf_format'`."
- "Diagnosis includes the string `transformers` and a specific version pin."

If an expectation would pass for a clearly wrong output, rewrite it. If a bug you observed in real use would not be caught by any expectation, add one.

## Anti-patterns

- **Evals that test Claude's capabilities, not the skill.** "Claude can read a JSON file" is not an eval for `quark-torch-quant-plan` — that's just Claude. Test the skill's specific decisions.
- **Trigger-only evals for non-routing categories.** A planning eval whose only expectation is "the right skill was invoked" should be a routing eval.
- **Evals that depend on external network state.** No live HuggingFace downloads, no remote model URLs that may disappear. Pin everything to local fixtures under `evals/inputs/`.
- **Evals without expectations.** A prompt with no expectations records execution but cannot fail — it has no signal.
