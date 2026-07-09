---
name: quark-skill-creator
description: >
  Author or restructure a Quark Agent Skill so it conforms to this project's
  template, contracts, and layer rules. Use when a maintainer says "create a
  new skill", "add a skill for X", "scaffold a skill", "draft a SKILL.md", or
  when an existing skill needs a structural rewrite to match the format
  contract. Walks the author through capture, draft, test, iterate, and
  governance handoff. This is a maintainer-facing meta skill, not an end-user
  PTQ tool.
layer: meta
primary_artifact: validation_report.md
source_knowledge:
  - .claude/skills-impl/CONTRIBUTING.md
  - .claude/skills-impl/shared/templates/skill-template.md
  - docs/agent_skills/skill-format-contract.md
  - docs/agent_skills/interaction-contract.md
  - docs/agent_skills/artifact-contracts.md
  - docs/agent_skills/architecture.md
  - docs/agent_skills/governance.md
---

# quark-skill-creator

## Purpose

Codify this repo's authoring conventions so a new skill is correct on the first try. The Quark skill system is governed by five separate documents under `docs/agent_skills/` (`skill-format-contract.md`, `interaction-contract.md`, `artifact-contracts.md`, `architecture.md`, `governance.md`) plus the contributing guide. Reading them all before every authoring task is slow and error-prone — missed `primary_artifact`, wrong layer, hand-invented artifacts that bypass the contract surface. This skill replays the rules in workflow order and pairs the conversation with deterministic scaffolding + validation scripts so structural compliance is mechanical and the human focus stays on content.

## Inputs

- **Required**: skill name (`^[a-z][a-z0-9-]{0,63}$`), one-sentence intent, target layer, primary artifact filename, `description` draft (≤ 100 words, third-person, with trigger phrases).
- **Optional**: `source_knowledge` repo-relative paths, whether to emit a user-facing entry stub, whether to scaffold evals, prior conversation context to capture as a workflow.

## Outputs

- **Primary artifact**: `validation_report.md` — the report produced by `scripts/validate_skill.py` after Phase 5, capturing the mechanical-validation result of the freshly authored skill. The author should commit this report alongside the new skill so reviewers can see the validation evidence (per project governance practice for meta skills).
- `.claude/skills-impl/<layer>/<skill-name>/SKILL.md` — main authored skill (side-effect of the scaffold).
- `.claude/skills-impl/<layer>/<skill-name>/evals/evals.json` — optional starter evals (recommended for non-trivial skills; side-effect when `--with-evals`).
- `.claude/skills/<skill-name>/SKILL.md` — entry stub, only when the skill is user-facing (L1/L2/L3). Meta skills do not get a stub.

The skill file format itself is governed by `docs/agent_skills/skill-format-contract.md` (prose), not by a JSON schema in `shared/contracts/`.

## Interaction Flow

This skill follows the project's five-stage interaction contract; the **Authoring Workflow** below is the concrete sequence the Plan + Execute stages walk through.

1. **Intake** — collect skill name, intent, trigger phrases, layer hint, primary artifact filename. If the maintainer brought a draft from prior conversation, extract these from there first and only ask for gaps.
2. **Route** — apply the layer decision tree in `references/layer-decision.md`. Decide whether an entry stub is needed (user-facing → yes, internal/meta → no) and whether evals are needed (non-trivial skills → yes).
3. **Plan** — present the proposed frontmatter, section outline, and exact file paths *before* any write. Verify length budgets and that `source_knowledge` paths exist (or match the upstream Quark layout).
4. **Confirm** — explicit user approval is mandatory before running `scaffold_skill.py` (it writes files), before any `--force` overwrite, and before committing.
5. **Execute or Summarize** — run the scaffold, run the validator, run the reviewer subagent, summarize what was created and what governance follow-ups (`quark-torch-doc-drift-check`, `quark-torch-skill-sync`, `quark-torch-eval-runner`) the maintainer should consider — but never auto-trigger them.

## Authoring Workflow

The Plan and Execute stages walk through these phases in order. Each phase has a clear boundary so the maintainer can pause, correct, or hand off.

### Phase 1: Capture Intent

Pull what's already known from the conversation before asking. The current conversation may already contain a workflow the maintainer wants to capture ("turn this into a skill"). If so, extract: the tools used, the sequence of steps, corrections the maintainer made, input/output formats. Then confirm what's missing.

Always confirm:

1. What does this skill do? (one sentence)
2. When should it trigger? (concrete user phrases)
3. What's the primary artifact?
4. Does the skill need empirical evals, or is it purely structural? (See `references/evals-schema.md`.)

### Phase 2: Interview & Research

Proactively ask about edge cases, input/output formats, dependencies on existing artifacts, recovery paths. Read at least one comparable existing skill — for a meta skill read `meta/torch/quark-torch-skill-sync/SKILL.md`; for L1 atomic read `l1-atomic/torch/quark-torch-quant-plan/SKILL.md`; for L2 workflow read `l2-workflows/torch/quark-torch-llm-ptq-workflow/SKILL.md`.

If the skill consumes or produces a contract artifact, read `docs/agent_skills/artifact-contracts.md` to confirm the producer/consumer slot is consistent. If the skill needs a new artifact, **stop and surface the governance question** — do not invent.

### Phase 3: Draft

Present the proposed frontmatter, section outline, and file list before writing anything. Apply the rules in `references/format-rules.md` (frontmatter, length) and the patterns in `references/writing-patterns.md` (template, examples, workflow, conditional, feedback loop).

Then run the scaffold:

```bash
python3 .claude/skills-impl/meta/shared/quark-skill-creator/scripts/scaffold_skill.py \
  --name quark-foo \
  --layer l1-atomic \
  --primary-artifact foo_result.json \
  --source-knowledge docs/source/install.rst quark/torch/foo.py \
  --description "..." \
  [--with-stub] \
  [--with-evals] \
  [--force]
```

The script writes `SKILL.md` (and optionally an entry stub and an `evals/` scaffold) with all required sections pre-stubbed `TODO:`. Fill the TODOs as authored content — the scaffold guarantees structural compliance, not content quality.

### Phase 4: Test Cases

For non-trivial skills, scaffold the evals:

```bash
python3 .claude/skills-impl/meta/shared/quark-skill-creator/scripts/scaffold_evals.py \
  .claude/skills-impl/<layer>/<name> \
  --categories routing planning artifact recovery
```

Fill 1–4 evals per applicable category with **realistic** prompts (concrete model names, paths, error messages — not abstract requests) and **discriminating** expectations (an expectation that would also pass for a clearly wrong output is worse than no expectation). See `references/evals-schema.md` for the schema and the writing-good-expectations guidance.

Pure-routing meta skills (like `quark-skill-creator` itself) typically only need routing evals — others can be skipped.

### Phase 5: Iterate

Run validation in this order:

1. **Mechanical** — `python3 scripts/validate_skill.py <skill-dir>`. Catches frontmatter, length, and section issues. Fix any blocking findings before continuing.
2. **Contractual** — spawn a subagent with the prompt in `agents/reviewer.md`. It reads the skill against the format and artifact contracts and returns a structured report. Apply blocking findings; consider non-blocking suggestions.
3. **Empirical** (if evals exist) — hand the skill + evals to `quark-torch-eval-runner` (see `meta/torch/quark-torch-eval-runner/SKILL.md`). It walks each prompt manually and records pass/fail. Treat failures like the official skill-creator's iteration loop: improve the skill body or sharpen the eval. Re-run.

Stop iterating when validator passes, reviewer reports no blocking findings, and (if applicable) all eval expectations pass.

### Phase 6: Governance Handoff

Summarize what was created. List — but do not auto-trigger — the governance follow-ups the maintainer may want:

- `quark-torch-doc-drift-check` — confirms `source_knowledge` is still aligned with current upstream Quark.
- `quark-torch-skill-sync` — re-audits all skills if the new one introduces a new contract dependency.
- `quark-torch-eval-runner` — re-runs the four MVP eval categories if the new skill changes routing or planning behavior.

Update the index lists in `.claude/skills-impl/README.md` and `CLAUDE.md` if the new skill is user-facing or meta.

## Anti-Patterns

These are project-specific traps the official skill-creator does not warn about:

- **Skipping `primary_artifact`** because "this skill produces a report". Pick a concrete filename — for meta skills the project default is `validation_report.md`. Empty or `<TBD>` fails the format contract.
- **Inventing a new artifact name** outside the eight canonical ones in `docs/agent_skills/artifact-contracts.md`. New artifacts require a JSON schema in `shared/contracts/` and a producer/consumer entry in the doc — never a unilateral addition.
- **Wrong layer placement** — a workflow orchestrator in `l1-atomic`, or routing/drift logic in `l1-atomic` instead of `meta`. Re-read `references/layer-decision.md` when uncertain.
- **External `source_knowledge` paths** — `docs/agent_skills/governance.md` only accepts the surrounding Quark repo as upstream. URLs, sibling-repo paths, or absolute filesystem paths are rejected.
- **Silently overwriting** an existing skill with `--force` — always re-confirm with the maintainer.
- **Trigger-only evals for non-routing categories** — a planning eval whose only expectation is "the right skill was invoked" should be a routing eval. See `references/evals-schema.md`.

## Recovery

- **Layer ambiguous** — default to drafting at `l1-atomic` with a `TODO: layer review` note in `## Notes`, then surface the ambiguity to the maintainer for a second opinion before commit.
- **Proposed artifact not in the canonical list** — stop. Suggest the closest existing artifact (`validation_report.md`, `model_analysis.json`, etc.). If a new one is genuinely needed, write the schema in `shared/contracts/` and the doc entry in `docs/agent_skills/artifact-contracts.md` *first*, then return to the skill.
- **Description blows past 100 words** — return the draft to the maintainer with the word count and the over-budget portions highlighted; do not silently truncate.
- **Target directory exists** — refuse the write, list existing files, ask whether the maintainer wants `--force` or a different name.
- **`source_knowledge` path missing in repo and not matching the upstream Quark layout** — block. The path is the basis of governance; phantom references cannot be committed. (Paths under `docs/source/`, `quark/`, `examples/`, `tools/` are warned but not blocked, since they live in the Quark host repo where the skills are deployed.)
- **Validator or reviewer reports blocking findings** — fix the skill, do not bypass the check. The cost of a one-time fix is far less than the cost of inconsistent skills compounding across the system.

## Notes

- Worked examples to read before authoring: `meta/torch/quark-torch-skill-sync/SKILL.md`, `meta/torch/quark-torch-doc-drift-check/SKILL.md`, `meta/torch/quark-torch-eval-runner/SKILL.md`, and any L1 atomic skill (e.g., `quark-torch-quant-plan`) for input/output discipline.
- Bundled resources:
  - `references/format-rules.md` — frontmatter / length / artifact rules
  - `references/layer-decision.md` — layer admission tree with rubrics and examples
  - `references/writing-patterns.md` — template / examples / workflow / conditional / feedback patterns
  - `references/evals-schema.md` — per-skill `evals.json` schema and integration with `quark-torch-eval-runner`
  - `agents/reviewer.md` — sub-agent prompt for structural + contract review
  - `scripts/scaffold_skill.py` — generate skill directory + SKILL.md (+ optional stub, evals)
  - `scripts/scaffold_evals.py` — generate `evals/evals.json` standalone
  - `scripts/validate_skill.py` — mechanical validation (frontmatter, length, sections)
- Governance is **not auto-triggered**. After creating a skill that touches new upstream Quark files, run `quark-torch-doc-drift-check` manually.
- This skill itself is scaffolded against and validated by the rules it enforces — eat your own dog food when modifying it.
