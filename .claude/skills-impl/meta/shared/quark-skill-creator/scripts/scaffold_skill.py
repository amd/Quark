#!/usr/bin/env python3
"""Scaffold a Quark Agent Skill that conforms to the project format contract.

Mechanically produces the directory layout, frontmatter, and required sections
so the author cannot forget a field or a section. Skill *content* is filled in
by the author (or by Claude during the `quark-skill-creator` interactive
flow) — this script only writes scaffolding.

Run from any working directory; the script locates the repo root from its own
path (it lives at .claude/skills-impl/meta/shared/quark-skill-creator/scripts/).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

VALID_LAYERS = ("l0-foundation", "l1-atomic", "l2-workflows", "l3-recipes", "meta")
NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
MAX_DESCRIPTION_WORDS = 100

# Paths that resolve in the deployed Quark host repo but may be absent in a
# standalone skills checkout. validate_skill.py warns rather than blocks for
# these; scaffold_skill.py mirrors that policy so source_knowledge entries
# pointing at upstream Quark code do not unconditionally block the scaffold.
UPSTREAM_PREFIXES = (
    "docs/source/",
    "quark/",
    "examples/",
    "tools/",
    "pyproject.toml",
    "requirements.txt",
)

REPO_ROOT = Path(__file__).resolve().parents[5]
SKILLS_IMPL = REPO_ROOT / ".claude" / "skills-impl"
SKILLS_STUBS = REPO_ROOT / ".claude" / "skills"


EN_SECTIONS = [
    ("## Purpose", "TODO: One paragraph describing the stable responsibility boundary of the skill."),
    ("## Inputs", "- TODO: required inputs\n- TODO: optional defaults"),
    (
        "## Outputs",
        "TODO: describe the primary artifact and any side effects.\n\nSchema (if applicable): [`<primary_artifact>.schema.json`](../../shared/contracts/<primary_artifact>.schema.json)",
    ),
    (
        "## Interaction Flow",
        "1. Intake — TODO\n2. Route — TODO\n3. Plan — TODO\n4. Confirm — TODO\n5. Execute or Summarize — TODO",
    ),
    ("## Recovery", "- TODO: common failure modes\n- TODO: recovery advice"),
    (
        "## Notes",
        "- TODO: traceability to upstream Quark docs, CLI, or code\n- TODO: guardrails that should not be silently skipped",
    ),
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for skill scaffolding.

    Args:
        argv: Command-line argument list (typically ``sys.argv[1:]``).

    Returns:
        argparse.Namespace with the scaffold options: ``name``, ``layer``,
        ``primary_artifact``, ``description``, ``source_knowledge``,
        ``with_stub``, ``with_evals``, ``eval_categories``, ``force``.
    """
    p = argparse.ArgumentParser(
        description="Scaffold a Quark Agent Skill (SKILL.md, optional entry stub).",
    )
    p.add_argument("--name", required=True, help="Skill name, ^[a-z][a-z0-9-]{0,63}$")
    p.add_argument("--layer", required=True, choices=VALID_LAYERS)
    p.add_argument(
        "--primary-artifact",
        required=True,
        help="Filename of the main output artifact, e.g. validation_report.md or foo_result.json",
    )
    p.add_argument(
        "--description", required=True, help="≤ 100 words, third-person, includes WHAT + WHEN + trigger phrases"
    )
    p.add_argument(
        "--source-knowledge",
        nargs="+",
        default=[],
        help="Repo-root-relative upstream Quark paths the skill is derived from",
    )
    p.add_argument(
        "--with-stub", action="store_true", help="Also write .claude/skills/<name>/SKILL.md (user-facing entry)"
    )
    p.add_argument(
        "--with-evals", action="store_true", help="Also scaffold evals/evals.json with one placeholder per category"
    )
    p.add_argument(
        "--eval-categories",
        nargs="+",
        choices=("routing", "planning", "artifact", "recovery"),
        default=["routing", "planning", "artifact", "recovery"],
        help="Eval categories to scaffold when --with-evals is set",
    )
    p.add_argument("--force", action="store_true", help="Overwrite an existing skill directory")
    return p.parse_args(argv)


def validate_inputs(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """Validate scaffold inputs.

    Returns a (errors, warnings) tuple. Errors block the scaffold (caller
    returns exit code 2); warnings are surfaced to stderr but do not block.
    The warn/block policy mirrors validate_skill.py so a source_knowledge
    entry pointing at upstream Quark code (e.g. ``quark/torch/foo.py``) only
    warns when absent locally — those paths are expected to resolve in the
    deployed Quark host repo, not in a standalone skills checkout.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not NAME_RE.match(args.name):
        errors.append(f"--name {args.name!r} does not match {NAME_RE.pattern}")
    word_count = len(args.description.split())
    if word_count > MAX_DESCRIPTION_WORDS:
        errors.append(f"--description is {word_count} words; format contract caps it at {MAX_DESCRIPTION_WORDS}")
    for sk in args.source_knowledge:
        if sk.startswith(("/", "http://", "https://")) or ".." in Path(sk).parts:
            errors.append(f"--source-knowledge {sk!r} is not repo-root-relative")
            continue
        if not (REPO_ROOT / sk).exists():
            if any(sk.startswith(p) for p in UPSTREAM_PREFIXES):
                warnings.append(
                    f"--source-knowledge {sk!r} not present locally — expected to resolve in the "
                    f"deployed Quark host repo (matches upstream prefix)"
                )
            else:
                errors.append(f"--source-knowledge {sk!r} not found in repo at {REPO_ROOT / sk}")
    return errors, warnings


def render_frontmatter(args: argparse.Namespace) -> str:
    """Render the YAML frontmatter block for a scaffolded SKILL.md.

    Args:
        args: Parsed command-line arguments containing ``name``,
            ``description``, ``layer``, ``primary_artifact``, and
            ``source_knowledge`` fields.

    Returns:
        Formatted YAML frontmatter string enclosed in ``---`` delimiters,
        ready to be concatenated with the rendered body.
    """
    sk_lines = "\n".join(f"  - {s}" for s in args.source_knowledge) or "  - TODO: repo-root-relative upstream path"
    return (
        "---\n"
        f"name: {args.name}\n"
        "description: >\n"
        f"  {args.description.strip()}\n"
        f"layer: {args.layer}\n"
        f"primary_artifact: {args.primary_artifact}\n"
        "source_knowledge:\n"
        f"{sk_lines}\n"
        "---\n"
    )


def render_body(name: str, sections: list[tuple[str, str]]) -> str:
    """Render the markdown body for a scaffolded SKILL.md.

    Args:
        name: Skill name to use as the top-level ``# <name>`` heading.
        sections: Ordered list of ``(heading, body)`` tuples — each emits
            the heading followed by the body separated by blank lines.

    Returns:
        Formatted markdown body ending in a single trailing newline.
    """
    parts = [f"# {name}", ""]
    for heading, body in sections:
        parts.extend([heading, "", body, ""])
    return "\n".join(parts).rstrip() + "\n"


def write_file(path: Path, content: str, force: bool) -> None:
    """Write ``content`` to ``path``, creating parent directories as needed.

    Args:
        path: Target file location.
        content: String content to write (UTF-8).
        force: If False and the target already exists, refuse to overwrite
            and raise ``SystemExit`` with a hint to pass ``--force``.

    Raises:
        SystemExit: If the file exists and ``force`` is False.
    """
    if path.exists() and not force:
        raise SystemExit(f"refuse to overwrite existing file: {path} (use --force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_stub(args: argparse.Namespace) -> str:
    """Render the entry-stub SKILL.md for a user-facing skill.

    The stub is a thin wrapper that Claude Code auto-discovers under
    ``.claude/skills/<name>/SKILL.md`` and that delegates to the impl
    skill at ``.claude/skills-impl/<layer>/<name>/SKILL.md``. Only
    user-facing skills (most L1 atomic, all L2 / L3) get a stub —
    internal and meta skills are invoked via ``/skill <name>`` and
    do not have one.

    Args:
        args: Parsed command-line arguments containing ``name``,
            ``description``, and ``layer``.

    Returns:
        Stub markdown content (frontmatter + one-line read-and-follow body).
    """
    return (
        "---\n"
        f"name: {args.name}\n"
        "description: >\n"
        f"  {args.description.strip()}\n"
        "---\n"
        "\n"
        f"Read and follow the instructions in `.claude/skills-impl/{args.layer}/{args.name}/SKILL.md`.\n"
    )


def main(argv: list[str]) -> int:
    """Entry point: parse args, validate, scaffold the skill, print summary.

    Args:
        argv: Command-line arguments excluding the program name (i.e.
            ``sys.argv[1:]``).

    Returns:
        Exit code: 0 on success, 2 on validation error (bad name, over-budget
        description, unresolvable non-upstream source_knowledge path) or
        when the target directory already exists without ``--force``.
    """
    args = parse_args(argv)
    errors, warnings = validate_inputs(args)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        return 2

    target_dir = SKILLS_IMPL / args.layer / args.name
    if target_dir.exists() and not args.force:
        print(
            f"error: target directory already exists: {target_dir}\n       pass --force to overwrite",
            file=sys.stderr,
        )
        return 2

    frontmatter = render_frontmatter(args)
    en_path = target_dir / "SKILL.md"
    write_file(en_path, frontmatter + "\n" + render_body(args.name, EN_SECTIONS), args.force)
    written = [en_path]

    if args.with_stub:
        stub_path = SKILLS_STUBS / args.name / "SKILL.md"
        write_file(stub_path, render_stub(args), args.force)
        written.append(stub_path)

    if args.with_evals:
        # Delegate to scaffold_evals.py to keep the evals format authoritative
        # in one place; this script just shells out.
        evals_cmd = [
            sys.executable,
            str(Path(__file__).parent / "scaffold_evals.py"),
            str(target_dir),
            "--categories",
            *args.eval_categories,
        ]
        if args.force:
            evals_cmd.append("--force")
        result = subprocess.run(evals_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"warning: scaffold_evals.py failed: {result.stderr}", file=sys.stderr)
        else:
            # Surface the delegate's stdout so the user sees its 'Scaffolded:'
            # and 'Next steps:' output (capture_output=True would otherwise
            # swallow it).
            if result.stdout:
                print(result.stdout, end="")
            written.append(target_dir / "evals" / "evals.json")

    print("Scaffolded:")
    for p in written:
        line_count = len(p.read_text(encoding="utf-8").splitlines())
        print(f"  {p.relative_to(REPO_ROOT)}  ({line_count} lines)")

    print()
    print("Next steps:")
    print("  - Fill TODO sections with real content (Claude does this in the quark-skill-creator flow).")
    print("  - Confirm SKILL.md body stays ≤ 315 lines and description ≤ 100 words.")
    print("  - Run scripts/validate_skill.py <skill-dir> to mechanically check the format contract.")
    if args.source_knowledge:
        print("  - Run quark-doc-drift-check to confirm source_knowledge alignment.")
    if args.with_evals:
        print("  - Fill the evals/evals.json TODOs and hand to quark-eval-runner for the iteration loop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
