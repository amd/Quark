#!/usr/bin/env python3
"""Validate a Quark Agent Skill against this project's format contract.

Mechanical checks only — semantic / contractual review is the job of
agents/reviewer.md. The validator is the fast pass that runs in seconds and
gates every commit.

Usage:
    python3 validate_skill.py <skill-dir>
    python3 validate_skill.py .claude/skills-impl/meta/shared/quark-skill-creator

Exit codes:
    0 — pass
    1 — warnings only (e.g., source_knowledge path absent locally but matches
        the upstream Quark layout)
    2 — blocking failures
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    print(
        "error: validate_skill.py requires PyYAML. Install with: pip install pyyaml",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

VALID_LAYERS = ("l0-foundation", "l1-atomic", "l2-workflows", "l3-recipes", "meta")
NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
REQUIRED_FRONTMATTER = ("name", "description", "layer", "primary_artifact", "source_knowledge")
REQUIRED_SECTIONS = ("## Purpose", "## Inputs", "## Outputs", "## Interaction Flow", "## Recovery")
MAX_DESCRIPTION_WORDS = 100
MAX_BODY_LINES = 315
CANONICAL_ARTIFACTS = {
    "session_context.json",
    "env_context.json",
    "workspace_context.json",
    "pytorch_install_result.json",
    "quark_install_result.json",
    "model_analysis.json",
    "quant_plan.json",
    "run_manifest.yaml",
    "validation_report.md",
}
# Path prefixes that are always considered "in the surrounding Quark repo"
# even if a particular file is absent from a standalone skills checkout.
# A source_knowledge entry under any of these prefixes is downgraded from
# BLOCK to WARN when not found locally — those paths are expected to
# resolve in the deployed Quark host repo.
UPSTREAM_PREFIXES = (
    "docs/source/",
    "docs/agent_skills/",
    ".claude/skills-impl/",
    "quark/",
    "examples/",
    "tools/",
    "pyproject.toml",
    "requirements.txt",
)

REPO_ROOT = Path(__file__).resolve().parents[5]


@dataclass
class Report:
    skill_path: Path
    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def block(self, msg: str) -> None:
        """Record a blocking finding.

        Blocking findings cause the validator to exit with code 2 and must
        be fixed before the skill can ship.

        Args:
            msg: Human-readable description of the blocking issue.
        """
        self.blocking.append(msg)

    def warn(self, msg: str) -> None:
        """Record a non-blocking warning.

        Warnings cause the validator to exit with code 1 (advisory) but do
        not prevent the skill from shipping. Typical use: an upstream-prefix
        ``source_knowledge`` path absent from the local checkout.

        Args:
            msg: Human-readable description of the warning.
        """
        self.warnings.append(msg)

    def exit_code(self) -> int:
        """Compute the validator exit code based on findings.

        Returns:
            ``2`` if any blocking findings were recorded;
            ``1`` if only warnings were recorded;
            ``0`` if no findings were recorded.
        """
        if self.blocking:
            return 2
        if self.warnings:
            return 1
        return 0

    def render(self) -> str:
        """Format the report as a multi-line human-readable string.

        Returns:
            A header line ``validate_skill: <repo-relative-path>`` followed
            by either ``PASS — no findings`` or one ``BLOCK:`` / ``WARN:``
            line per recorded finding.
        """
        lines = [f"validate_skill: {self.skill_path.relative_to(REPO_ROOT)}"]
        if not self.blocking and not self.warnings:
            lines.append("  PASS — no findings")
        for m in self.blocking:
            lines.append(f"  BLOCK: {m}")
        for m in self.warnings:
            lines.append(f"  WARN:  {m}")
        return "\n".join(lines)


def parse_frontmatter(text: str) -> tuple[dict, int]:
    """Parse the YAML frontmatter at the top of a skill file.

    Uses PyYAML's safe_load so any standard YAML construct (folded
    scalars `key: >`, block scalars `key: |`, lists with arbitrary but
    consistent indent, scalar `key: value`) is handled correctly. The
    previous hand-rolled parser silently dropped 4-space-indented list
    items, which let buggy frontmatter pass validation.

    Returns (frontmatter_dict, body_start_line_index).
    """
    if not text.startswith("---\n"):
        raise ValueError("missing opening --- delimiter")
    lines = text.splitlines()
    end = None
    for i in range(1, len(lines)):
        if lines[i] == "---":
            end = i
            break
    if end is None:
        raise ValueError("missing closing --- delimiter")
    body_yaml = "\n".join(lines[1:end])
    try:
        fm = yaml.safe_load(body_yaml)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML parse error: {exc}") from exc
    if fm is None:
        fm = {}
    if not isinstance(fm, dict):
        raise ValueError(f"frontmatter must be a mapping, got {type(fm).__name__}")
    return fm, end + 1


def check_skill_md(path: Path, report: Report) -> dict | None:
    """Validate a single SKILL.md against the project format contract.

    Performs all mechanical checks in one pass: presence, parseable
    frontmatter, all five required frontmatter fields, name regex, layer
    membership, description word cap, primary_artifact placeholder
    rejection and canonical-set check, source_knowledge resolvability
    (with upstream-prefix warn vs. block split), body line cap, and the
    five required ``##`` section headings.

    Findings are appended to ``report`` rather than raised, so callers can
    aggregate results and produce a single exit code.

    Args:
        path: Path to the SKILL.md file to validate.
        report: ``Report`` instance to collect blocking errors and
            warnings into.

    Returns:
        Parsed frontmatter mapping if the file exists and parses
        successfully; ``None`` if the file is missing or unparseable
        (in which case the caller has already received the corresponding
        BLOCK finding via ``report``).
    """
    label = path.name
    if not path.is_file():
        report.block(f"{label} is missing")
        return None
    text = path.read_text(encoding="utf-8")
    try:
        fm, body_start = parse_frontmatter(text)
    except ValueError as e:
        report.block(f"{label} frontmatter: {e}")
        return None

    for field_name in REQUIRED_FRONTMATTER:
        if field_name not in fm:
            report.block(f"{label} frontmatter missing required field: {field_name}")

    name = fm.get("name", "")
    if isinstance(name, str) and not NAME_RE.match(name):
        report.block(f"{label} name {name!r} does not match {NAME_RE.pattern}")

    layer = fm.get("layer", "")
    if isinstance(layer, str) and layer not in VALID_LAYERS:
        report.block(f"{label} layer {layer!r} not in {VALID_LAYERS}")

    desc = fm.get("description", "")
    if isinstance(desc, str):
        word_count = len(desc.split())
        if word_count > MAX_DESCRIPTION_WORDS:
            report.block(f"{label} description is {word_count} words; cap is {MAX_DESCRIPTION_WORDS}")
        if "<" in desc or ">" in desc:
            report.warn(f"{label} description contains angle brackets — these often break YAML parsers")

    pa = fm.get("primary_artifact", "")
    if isinstance(pa, str):
        stripped = pa.strip()
        if stripped.startswith("<") or stripped.endswith(">") or "<TBD>" in stripped or "TODO" in stripped:
            report.block(
                f"{label} primary_artifact {pa!r} looks like a placeholder — "
                f"pick a concrete filename (e.g. validation_report.md, quant_plan.json)"
            )
        else:
            leaf = stripped.split("/")[-1]
            if leaf not in CANONICAL_ARTIFACTS and not leaf.endswith((".json", ".yaml", ".md")):
                report.warn(
                    f"{label} primary_artifact {pa!r} is not in the canonical artifact set; "
                    f"add a schema in shared/contracts/ and an entry in docs/agent_skills/artifact-contracts.md"
                )

    sk = fm.get("source_knowledge", [])
    if isinstance(sk, list):
        for item in sk:
            if not isinstance(item, str):
                report.block(f"{label} source_knowledge entry not a string: {item!r}")
                continue
            if item.startswith(("/", "http://", "https://")) or ".." in Path(item).parts:
                report.block(f"{label} source_knowledge {item!r} is not repo-root-relative")
                continue
            full = REPO_ROOT / item
            if not full.exists():
                if any(item.startswith(p) for p in UPSTREAM_PREFIXES):
                    report.warn(
                        f"{label} source_knowledge {item!r} not present locally — "
                        f"expected to resolve in the deployed Quark host repo"
                    )
                else:
                    report.block(f"{label} source_knowledge {item!r} not found in repo")

    body = "\n".join(text.splitlines()[body_start:])
    body_lines = len(body.splitlines())
    if body_lines > MAX_BODY_LINES:
        report.block(f"{label} body is {body_lines} lines; cap is {MAX_BODY_LINES}")

    found_section_keys: set[str] = set()
    for line in body.splitlines():
        if line.startswith("## "):
            # Treat "## Outputs: validation_report.md" as satisfying "## Outputs".
            key = line.split(":", 1)[0].strip()
            found_section_keys.add(key)
    for required in REQUIRED_SECTIONS:
        if required not in found_section_keys:
            report.block(f"{label} missing required section: {required}")

    return fm


def main(argv: list[str]) -> int:
    """Entry point: validate one skill directory and print the report.

    Args:
        argv: Command-line arguments excluding the program name (i.e.
            ``sys.argv[1:]``). Expects exactly one positional: the path
            to the skill directory containing ``SKILL.md``.

    Returns:
        Exit code: ``0`` on no findings, ``1`` on warnings only, ``2`` on
        any blocking finding (including a missing/non-directory target).
    """
    p = argparse.ArgumentParser(description="Validate a Quark Agent Skill against the project format contract.")
    p.add_argument("skill_dir", help="Path to the skill directory (containing SKILL.md)")
    args = p.parse_args(argv)

    skill_dir = Path(args.skill_dir).resolve()
    report = Report(skill_path=skill_dir)

    if not skill_dir.is_dir():
        print(f"error: {skill_dir} is not a directory", file=sys.stderr)
        return 2

    en_path = skill_dir / "SKILL.md"
    check_skill_md(en_path, report)

    print(report.render())
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
