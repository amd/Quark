#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Command-line review and promotion of runtime knowledge experience."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from quark.experimental.torch.quant_perf import config
from quark.experimental.torch.quant_perf.knowledge.store import ExperienceStore

from .provider import _StrictSafeLoader, _validate_record_data
from .review import KnowledgeReviewService


def _parser() -> argparse.ArgumentParser:
    """Build the knowledge-management command parser.

    :return: Configured argument parser.
    """
    parser = argparse.ArgumentParser(prog="quark-quant-perf knowledge")
    commands = parser.add_subparsers(dest="command", required=True)

    migrate_parser = commands.add_parser(
        "migrate",
        help="Migrate repair signatures using original session evidence",
        description="Back up and migrate recoverable repair signatures without changing source sessions or review history. Report records still missing original evidence.",
    )
    migrate_parser.add_argument(
        "--session", action="append", type=Path, default=[], help="Original session directory; may be repeated"
    )

    list_parser = commands.add_parser("list")
    list_parser.add_argument("--session", default="")
    list_parser.add_argument(
        "--domain",
        choices=["quantization", "repair", "kernel_optimization"],
        default="",
    )
    list_parser.add_argument("--include-reviewed", action="store_true")

    show_parser = commands.add_parser("show")
    show_parser.add_argument("experience_key")

    review_parser = commands.add_parser("review")
    review_parser.add_argument("experience_key")
    review_parser.add_argument("--output")

    approve_parser = commands.add_parser("approve")
    approve_parser.add_argument("candidate")
    approve_parser.add_argument("--repo-root", required=True)

    reject_parser = commands.add_parser("reject")
    reject_parser.add_argument("experience_key")
    reject_parser.add_argument("--reason", required=True)

    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("candidate")
    return parser


def _resolve_session_id(value: str) -> str:
    """Resolve a session ID from either a literal ID or session directory.

    :param value: Literal session ID or session directory.
    :return: Resolved session ID.
    """
    if not value:
        return ""
    path = Path(value)
    state_path = path / "state.json"
    if not state_path.is_file():
        return value
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return value
    return str(state.get("session_id") or value)


def main(argv: list[str] | None = None) -> int:
    """Run the knowledge review and promotion command.

    :param argv: Optional command arguments.
    :return: Process exit code.
    """
    args = _parser().parse_args(argv)
    with ExperienceStore(config.experience_store_path()) as store:
        if args.command == "migrate":
            summary = store.migrate_repair(tuple(args.session))
            summary["migrated"] += store.migration_summary["migrated"]
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        service = KnowledgeReviewService(store)
        if args.command == "list":
            rows = service.list_reviewable(
                session_id=_resolve_session_id(args.session),
                domain=args.domain,
                include_reviewed=args.include_reviewed,
            )
            print(json.dumps(rows, indent=2, sort_keys=True))
            return 0
        if args.command in {"show", "review"}:
            candidate = service.build_candidate(args.experience_key)
            text = yaml.safe_dump(candidate, sort_keys=False)
            output = getattr(args, "output", None)
            if output:
                path = Path(output).resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                print(str(path))
            else:
                print(text, end="")
            return 0
        if args.command == "approve":
            path = service.approve(
                candidate_path=args.candidate,
                repo_root=args.repo_root,
            )
            print(str(path))
            return 0
        if args.command == "reject":
            service.reject(
                args.experience_key,
                reason=args.reason,
            )
            print(
                json.dumps(
                    {
                        "experience_key": args.experience_key,
                        "decision": "rejected",
                        "reason": args.reason,
                    },
                    indent=2,
                )
            )
            return 0
        else:
            candidate_path = Path(args.candidate).resolve()
            raw = yaml.load(
                candidate_path.read_text(encoding="utf-8"),
                Loader=_StrictSafeLoader,
            )
            required = {
                "schema_version",
                "id",
                "domain",
                "kind",
                "status",
                "summary",
                "evidence",
                "provenance",
            }
            missing = sorted(required - set(raw or {}))
            if missing:
                raise SystemExit("invalid knowledge candidate; missing: " + ", ".join(missing))
            validation_copy = dict(raw)
            if validation_copy.get("status") == "proposed":
                validation_copy["status"] = "validated"
            _validate_record_data(validation_copy, path=candidate_path)
            print(json.dumps({"status": "valid", "path": str(candidate_path)}))
            return 0
