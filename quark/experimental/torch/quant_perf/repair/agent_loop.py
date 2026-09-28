#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf import config
from quark.experimental.torch.quant_perf.evaluation.execution import subprocess_text
from quark.experimental.torch.quant_perf.llm.audit import append_llm_call
from quark.experimental.torch.quant_perf.repair.evidence import (
    extract_failure_evidence,
    render_failure_evidence,
    save_failure_evidence,
)
from quark.experimental.torch.quant_perf.repair.types import FailureEvidence, RepairKnowledgeQuery
from quark.experimental.torch.quant_perf.session.progress import write_progress
from quark.experimental.torch.quant_perf.workspace.git import (
    commit_baseline,
    commit_selected_changes,
    reset_hard_to,
)

logger = logging.getLogger(__name__)

MAX_ROUNDS = 3
BASE_ROUNDS = 2
TIMEOUT_S = 300
# Soft elapsed-time budget checked before starting each new agent round.
# Verifiers have their own timeout and are never interrupted mid-verification.
AGENT_ROUND_START_BUDGET_S = 900
_GENERATED_PATH_PARTS = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


def _is_generated_path(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    return any(part in _GENERATED_PATH_PARTS for part in parts) or path.endswith((".pyc", ".pyo"))


@dataclass(frozen=True)
class CandidateSnapshot:
    diff: str
    changed_files: list[str]


def _read_candidate(repo: str, revision: str) -> CandidateSnapshot:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "--literal-pathspecs", *args], cwd=repo, capture_output=True, text=True, check=True
        ).stdout

    # Do not filter tracked changes: export and promotion use the full Git diff.
    paths = [path for path in git("diff", "--no-renames", "--name-only", "-z", revision, "--").split("\0") if path]
    diff = git("diff", "--no-renames", "--binary", "--full-index", revision, "--", *paths) if paths else ""
    return CandidateSnapshot(diff, paths)


def _capture_candidate(repo: str, baseline_sha: str) -> CandidateSnapshot:
    # Make new source files visible to diff without staging their contents.
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    paths = [path for path in untracked if path and not _is_generated_path(path)]
    if paths:
        subprocess.run(
            ["git", "--literal-pathspecs", "add", "--intent-to-add", "--", *paths],
            cwd=repo,
            capture_output=True,
            check=True,
        )
    return _read_candidate(repo, baseline_sha)


def parse_summary(output: str) -> dict[str, Any]:
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    return {}


def run_agent_rounds(
    framework_repo: str,
    make_prompt: Callable[[int, str, FailureEvidence | None], str],
    verify: Callable[[], tuple[bool, str | FailureEvidence]],
    commit_msg: str,
    label: str = "repair",
    timeout_s: int = TIMEOUT_S,
    agent_round_start_budget_s: int = AGENT_ROUND_START_BUDGET_S,
    initial_context: str = "",
    session_dir: str = "",
    knowledge_ids: list[str] | None = None,
    phase_callback: Callable[[dict[str, Any]], None] | None = None,
    initial_evidence: FailureEvidence | None = None,
    knowledge_text: str = "",
    knowledge_for_round: RepairKnowledgeQuery | None = None,
) -> dict[str, Any]:
    agent_env = config.build_subprocess_env()
    agent_env["IS_SANDBOX"] = "1"
    context = initial_context
    started = time.monotonic()
    repair_started_at = datetime.now(UTC).isoformat()
    success = False
    interrupted = False
    rounds_done = 0
    last_summary: dict[str, Any] = {}
    attempts: list[dict[str, Any]] = []
    current_evidence = initial_evidence
    seen_failure_signatures = {initial_evidence.signature} if initial_evidence else set()
    round_knowledge_ids = list(knowledge_ids or [])
    allow_progress_round = False
    run_dir = Path(session_dir) / "repair" / f"{label}-{time.time_ns()}" if session_dir else None
    round_dir: Path | None = None
    round_evidence: FailureEvidence | None = None
    patch_path = ""
    retry_candidate_diff = ""
    candidate = CandidateSnapshot("", [])
    phase_seconds: dict[str, float] = {}

    def publish(event: dict[str, Any]) -> None:
        event.update(
            round=rounds_done,
            max_rounds=MAX_ROUNDS,
            label=label,
            repair_started_at=repair_started_at,
            total_elapsed_seconds=time.monotonic() - started,
            evidence_dir=str(round_dir or ""),
        )
        if session_dir:
            write_progress(
                Path(session_dir),
                detail=f"{label} round {rounds_done}/{MAX_ROUNDS}: {event['phase']} ({event['status']})",
                repair=event,
            )
        if phase_callback is not None:
            phase_callback(dict(event))

    @contextmanager
    def phase(name: str) -> Iterator[None]:
        phase_started = time.monotonic()
        event = {
            "phase": name,
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": 0.0,
        }
        publish(event)
        try:
            yield
        except (KeyboardInterrupt, SystemExit):
            event["status"] = "interrupted"
            raise
        except BaseException:
            event["status"] = "failed"
            raise
        else:
            event["status"] = "completed"
        finally:
            event["finished_at"] = datetime.now(UTC).isoformat()
            elapsed = time.monotonic() - phase_started
            event["elapsed_seconds"] = elapsed
            phase_seconds[name] = elapsed
            publish(event)

    def save_output(stdout: object, stderr: object) -> None:
        if round_dir is not None:
            (round_dir / "agent.stdout.log").write_text(subprocess_text(stdout))
            (round_dir / "agent.stderr.log").write_text(subprocess_text(stderr))

    def save_candidate() -> tuple[CandidateSnapshot, str]:
        snapshot = _capture_candidate(framework_repo, baseline_sha)
        if round_dir is None:
            return snapshot, ""
        path = round_dir / "candidate.patch"
        path.write_text(snapshot.diff)
        return snapshot, str(path)

    def note(
        round_num: int,
        changed_files: list[str],
        tried: str,
        failed_because: str,
        *,
        outcome: str = "no_progress",
    ) -> None:
        attempts.append(
            {
                "round": round_num,
                "changed_files": changed_files or [],
                "tried": (tried or "").strip()[:500] or "(no summary reported)",
                "failed_because": (failed_because or "").strip()[:500],
                "outcome": outcome,
                "baseline_sha": baseline_sha,
                "knowledge_ids": list(round_knowledge_ids),
                "patch_path": patch_path,
                "rolled_back": False,
                "evidence_signature": round_evidence.signature if round_evidence else "",
                "evidence_paths": list(round_evidence.evidence_paths) if round_evidence else [],
                "generation_seconds": phase_seconds.get("generating", 0.0),
                "verification_seconds": phase_seconds.get("verifying", 0.0),
            }
        )

    baseline_sha = commit_baseline(
        framework_repo,
        f"quant_perf: {label} baseline (pre-fix state)",
    )
    try:
        for round_num in range(1, MAX_ROUNDS + 1):
            if round_num > BASE_ROUNDS and not allow_progress_round:
                break
            if time.monotonic() - started >= agent_round_start_budget_s:
                logger.warning(
                    "[%s] agent round-start budget exhausted after %d rounds",
                    label,
                    round_num - 1,
                )
                break
            rounds_done = round_num
            if retry_candidate_diff:
                # The next error was observed on this candidate, not on baseline.
                # Restore that exact state, including new files, after rollback.
                subprocess.run(
                    ["git", "apply", "--index", "-"],
                    input=retry_candidate_diff,
                    cwd=framework_repo,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            round_dir = run_dir / f"round-{round_num}" if run_dir else None
            if round_dir is not None:
                round_dir.mkdir(parents=True, exist_ok=True)
            round_evidence = None
            patch_path = ""
            phase_seconds.clear()
            round_knowledge_text = knowledge_text
            round_knowledge_ids = list(knowledge_ids or [])
            if knowledge_for_round is not None and current_evidence is not None:
                round_knowledge_text, round_knowledge_ids = knowledge_for_round(round_num, current_evidence)
            prompt = make_prompt(round_num, context, current_evidence)
            if round_knowledge_text:
                prompt += "\n\n" + round_knowledge_text
            if round_dir is not None:
                (round_dir / "agent.prompt.txt").write_text(prompt, encoding="utf-8")
            try:
                with phase("generating"):
                    result = subprocess.run(
                        [
                            "claude",
                            "-p",
                            "--model",
                            config.codegen_model(),
                            "--dangerously-skip-permissions",
                            prompt,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=timeout_s,
                        env=agent_env,
                        cwd=framework_repo,
                    )
            except subprocess.TimeoutExpired as error:
                save_output(error.stdout, error.stderr)
                candidate, patch_path = save_candidate()
                partial_output = "\n".join(
                    text
                    for text in (
                        subprocess_text(error.stdout),
                        subprocess_text(error.stderr),
                    )
                    if text
                ).strip()
                changed_files = candidate.changed_files
                if session_dir:
                    append_llm_call(
                        session_dir,
                        call_type=label,
                        model=config.codegen_model(),
                        round_id=round_num,
                        prompt=prompt,
                        output=partial_output,
                        outcome="timeout",
                        knowledge_ids=round_knowledge_ids,
                    )
                note(
                    round_num,
                    changed_files,
                    "(agent round timed out)",
                    "exceeded per-round timeout",
                )
                context_parts = [
                    "The previous repair round timed out before returning a final summary.",
                ]
                if partial_output:
                    context_parts.append("Partial agent output:\n" + partial_output[-2000:])
                if changed_files:
                    context_parts.append("Files touched before rollback: " + ", ".join(changed_files))
                context = "\n".join(context_parts)[-3000:]
                reset_hard_to(framework_repo, baseline_sha)
                attempts[-1]["rolled_back"] = True
                continue
            save_output(result.stdout, result.stderr)
            candidate, patch_path = save_candidate()
            summary = parse_summary(result.stdout)
            if summary.get("changed_files"):
                last_summary = summary
            if not summary.get("success", False) or result.returncode != 0:
                if session_dir:
                    append_llm_call(
                        session_dir,
                        call_type=label,
                        model=config.codegen_model(),
                        round_id=round_num,
                        prompt=prompt,
                        output=result.stdout or result.stderr,
                        outcome="agent_failed",
                        knowledge_ids=round_knowledge_ids,
                    )
                context = (result.stderr or result.stdout)[-2000:]
                note(
                    round_num,
                    candidate.changed_files,
                    summary.get("summary", ""),
                    "agent did not complete a working fix",
                )
                reset_hard_to(framework_repo, baseline_sha)
                attempts[-1]["rolled_back"] = True
                continue
            with phase("verifying"):
                ok, verification_failure = verify()
            candidate_changed_files = candidate.changed_files
            if not ok:
                round_evidence = extract_failure_evidence(verification_failure)
                if round_dir is not None and not round_evidence.evidence_paths:
                    round_evidence = save_failure_evidence(round_evidence, round_dir)
                fail_context = render_failure_evidence(round_evidence)
            if ok:
                try:
                    if _capture_candidate(framework_repo, baseline_sha) != candidate:
                        raise RuntimeError("candidate source changed during verification")
                    if _read_candidate(framework_repo, "HEAD").diff:
                        if not commit_selected_changes(framework_repo, candidate_changed_files, commit_msg):
                            raise RuntimeError("could not commit verified candidate")
                    if (
                        _read_candidate(framework_repo, f"{baseline_sha}..HEAD") != candidate
                        or _capture_candidate(framework_repo, "HEAD").diff
                    ):
                        raise RuntimeError("committed source differs from verified candidate")
                except Exception as error:
                    note(
                        round_num,
                        candidate_changed_files,
                        summary.get("summary", ""),
                        str(error),
                        outcome="candidate_rejected",
                    )
                    raise
                note(
                    round_num,
                    candidate_changed_files,
                    summary.get("summary", ""),
                    "",
                    outcome="verified",
                )
                if session_dir:
                    append_llm_call(
                        session_dir,
                        call_type=label,
                        model=config.codegen_model(),
                        round_id=round_num,
                        prompt=prompt,
                        output=result.stdout,
                        outcome="verified",
                        knowledge_ids=round_knowledge_ids,
                    )
                success = True
                return {
                    "success": True,
                    "rounds": round_num,
                    "changed_files": candidate_changed_files,
                    "summary": summary.get("summary", ""),
                    "attempts": attempts,
                    "baseline_sha": baseline_sha,
                }
            assert round_evidence is not None
            current_evidence = round_evidence
            verifier_signature = round_evidence.signature
            outcome = (
                "new_failure"
                if (
                    seen_failure_signatures and verifier_signature and verifier_signature not in seen_failure_signatures
                )
                else "no_progress"
            )
            if outcome == "new_failure":
                allow_progress_round = True
            if verifier_signature:
                seen_failure_signatures.add(verifier_signature)
            note(
                round_num,
                candidate_changed_files,
                summary.get("summary", ""),
                fail_context,
                outcome=outcome,
            )
            if session_dir:
                append_llm_call(
                    session_dir,
                    call_type=label,
                    model=config.codegen_model(),
                    round_id=round_num,
                    prompt=prompt,
                    output=result.stdout,
                    outcome="verifier_rejected",
                    knowledge_ids=round_knowledge_ids,
                )
            retry_candidate_diff = candidate.diff
            reset_hard_to(framework_repo, baseline_sha)
            attempts[-1]["rolled_back"] = True
            context = (
                f"Previous candidate ({patch_path or 'in-memory snapshot'}) has been restored for this round. "
                "This failure was observed on that candidate. Preserve its necessary changes while fixing "
                "the current failure. The combined changes still require independent verification.\n" + fail_context
            )
    except (KeyboardInterrupt, SystemExit):
        interrupted = True
        raise
    except Exception as error:
        logger.error("[%s] unexpected error: %s", label, error)
        reset_hard_to(framework_repo, baseline_sha)
        if attempts:
            attempts[-1]["rolled_back"] = True
    finally:
        publish(
            {
                "phase": "finished",
                "status": "interrupted" if interrupted else ("verified" if success else "not_fixed"),
                "started_at": repair_started_at,
                "finished_at": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
    return {
        "success": False,
        "rounds": rounds_done,
        "changed_files": candidate.changed_files,
        "summary": last_summary.get("summary", ""),
        "attempts": attempts,
        "baseline_sha": baseline_sha,
    }
