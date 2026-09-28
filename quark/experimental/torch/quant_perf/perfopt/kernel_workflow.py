#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Drive GEAK's single-kernel `kernel_workflow` via the Claude Agent SDK.

Replaces the old whole-model `interface/run_e2e.py` handoff (which self-profiled
and self-selected kernels, ignoring Quark Quant-Perf's differential choice). This path
feeds a SPECIFIC kernel (Quark Quant-Perf's differential/located pick) to GEAK's
kernel_workflow, which builds its own compile/correctness/benchmark COMMANDMENT
from the kernel source + a workload spec, optimizes it, and writes verified
patches to disk incrementally.

Key behaviours:
- Input = kernel source location (map_kernel_to_source) + a workload spec
  (parse_profile.py over Quark Quant-Perf's collected trace, so the perf harness times
  the REAL production shapes) + a task hint. No pre-built oracle is needed;
  the benchmark_engineer authors the correctness harness against the original.
- COMPLETE-OR-SALVAGE: the driver lets kernel_workflow run its full acceptance
  chain within a hard timeout and returns the FINAL result (Director-validated
  speedup + final_patch) when it finishes; if the timeout hits first it returns
  available patch and evidence without treating an unfinished candidate as verified.
- Invoked exactly like run_e2e.py drives e2e_workflow (Workflow tool +
  enableWorkflows/ultracode settings, bypassPermissions).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from quark.experimental.torch.quant_perf import config
from quark.experimental.torch.quant_perf.perfopt.workload_contract import WorkloadContract
from quark.experimental.torch.quant_perf.session.persistence import write_json_atomic
from quark.experimental.torch.quant_perf.workspace.git import source_only_git_env

_WORKFLOW_SETTINGS = '{"enableWorkflows": true, "ultracode": true}'
# Preserve unfinished candidates for diagnosis; only final validation grants eligibility.
_EMIT_DIFF_NAMES = ("final_patch.diff", "current_best.diff", "integrated_patch.diff", "best_patch.diff")
_RESULT_GRACE_S = 60.0
_COMPILED_SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hip",
    ".hpp",
}
_EDITABLE_SOURCE_SUFFIXES = {
    *_COMPILED_SOURCE_SUFFIXES,
    ".S",
    ".asm",
    ".inc",
    ".py",
    ".pyi",
    ".s",
}
_GENERATED_ARTIFACT_DIR_PREFIXES = (
    ".aiter_jit",
    ".aiter_meta",
    ".aiter_shadow_meta",
    ".profile",
    ".rocprof",
    ".torch_ext",
)
_GENERATED_ARTIFACT_DIR_NAMES = {
    ".harness",
    "__pycache__",
    "build",
}
_GENERATED_ARTIFACT_FILES = {
    "test_harness.py",
}


@dataclass(frozen=True)
class KernelPatchPreparation:
    path: str = ""
    changed_files: tuple[str, ...] = ()
    filtered_artifacts: tuple[str, ...] = ()
    rejected_new_files: tuple[str, ...] = ()
    requires_rebuild: bool = False
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return bool(self.path)


def persist_kernel_workflow_artifacts(
    report: dict[str, Any],
    run_dir: str | Path,
    *,
    kernel_src: str,
    source_repo: str,
    explicitly_allowed_files: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Copy transient GEAK outputs into the session before cleanup."""
    persisted = dict(report)
    artifacts = Path(run_dir) / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    patch_value = report.get("best_patch", "")
    patch = Path(patch_value) if patch_value else None
    if patch is not None and patch.is_file() and patch.stat().st_size > 0:
        raw_patch = artifacts / "raw_patch.diff"
        shutil.copy2(patch, raw_patch)
        raw_digest = hashlib.sha256(raw_patch.read_bytes()).hexdigest()
        if report.get("patch_sha256") and report["patch_sha256"] != raw_digest:
            raise ValueError("kernel patch changed after its validation result was read")
        persisted["raw_patch_sha256"] = raw_digest
        preparation = prepare_kernel_patch(
            str(raw_patch),
            kernel_src,
            source_repo,
            explicitly_allowed_files=explicitly_allowed_files,
        )
        if preparation.accepted:
            target = artifacts / "final_patch.diff"
            os.replace(preparation.path, target)
            persisted["best_patch"] = str(target)
            persisted["patch_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
            persisted["patch_changed_files"] = list(preparation.changed_files)
            persisted["patch_requires_rebuild"] = preparation.requires_rebuild
            persisted["patch_validation"] = {
                "status": "accepted",
                "reason": preparation.reason,
                "raw_patch": str(raw_patch),
                "filtered_artifacts": list(preparation.filtered_artifacts),
                "deliverable_changed_files": list(preparation.changed_files),
                "rejected_new_files": [],
            }
        else:
            persisted["best_patch"] = ""
            persisted["patch_changed_files"] = []
            persisted["patch_requires_rebuild"] = False
            persisted["patch_validation"] = {
                "status": "rejected",
                "reason": preparation.reason,
                "raw_patch": str(raw_patch),
                "filtered_artifacts": list(preparation.filtered_artifacts),
                "deliverable_changed_files": [],
                "rejected_new_files": list(preparation.rejected_new_files),
            }

    correctness = report.get("round_evaluation", {}).get("correctness")
    if correctness is not None:
        write_json_atomic(artifacts / "correctness.json", correctness)
        validation = correctness.get("result")
        if isinstance(validation, dict):
            write_json_atomic(artifacts / "director_validation.json", validation)
            if report.get("verified_speedup") is not None:
                write_json_atomic(
                    artifacts / "benchmark.json",
                    {
                        "source": "director_validation.json",
                        "verified_speedup": report["verified_speedup"],
                        "per_case": validation.get("per_case", []),
                        "timing_receipt": validation.get("timing_receipt"),
                    },
                )
    persisted["artifacts_dir"] = str(artifacts)
    write_json_atomic(artifacts / "result.json", persisted)
    return persisted


def cleanup_kernel_workflow_eval_dir(eval_dir: str) -> None:
    if not eval_dir:
        return
    path = Path(eval_dir).resolve()
    root = (Path(tempfile.gettempdir()) / "quark_quant_perf_kw_eval").resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return
    shutil.rmtree(path, ignore_errors=True)


def _create_kernel_workflow_eval_dir(run_dir: str) -> str:
    root = Path(tempfile.gettempdir()) / "quark_quant_perf_kw_eval"
    root.mkdir(parents=True, exist_ok=True)
    tag = hashlib.md5(
        os.path.abspath(run_dir).encode(),
    ).hexdigest()[:12]
    return tempfile.mkdtemp(prefix=f"{tag}-", dir=root)


def gen_workload_spec(trace_dir: str, kernel_name: str, out_path: str) -> str | None:
    """Build a workload-v1 spec for `kernel_name` from Quark Quant-Perf's newest torch
    trace via GEAK's parse_profile.py, so kernel_workflow benchmarks the exact
    (shape,dtype) cases the kernel hits in production, weighted by total time.

    Returns the spec path on success, else None (kernel_workflow then falls back
    to unweighted small/medium/large cases — still correct, just not aligned).
    """
    tdir = Path(trace_dir)
    if not tdir.is_dir():
        return None
    traces = sorted(
        [p for pat in ("*.pt.trace.json.gz", "*.json.gz", "*.json") for p in tdir.rglob(pat)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not traces:
        return None
    parser = Path(config.geak_root()) / "e2e_workflow" / "scripts" / "parse_profile.py"
    if not parser.exists():
        return None
    # kernel_name may be a full C++ signature; parse_profile --target does a
    # substring match, so pass a short stable token (the demangled symbol stem).
    target = _short_kernel_token(kernel_name)
    try:
        subprocess.run(
            [
                sys.executable,
                str(parser),
                "--torch-trace",
                str(traces[0]),
                "--workload-out",
                out_path,
                "--target",
                target,
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    spec_path = Path(out_path)
    if not spec_path.exists():
        return None
    try:
        workload = json.loads(spec_path.read_text())
    except (OSError, ValueError):
        return None
    has_dimensions = any(
        case.get("dims")
        for kernel in workload.get("kernels", [])
        for case in kernel.get("cases", [])
        if isinstance(case, dict)
    )
    return out_path if has_dimensions else None


def _short_kernel_token(kernel_name: str) -> str:
    """Extract a stable substring for parse_profile --target / prompts from a
    possibly-mangled C++ kernel signature (e.g.
    'void vllm::scaled_fp8_quant_kernel_strided_group_shape<...>' ->
    'scaled_fp8_quant_kernel_strided_group_shape')."""
    name = kernel_name
    for prefix in ("void ", "__global__ ", "__device__ "):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    name = name.split("<")[0].split("(")[0]
    if "::" in name:
        name = name.split("::")[-1]
    return name.strip()[:80]


def _scan_emit(
    eval_dir: str,
    emit_names: tuple[str, ...] = _EMIT_DIFF_NAMES,
) -> dict[str, Any] | None:
    """Recover an unfinished patch without borrowing another candidate's evidence."""
    root = Path(eval_dir)
    if not root.exists():
        return None
    for name in emit_names:
        hits = [p for p in root.rglob(name) if p.is_file() and p.stat().st_size > 0]
        if not hits:
            continue
        patch = max(hits, key=lambda p: p.stat().st_mtime)
        return {
            "verified_speedup": None,
            "micro_speedup_source": "unverified_patch",
            "best_patch": str(patch),
        }
    return None


# Authoritative final-result speedup fields, in priority order (Director's
# arbitrated number first, then Finalize's).
_FINAL_SPEEDUP_FIELDS = (
    "director_verified_speedup_weighted",
    "director_verified_speedup_geomean",
    "final_speedup_weighted",
    "final_speedup_geomean",
)


def parse_director_validation(validation: dict[str, Any]) -> dict[str, Any]:
    """Accept a device speedup only with correctness, case timings and a receipt."""
    correctness = {"pass": True, "fail": False}.get(str(validation.get("correctness")).strip().lower())
    status = validation.get("validation_status")
    status = status.strip().lower() if isinstance(status, str) else "unknown"
    if status == "accept":
        status = "accepted"
    if status not in {"accepted", "flagged", "rejected"}:
        status = "unknown"
    validated = correctness is True and status in {"accepted", "flagged"}
    cases = validation.get("per_case")
    receipt = validation.get("timing_receipt")

    def positive_number(value: Any) -> bool:
        return type(value) in (int, float) and math.isfinite(value) and value > 0

    measured = (
        validated
        and isinstance(receipt, dict)
        and receipt.get("all_primed") is True
        and receipt.get("timer_unprimed") is False
        and validation.get("timing_basis") not in {"host_bound", "unprimed", "unknown"}
        and isinstance(cases, list)
        and bool(cases)
        and all(
            isinstance(case, dict)
            and positive_number(case.get("baseline_ms"))
            and positive_number(case.get("optimized_ms"))
            for case in cases
        )
    )
    speedup = (
        next((float(validation[key]) for key in _FINAL_SPEEDUP_FIELDS if positive_number(validation.get(key))), None)
        if measured
        else None
    )
    return {
        "director_status": status,
        "verified_speedup": speedup,
        "micro_speedup_source": (
            "director" if speedup is not None else "unmeasured_verified_patch" if validated else "unverified_patch"
        ),
        "round_evaluation": {
            "correctness": {"success": correctness, "source": "director_validation", "result": validation},
        },
    }


def _scan_final(eval_dir: str) -> dict[str, Any] | None:
    """Return the COMPLETED result if kernel_workflow finished its full
    acceptance chain (Director validation / Finalize), else None.

    Only `director_validation.json` marks completion. GEAK writes
    `final_patch.diff` before independent validation starts, so the patch alone
    must not end the SDK message pump. Preserve the Director's correctness
    result for the downstream candidate gate."""
    root = Path(eval_dir)
    if not root.exists():
        return None

    for jp in root.rglob("director_validation.json"):
        try:
            d = json.loads(jp.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict) or not ({"correctness", "validation_status"} & d.keys()):
            continue
        patch_path = Path(str(d.get("final_patch") or d.get("best_patch") or "final_patch.diff"))
        if not patch_path.is_absolute():
            patch_path = jp.parent / patch_path
        patch = (
            str(patch_path)
            if (
                patch_path.resolve().is_relative_to(root.resolve())
                and patch_path.is_file()
                and patch_path.stat().st_size > 0
            )
            else ""
        )
        return {
            **parse_director_validation(d),
            "best_patch": patch,
            "patch_sha256": hashlib.sha256(Path(patch).read_bytes()).hexdigest() if patch else "",
            "final": True,
        }
    return None


def _rebase_diff(diff_text: str, repo_rel_dir: str) -> str:
    """Rewrite a kernel_workflow patch (paths relative to GEAK's isolated
    workspace, e.g. `+++ b/./common.cu`, `+++ b/amd/quant_utils.cuh`) so its
    file paths are relative to `framework_repo` at `repo_rel_dir`
    (the kernel's directory inside the repo, e.g. csrc/quantization/w8a8/fp8).
    Only the `diff --git`, `---` and `+++` header lines are touched; hunks are
    left verbatim. The result applies cleanly with `git apply -p1`."""
    rel_dir = repo_rel_dir.strip("/")

    def _repo_path(raw: str) -> str:
        # raw is the path after the a/ or b/ prefix, e.g. "./common.cu",
        # "home/.../workspace/common.cu", or "amd/quant_utils.cuh".
        if "/workspace/" in raw:  # a-side absolute path into GEAK's
            p = raw.rsplit("/workspace/", 1)[-1]  # isolated workspace (its LAST
        else:  # segment is the kernel dir root)
            p = raw[2:] if raw.startswith("./") else raw.lstrip("/")
        return f"{rel_dir}/{p}" if rel_dir else p

    out = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                a = _repo_path(parts[2][2:])
                b = _repo_path(parts[3][2:])
                out.append(f"diff --git a/{a} b/{b}")
                continue
        elif line.startswith("--- ") and not line.startswith("--- /dev/null"):
            out.append(f"--- a/{_repo_path(line[4:].lstrip()[2:])}")
            continue
        elif line.startswith("+++ ") and not line.startswith("+++ /dev/null"):
            out.append(f"+++ b/{_repo_path(line[4:].lstrip()[2:])}")
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def _git_apply(
    repo: str,
    patch_path: str,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", "-C", repo, "apply", "--index", "--recount", "-p1", patch_path],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        proc = subprocess.run(
            ["git", "-C", repo, "apply", "--recount", "-p1", patch_path],
            capture_output=True,
            text=True,
        )
    return proc


def _patch_sections(diff_text: str) -> list[tuple[str, str]]:
    sections = re.split(
        r"(?=^diff --git )",
        diff_text,
        flags=re.MULTILINE,
    )
    parsed: list[tuple[str, str]] = []
    for section in sections:
        if not section.startswith("diff --git "):
            continue
        header = section.splitlines()[0].split()
        path = header[3][2:] if len(header) >= 4 else ""
        parsed.append((path, section))
    if parsed:
        return parsed

    lines = diff_text.splitlines(keepends=True)
    starts = [
        index
        for index in range(len(lines) - 1)
        if (lines[index].startswith("--- ") and lines[index + 1].startswith("+++ "))
    ]
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        old_path = _unified_diff_header_path(lines[start], "--- ")
        new_path = _unified_diff_header_path(lines[start + 1], "+++ ")
        path = new_path if new_path != "/dev/null" else old_path
        parsed.append((path, "".join(lines[start:end])))
    return parsed


def _unified_diff_header_path(line: str, prefix: str) -> str:
    raw = line[len(prefix) :].strip().split("\t", 1)[0]
    if raw.startswith(("a/", "b/")):
        return raw[2:]
    return raw


def _is_new_file_section(section: str) -> bool:
    return "\nnew file mode " in section or "--- /dev/null" in section


def _is_generated_evaluation_artifact(
    path: str,
    kernel_relpath: str,
) -> bool:
    candidate = PurePosixPath(path)
    parts = candidate.parts
    if any(
        part in _GENERATED_ARTIFACT_DIR_NAMES or part.startswith(_GENERATED_ARTIFACT_DIR_PREFIXES) for part in parts
    ):
        return True
    if candidate.name in _GENERATED_ARTIFACT_FILES or "harness_pybind" in candidate.stem:
        return True

    kernel = PurePosixPath(kernel_relpath)
    return (
        kernel.suffix == ".cu"
        and candidate.parent == kernel.parent
        and candidate.stem == kernel.stem
        and candidate.suffix == ".hip"
    )


def _git_patch_state(repo: str, patch_path: str) -> tuple[bool, str]:
    forward = subprocess.run(
        [
            "git",
            "-C",
            repo,
            "apply",
            "--check",
            "--recount",
            "-p1",
            patch_path,
        ],
        capture_output=True,
        text=True,
    )
    if forward.returncode == 0:
        return True, "applies_to_candidate"
    reverse = subprocess.run(
        [
            "git",
            "-C",
            repo,
            "apply",
            "--check",
            "--reverse",
            "--recount",
            "-p1",
            patch_path,
        ],
        capture_output=True,
        text=True,
    )
    if reverse.returncode == 0:
        return True, "already_applied_to_candidate"
    error = forward.stderr.strip() or reverse.stderr.strip()
    return False, error[:300]


def prepare_kernel_patch(
    patch_path: str,
    kernel_src: str,
    source_repo: str,
    *,
    explicitly_allowed_files: tuple[str, ...] = (),
) -> KernelPatchPreparation:
    patch_file = Path(patch_path).resolve()
    try:
        diff = patch_file.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return KernelPatchPreparation(reason=f"read patch failed: {exc}")

    repo = os.path.abspath(source_repo)
    kernel_path = os.path.abspath(kernel_src)
    kernel_rel = os.path.relpath(kernel_path, repo).replace(os.sep, "/")
    kernel_parent = PurePosixPath(kernel_rel).parent
    kernel_dir = "" if kernel_parent == PurePosixPath(".") else str(kernel_parent)
    if kernel_rel.startswith("../"):
        return KernelPatchPreparation(reason=(f"kernel source {kernel_src} is outside source repo {source_repo}"))

    rebased = _rebase_diff(diff, kernel_dir)
    sections = _patch_sections(rebased)
    if not sections:
        return KernelPatchPreparation(reason="patch contains no file changes")

    explicit = {str(PurePosixPath(path)) for path in explicitly_allowed_files if path}
    changed_files: list[str] = []
    filtered_artifacts: list[str] = []
    rejected_new_files: list[str] = []
    canonical_sections: list[str] = []
    for path, section in sections:
        normalized = str(PurePosixPath(path))
        parts = PurePosixPath(normalized).parts
        if not normalized or normalized.startswith("../") or any(part in {"", ".."} for part in parts):
            return KernelPatchPreparation(
                filtered_artifacts=tuple(filtered_artifacts),
                rejected_new_files=tuple(rejected_new_files),
                reason=f"invalid patch path: {path}",
            )
        tracked = subprocess.run(
            [
                "git",
                "-C",
                repo,
                "ls-files",
                "--error-unmatch",
                "--",
                normalized,
            ],
            capture_output=True,
            text=True,
        )
        # Optimizers may commit generated files in their private workspace, so
        # a modification section alone does not prove the file is real source.
        if tracked.returncode != 0 and _is_generated_evaluation_artifact(
            normalized,
            kernel_rel,
        ):
            filtered_artifacts.append(normalized)
            continue
        if any(
            marker in section
            for marker in (
                "\ndeleted file mode ",
                "\nrename from ",
                "\nrename to ",
                "\ncopy from ",
                "\ncopy to ",
                "\nGIT binary patch",
                "\nBinary files ",
            )
        ):
            return KernelPatchPreparation(
                filtered_artifacts=tuple(filtered_artifacts),
                rejected_new_files=tuple(rejected_new_files),
                reason=(f"non-modification patch operation is not allowed: {normalized}"),
            )
        if _is_new_file_section(section):
            rejected_new_files.append(normalized)
            continue
        if any(part.startswith(".") for part in parts):
            return KernelPatchPreparation(
                filtered_artifacts=tuple(filtered_artifacts),
                rejected_new_files=tuple(rejected_new_files),
                reason=f"hidden patch paths are not allowed: {normalized}",
            )
        in_kernel_dir = not kernel_dir or normalized == kernel_dir or normalized.startswith(f"{kernel_dir}/")
        if not in_kernel_dir and normalized not in explicit:
            return KernelPatchPreparation(
                filtered_artifacts=tuple(filtered_artifacts),
                rejected_new_files=tuple(rejected_new_files),
                reason=(f"patch path is outside the authorized source scope: {normalized}"),
            )
        if tracked.returncode != 0:
            return KernelPatchPreparation(
                filtered_artifacts=tuple(filtered_artifacts),
                rejected_new_files=tuple(rejected_new_files),
                reason=f"patch path is not tracked at baseline: {normalized}",
            )
        if (
            normalized != kernel_rel
            and normalized not in explicit
            and PurePosixPath(normalized).suffix not in _EDITABLE_SOURCE_SUFFIXES
        ):
            return KernelPatchPreparation(
                filtered_artifacts=tuple(filtered_artifacts),
                rejected_new_files=tuple(rejected_new_files),
                reason=(f"patch path is not an editable source file: {normalized}"),
            )
        changed_files.append(normalized)
        canonical_sections.append(section)

    if rejected_new_files:
        return KernelPatchPreparation(
            filtered_artifacts=tuple(filtered_artifacts),
            rejected_new_files=tuple(rejected_new_files),
            reason=("unknown new files are not allowed: " + ", ".join(rejected_new_files)),
        )

    if kernel_rel not in changed_files:
        return KernelPatchPreparation(
            filtered_artifacts=tuple(filtered_artifacts),
            reason=f"patch does not modify target kernel source: {kernel_rel}",
        )

    prepared_path = str(patch_file.with_suffix(".repo.diff"))
    Path(prepared_path).write_text(
        "".join(canonical_sections),
        encoding="utf-8",
    )
    valid_patch_state, patch_state = _git_patch_state(
        repo,
        prepared_path,
    )
    if not valid_patch_state:
        Path(prepared_path).unlink(missing_ok=True)
        return KernelPatchPreparation(
            filtered_artifacts=tuple(filtered_artifacts),
            reason=f"canonical patch does not apply: {patch_state}",
        )
    requires_rebuild = any(PurePosixPath(path).suffix in _COMPILED_SOURCE_SUFFIXES for path in changed_files)
    unique_changed_files = tuple(dict.fromkeys(changed_files))
    unique_filtered_artifacts = tuple(dict.fromkeys(filtered_artifacts))
    reason = f"prepared tracked source patch ({patch_state})"
    if unique_filtered_artifacts:
        reason += f"; filtered {len(unique_filtered_artifacts)} generated evaluation artifact(s)"
    return KernelPatchPreparation(
        path=prepared_path,
        changed_files=unique_changed_files,
        filtered_artifacts=unique_filtered_artifacts,
        requires_rebuild=requires_rebuild,
        reason=reason,
    )


def apply_kernel_patch(patch_path: str, kernel_src: str, framework_repo: str) -> tuple[bool, str]:
    """Prepare and atomically apply one tracked source-only kernel patch."""
    preparation = prepare_kernel_patch(
        patch_path,
        kernel_src,
        framework_repo,
    )
    if not preparation.accepted:
        return False, preparation.reason
    return apply_prepared_kernel_patch(
        preparation.path,
        framework_repo,
    )


def apply_prepared_kernel_patch(
    patch_path: str,
    source_repo: str,
) -> tuple[bool, str]:
    """Apply a canonical source-only patch already validated for this repo."""
    proc = _git_apply(source_repo, patch_path)
    if proc.returncode != 0:
        return False, f"git apply failed: {proc.stderr.strip()[:300]}"
    return True, f"applied {patch_path}"


def rebuild_framework(framework_repo: str, timeout_s: float = 3600.0) -> tuple[bool, str]:
    """Rebuild the framework's compiled extension after a C-source patch so the
    change is live for the throughput re-measure. Assumes an editable install
    (vLLM's default); the CMake cache makes this an incremental recompile of the
    touched translation unit. Returns (ok, message). Best-effort: on failure the
    caller falls back to the kernel-level verified gain and logs the miss."""
    env = dict(os.environ)
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation"],
            cwd=framework_repo,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, f"rebuild timed out after {timeout_s:.0f}s"
    except OSError as e:
        return False, f"rebuild spawn failed: {e}"
    if proc.returncode != 0:
        return False, f"rebuild failed: {proc.stderr.strip()[-300:]}"
    return True, "rebuilt framework extension"


def _build_kernel_workflow_sdk_env(
    *,
    gpu_id: int,
    geak_root: str,
    source_repo: str = "",
    kernel_path: str = "",
    run_dir: str = "",
) -> dict[str, str]:
    env = {
        # ROCR-only visibility (setting HIP_VISIBLE_DEVICES too collides with
        # ROCR and yields "No HIP GPUs available").
        "ROCR_VISIBLE_DEVICES": str(gpu_id),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "GEAK_ROOT": geak_root,
        # GEAK compares source variants in separate processes. FlyDSL's disk
        # cache can reuse a binary when a builder-only source change does not
        # alter the inner @kernel cache key, invalidating that comparison.
        # Compile each isolated candidate from its actual source instead.
        "FLYDSL_RUNTIME_ENABLE_CACHE": "0",
    }
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        env["IS_SANDBOX"] = "1"
    if source_repo:
        source_root = Path(source_repo).resolve()
        if (source_root / "aiter").is_dir():
            # Check source changes with Ninja before loading a cached module.
            env["AITER_REBUILD"] = "2"
            if run_dir:
                env["AITER_JIT_DIR"] = str(Path(run_dir).resolve() / "build" / "aiter")
            if (source_root / "csrc").is_dir():
                env["AITER_ROOT_DIR"] = str(source_root)
        if kernel_path and run_dir:
            env.update(source_only_git_env(source_repo, kernel_path, run_dir))
    return config.build_subprocess_env(env)


def run_kernel_workflow(
    kernel_path: str,
    task: str,
    gpu_id: int,
    run_dir: str,
    *,
    workload_contract: WorkloadContract | None = None,
    source_repo: str = "",
    budget: int = 3,
    model: str = "claude-opus-4-8",
    hard_timeout_s: float = 5400.0,
) -> dict[str, Any]:
    """Optimize a single kernel with GEAK kernel_workflow.

    Runs the workflow to its final acceptance result within hard_timeout_s;
    on timeout, preserves available patches and validation evidence from disk.
    Recovered patches retain their execution status and tri-state correctness.
    Returns Quark Quant-Perf's standard report dict:
      {verified_speedup, best_patch, watchdog_status, approach_summary}
    """
    geak_root = config.geak_root()
    if not geak_root:
        return {
            "verified_speedup": None,
            "watchdog_status": "kernel_workflow_not_found",
            "error": "GEAK_ROOT is not configured.",
        }
    kw_dir = Path(geak_root) / "kernel_workflow"
    kw_js = kw_dir / "kernel_workflow.js"
    if not kw_js.exists():
        return {
            "verified_speedup": None,
            "watchdog_status": "kernel_workflow_not_found",
            "error": f"kernel_workflow.js not found under {kw_dir}. Set GEAK_ROOT.",
        }

    try:
        import anyio
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
    except ImportError as exc:
        return {
            "verified_speedup": None,
            "watchdog_status": "kernel_workflow_dependency_missing",
            "error": (
                "GEAK kernel optimization requires the Quant-Perf optional dependencies. "
                "Install them with `pip install 'amd-quark[quant_perf]'`."
            ),
            "diagnostic": f"{type(exc).__name__}: {exc}",
        }

    os.makedirs(run_dir, exist_ok=True)

    # eval_dir MUST live OUTSIDE the Quark Quant-Perf repo's git tree. kernel_workflow's
    # engineers save patches via `cd <workspace> && git diff`; if the workspace
    # sits under Quark Quant-Perf's .git and GEAK's own per-workspace `git init` hasn't
    # taken effect, that git diff walks UP to Quark Quant-Perf's .git and captures
    # Quark Quant-Perf's own uncommitted changes instead of the kernel diff (observed:
    # a patch full of quant_perf/*.py). A tmp dir has no parent .git, so the
    # workspace is always isolated.
    eval_dir = _create_kernel_workflow_eval_dir(run_dir)

    wf_args: dict[str, Any] = {
        "kernel_path": kernel_path,
        "workflow_dir": str(kw_dir),
        "budget": budget,
        "gpu_ids": "0",  # ROCR mask below pins the physical GPU; 0 == that GPU
        "eval_dir": eval_dir,
        "task": task,
    }
    if workload_contract is not None:
        wf_args["workload_spec_path"] = workload_contract.spec_path
        measurement_rules = Path(__file__).resolve().parents[1] / "agent-instructions" / "KERNEL_MEASUREMENT.md"
        addendum = Path(run_dir).resolve() / "harness_addendum.md"
        addendum.write_text(
            workload_contract.prompt_suffix() + "\n\n" + measurement_rules.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        wf_args["harness_addendum"] = str(addendum)

    prompt = (
        "Call the Workflow tool ONCE with these exact parameters, run it in the "
        "background, and then just wait — do not summarise, re-plan, or call any "
        "other tool.\n"
        f"scriptPath: {kw_js}\n"
        f"args: {json.dumps(wf_args)}\n"
    )

    log_path = Path(run_dir) / "kernel_workflow.log"

    def _opts() -> Any:
        sdk_env = _build_kernel_workflow_sdk_env(
            gpu_id=gpu_id,
            geak_root=geak_root,
            source_repo=source_repo,
            kernel_path=kernel_path,
            run_dir=run_dir,
        )
        claude_bin = config.claude_bin() if hasattr(config, "claude_bin") else ""
        kw = dict(
            model=model,
            allowed_tools=["Workflow", "Bash", "Read", "Write"],
            permission_mode="bypassPermissions",
            settings=_WORKFLOW_SETTINGS,
            cwd=str(kw_dir),
            env=sdk_env,
        )
        if claude_bin:
            kw["cli_path"] = claude_bin
        return ClaudeAgentOptions(**kw)

    emitted: dict[str, Any] = {}
    t0 = time.monotonic()

    execution_status = "incomplete"
    error = ""
    contract_violation = ""

    def _current_contract_violation() -> str:
        if workload_contract is None:
            return ""
        return workload_contract.analysis_violation(eval_dir) or workload_contract.baseline_violation(eval_dir)

    async def _drive() -> None:
        nonlocal contract_violation
        pending: set[str] = set()
        background_started = False
        turn_finished = False
        # Keep query and receive_messages in the same task. Closing the client
        # on the main turn's result also tears down its background workflow.
        with open(log_path, "w") as lf:
            with anyio.fail_after(hard_timeout_s):
                async with ClaudeSDKClient(options=_opts()) as client:
                    await client.query(prompt)
                    # SDK versions expose different message unions; dispatch by runtime type.
                    msg: Any
                    async for msg in client.receive_messages():
                        for b in getattr(msg, "content", []) or []:
                            t = getattr(b, "text", None)
                            if t:
                                lf.write(t + "\n")
                                lf.flush()
                        name = type(msg).__name__
                        if name == "TaskStartedMessage":
                            pending.add(msg.task_id)
                            background_started = True
                        elif name in {"TaskNotificationMessage", "TaskUpdatedMessage"}:
                            status = msg.status if name == "TaskNotificationMessage" else msg.patch.get("status")
                            if status in {"completed", "failed", "stopped", "killed"}:
                                pending.discard(msg.task_id)
                                if status != "completed" and not pending:
                                    raise RuntimeError(f"workflow task {status}: {getattr(msg, 'summary', '')}")
                        elif name == "ResultMessage":
                            turn_finished = True
                            if msg.is_error:
                                raise RuntimeError(f"workflow turn failed: {getattr(msg, 'result', '')}")
                        contract_violation = _current_contract_violation()
                        if contract_violation:
                            return
                        if pending:
                            continue
                        done = _scan_final(eval_dir)
                        if done:
                            emitted.update(done)
                            return
                        if turn_finished or background_started:
                            break

                    # A terminal notification can precede the final file write.
                    # Keep the client alive for a bounded grace period; active
                    # tasks retain the original overall deadline.
                    deadline = (
                        t0 + hard_timeout_s if pending else min(t0 + hard_timeout_s, time.monotonic() + _RESULT_GRACE_S)
                    )
                    while background_started and time.monotonic() < deadline:
                        contract_violation = _current_contract_violation()
                        if contract_violation:
                            return
                        done = _scan_final(eval_dir)
                        if done:
                            emitted.update(done)
                            return
                        await anyio.sleep(1)

    try:
        anyio.run(_drive)
    except TimeoutError:
        execution_status = "timed_out"
    except Exception as exc:  # noqa: BLE001
        execution_status = "error"
        error = f"{type(exc).__name__}: {exc}"

    # Completed but the marker landed after the last message; or ended/timed out
    # without completing -> salvage the best result available (final if present,
    # else the best committed/candidate patch so far).
    if not emitted:
        emitted.update(_scan_final(eval_dir) or _scan_emit(eval_dir) or {})

    if execution_status == "incomplete" and emitted.get("final"):
        execution_status = "completed"
    patch = emitted.get("best_patch", "")
    status = {
        "completed": "success",
        "timed_out": "timeout_salvage" if patch else "timeout_no_patch",
        "error": "sdk_error",
        "incomplete": "partial_salvage" if patch else "no_patch",
    }[execution_status]
    report: dict[str, Any] = {
        **emitted,
        "verified_speedup": emitted.get("verified_speedup"),
        "best_patch": patch,
        "watchdog_status": status,
        "execution_status": execution_status,
        "timed_out": execution_status == "timed_out",
        "error": error,
        "eval_dir": eval_dir,
        "round_evaluation": emitted.get("round_evaluation", {"correctness": {"success": None}}),
        "approach_summary": f"kernel_workflow {execution_status} ({time.monotonic() - t0:.0f}s)",
    }
    contract_violation = contract_violation or _current_contract_violation()
    workload_valid, workload_reason = (
        workload_contract.validate_alignment(eval_dir)
        if emitted and workload_contract is not None and not contract_violation
        else (True, "")
    )
    if contract_violation or not workload_valid:
        report.update(
            verified_speedup=None,
            best_patch="",
            watchdog_status="workload_contract_violation" if contract_violation else "workload_unaligned",
            micro_speedup_source="invalid_workload",
            approach_summary=contract_violation or workload_reason,
        )
    return report
