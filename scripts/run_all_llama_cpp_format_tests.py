#!/usr/bin/env python3
#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Export Quark AWQ to every llama.cpp format and validate with llama-cli."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from quark.torch import export_llama_cpp_gguf, list_llama_cpp_export_formats
from quark.torch.export.llama_cpp_export.formats import get_export_format
from quark.torch.export.llama_cpp_export.validate import (
    run_llama_cpp_load_test,
    validate_gguf_metadata,
)

DEFAULT_MODEL = (
    "/home/l/.cache/huggingface/hub/"
    "models--amd--Qwen3.5-35B-A3B-AWQ-INT4-WO-128-FLOAT16/snapshots/"
    "892e9ce6c763f008eb04dadcafe017503cbefb8c"
)
DEFAULT_OUT = "/home/l/work/gguf/qwen35-awq"
DEFAULT_LLAMA_CPP = "/home/l/work/llama.cpp"
DEFAULT_LLAMA_CLI = "/home/l/work/llama.cpp/build-hip/bin/llama-cli"
DEFAULT_RESULTS = "/home/l/work/gguf/qwen35-awq/format-test-results.json"
VALIDATED_JSON = Path(__file__).resolve().parents[1] / "test/test_for_torch/llama_cpp_export_e2e_validated.json"


def _format_passed(entry: dict, skip_load_formats: list[str]) -> bool:
    if entry.get("export") not in {"ok", "skipped (exists)"}:
        return False
    if entry.get("metadata") != "ok":
        return False
    load = entry.get("load_test")
    if load == "ok":
        return True
    if load == "skipped (large float format)" and entry.get("format") in skip_load_formats:
        return True
    return False


def _commit_validated_format(fmt_name: str, entry: dict, *, repo_root: Path) -> None:
    validated_path = repo_root / "test/test_for_torch/llama_cpp_export_e2e_validated.json"
    validated: dict = {}
    if validated_path.exists():
        validated = json.loads(validated_path.read_text(encoding="utf-8"))

    record = {
        "model": "amd/Qwen3.5-35B-A3B-AWQ-INT4-WO-128-FLOAT16",
        "export": entry.get("export"),
        "metadata": entry.get("metadata"),
        "load_test": entry.get("load_test"),
        "answer_check": entry.get("answer_check"),
        "meta_summary": entry.get("meta_summary"),
        "elapsed_s": entry.get("elapsed_s"),
    }
    validated[fmt_name] = {k: v for k, v in record.items() if v is not None}
    validated_path.write_text(
        json.dumps(validated, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "l",
        "GIT_AUTHOR_EMAIL": "l@local",
        "GIT_COMMITTER_NAME": "l",
        "GIT_COMMITTER_EMAIL": "l@local",
    }
    subprocess.run(
        ["git", "add", str(validated_path.relative_to(repo_root))],
        cwd=repo_root,
        check=True,
        env=env,
    )
    answer = entry.get("answer_check", entry.get("load_test", "ok"))
    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            (
                f"test(llama_cpp_export): e2e validate {fmt_name} "
                f"on Qwen3.5-35B-AWQ\n\n"
                f"export={entry.get('export')} metadata=ok "
                f"load={entry.get('load_test')} answer={answer}\n\n"
                f"Co-authored-by: Cursor\n"
            ),
        ],
        cwd=repo_root,
        check=True,
        env=env,
    )
    print(f"  git commit: validated {fmt_name}", flush=True)


def _stop_llama_servers() -> None:
    subprocess.run(
        ["pkill", "-f", "llama.cpp/build-hip/bin/llama-server"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)


def _shard_glob(out_dir: Path, name: str, fmt: str) -> str:
    del out_dir
    return f"*{name}-{fmt}-{fmt}*.gguf"


def _first_shard(out_dir: Path, name: str, fmt: str) -> Path | None:
    matches = sorted(out_dir.glob(_shard_glob(out_dir, name, fmt)))
    return matches[0] if matches else None


def _all_shards(out_dir: Path, name: str, fmt: str) -> list[Path]:
    return sorted(out_dir.glob(_shard_glob(out_dir, name, fmt)))


def _already_exported(out_dir: Path, name: str, fmt: str) -> bool:
    return _first_shard(out_dir, name, fmt) is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument("--name", default="qwen35-awq-main")
    parser.add_argument("--llama-cpp-dir", default=DEFAULT_LLAMA_CPP)
    parser.add_argument("--llama-cli", default=DEFAULT_LLAMA_CLI)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-load-test", action="store_true")
    parser.add_argument(
        "--formats",
        nargs="*",
        default=None,
        help="Subset of formats; default is all",
    )
    parser.add_argument("--force", action="store_true", help="Re-export even if shards exist")
    parser.add_argument(
        "--cleanup-after-test",
        action="store_true",
        help="Delete GGUF shards after successful load test to save disk",
    )
    parser.add_argument(
        "--skip-load-formats",
        nargs="*",
        default=["f16", "bf16", "f32"],
        help="Formats too large for local GPU load smoke test (metadata only)",
    )
    parser.add_argument(
        "--keep-formats",
        nargs="*",
        default=["q4_k_m", "q8_0"],
        help="Never delete these formats when --cleanup-after-test is set",
    )
    parser.add_argument(
        "--git-commit",
        action="store_true",
        help="Git commit in Quark repo after each passing format",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = args.formats or list_llama_cpp_export_formats()
    results_path = Path(args.results)
    results: dict[str, dict] = {}
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))

    llama_cli = Path(args.llama_cli)
    _stop_llama_servers()
    for fmt_name in formats:
        fmt = get_export_format(fmt_name)
        entry: dict = results.get(fmt_name, {})
        entry["format"] = fmt_name
        t0 = time.time()

        shard = _first_shard(out_dir, args.name, fmt_name)
        need_export = args.force or shard is None
        if need_export and not args.skip_export:
            print(f"\n=== EXPORT {fmt_name} ===", flush=True)
            try:
                export_llama_cpp_gguf(
                    quark_model_dir=args.model,
                    output_dir=out_dir,
                    export_format=fmt_name,
                    name=args.name,
                    llama_cpp_dir=args.llama_cpp_dir,
                )
                entry["export"] = "ok"
            except Exception as exc:  # noqa: BLE001
                entry["export"] = f"fail: {exc}"
                results[fmt_name] = entry
                results_path.write_text(
                    json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                print(f"EXPORT FAIL {fmt_name}: {exc}", flush=True)
                continue
        elif shard is not None:
            entry["export"] = "skipped (exists)"
        else:
            entry["export"] = "skipped (--skip-export)"

        shard = _first_shard(out_dir, args.name, fmt_name)
        if shard is None:
            entry["metadata"] = "fail: no shard"
            results[fmt_name] = entry
            continue

        try:
            meta = validate_gguf_metadata(shard, fmt)
            entry["metadata"] = "ok"
            entry["meta_summary"] = meta
        except Exception as exc:  # noqa: BLE001
            entry["metadata"] = f"fail: {exc}"

        if (
            not args.skip_load_test
            and fmt_name not in args.skip_load_formats
            and llama_cli.exists()
        ):
            _stop_llama_servers()
            try:
                text = run_llama_cpp_load_test(
                    shard,
                    llama_cli,
                    timeout_s=600,
                )
                entry["load_test"] = "ok"
                entry["response"] = text[:500]
                if "卢伟冰" in text:
                    entry["answer_check"] = "ok"
                else:
                    entry["answer_check"] = f"unexpected: {text[:120]}"
            except Exception as exc:  # noqa: BLE001
                entry["load_test"] = f"fail: {exc}"
        elif fmt_name in args.skip_load_formats:
            entry["load_test"] = "skipped (large float format)"

        entry["elapsed_s"] = round(time.time() - t0, 1)
        results[fmt_name] = entry
        results_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"=== {fmt_name}: export={entry.get('export')} "
            f"meta={entry.get('metadata')} load={entry.get('load_test')} ===",
            flush=True,
        )

        repo_root = Path(__file__).resolve().parents[1]
        validated_path = repo_root / "test/test_for_torch/llama_cpp_export_e2e_validated.json"
        already_committed = False
        if validated_path.exists():
            validated_data = json.loads(validated_path.read_text(encoding="utf-8"))
            already_committed = fmt_name in validated_data

        if (
            args.git_commit
            and _format_passed(entry, args.skip_load_formats)
            and not already_committed
        ):
            try:
                _commit_validated_format(fmt_name, entry, repo_root=repo_root)
            except subprocess.CalledProcessError as exc:
                print(f"  git commit failed for {fmt_name}: {exc}", flush=True)

        if (
            args.cleanup_after_test
            and fmt_name not in args.keep_formats
            and (
                entry.get("load_test") == "ok"
                or entry.get("load_test") == "skipped (large float format)"
            )
            and entry.get("metadata") == "ok"
        ):
            for shard_path in _all_shards(out_dir, args.name, fmt_name):
                shard_path.unlink(missing_ok=True)
                print(f"  cleaned {shard_path.name}", flush=True)

    failed = [
        f
        for f, e in results.items()
        if e.get("export", "").startswith("fail")
        or e.get("metadata", "").startswith("fail")
        or (
            e.get("load_test", "").startswith("fail")
            and e.get("format") not in args.skip_load_formats
        )
    ]
    print(f"\nDone. {len(results)} formats, {len(failed)} failures.")
    if failed:
        print("Failed:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
