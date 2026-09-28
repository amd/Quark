#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Filter and deterministically split target-generated on-policy JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _stable_key(row: dict[str, Any], seed: int, namespace: str) -> str:
    identity = str(row.get("id") or row.get("data_id") or json.dumps(row, sort_keys=True))
    return hashlib.sha256(f"{seed}\0{namespace}\0{identity}".encode()).hexdigest()


_DETACHED_REASONING_FIELDS = ("thinking", "reasoning", "reasoning_content")


def _balanced(row: dict[str, Any], *, prompt_opens_think: bool = False) -> bool:
    """Report whether every assistant turn closes the thinking spans it opens.

    Some chat templates pre-fill ``<think>\\n`` at the end of the assistant
    header, i.e. into the prompt. The OpenAI API never echoes the prompt back in
    ``message.content``, so a well-formed completion for such a target is
    ``{reasoning}</think>\\n\\n{answer}`` -- a closing tag with no opener. Under
    the default expectation those complete rows are all rejected and only
    truncated ones (which never reach the closing tag) survive, inverting the
    filter. ``prompt_opens_think`` states that the opener lives in the prompt so
    the count is off by exactly one. Turns are checked individually rather than
    over their concatenation so one truncated turn cannot be masked by a
    neighbour.
    """
    expected = 1 if prompt_opens_think else 0
    for message in row.get("conversations", []):
        if message.get("role") != "assistant":
            continue
        text = str(message.get("content") or "")
        if text.count("</think>") - text.count("<think>") != expected:
            return False
        if text.count("</mm:think>") != text.count("<mm:think>"):
            return False
    return True


def _detached_reasoning(row: dict[str, Any]) -> bool:
    """Report assistant turns whose reasoning was split out of ``content``.

    A serve started with a reasoning parser returns the thinking span in a
    separate field, so the recorded text is no longer the token sequence the
    target produces. Training on it silently lowers acceptance length.
    """
    return any(
        message.get("role") == "assistant" and any(message.get(field) for field in _DETACHED_REASONING_FIELDS)
        for message in row.get("conversations", [])
    )


def _eval_quotas(groups: dict[str, list[dict[str, Any]]], eval_size: int) -> dict[str, int]:
    total = sum(len(rows) for rows in groups.values())
    raw = {domain: len(rows) * eval_size / total for domain, rows in groups.items()}
    quotas = {domain: math.floor(value) for domain, value in raw.items()}
    remaining = eval_size - sum(quotas.values())
    order = sorted(groups, key=lambda domain: (-(raw[domain] - quotas[domain]), domain))
    for domain in order[:remaining]:
        quotas[domain] += 1
    return quotas


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--requested", type=int, required=True)
    parser.add_argument("--eval-size", type=int, required=True)
    parser.add_argument("--generation-max-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-shortfall", action="store_true")
    parser.add_argument(
        "--prompt-opens-think",
        action="store_true",
        help="the target's chat template opens <think> in the assistant header, so completions carry only the closing tag",
    )
    args = parser.parse_args()

    # Split on "\n" only. Generations carry raw U+2028/U+0085, which JSON allows
    # inside a string but splitlines() and universal newlines treat as record
    # separators, tearing valid records into unparseable fragments.
    with open(args.input, encoding="utf-8", newline="\n") as handle:
        raw = [json.loads(line) for line in handle if line.strip()]
    detached = sum(1 for row in raw if _detached_reasoning(row))
    if detached:
        raise ValueError(
            f"{detached} generations carry reasoning outside assistant content. "
            "Regenerate on-policy data with the target served without a reasoning parser "
            "so the recorded text matches what the target emits."
        )
    valid = [
        row for row in raw if row.get("conversations") and _balanced(row, prompt_opens_think=args.prompt_opens_think)
    ]
    if not args.allow_shortfall and len(valid) < args.requested:
        raise ValueError(f"only {len(valid)} valid generations; need {args.requested}")
    selected_count = min(args.requested, len(valid))
    if selected_count <= args.eval_size:
        raise ValueError(f"need more than {args.eval_size} valid generations, got {selected_count}")

    has_domains = any(isinstance(row.get("domain"), str) for row in valid)
    if has_domains:
        ordered = sorted(valid, key=lambda row: _stable_key(row, args.seed, "selection"))
        selected = ordered[:selected_count]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            groups[str(row.get("domain") or "general_instruction")].append(row)
        for domain, rows in groups.items():
            rows.sort(key=lambda row: _stable_key(row, args.seed, f"domain:{domain}"))
        quotas = _eval_quotas(groups, args.eval_size)
        eval_rows = [row for domain in sorted(groups) for row in groups[domain][: quotas[domain]]]
        train_rows = [row for domain in sorted(groups) for row in groups[domain][quotas[domain] :]]
        eval_rows.sort(key=lambda row: _stable_key(row, args.seed, "eval"))
        train_rows.sort(key=lambda row: _stable_key(row, args.seed, "train"))
    else:
        selected = list(valid)
        random.Random(args.seed).shuffle(selected)
        selected = selected[:selected_count]
        eval_rows = selected[: args.eval_size]
        train_rows = selected[args.eval_size :]

    _write(Path(args.train), train_rows)
    _write(Path(args.eval), eval_rows)
    domain_counts: dict[str, int] = defaultdict(int)
    for row in selected:
        domain_counts[str(row.get("domain") or "unspecified")] += 1
    Path(args.manifest_out).write_text(
        json.dumps(
            {
                "requested": args.requested,
                "source_generations": len(raw),
                "valid_generations": len(valid),
                "used": len(selected),
                "train": len(train_rows),
                "eval": len(eval_rows),
                "seed": args.seed,
                "generation_max_tokens": args.generation_max_tokens,
                "dropped_unbalanced_thinking": len(raw) - len(valid),
                "domains": dict(sorted(domain_counts.items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"train {len(train_rows)} eval {len(eval_rows)}")


if __name__ == "__main__":
    main()
