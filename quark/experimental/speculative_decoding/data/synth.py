#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""On-policy data synthesis.

The single biggest lever on acceptance rate is training the draft on data drawn
from the *target's own* output distribution. This module drives a running,
OpenAI-compatible target serve (e.g. ``vllm serve``) to regenerate responses for
a prompt set.

Two generation paths (per the design doc):

* ``mode="chat"``  -> ``/v1/chat/completions`` with the target's exact chat
  template. Use this for the bulk of the data; the template MUST match serving.
* ``mode="raw"``   -> ``/v1/completions`` (template bypassed) for non-chat /
  out-of-distribution robustness.

Only depends on the stdlib (``urllib``) + ``tqdm`` so it has no hard client dep.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm


def _post_json(url: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _read_prompts(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _first_user_turn(row: dict[str, Any]) -> list[dict[str, str]]:
    """Extract the leading user (and optional system) messages from a record."""
    convs = row.get("conversations") or row.get("messages") or []
    msgs: list[dict[str, str]] = []
    for m in convs:
        role = m.get("role") or m.get("from")
        content = m.get("content") or m.get("value") or ""
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        if role in ("system", "user"):
            msgs.append({"role": role, "content": content})
        if role == "user":
            break
    if not msgs and "prompt" in row:
        msgs = [{"role": "user", "content": row["prompt"]}]
    return msgs


def synthesize(
    prompts: str,
    target_endpoint: str,
    served_model_name: str,
    out: str,
    mode: str = "chat",
    max_tokens: int = 4096,
    temperature: float = 0.8,
    top_p: float = 0.95,
    concurrency: int = 64,
    limit: int | None = None,
    request_timeout: float = 600.0,
) -> str:
    """Generate on-policy responses and write standardized conversation JSONL.

    Output rows are ``{"conversations": [{"role","content"}...]}`` where the
    assistant turn is produced by the target itself.

    Returns the output path.
    """
    endpoint = target_endpoint.rstrip("/")
    chat_url = f"{endpoint}/chat/completions"
    comp_url = f"{endpoint}/completions"

    rows = _read_prompts(prompts)
    if limit is not None:
        rows = rows[:limit]

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    def _one(row: dict[str, Any]) -> dict[str, Any] | None:
        try:
            if mode == "chat":
                msgs = _first_user_turn(row)
                if not msgs:
                    return None
                resp = _post_json(
                    chat_url,
                    {
                        "model": served_model_name,
                        "messages": msgs,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    },
                    timeout=request_timeout,
                )
                text = resp["choices"][0]["message"]["content"]
                return {"conversations": msgs + [{"role": "assistant", "content": text}]}
            else:  # raw / completions (template bypassed)
                prompt_text = row.get("text") or row.get("prompt") or ""
                if not prompt_text:
                    msgs = _first_user_turn(row)
                    prompt_text = msgs[-1]["content"] if msgs else ""
                if not prompt_text:
                    return None
                resp = _post_json(
                    comp_url,
                    {
                        "model": served_model_name,
                        "prompt": prompt_text,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    },
                    timeout=request_timeout,
                )
                text = resp["choices"][0]["text"]
                return {
                    "conversations": [
                        {"role": "user", "content": prompt_text},
                        {"role": "assistant", "content": text},
                    ],
                    "raw": True,
                }
        except (urllib.error.URLError, KeyError, TimeoutError, ConnectionError):
            return None

    n_ok = 0
    with open(out, "w", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_one, r) for r in rows]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"on-policy[{mode}]"):
            res = fut.result()
            if res is not None:
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                n_ok += 1
    print(f"[synthesize] wrote {n_ok}/{len(rows)} responses -> {out}")
    return out


def wait_for_endpoint(target_endpoint: str, served_model_name: str, timeout_s: float = 1800.0) -> bool:
    """Block until the target serve answers ``/models`` (used by the example)."""
    url = target_endpoint.rstrip("/") + "/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(5)
    return False
