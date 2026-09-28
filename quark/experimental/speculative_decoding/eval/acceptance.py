#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Acceptance-length (AL) evaluation.

AL = expected number of tokens emitted per target verification step
(``1 + accepted/drafts``). AL = 1 means no speedup.

Served AL is measured by running the draft under vLLM speculative decoding and
reading the engine's spec-decode counters from ``/metrics`` (Prometheus):
(``vllm:spec_decode_num_accepted_tokens_total`` /
``vllm:spec_decode_num_drafts_total``).
"""

from __future__ import annotations

import re
import urllib.request
from typing import Any

from quark.experimental.speculative_decoding.data.synth import _first_user_turn, _post_json, _read_prompts

_METRIC_RE = re.compile(r"^(vllm:[a-z_]+)\{[^}]*\}\s+([0-9eE.+-]+)", re.MULTILINE)


def _scrape_metrics(metrics_url: str) -> dict[str, float]:
    with urllib.request.urlopen(metrics_url, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    out: dict[str, float] = {}
    for name, val in _METRIC_RE.findall(text):
        out[name] = out.get(name, 0.0) + float(val)
    return out


def _al_from_metrics(m: dict[str, float]) -> float | None:
    accepted = m.get("vllm:spec_decode_num_accepted_tokens_total")
    # The AL denominator must be the number of draft *steps* (verification steps).
    # `num_draft_tokens_total` (total drafted tokens = steps * num_speculative_tokens)
    # is a different quantity and is NOT a valid AL denominator, so we deliberately do
    # not fall back to it; if the step counter is absent we return None and let the
    # caller surface it rather than report a wrong (deflated) AL.
    n_drafts = m.get("vllm:spec_decode_num_drafts_total")
    if accepted is None or not n_drafts:
        return None
    # AL = 1 (the always-emitted bonus token) + accepted per draft step.
    return 1.0 + accepted / n_drafts


def acceptance_via_serve(
    draft_dir: str,
    target_endpoint: str,
    served_model_name: str,
    dataset: str,
    nst: int = 3,
    max_prompts: int = 200,
    max_tokens: int = 512,
    metrics_url: str | None = None,
) -> float:
    """Drive a running spec-decode serve over ``dataset`` and read AL from metrics.

    The serve must already be running with this draft (see
    ``examples/.../deploy/serve_spec.sh``). Returns the measured AL.
    """
    endpoint = target_endpoint.rstrip("/")
    chat_url = f"{endpoint}/chat/completions"
    if metrics_url is None:
        # /metrics is served at the server root (sibling of /v1).
        base = endpoint[: -len("/v1")] if endpoint.endswith("/v1") else endpoint
        metrics_url = base + "/metrics"

    before = _scrape_metrics(metrics_url)
    rows = _read_prompts(dataset)[:max_prompts]
    for row in rows:
        msgs = _first_user_turn(row)
        if not msgs:
            continue
        try:
            _post_json(
                chat_url,
                {
                    "model": served_model_name,
                    "messages": msgs,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
            )
        except Exception:  # noqa: BLE001 - keep sweeping on transient errors
            continue
    after = _scrape_metrics(metrics_url)

    delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in set(after) | set(before)}
    al = _al_from_metrics(delta)
    if al is None:
        raise RuntimeError(
            "could not read spec-decode counters from /metrics; ensure the serve runs with "
            "speculative decoding enabled and exposes Prometheus metrics."
        )
    print(f"[acceptance_via_serve] AL={al:.3f} over {len(rows)} prompts (NST={nst})")
    return al


def acceptance(
    draft_hf: str,
    target: Any,
    dataset: str,
    nst: int = 3,
    target_endpoint: str | None = None,
    served_model_name: str = "target",
) -> dict[str, float | str]:
    """Measure AL@NST.

    If ``target_endpoint`` is given (a running spec-decode serve), returns the
    real served AL. Otherwise raises with guidance -- served AL is the metric we
    trust for ship decisions (in-framework accuracy overestimates AL).
    """
    if target_endpoint:
        al = acceptance_via_serve(draft_hf, target_endpoint, served_model_name, dataset, nst=nst)
        return {"al": al, "nst": nst, "source": "serve"}
    raise RuntimeError(
        "acceptance() needs a running spec-decode serve (target_endpoint). Start one with "
        "deploy/serve_spec.sh, then pass its /v1 endpoint. See eval/throughput_sweep.py for the "
        "deployment-value metric."
    )
