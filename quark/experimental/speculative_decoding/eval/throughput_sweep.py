#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Per-GPU throughput sweep (deployment value).

Throughput -- not AL -- is the SLA metric. This driver hits a running
OpenAI-compatible serve at several concurrency levels and reports output
tokens/s (and tokens/s/GPU). Sweep speculative vs. no-speculative serves to get
the speedup, and remember the design caveat: speculative decoding helps most at
*low* concurrency; report the optimal operating point.

Report a small noise floor (~0.01-0.02 AL from TP all-reduce FP nondeterminism)
so tiny deltas are not over-read.
"""

from __future__ import annotations

import json
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from quark.experimental.speculative_decoding.data.synth import _post_json


def _random_prompt(approx_tokens: int) -> str:
    # ~4 chars/token heuristic for a random-input stress prompt.
    n = max(8, approx_tokens * 4)
    return "".join(random.choice(string.ascii_lowercase + " ") for _ in range(n))


def _one_request(comp_url: str, model: str, prompt: str, osl: int) -> tuple[int, float]:
    t0 = time.time()
    resp = _post_json(
        comp_url,
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": osl,
            "temperature": 0.0,
            "ignore_eos": True,
        },
    )
    dt = time.time() - t0
    usage = resp.get("usage", {})
    out_tokens = usage.get("completion_tokens", osl)
    return out_tokens, dt


def throughput_sweep(
    endpoint: str,
    served_model_name: str,
    conc: list[int],
    isl_osl: list[tuple[int, int]],
    num_gpus: int = 1,
    prompts_per_cell: int | None = None,
    warmup: int = 2,
    out_json: str | None = None,
) -> dict[str, float]:
    """Sweep concurrency x (ISL,OSL) and record output tok/s (per GPU).

    Returns a flat dict of ``"conc{C}_isl{I}_osl{O}": tok_s_per_gpu`` and, if
    ``out_json`` is set, writes the full table.
    """
    endpoint = endpoint.rstrip("/")
    comp_url = f"{endpoint}/completions"
    results: dict[str, float] = {}
    table: list[dict[str, float]] = []

    for isl, osl in isl_osl:
        for c in conc:
            n = prompts_per_cell or max(c * 4, 8)
            prompts = [_random_prompt(isl) for _ in range(n + warmup)]
            # Warmup (not measured).
            with ThreadPoolExecutor(max_workers=c) as ex:
                for _ in as_completed(
                    [ex.submit(_one_request, comp_url, served_model_name, p, osl) for p in prompts[:warmup]]
                ):
                    pass
            # Measured window.
            t0 = time.time()
            total_out = 0
            with ThreadPoolExecutor(max_workers=c) as ex:
                futs = [ex.submit(_one_request, comp_url, served_model_name, p, osl) for p in prompts[warmup:]]
                for fut in as_completed(futs):
                    try:
                        out_tokens, _ = fut.result()
                        total_out += out_tokens
                    except Exception:  # noqa: BLE001
                        continue
            wall = time.time() - t0
            tok_s = total_out / max(wall, 1e-6)
            tok_s_gpu = tok_s / max(num_gpus, 1)
            key = f"conc{c}_isl{isl}_osl{osl}"
            results[key] = tok_s_gpu
            table.append({"conc": c, "isl": isl, "osl": osl, "tok_s": tok_s, "tok_s_per_gpu": tok_s_gpu})
            print(f"[throughput] {key}: {tok_s_gpu:.1f} tok/s/GPU (wall {wall:.1f}s)")

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(table, f, indent=2)
    return results
