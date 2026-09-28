#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json


def test_append_llm_call_writes_structured_jsonl(tmp_path):
    from quark.experimental.torch.quant_perf.llm.audit import append_llm_call

    append_llm_call(
        tmp_path,
        call_type="decision",
        model="model",
        round_id=2,
        prompt="prompt",
        output="output",
        outcome="accepted",
        knowledge_ids=["k1"],
    )

    row = json.loads((tmp_path / "llm_calls.jsonl").read_text())
    assert row["call_type"] == "decision"
    assert row["round"] == 2
    assert row["knowledge_ids"] == ["k1"]
    assert row["prompt_hash"] != row["output_hash"]
