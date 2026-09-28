#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import json

import yaml

from quark.experimental.torch.quant_perf import config
from quark.experimental.torch.quant_perf.knowledge.cli import _resolve_session_id, main


def test_resolve_session_id_reads_session_state(tmp_path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "state.json").write_text(json.dumps({"session_id": "session-123"}), encoding="utf-8")

    assert _resolve_session_id(str(session_dir)) == "session-123"


def test_main_validates_candidate(monkeypatch, tmp_path, capsys) -> None:
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "candidate",
                "domain": "repair",
                "kind": "diagnostic_playbook",
                "status": "proposed",
                "summary": "Test candidate",
                "evidence": {},
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "experience_store_path", lambda: str(tmp_path / "experience.sqlite"))

    assert main(["validate", str(candidate)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "valid"
