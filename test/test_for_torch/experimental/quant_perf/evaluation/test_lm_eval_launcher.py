#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Exercise model-import compatibility across real evaluator process boundaries."""

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

import quark


@pytest.mark.parametrize("exit_code", [0, 7])
def test_lm_eval_launcher_preserves_cli_and_patches_spawned_workers(tmp_path, exit_code):
    (tmp_path / "recorder_probe.py").write_text(
        dedent(
            """\
            import sys

            def recorder_details():
                from transformers.utils.generic import OutputRecorder
                try:
                    from transformers.utils.output_capturing import OutputRecorder as upstream
                except ModuleNotFoundError:
                    upstream = OutputRecorder
                return {
                    "same_upstream_class": OutputRecorder is upstream,
                    "main_module": sys.modules["__main__"].__spec__.name,
                    "quantizer_loaded": "quark.experimental.torch.mix_precision.quantizer" in sys.modules,
                }

            def send_recorder_details(queue):
                queue.put(recorder_details())
            """
        )
    )
    evaluator = tmp_path / "lm_eval"
    evaluator.mkdir()
    (evaluator / "__init__.py").write_text("")
    (evaluator / "__main__.py").write_text(
        dedent(
            """\
            import json
            import multiprocessing
            import sys

            # Remote modeling code can import this before the CLI is called.
            from transformers.utils.generic import OutputRecorder
            from recorder_probe import recorder_details, send_recorder_details

            def cli_evaluate():
                context = multiprocessing.get_context("spawn")
                queue = context.Queue()
                worker = context.Process(target=send_recorder_details, args=(queue,))
                worker.start()
                child = queue.get(timeout=30)
                worker.join(timeout=30)
                assert worker.exitcode == 0
                queue.close()
                queue.join_thread()
                print("RECORDER_PROBE=" + json.dumps({
                    "argv": sys.argv[1:],
                    "parent": recorder_details(),
                    "child": child,
                }))
                raise SystemExit(int(sys.argv[sys.argv.index("--exit-code") + 1]))
            """
        )
    )
    launcher = "quark.experimental.torch.quant_perf.evaluation.lm_eval_launcher"
    arguments = ["--model", "vllm", "--model_args", "pretrained=/models/kimi,trust_remote_code=True"]
    arguments.extend(["--exit-code", str(exit_code)])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(Path(quark.__file__).resolve().parent.parent), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-m", launcher, *arguments],
        capture_output=True,
        text=True,
        env=env,
        timeout=90,
    )

    assert result.returncode == exit_code, result.stderr
    payload = json.loads(
        next(
            line.removeprefix("RECORDER_PROBE=")
            for line in result.stdout.splitlines()
            if line.startswith("RECORDER_PROBE=")
        )
    )
    assert payload["argv"] == arguments
    for process in ("parent", "child"):
        assert payload[process] == {
            "same_upstream_class": True,
            "main_module": launcher,
            "quantizer_loaded": False,
        }
