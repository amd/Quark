#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Run lm-eval with model-import compatibility in the evaluator and its workers."""

if __name__ in {"__main__", "__mp_main__"}:
    # A spawned worker reimports this module before unpickling its target and
    # model config. Hold the alias for the process lifetime, including lazy
    # model imports after startup. Forked workers inherit the same alias.
    # Keep setup local so evaluator workers do not import the quantization stack.
    from transformers.utils import generic

    if not hasattr(generic, "OutputRecorder"):
        try:
            from transformers.utils.output_capturing import OutputRecorder
        except ModuleNotFoundError as error:
            if error.name != "transformers.utils.output_capturing":
                raise
        else:
            generic.OutputRecorder = OutputRecorder

if __name__ == "__main__":
    from lm_eval.__main__ import cli_evaluate

    # Keep this module as __main__; runpy(alter_sys=True) would lose the
    # compatibility setup when multiprocessing reconstructs spawned workers.
    cli_evaluate()
