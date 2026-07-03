#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import argparse
import importlib
import sys
from unittest.mock import MagicMock, patch

from quark.common.utils.import_utils import UnavailableObject
from quark.contrib.llm_eval.evaluation import eval_model


@patch("quark.contrib.llm_eval.evaluation.AutoTokenizer")
def test_eval_model(_mock_auto_tokenizer: MagicMock) -> None:
    args = argparse.Namespace(
        model_dir="/fake",
        num_eval_data=-1,
        use_mlperf_rouge=False,
        use_ppl_eval_for_kv_cache=False,
        use_ppl_eval_model=False,
        tasks=None,
    )
    eval_model(args=args, model=MagicMock(), main_device="cpu")


def test_evaluate_unavailable_falls_back_to_unavailable_object() -> None:
    import quark.contrib.llm_eval.evaluation as mod

    with patch.dict(sys.modules, {"evaluate": None}):
        importlib.reload(mod)
        assert isinstance(mod.evaluate, UnavailableObject)

    importlib.reload(mod)
