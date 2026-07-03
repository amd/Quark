#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import argparse
from unittest.mock import MagicMock, patch

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
