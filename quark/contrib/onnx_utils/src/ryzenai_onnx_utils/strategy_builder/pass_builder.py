# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path
from typing import Any

import ryzenai_onnx_utils.partitioner as partitioner


class PassBuilder:
    def __init__(self):
        self.setup_passes = []
        self.passes = []
        self.cleanup_passes = []

    def add_step(self, step: Path):
        strategy = partitioner._load_strategy(step)

        setup_passes = strategy.get("setup", [])
        for s in setup_passes:
            if s not in self.setup_passes:
                self.setup_passes.append(s)

        self.passes.extend(strategy.get("passes", []))

        cleanup_passes = strategy.get("cleanup", [])
        for s in cleanup_passes:
            if s not in self.cleanup_passes:
                self.cleanup_passes.append(s)

    def add_pass(self, pass_name: str):
        self.passes.append(pass_name)

    def add_conditional_pass(self, pass_name: str, predicate: str):
        if predicate in self.setup_passes or predicate in self.passes:
            self.passes.append(pass_name)

    def finalize(self) -> dict[str, Any]:
        steps = self.setup_passes.copy()
        for p in self.passes:
            if p not in self.setup_passes:
                steps.append(p)
        for p in self.cleanup_passes:
            if p not in steps:
                steps.append(p)
        return steps
