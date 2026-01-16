# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any

from .pattern import Pattern


class Lexer:
    def __init__(self, patterns: str | list[str]) -> None:
        # Remove spaces and split patterns by the break line.
        if isinstance(patterns, str):
            lines = [item for item in patterns.replace(" ", "").split("\n") if item != ""]
        else:
            lines = [item.replace(" ", "") for item in patterns]
        self.lines = lines

        # Parsing patterns by lexical analyzer.
        self.patterns: list[Pattern] = []
        self.edges = {}
        self.operators: dict[str, int] = {}
        for line in lines:
            pattern = Pattern(line)
            self.patterns.append(pattern)
            if pattern.name not in self.operators:
                self.operators[pattern.name] = 0
            self.operators[pattern.name] += 1

        named_operators = {k: v for k, v in self.operators.items() if k != "?"}
        self.anchor: str = min(named_operators, key=lambda k: self.operators[k]) if named_operators else "?"

        self.edges = self._build_edges()

    def _build_edges(self) -> dict[str, Any]:
        edges: dict[str, Any] = {}
        for index, pattern in enumerate(self.patterns):
            operator_names = pattern.name
            inputs = pattern.inputs
            outputs = pattern.outputs
            for io_num, io in enumerate(inputs):
                if io != "?":
                    if io not in edges:
                        edges[io] = {"src": [], "dst": []}
                    edges[io]["dst"].append(
                        {
                            "name": operator_names,
                            "pattern_index": index,
                            "io_index": io_num,
                        }
                    )
            for io_num, io in enumerate(outputs):
                if io != "?":
                    if io not in edges:
                        edges[io] = {"src": [], "dst": []}
                    edges[io]["src"].append(
                        {
                            "name": operator_names,
                            "pattern_index": index,
                            "io_index": io_num,
                        }
                    )
        return edges

    def op_in_pattern(self, op: str) -> bool:
        return op in self.operators or "?" in self.operators

    def get_patterns_by_name(self, op: str, include_wildcards: bool = True) -> list[tuple[Pattern, int]]:
        patterns = []
        for index, pattern in enumerate(self.patterns):
            if (include_wildcards and pattern.name == "?") or pattern.name == op:
                patterns.append((pattern, index))
        return patterns

    def get_pattern_by_index(self, pattern_index: int) -> list[tuple[Pattern, int]]:
        return [(self.patterns[pattern_index], pattern_index)]

    def get_named_io(self, pattern: Pattern) -> tuple[list[str], list[str]]:
        named_inputs = []
        named_outputs = []
        for io in pattern.inputs:
            if io != "?":
                named_inputs.append(io)
        for io in pattern.outputs:
            if io != "?":
                named_outputs.append(io)

        return named_inputs, named_outputs
