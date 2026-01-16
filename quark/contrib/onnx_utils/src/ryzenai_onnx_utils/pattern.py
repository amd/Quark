# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import re

RE_PATTERN = re.compile(r"(.+)\((.+)\)")


class Pattern:
    def __init__(self, pattern: str) -> None:
        pattern = pattern.replace(" ", "")
        matched_groups = RE_PATTERN.findall(pattern)
        assert len(matched_groups) == 1, (
            f"Unexpected line: {pattern}. The valid symbol is: name(input_argument, output_argument)"
        )
        operator_name, arguments = matched_groups[0]
        inputs, outputs = self._parse_arguments(arguments)
        self.pattern_str = pattern
        self.name = operator_name
        self.inputs = inputs
        self.outputs = outputs

    @classmethod
    def _parse_variable(cls, symbols: str, token_index: int) -> tuple[str, int]:
        variable_name = ""
        while token_index < len(symbols):
            token = symbols[token_index]

            # If a valid token(alpha/number/_ or ?) for variable.
            if token.isalnum() or token == "?" or token == "_":
                variable_name += token
            else:
                break

            token_index += 1
        return variable_name, token_index

    @classmethod
    def _parse_list(cls, symbols: str, token_index: int) -> tuple[list[str], int]:
        var, token_index = cls._parse_variable(symbols, token_index)
        variables = [var]
        token = symbols[token_index]
        while token_index < len(symbols):
            token = symbols[token_index]
            if token == ",":
                token_index += 1
                name, token_index = cls._parse_variable(symbols, token_index)
                variables.append(name)
            elif token == "]":
                token_index += 1
                break
            else:
                raise ValueError(f"Unexpected token: {token}")
        assert token == "]", f"Unexpected end token for list: ], pos: {token_index}"
        return variables, token_index

    @classmethod
    def _parse_arguments(cls, symbols: str) -> list[list[str]]:
        token_index = 0

        lists = []
        while token_index < len(symbols):
            token = symbols[token_index]
            if token == "[":
                arguments, token_index = cls._parse_list(symbols, token_index + 1)
                lists.append(arguments)
            else:
                argument, token_index = cls._parse_variable(symbols, token_index)
                lists.append([argument])
            token_index += 1
        assert len(lists) == 2, f"Unexpected number of params: {len(lists)}"
        return lists

    def __str__(self) -> str:
        input_str = ",".join(self.inputs)
        output_str = ",".join(self.outputs)
        return f"{self.name}([{input_str}],[{output_str}])"

    def __repr__(self) -> str:
        return str(self)
