# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard, TypeVar, Union, overload

import onnx

T = TypeVar("T")


@overload
def is_sequence_of(val: list[Any], type: type[T]) -> TypeGuard[list[T]]: ...


@overload
def is_sequence_of(val: tuple[Any], type: type[T]) -> TypeGuard[tuple[T]]: ...


@overload
def is_sequence_of(val: Sequence[Any], type: type[T]) -> TypeGuard[Sequence[T]]: ...


def is_sequence_of(val: Sequence[Any], type: type[T]) -> TypeGuard[Sequence[T]]:
    return all(isinstance(x, type) for x in val)


@overload
def is_static_shape(val: tuple[int | str | None, ...]) -> TypeGuard[tuple[int, ...]]: ...


@overload
def is_static_shape(val: list[int | str | None]) -> TypeGuard[list[int]]: ...


@overload
def is_static_shape(val: Sequence[int | str | None]) -> TypeGuard[Sequence[int]]: ...


def is_static_shape(val: Sequence[int | str | None]) -> TypeGuard[Sequence[int]]:
    return is_sequence_of(val, int)


def are_static_shapes(val: Sequence[tuple[int | str | None, ...]]) -> TypeGuard[Sequence[tuple[int, ...]]]:
    return all(is_static_shape(shape) for shape in val)


class ReplaceParams:
    """
    Extra arguments that are passed to the replace function from the top-level
    """

    def __init__(
        self, attributes: dict[str, Any], output_path: Path, dd_files_path: Path, abs_dd_files_path: Path
    ) -> None:
        self.attributes = attributes
        self.output_path = output_path
        self.dd_files_path = dd_files_path
        self.abs_dd_files_path = abs_dd_files_path

    def get_domain(self, name: str) -> str:
        return self._get_property("domains", name)

    def get_domains(self) -> list[str]:
        domains = self._get_properties("domains")
        assert is_sequence_of(domains, str)
        return domains

    def get_xclbin(self, name: str) -> str:
        return self._get_property("xclbins", name)

    def get_op_namespace(self, name: str) -> str:
        return self._get_property("op_namespaces", name)

    def get_op_namespaces(self) -> list[str]:
        op_namespaces = self._get_properties("op_namespaces")
        assert is_sequence_of(op_namespaces, str)
        return op_namespaces

    def get_subgraph_domain(self, subgraph: list[onnx.NodeProto]) -> str:
        """
        Given a subgraph of nodes, validate that they all share the same domain
        and return this domain

        Args:
            subgraph (list[onnx.NodeProto]): subgraph to get domain for
        """
        return self._get_subgraph_property("domains", subgraph)

    def get_subgraph_xclbin(self, subgraph: list[onnx.NodeProto]) -> str:
        """
        Given a subgraph of nodes, validate that they all share the same xclbin
        and return this xclbin

        Args:
            subgraph (list[onnx.NodeProto]): subgraph to get xclbin for
        """
        return self._get_subgraph_property("xclbins", subgraph)

    def get_subgraph_op_namespace(self, subgraph: list[onnx.NodeProto]) -> str:
        """
        Given a subgraph of nodes, validate that they all share the same op
        namespace and return this namespace

        Args:
            subgraph (list[onnx.NodeProto]): subgraph to get op_namespace for
        """
        return self._get_subgraph_property("op_namespaces", subgraph)

    def get_bool_attr(self, attr_name: str, default_value: bool | None = None) -> bool:
        if default_value is None:
            assert attr_name in self.attributes
        value = self.attributes.get(attr_name, default_value)
        if isinstance(value, bool):
            return value
        assert isinstance(value, str)
        value = value.lower()
        return value in {"true", "yes", "on", "y"}

    def _get_subgraph_property(self, prop_name: str, subgraph: list[onnx.NodeProto]) -> str:
        prop = None
        for node in subgraph:
            new_prop = self._get_property(prop_name, node.op_type)
            if prop is None:
                prop = new_prop
            elif new_prop != prop:
                raise ValueError(f"Multiple {prop_name}s detected in subgraph: {prop} and {new_prop}")
            # else case they match so nothing to do
        assert prop is not None
        return prop

    def _get_property(self, prop_name: str, key: str) -> str:
        prop = self.attributes[prop_name]
        if isinstance(prop, str):
            return prop
        raise ValueError(f"Unsupported {prop_name} type saved or unhandled key {key}")

    def _get_properties(self, prop_name: str) -> list[str | int | float]:
        if prop_name not in self.attributes:
            return []
        prop = self.attributes[prop_name]
        if isinstance(prop, str | int | float):
            return [prop]
        if isinstance(prop, dict):
            props = set()
            for _key, value in prop.items():
                props.add(value)
            return list(props)
        raise ValueError(f"Unsupported {prop_name} type saved")


@dataclass
class SubPass:
    name: str
    pattern: list[str]

    def __len__(self) -> int:
        return len(self.pattern)


PassInputArgs = tuple[onnx.utils.Extractor, str, list[onnx.NodeProto], ReplaceParams]
PassOutputArgs = tuple[list[onnx.NodeProto] | None, list[onnx.TensorProto], list[onnx.ValueInfoProto] | None]
StrictPassOutputArgs = tuple[list[onnx.NodeProto], list[onnx.TensorProto], list[onnx.ValueInfoProto]]
PassFunction = Callable[[onnx.utils.Extractor, str, list[onnx.NodeProto], ReplaceParams], PassOutputArgs]
PatternType = str | list[str] | list[SubPass]
ShapeType = Sequence[int | str | None]
# newer versions of ONNX have onnx.TensorProto.DataType with a custom metaclass
# from Protobuf that doesn't implement | support in Protobuf<=5.28. Using Union
# is a workaround to prevent TypeError in this case. Putting quotes on it is
# further necessary for Python 3.10 compatibility. Once we drop support for
# old protobuf, this should be replaced with a | again
DtypeType = Union["onnx.TensorProto.DataType", int]  # noqa: UP007

TupleInts4 = tuple[int, int, int, int]
