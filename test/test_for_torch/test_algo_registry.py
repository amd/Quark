#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Tests for the algorithm registry.

No model downloads and no GPU: everything here is ``QuarkAlgorithm``, registry and dispatch
behaviour.

Quark registers no algorithms yet, so every test that needs one installs a test double through
:py:func:`register` rather than relying on what the registry happens to hold.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from quark.torch.algorithm import ALGORITHM_REGISTRY, QuarkAlgorithm
from quark.torch.algorithm.api import PROCESSOR_MAP, get_processor
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.quantization.config.algo_configs import (
    ALGORITHM_CONFIG_MAPS,
    get_algo_config,
    get_supported_algorithm_types,
)
from quark.torch.quantization.config.config import (
    AlgoConfig,
    PreQuantOptConfig,
    RotationConfig,
    _load_pre_optimization_config_from_dict,
    _load_quant_algo_config_from_dict,
)
from quark.torch.quantization.config.template import LLMTemplate

# ------------------------------------------------------------------ test doubles


@dataclass
class DummyAlgoConfig(AlgoConfig):
    """The config class of a fictional quantization algorithm."""

    name: str = "dummyalgo"
    magic: int = 7


@dataclass
class DummyPreOptConfig(PreQuantOptConfig):
    """The config class of a fictional pre-quantization optimization."""

    name: str = "dummypreopt"
    magic: int = 7


class DummyProcessor(BaseAlgoProcessor):
    def __init__(self, model: Any, quant_algo_config: Any, calib_data: Any) -> None:
        self.model = model
        self.config = quant_algo_config
        self.calib_data = calib_data

    def apply(self) -> None:
        self.applied = True


def make_algo(name: str = "dummyalgo", **kwargs: Any) -> QuarkAlgorithm:
    kwargs.setdefault("algo_config", DummyAlgoConfig)
    kwargs.setdefault("algo_processor", DummyProcessor)
    return QuarkAlgorithm(name=name, **kwargs)


Register = Callable[..., None]


@pytest.fixture
def register(monkeypatch: pytest.MonkeyPatch) -> Register:
    """Install algorithms into the registry for the duration of one test.

    The registry holds its algorithms privately, so the fixture reaches in and patches them to a
    *copy*: a test which calls ``register`` then mutates the copy and leaves the real registry
    alone.
    """

    def _register(*algos: QuarkAlgorithm) -> None:
        monkeypatch.setattr(ALGORITHM_REGISTRY, "_algorithms", {algo.name: algo for algo in algos})

    return _register


# ------------------------------------------------------------------ QuarkAlgorithm


def test_algo_normalizes_names() -> None:
    assert make_algo(name="  DummyAlgo  ").name == "dummyalgo"


def test_algo_rejects_non_config_class() -> None:
    with pytest.raises(TypeError, match="algo_config"):
        QuarkAlgorithm(name="x", algo_config=dict, algo_processor=DummyProcessor)  # type: ignore[arg-type]


def test_algo_rejects_non_processor_class() -> None:
    with pytest.raises(TypeError, match="algo_processor"):
        QuarkAlgorithm(name="x", algo_config=DummyAlgoConfig, algo_processor=dict)  # type: ignore[arg-type]


def test_algo_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        make_algo(name="   ")


def test_build_config_returns_the_algos_config_class() -> None:
    config = make_algo().build_config({"name": "dummyalgo", "magic": 3})
    assert isinstance(config, DummyAlgoConfig)
    assert config.magic == 3


def test_build_config_does_not_mutate_caller_dict() -> None:
    config_dict = {"name": "dummyalgo", "magic": 11}
    make_algo().build_config(config_dict)
    assert config_dict == {"name": "dummyalgo", "magic": 11}


def test_algo_config_map_defaults_to_empty() -> None:
    assert make_algo().algo_config_map == {}


# ------------------------------------------------------------------ the registry


def test_the_registry_exposes_only_register_get_and_get_algorithms() -> None:
    """The point of the singleton: one way in, two ways out, and no way to remove an entry."""
    assert {name for name in vars(type(ALGORITHM_REGISTRY)) if not name.startswith("_")} == {
        "register",
        "get",
        "get_algorithms",
    }


def test_registered_algo_names_are_unique() -> None:
    names = [algo.name for algo in ALGORITHM_REGISTRY.get_algorithms()]
    assert len(names) == len(set(names))


def test_lookup_is_case_insensitive(register: Register) -> None:
    algo = make_algo()
    register(algo)
    assert ALGORITHM_REGISTRY.get("dummyalgo") is algo
    assert ALGORITHM_REGISTRY.get("DUMMYALGO") is algo
    assert ALGORITHM_REGISTRY.get("  DummyAlgo  ") is algo


def test_lookup_of_an_unregistered_name_is_none() -> None:
    assert ALGORITHM_REGISTRY.get("nosuchalgo") is None


def test_get_algorithms_returns_every_algo_in_registration_order(register: Register) -> None:
    first, second = make_algo(name="first"), make_algo(name="second")
    register(first, second)
    assert ALGORITHM_REGISTRY.get_algorithms() == (first, second)


# ------------------------------------------------------------------ register


def test_register_makes_an_algo_findable(register: Register) -> None:
    register()
    algo = make_algo()
    ALGORITHM_REGISTRY.register(algo)
    assert ALGORITHM_REGISTRY.get("dummyalgo") is algo
    assert ALGORITHM_REGISTRY.get_algorithms() == (algo,)


def test_registering_the_same_algo_twice_is_a_noop(register: Register) -> None:
    """A module that registers at import time must survive being imported twice."""
    register()
    algo = make_algo()
    ALGORITHM_REGISTRY.register(algo)
    ALGORITHM_REGISTRY.register(algo)
    assert ALGORITHM_REGISTRY.get_algorithms() == (algo,)


def test_registering_a_different_algo_under_a_taken_name_raises(register: Register) -> None:
    register()
    ALGORITHM_REGISTRY.register(make_algo())
    with pytest.raises(ValueError, match="already registered"):
        ALGORITHM_REGISTRY.register(make_algo(algo_config_map={"llama": DummyAlgoConfig()}))


def test_register_rejects_a_non_algo(register: Register) -> None:
    register()
    with pytest.raises(TypeError, match="QuarkAlgorithm"):
        ALGORITHM_REGISTRY.register(DummyAlgoConfig())  # type: ignore[arg-type]


def test_a_registered_algo_reaches_every_seam(register: Register) -> None:
    """The end-to-end point of the registry: one call, and all four lookups see it."""
    register()
    default = DummyAlgoConfig(magic=9)
    ALGORITHM_REGISTRY.register(make_algo(algo_config_map={"llama": default}))

    assert "dummyalgo" in get_supported_algorithm_types()
    assert get_algo_config("dummyalgo", "llama") is default
    assert get_processor("dummyalgo") is DummyProcessor
    assert isinstance(_load_quant_algo_config_from_dict({"name": "dummyalgo"}), DummyAlgoConfig)


# ------------------------------------------------------------------ the seam into core


def test_an_algo_is_listed_among_the_supported_algorithm_types(register: Register) -> None:
    register(make_algo())
    supported = get_supported_algorithm_types()
    assert "dummyalgo" in supported
    # Core's own algorithms are still listed, and still first.
    assert supported[: len(ALGORITHM_CONFIG_MAPS)] == list(ALGORITHM_CONFIG_MAPS)


def test_an_algo_claiming_a_core_name_is_listed_once(register: Register) -> None:
    register(make_algo(name="rotation"))
    assert get_supported_algorithm_types().count("rotation") == 1


def test_get_algo_config_returns_the_algos_per_model_default(register: Register) -> None:
    default = DummyAlgoConfig(magic=1)
    register(make_algo(algo_config_map={"llama": default}))
    assert get_algo_config("dummyalgo", "llama") is default
    assert get_algo_config("DummyAlgo", "llama") is default
    # A model the algorithm ships no default for is a miss, not an error — as for core algorithms.
    assert get_algo_config("dummyalgo", "some_unknown_arch") is None


def test_get_algo_config_prefers_the_registry_over_cores_own_map(register: Register) -> None:
    default = DummyAlgoConfig(magic=2)
    register(make_algo(name="awq", algo_config_map={"llama": default}))
    assert get_algo_config("awq", "llama") is default


def test_get_algo_config_still_serves_core_algorithms(register: Register) -> None:
    register(make_algo())
    assert get_algo_config("awq", "llama") is ALGORITHM_CONFIG_MAPS["awq"]["llama"]


def test_get_algo_config_still_rejects_an_unknown_algorithm(register: Register) -> None:
    register(make_algo())
    with pytest.raises(ValueError, match="Unsupported algorithm type"):
        get_algo_config("nosuchalgo", "llama")


def test_an_algos_processor_is_dispatched(register: Register) -> None:
    register(make_algo())
    assert get_processor("dummyalgo") is DummyProcessor


def test_a_registered_processor_wins_over_the_processor_map(register: Register) -> None:
    register(make_algo(name="awq"))
    assert get_processor("awq") is DummyProcessor


def test_core_processors_are_still_dispatched(register: Register) -> None:
    register(make_algo())
    assert get_processor("gptq") is PROCESSOR_MAP["gptq"]


def test_dispatching_an_unknown_algorithm_raises() -> None:
    with pytest.raises(KeyError):
        get_processor("nosuchalgo")


def test_an_algo_claims_its_name_in_the_quant_algo_loader(register: Register) -> None:
    register(make_algo())
    config = _load_quant_algo_config_from_dict({"name": "dummyalgo", "magic": 5})
    assert isinstance(config, DummyAlgoConfig)
    assert config.magic == 5


def test_an_algo_claims_its_name_in_the_pre_optimization_loader(register: Register) -> None:
    register(make_algo(name="dummypreopt", algo_config=DummyPreOptConfig))
    config = _load_pre_optimization_config_from_dict({"name": "dummypreopt", "magic": 5})
    assert isinstance(config, DummyPreOptConfig)


def test_core_algorithms_still_load_when_nothing_claims_the_name(register: Register) -> None:
    register(make_algo())
    config = _load_quant_algo_config_from_dict({"name": "rotation", "scaling_layers": []})
    assert isinstance(config, RotationConfig)


def test_the_registry_is_consulted_before_cores_own_branches(register: Register) -> None:
    """An algorithm claiming a core name wins, and sees the dict as it was written.

    Nothing shipped does this — it is the ordering that lets a ``QuarkAlgorithm`` own an algorithm
    core already knows a deprecated spelling of, without core's rewrite running first.
    """
    register(make_algo(name="rotation"))
    config = _load_quant_algo_config_from_dict({"name": "rotation", "magic": 2})
    assert isinstance(config, DummyAlgoConfig)
    assert config.magic == 2


# ------------------------------------------------------------------ LLMTemplate


def test_llm_template_selects_an_algo_registered_after_import(register: Register) -> None:
    """``LLMTemplate`` must not cache the supported-algorithm list or the per-model defaults.

    Registration is always "late": importing ``quark.torch`` imports ``template``, and the
    registry lives under ``quark.torch.algorithm``, so no import order lets a contributed
    algorithm register before the templates are built. A snapshot taken at import time therefore
    never contains one, which is what made ``--quant_algo <name>`` unreachable for them.
    """
    default = DummyAlgoConfig(magic=3)
    register(make_algo(algo_config_map={"llama": default}))

    template = LLMTemplate.get("llama")
    config = template.get_config(scheme="int4_wo_128", algorithm=["dummyalgo"])

    assert config.algo_config == [default]
    # Core's own algorithms are untouched by the lookup change.
    assert template.get_config(scheme="int4_wo_128", algorithm=["awq"]).algo_config == [
        ALGORITHM_CONFIG_MAPS["awq"]["llama"]
    ]


def test_llm_template_requires_a_config_for_an_algo_with_no_default(register: Register) -> None:
    """A missing per-model default is still the caller's problem, not a silent fallback."""
    register(make_algo())
    template = LLMTemplate.get("llama")

    with pytest.raises(NotImplementedError, match="No built-in dummyalgo configuration"):
        template.get_config(scheme="int4_wo_128", algorithm=["dummyalgo"])

    supplied = DummyAlgoConfig(magic=9)
    config = template.get_config(scheme="int4_wo_128", algorithm=["dummyalgo"], algo_configs={"dummyalgo": supplied})
    assert config.algo_config == [supplied]


def test_llm_template_still_rejects_an_unknown_algorithm(register: Register) -> None:
    register(make_algo())
    with pytest.raises(ValueError, match="Unsupported algorithm: nosuchalgo"):
        LLMTemplate.get("llama").get_config(scheme="int4_wo_128", algorithm=["nosuchalgo"])
