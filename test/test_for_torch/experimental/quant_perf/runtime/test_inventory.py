#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys
from types import SimpleNamespace

import quark.experimental.torch.quant_perf.runtime.inventory as runtime_inventory
from quark.experimental.torch.quant_perf.runtime.inventory import collect_runtime_inventory
from quark.experimental.torch.quant_perf.workspace.git import find_git_root


def test_find_git_root_accepts_files_and_nested_directories(tmp_path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "package" / "module.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("", encoding="utf-8")
    (repo / ".git").mkdir()

    assert find_git_root(nested) == repo
    assert find_git_root(nested.parent) == repo
    assert find_git_root(None) is None


def test_runtime_inventory_records_relevant_versions() -> None:
    inventory = collect_runtime_inventory()

    assert inventory["python"]
    assert inventory["platform"]
    assert inventory["packages"]["torch"]["version"]
    assert "rocm" in inventory["accelerator"]
    assert "gpu_models" in inventory["accelerator"]


def test_runtime_inventory_does_not_initialize_torch_accelerator(monkeypatch) -> None:
    class CudaProbe:
        accessed = False

        def is_available(self) -> bool:
            self.accessed = True
            return False

    cuda = CudaProbe()
    torch = SimpleNamespace(
        version=SimpleNamespace(hip="7.1", cuda=None),
        cuda=cuda,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(runtime_inventory, "_rocm_smi_gpu_models", lambda: ["AMD Instinct MI355X"])

    inventory = collect_runtime_inventory()

    assert cuda.accessed is False
    assert inventory["accelerator"]["rocm"] == "7.1"
    assert inventory["accelerator"]["gpu_models"] == ["AMD Instinct MI355X"]


def test_package_identity_reads_generated_version_module_when_distribution_metadata_is_missing(
    monkeypatch, tmp_path
) -> None:
    package_dir = tmp_path / "aiter"
    package_dir.mkdir()
    init_file = package_dir / "__init__.py"
    init_file.write_text("", encoding="utf-8")
    (package_dir / "_version.py").write_text("__version__ = '0.1.13.post1'\n", encoding="utf-8")

    def missing_distribution(_distribution: str) -> str:
        raise runtime_inventory.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(runtime_inventory.importlib.metadata, "version", missing_distribution)
    monkeypatch.setattr(
        runtime_inventory.importlib.util,
        "find_spec",
        lambda _module: SimpleNamespace(origin=str(init_file), submodule_search_locations=None),
    )

    identity = runtime_inventory._package_identity("amd-aiter", "aiter")

    assert identity["version"] == "0.1.13.post1"
