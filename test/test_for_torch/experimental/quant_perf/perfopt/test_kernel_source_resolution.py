#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
import subprocess

import pytest

from quark.experimental.torch.quant_perf.perfopt.kernel_source import (
    SOURCE_RESOLVER_VERSION,
    MappingKind,
    ResolutionConfidence,
    classify_kernel,
    resolve_kernel_source_repo,
)


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_operator_rule_maps_anonymous_w4a8_flydsl_kernel(tmp_path):
    framework_repo = tmp_path / "vllm"
    kernel_repo = tmp_path / "aiter"
    framework_repo.mkdir()
    expected = _write(
        kernel_repo / "aiter" / "ops" / "flydsl" / "kernels" / "mxfp4_preshuffle.py",
        "@flyc.kernel\ndef kernel_gemm(x):\n    pass\n",
    )
    _write(
        kernel_repo / "aiter" / "ops" / "flydsl" / "kernels" / "preshuffle_gemm.py",
        "@flyc.kernel\ndef kernel_gemm(x):\n    pass\n",
    )
    launcher = _write(
        kernel_repo / "aiter" / "ops" / "flydsl" / "batched_gemm_mxfp4.py",
        "from .kernels.mxfp4_preshuffle import launch_gemm\n"
        "def flydsl_gemm_a8w4_per_tensor(*args):\n"
        "    return launch_gemm(*args)\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "kernel_gemm_0.kd",
            "kernel_names": ["kernel_gemm_0.kd"],
            "parent_op_name": ("vllm::rocm_aiter_flydsl_gemm_a8w4_per_tensor"),
        },
        str(framework_repo),
        str(kernel_repo),
    )

    assert resolution.source_file == str(expected)
    assert resolution.source_symbol == "kernel_gemm"
    assert resolution.builder_symbol == "launch_gemm"
    assert resolution.launcher_source_file == str(launcher)
    assert resolution.compiler == "flydsl"
    assert resolution.confidence is ResolutionConfidence.OPERATOR_RULE
    assert resolution.resolver_version == SOURCE_RESOLVER_VERSION


def test_operator_rule_maps_anonymous_a4w4_flydsl_kernel(tmp_path):
    framework_repo = tmp_path / "vllm"
    kernel_repo = tmp_path / "aiter"
    framework_repo.mkdir()
    expected = _write(
        kernel_repo / "aiter" / "ops" / "flydsl" / "kernels" / "mxfp4_preshuffle.py",
        "@flyc.kernel\ndef kernel_gemm(x):\n    pass\n",
    )
    _write(
        kernel_repo / "aiter" / "ops" / "flydsl" / "kernels" / "preshuffle_gemm.py",
        "@flyc.kernel\ndef kernel_gemm(x):\n    pass\n",
    )
    launcher = _write(
        kernel_repo / "aiter" / "ops" / "flydsl" / "batched_gemm_mxfp4.py",
        "from .kernels.mxfp4_preshuffle import launch_gemm\n"
        "def flydsl_gemm_a4w4_dynamic(*args):\n"
        "    return launch_gemm(*args)\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "kernel_gemm_0.kd",
            "kernel_names": ["kernel_gemm_0.kd"],
            "parent_op_name": ("vllm::rocm_aiter_flydsl_gemm_a4w4_dynamic"),
        },
        str(framework_repo),
        str(kernel_repo),
    )

    assert resolution.source_file == str(expected)
    assert resolution.source_symbol == "kernel_gemm"
    assert resolution.builder_symbol == "launch_gemm"
    assert resolution.launcher_source_file == str(launcher)
    assert resolution.confidence is ResolutionConfidence.OPERATOR_RULE


def test_runtime_context_maps_anonymous_dense_mxfp4_flydsl_kernel(
    tmp_path,
):
    kernel_repo = tmp_path / "aiter"
    expected = _write(
        kernel_repo / "aiter" / "ops" / "flydsl" / "kernels" / "mxfp4_preshuffle.py",
        "@flyc.kernel\ndef kernel_gemm(x):\n    pass\n",
    )
    _write(
        kernel_repo / "aiter" / "ops" / "flydsl" / "kernels" / "preshuffle_gemm.py",
        "@flyc.kernel\ndef kernel_gemm(x):\n    pass\n",
    )
    launcher = _write(
        kernel_repo / "aiter" / "ops" / "flydsl" / "batched_gemm_mxfp4.py",
        "from .kernels.mxfp4_preshuffle import launch_gemm\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "kernel_gemm_7.kd",
            "kernel_names": ["kernel_gemm_7.kd"],
        },
        "",
        str(kernel_repo),
        context_tags=("flydsl_dense_mxfp4",),
    )

    assert resolution.source_file == str(expected)
    assert resolution.launcher_source_file == str(launcher)
    assert resolution.confidence is ResolutionConfidence.OPERATOR_RULE
    assert resolution.to_dict()["mapping_kind"] == "editable_source"
    assert resolution.to_dict()["patchable"] is True


def test_operator_rule_maps_parameterized_moe_reduction_kernel(tmp_path):
    framework_repo = tmp_path / "vllm"
    kernel_repo = tmp_path / "aiter"
    framework_repo.mkdir()
    expected = _write(
        kernel_repo / "aiter" / "ops" / "flydsl" / "kernels" / "moe_gemm_2stage.py",
        "def compile_moe_reduction():\n"
        "    @flyc.kernel(name='dynamic')\n"
        "    def moe_reduction_kernel(x):\n"
        "        pass\n",
    )
    launcher = _write(
        kernel_repo / "aiter" / "ops" / "flydsl" / "moe_kernels.py",
        "from .kernels.moe_gemm_2stage import compile_moe_reduction\n"
        "def _run_moe_reduction():\n"
        "    return compile_moe_reduction()\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": ("moe_reduction_kernel_plain_bf16_topk10_md4096.kd"),
            "kernel_names": ["moe_reduction_kernel_plain_bf16_topk10_md4096.kd"],
            "parent_op_name": "aiter::fused_moe_",
        },
        str(framework_repo),
        str(kernel_repo),
    )

    assert resolution.source_file == str(expected)
    assert resolution.source_symbol == "moe_reduction_kernel"
    assert resolution.builder_symbol == "compile_moe_reduction"
    assert resolution.launcher_source_file == str(launcher)
    assert resolution.confidence is ResolutionConfidence.OPERATOR_RULE


def test_operator_rule_maps_mixed_moe_kernel_launcher(tmp_path):
    kernel_repo = tmp_path / "aiter"
    expected = _write(
        kernel_repo / "aiter" / "ops" / "flydsl" / "kernels" / "mixed_moe_gemm_2stage.py",
        "def compile_mixed_moe_gemm():\n    pass\n",
    )
    launcher = _write(
        kernel_repo / "aiter" / "ops" / "flydsl" / "moe_kernels.py",
        "from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": ("mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256_vscale_fix3_fp4opt_v1_pm1_acc0.kd"),
            "kernel_names": ["mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256_vscale_fix3_fp4opt_v1_pm1_acc0.kd"],
        },
        "",
        str(kernel_repo),
    )

    assert resolution.source_file == str(expected)
    assert resolution.builder_symbol == "compile_mixed_moe_gemm"
    assert resolution.launcher_source_file == str(launcher)
    assert resolution.live_call_seam == "aiter::fused_moe_"
    assert resolution.confidence is ResolutionConfidence.OPERATOR_RULE


def test_anonymous_flydsl_kernel_without_parent_context_is_ambiguous(tmp_path):
    kernel_repo = tmp_path / "aiter"
    _write(
        kernel_repo / "one.py",
        "@flyc.kernel\ndef kernel_gemm(x):\n    pass\n",
    )
    _write(
        kernel_repo / "two.py",
        "@flyc.kernel\ndef kernel_gemm(x):\n    pass\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "kernel_gemm_0.kd",
            "kernel_names": ["kernel_gemm_0.kd"],
        },
        "",
        str(kernel_repo),
    )

    assert resolution.source_file is None
    assert resolution.confidence is ResolutionConfidence.AMBIGUOUS
    assert len(resolution.alternatives) == 2


def test_operator_rule_rejects_source_missing_expected_symbol(tmp_path):
    kernel_repo = tmp_path / "aiter"
    _write(
        kernel_repo / "aiter" / "ops" / "flydsl" / "kernels" / "mxfp4_preshuffle.py",
        "def unrelated():\n    pass\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "kernel_gemm_0.kd",
            "kernel_names": ["kernel_gemm_0.kd"],
            "parent_op_name": ("vllm::rocm_aiter_flydsl_gemm_a8w4_per_tensor"),
        },
        "",
        str(kernel_repo),
    )

    assert resolution.source_file is None
    assert resolution.confidence is ResolutionConfidence.UNRESOLVED


def test_missing_upstream_source_does_not_bypass_resolution(tmp_path):
    kernel_repo = tmp_path / "aiter"
    kernel_repo.mkdir()

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "unknown_kernel.kd",
            "kernel_names": ["unknown_kernel.kd"],
            "source_file": str(kernel_repo / "missing.py"),
        },
        "",
        str(kernel_repo),
    )

    assert resolution.source_file is None
    assert resolution.confidence is ResolutionConfidence.UNRESOLVED


def test_unverified_upstream_source_falls_through_to_definition_search(tmp_path):
    kernel_repo = tmp_path / "aiter"
    hinted_source = _write(
        kernel_repo / "csrc" / "unrelated.cu",
        "__global__ void unrelated_kernel(float* out) {}\n",
    )
    expected = _write(
        kernel_repo / "csrc" / "target.cu",
        "__global__ void target_kernel(float* out) {}\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "target_kernel.kd",
            "kernel_names": [],
            "source_file": str(hinted_source),
        },
        "",
        str(kernel_repo),
    )

    assert resolution.source_file == str(expected)
    assert resolution.source_symbol == "target_kernel"
    assert resolution.method == "exact_gpu_definition"


def test_verified_upstream_source_records_matching_symbol(tmp_path):
    kernel_repo = tmp_path / "aiter"
    expected = _write(
        kernel_repo / "csrc" / "target.cu",
        "__global__ void target_kernel(float* out) {}\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "target_kernel.kd",
            "kernel_names": [],
            "source_file": str(expected),
        },
        "",
        str(kernel_repo),
    )

    assert resolution.source_file == str(expected)
    assert resolution.source_symbol == "target_kernel"
    assert resolution.method == "upstream_source_file"


def test_generated_kernel_provenance_records_persistent_artifacts(tmp_path):
    from quark.experimental.torch.quant_perf.perfopt.kernel_source import (
        generated_kernel_provenance,
    )

    kernel_name = "triton_red_fused_example_1.kd"
    stem = kernel_name.removesuffix(".kd")
    cache = tmp_path / "triton"
    artifact_dir = cache / "HASH"
    origin = tmp_path / "torchinductor" / "graph.py"
    origin.parent.mkdir()
    origin.write_text("def triton_red_fused_example_1():\n    pass\n")
    generated_source = _write(
        artifact_dir / f"{stem}.source",
        f'#loc = loc("{origin}":18:0)\nmodule {{}}\n',
    )
    generated_ttir = _write(
        artifact_dir / f"{stem}.ttir",
        "module {}\n",
    )

    provenance = generated_kernel_provenance(
        kernel_name,
        cache_roots=(cache,),
    )

    assert provenance["mapping_kind"] == "generated_artifact"
    assert provenance["patchable"] is False
    assert provenance["method"] == "torchinductor_generated"
    assert provenance["generated_source_file"] == str(generated_source)
    assert provenance["generated_ttir_file"] == str(generated_ttir)
    assert provenance["generated_origin_file"] == str(origin)
    assert provenance["retryable"] is False


def test_dynamic_group_quant_family_maps_without_parent_context(tmp_path):
    kernel_repo = tmp_path / "aiter"
    expected = _write(
        kernel_repo / "csrc" / "kernels" / "quant_kernels.cu",
        "template <typename T>\n"
        "__global__ void\n"
        "dynamic_per_group_scaled_quant_kernel(T* out) {\n"
        "}\n"
        "void dynamic_per_group_scaled_quant() {}\n",
    )
    _write(
        kernel_repo / "aiter" / "ops" / "quant.py",
        "def dynamic_per_group_scaled_quant(*args):\n    pass\n",
    )
    _write(
        kernel_repo / "csrc" / "include" / "quant.h",
        "void dynamic_per_group_scaled_quant();\n",
    )
    _write(
        kernel_repo / "csrc" / "kernels" / "dsv4_rotate_quant.cu",
        "// dynamic_per_group_scaled_quant_kernel(...)\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": (
                "_ZN5aiter37dynamic_per_group_scaled_quant_kernel"
                "IDF16bN4opus5fp4_tELi32ELi32ELb1ELi64ELb1EEEv"
                "PT0_PfPKT_PKfliilPKii.kd"
            ),
            "kernel_names": [],
        },
        "",
        str(kernel_repo),
    )

    assert resolution.source_file == str(expected)
    assert resolution.source_symbol == ("dynamic_per_group_scaled_quant_kernel")
    assert resolution.builder_symbol == "dynamic_per_group_scaled_quant"
    assert resolution.build_module == "module_quant"
    assert resolution.confidence is ResolutionConfidence.OPERATOR_RULE
    assert resolution.to_dict()["mapping_kind"] == "editable_source"


def test_compiler_provenance_maps_graph_only_anonymous_kernel(tmp_path):
    kernel_repo = tmp_path / "aiter"
    source = _write(
        kernel_repo / "aiter" / "ops" / "flydsl" / "kernels" / "mxfp4_preshuffle.py",
        "@flyc.kernel\ndef kernel_gemm(x):\n    pass\n",
    )
    records = [
        {
            "runtime_kernel_name": "kernel_gemm_0.kd",
            "compiler": "flydsl",
            "source_file": str(source),
            "source_repo": str(kernel_repo),
            "source_repo_role": "kernel",
            "source_relpath": ("aiter/ops/flydsl/kernels/mxfp4_preshuffle.py"),
            "source_symbol": "kernel_gemm",
            "builder_symbol": "launch_gemm",
            "gpu_arch": "MI355X",
            "repo_revision": "",
            "source_sha256": "",
            "cache_key_hash": "abc123",
            "artifact_sha256": "def456",
        }
    ]

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "kernel_gemm_0.kd",
            "kernel_names": ["kernel_gemm_0.kd"],
            "parent_op_names": [],
        },
        "",
        str(kernel_repo),
        provenance_records=records,
        gpu_arch="MI355X",
    )

    assert resolution.source_file == str(source)
    assert resolution.method == "compiler_manifest"
    assert resolution.confidence is ResolutionConfidence.EXACT
    assert resolution.to_dict()["cache_key_hash"] == "abc123"


@pytest.mark.parametrize("packaged_layout", [False, True])
def test_aiter_asm_code_object_is_classified_without_editable_source(
    tmp_path,
    packaged_layout,
):
    kernel_repo = tmp_path / "aiter"
    metadata_root = kernel_repo / "aiter_meta" if packaged_layout else kernel_repo
    config = _write(
        metadata_root / "hsa" / "gfx950" / "f4gemm" / "f4gemm_bf16_per1x32Fp4.csv",
        "tile_M,tile_N,splitK,bpreshuffle,knl_name,co_name\n"
        "192,128,0,1,"
        "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_192x128E,"
        "f4gemm_bf16_per1x32Fp4_BpreShuffle_192x128.co\n",
    )
    binary = _write(
        config.parent / "f4gemm_bf16_per1x32Fp4_BpreShuffle_192x128.co",
        "binary",
    )
    launcher = _write(
        metadata_root / "csrc" / "py_itfs_cu" / "asm_gemm_a4w4.cu",
        "AiterAsmKernel(name, co_name);\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": ("_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_192x128E.kd"),
            "kernel_names": [],
        },
        "",
        str(kernel_repo),
        gpu_arch="MI355X",
    )
    payload = resolution.to_dict()

    assert resolution.source_file is None
    assert payload["mapping_kind"] == "precompiled_binary"
    assert payload["patchable"] is False
    assert payload["retryable"] is False
    assert payload["binary_file"] == str(binary)
    assert payload["config_file"] == str(config)
    assert payload["launcher_source_file"] == str(launcher)


def test_exact_gpu_definition_outranks_wrapper_declaration_and_comment(
    tmp_path,
):
    kernel_repo = tmp_path / "aiter"
    implementation = _write(
        kernel_repo / "csrc" / "kernels" / "real.cu",
        "__global__ void\nexample_quant_kernel(float* out) {\n}\n",
    )
    _write(
        kernel_repo / "aiter" / "ops" / "wrapper.py",
        "def example_quant_kernel(*args):\n    pass\n",
    )
    _write(
        kernel_repo / "csrc" / "include" / "example.h",
        "void example_quant_kernel(float* out);\n",
    )
    _write(
        kernel_repo / "csrc" / "kernels" / "mention.cu",
        "// example_quant_kernel(...)\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": ("void aiter::example_quant_kernel<float>(float*)"),
            "kernel_names": [],
        },
        "",
        str(kernel_repo),
    )

    assert resolution.source_file == str(implementation)
    assert resolution.source_symbol == "example_quant_kernel"
    assert resolution.method == "exact_gpu_definition"
    assert resolution.confidence is ResolutionConfidence.UNIQUE_DEFINITION


def test_native_kernel_python_wrapper_is_launcher_only(tmp_path):
    framework_repo = tmp_path / "vllm"
    wrapper = _write(
        framework_repo / "vllm" / "_custom_ops.py",
        "def native_fp8_kernel(*args):\n    return torch.ops._C.native_fp8_kernel(*args)\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "native_fp8_kernel.kd",
            "kernel_names": [],
        },
        str(framework_repo),
    )

    assert resolution.mapping_kind is MappingKind.UNRESOLVED
    assert resolution.source_file is None
    assert resolution.launcher_source_file == str(wrapper)
    assert resolution.launcher_symbol == "native_fp8_kernel"
    assert resolution.patchable is False
    assert resolution.retryable is True
    assert resolution.reason == "full framework source repository required"


def test_native_kernel_upstream_python_source_is_launcher_only(tmp_path):
    framework_repo = tmp_path / "vllm"
    wrapper = _write(
        framework_repo / "vllm" / "_custom_ops.py",
        "def native_fp8_kernel(*args):\n    return torch.ops._C.native_fp8_kernel(*args)\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "native_fp8_kernel.kd",
            "kernel_names": [],
            "source_file": str(wrapper),
        },
        str(framework_repo),
    )

    assert resolution.mapping_kind is MappingKind.UNRESOLVED
    assert resolution.source_file is None
    assert resolution.launcher_source_file == str(wrapper)
    assert resolution.launcher_symbol == "native_fp8_kernel"
    assert resolution.patchable is False
    assert resolution.retryable is True
    assert resolution.reason == "full framework source repository required"


def test_exact_gpu_definition_ignores_runtime_build_mirror(tmp_path):
    kernel_repo = tmp_path / "aiter"
    _write(
        kernel_repo / "aiter_meta" / "3rdparty" / "composable_kernel" / "include" / "gridwise_moe_mx_gemm.hpp",
        "__global__ void kernel_moe_mxgemm_2lds(float* out) {}\n",
    )
    canonical = _write(
        kernel_repo
        / "aiter_meta"
        / "3rdparty"
        / "composable_kernel"
        / "include"
        / "gridwise_moe_mx_gemm_bpreshuffle.hpp",
        "__global__ void kernel_moe_mxgemm_2lds(float* out) {}\n",
    )
    _write(
        kernel_repo / "build" / "module_hash" / "include" / "gridwise_moe_mx_gemm_bpreshuffle.hpp",
        "__global__ void kernel_moe_mxgemm_2lds(float* out) {}\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": ("void ck::kernel_moe_mxgemm_2lds<ck::GridwiseMoeGemmMX_BPreshuffle<float>>(float)"),
            "kernel_names": [],
        },
        "",
        str(kernel_repo),
    )

    assert resolution.source_file == str(canonical)
    assert resolution.mapping_kind is MappingKind.DEPENDENCY_SOURCE
    assert resolution.patchable is False
    assert resolution.confidence is ResolutionConfidence.UNIQUE_DEFINITION
    assert resolution.method == "exact_gpu_definition_path_match"


def test_runtime_build_definition_is_not_patchable_source(tmp_path):
    kernel_repo = tmp_path / "aiter"
    generated = _write(
        kernel_repo / "build" / "module_hash" / "kernel.cu",
        "__global__ void generated_only_kernel(float* out) {}\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "generated_only_kernel.kd",
            "kernel_names": [],
            "source_file": str(generated),
        },
        "",
        str(kernel_repo),
    )

    assert resolution.source_file is None
    assert resolution.mapping_kind.value == "generated_artifact"
    assert resolution.patchable is False
    assert resolution.alternatives == (str(generated),)


def test_legitimate_generated_source_directory_remains_searchable(tmp_path):
    kernel_repo = tmp_path / "aiter"
    source = _write(
        kernel_repo / "generated" / "stable_kernel.cu",
        "__global__ void stable_generated_kernel(float* out) {}\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "stable_generated_kernel.kd",
            "kernel_names": [],
        },
        "",
        str(kernel_repo),
    )

    assert resolution.source_file == str(source)
    assert resolution.mapping_kind.value == "editable_source"


def test_multiple_exact_gpu_definitions_remain_ambiguous(tmp_path):
    kernel_repo = tmp_path / "aiter"
    first = _write(
        kernel_repo / "csrc" / "kernels" / "one.cu",
        "__global__ void duplicate_kernel(float* out) {}\n",
    )
    second = _write(
        kernel_repo / "csrc" / "kernels" / "two.cu",
        "__global__ void duplicate_kernel(float* out) {}\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "duplicate_kernel.kd",
            "kernel_names": [],
        },
        "",
        str(kernel_repo),
    )

    assert resolution.source_file is None
    assert resolution.mapping_kind.value == "ambiguous"
    assert resolution.method == "exact_gpu_definition"
    assert set(resolution.alternatives) == {
        str(first),
        str(second),
    }


def test_external_runtime_source_is_located_but_not_patchable(tmp_path):
    kernel_repo = tmp_path / "aiter"
    kernel_repo.mkdir()
    quark_repo = tmp_path / "Quark"
    source = _write(
        quark_repo / "quark" / "torch" / "kernel" / "dequantize_kernels_hip.cuh",
        "__global__ void dq_uint8_mxfp4_to_half_kernel(float* out) {}\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": ("void dq_uint8_mxfp4_to_half_kernel<float>(float*)"),
            "kernel_names": [],
        },
        "",
        str(kernel_repo),
        external_source_roots={"quantizer": str(quark_repo)},
    )
    payload = resolution.to_dict()

    assert resolution.source_file == str(source)
    assert payload["mapping_kind"] == "external_unmanaged_source"
    assert payload["source_repo_role"] == "quantizer"
    assert payload["patchable"] is False
    assert payload["retryable"] is False


@pytest.mark.parametrize("packaged_layout", [False, True])
def test_aiter_build_module_resolves_unique_gpu_definition(
    tmp_path,
    packaged_layout,
):
    kernel_repo = tmp_path / "aiter"
    metadata_root = kernel_repo / "aiter_meta" if packaged_layout else kernel_repo
    source = _write(
        metadata_root / "csrc" / "kernels" / "module_kernel.cu",
        "__global__ void module_owned_kernel(float* out) {}\n",
    )
    config = {
        "module_owned": {
            "srcs": [
                "f'{AITER_CSRC_DIR}/kernels/module_kernel.cu'",
            ],
        }
    }
    _write(
        kernel_repo / "aiter" / "jit" / "optCompilerConfig.json",
        json.dumps(config),
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "module_owned_kernel.kd",
            "kernel_names": [],
        },
        "",
        str(kernel_repo),
    )

    assert resolution.source_file == str(source)
    assert resolution.source_symbol == "module_owned_kernel"
    assert resolution.build_module == "module_owned"
    assert resolution.method == "aiter_build_module"


@pytest.mark.parametrize("packaged_layout", [False, True])
def test_ck_tile_kentry_maps_inner_kernel_type_as_readonly_dependency(
    tmp_path,
    packaged_layout,
):
    kernel_repo = tmp_path / "aiter"
    metadata_root = kernel_repo / "aiter_meta" if packaged_layout else kernel_repo
    ck_root = metadata_root / "3rdparty" / "composable_kernel"
    implementation = _write(
        ck_root / "include" / "ck_tile" / "ops" / "fused_moe" / "kernel" / "moe_sorting_kernel.hpp",
        "namespace ck_tile {\ntemplate <typename Problem>\nstruct MoeSortingMultiPhaseKernel_P23 {};\n}\n",
    )
    launcher = _write(
        ck_root / "include" / "ck_tile" / "host" / "kernel_launch.hpp",
        "namespace ck_tile {\n"
        "template <int N, typename Kernel, typename Args>\n"
        "__global__ void kentry(Args args) {}\n"
        "}\n",
    )
    _write(
        metadata_root / "csrc" / "include" / "moe_sorting_opus.h",
        "namespace aiter {\ntemplate <typename Problem>\nstruct MoeSortingMultiPhaseKernel_P23 {};\n}\n",
    )
    runtime_name = (
        "void ck_tile::kentry<2, "
        "ck_tile::MoeSortingMultiPhaseKernel_P23<"
        "ck_tile::MoeSortingProblemMp<int>>, "
        "ck_tile::MoeSortingMultiPhaseKernel_P23<"
        "ck_tile::MoeSortingProblemMp<int>>::Kargs>("
        "ck_tile::MoeSortingMultiPhaseKernel_P23<"
        "ck_tile::MoeSortingProblemMp<int>>::Kargs)"
    )

    resolution = resolve_kernel_source_repo(
        {
            "device_kernel_name": runtime_name,
            "kernel_names": [],
        },
        "",
        str(kernel_repo),
    )

    assert resolution.mapping_kind is MappingKind.DEPENDENCY_SOURCE
    assert resolution.source_file == str(implementation)
    assert resolution.source_repo == str(ck_root)
    assert resolution.source_repo_role == "composable_kernel"
    assert resolution.source_symbol == "MoeSortingMultiPhaseKernel_P23"
    assert resolution.launcher_source_file == str(launcher)
    assert resolution.launcher_symbol == "kentry"
    assert resolution.method == "template_kernel_type"
    assert resolution.patchable is False
    assert resolution.retryable is False


def test_generated_compiler_provenance_falls_back_to_canonical_source(
    tmp_path,
):
    kernel_repo = tmp_path / "aiter"
    canonical = _write(
        kernel_repo / "csrc" / "kernels" / "canonical.cu",
        "__global__ void canonical_kernel(float* out) {}\n",
    )
    generated = _write(
        kernel_repo / "build" / "module" / "canonical.cu",
        "__global__ void canonical_kernel(float* out) {}\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "canonical_kernel.kd",
            "kernel_names": [],
        },
        "",
        str(kernel_repo),
        provenance_records=[
            {
                "runtime_kernel_name": "canonical_kernel.kd",
                "source_file": str(generated),
                "source_repo": str(kernel_repo),
                "source_repo_role": "kernel",
                "source_symbol": "canonical_kernel",
            }
        ],
    )

    assert resolution.source_file == str(canonical)
    assert resolution.mapping_kind is MappingKind.EDITABLE_SOURCE
    assert resolution.method == "exact_gpu_definition"


def test_wvsplit_kernel_is_not_preclassified_as_vendor_binary():
    rewritable, reason = classify_kernel("void wvSplitKrc_<__hip_bfloat16>(int, int)")

    assert rewritable is True
    assert reason == ""


@pytest.mark.parametrize(
    ("kernel_name", "source_symbol", "expected_kind"),
    [
        ("Cijk_A_B.kd", "Cijk_A", MappingKind.PRECOMPILED_BINARY),
        (
            "triton_red_fused_add_0.kd",
            "triton_red_fused_add",
            MappingKind.GENERATED_ARTIFACT,
        ),
    ],
)
def test_resolver_rejects_non_patchable_runtime_kernels(
    tmp_path,
    kernel_name,
    source_symbol,
    expected_kind,
):
    framework_repo = tmp_path / "vllm"
    _write(
        framework_repo / "csrc" / "candidate.cu",
        f"__global__ void {source_symbol}(float* out) {{}}\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": kernel_name,
            "kernel_names": [],
        },
        str(framework_repo),
    )

    assert resolution.mapping_kind is expected_kind
    assert resolution.patchable is False
    assert resolution.method == "runtime_classification"


def test_exact_definition_checks_all_managed_repositories(tmp_path):
    framework_repo = tmp_path / "vllm"
    kernel_repo = tmp_path / "aiter"
    framework_source = _write(
        framework_repo / "csrc" / "shared.cu",
        "__global__ void shared_runtime_kernel(float* out) {}\n",
    )
    kernel_source = _write(
        kernel_repo / "csrc" / "shared.cu",
        "__global__ void shared_runtime_kernel(float* out) {}\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "shared_runtime_kernel.kd",
            "kernel_names": [],
        },
        str(framework_repo),
        str(kernel_repo),
    )

    assert resolution.source_file is None
    assert resolution.mapping_kind.value == "ambiguous"
    assert set(resolution.alternatives) == {
        str(framework_source),
        str(kernel_source),
    }


def test_generic_definition_search_checks_all_managed_repositories(
    tmp_path,
):
    framework_repo = tmp_path / "vllm"
    kernel_repo = tmp_path / "aiter"
    framework_source = _write(
        framework_repo / "python" / "shared.py",
        "def shared_python_kernel(x):\n    return x\n",
    )
    kernel_source = _write(
        kernel_repo / "python" / "shared.py",
        "def shared_python_kernel(x):\n    return x\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "shared_python_kernel.kd",
            "kernel_names": [],
        },
        str(framework_repo),
        str(kernel_repo),
    )

    assert resolution.source_file is None
    assert resolution.method == "definition_search"
    assert set(resolution.alternatives) == {
        str(framework_source),
        str(kernel_source),
    }


def test_generic_definition_search_ignores_cpp_declaration_and_comment(
    tmp_path,
):
    kernel_repo = tmp_path / "aiter"
    implementation = _write(
        kernel_repo / "csrc" / "impl.cpp",
        "void host_dispatch_helper(float* out) {\n  *out = 0.0f;\n}\n",
    )
    _write(
        kernel_repo / "csrc" / "include" / "helper.h",
        "void host_dispatch_helper(float* out);\n",
    )
    _write(
        kernel_repo / "csrc" / "mention.cpp",
        "// host_dispatch_helper(...)\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": "host_dispatch_helper.kd",
            "kernel_names": [],
        },
        "",
        str(kernel_repo),
    )

    assert resolution.source_file == str(implementation)
    assert resolution.method == "unique_definition"


def test_generic_definition_search_finds_hpp_kernel_definition(tmp_path):
    kernel_repo = tmp_path / "aiter"
    implementation = _write(
        kernel_repo / "include" / "gridwise_moe.hpp",
        "template <typename GridwiseGemm>\n"
        "__global__ void kernel_moe_mxgemm_2lds(typename GridwiseGemm::Argument arg) {}\n",
    )

    resolution = resolve_kernel_source_repo(
        {
            "op_name": ("void ck::kernel_moe_mxgemm_2lds<ck::GridwiseMoeGemmMX_BPreshuffle<float>>(float)"),
            "kernel_names": [],
        },
        "",
        str(kernel_repo),
    )

    assert resolution.source_file == str(implementation)
    assert resolution.method == "exact_gpu_definition"


def test_source_resolution_serialization_adds_revision_and_source_hash(
    tmp_path,
):
    kernel_repo = tmp_path / "aiter"
    source = _write(
        kernel_repo / "csrc" / "kernels" / "hash_kernel.cu",
        "__global__ void hash_kernel(float* out) {}\n",
    )
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=kernel_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=kernel_repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=kernel_repo,
        check=True,
    )
    subprocess.run(
        ["git", "add", "."],
        cwd=kernel_repo,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "base"],
        cwd=kernel_repo,
        check=True,
        capture_output=True,
    )

    resolution = resolve_kernel_source_repo(
        {"op_name": "hash_kernel.kd", "kernel_names": []},
        "",
        str(kernel_repo),
    )
    payload = resolution.to_dict()

    assert (
        payload["repo_revision"]
        == subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=kernel_repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert payload["source_sha256"]
    assert payload["source_file"] == str(source)
