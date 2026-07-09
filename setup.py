#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Anchor setup.py to its own directory:
# (1) so ``from quark.*`` works under PEP 517 frontends that exec
#     setup.py with the source root off ``sys.path`` (in particular
#     ``quark.common.torch_cpp_build_specs`` below); and
# (2) so relative paths in this file (``quark/version.txt``,
#     ``pyproject.toml``, ``requirements.txt``, etc.) resolve correctly
#     regardless of the caller's CWD. The second part matters on
#     Windows wheel builds where ``vcvarsx86_amd64.bat`` (sourced
#     before ``python -m build`` to set up MSVC for AOT-compiled
#     stable-ABI ``_C`` extensions) shifts CWD to the Visual Studio
#     install tree as a documented side effect.
_setup_py_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _setup_py_dir)
os.chdir(_setup_py_dir)


from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py

# ScreenLogger lives in quark.common.utils.log and only depends on stdlib, so it
# is safe to import before the optional torch import below.
from quark.common.utils.log import ScreenLogger

logger = ScreenLogger("quark.setup")

try:
    import torch
    from torch.utils.cpp_extension import (
        CUDA_HOME,
        ROCM_HOME,
        BuildExtension,
    )

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False
    # Reached when ``--no-build-isolation`` is set but torch isn't installed in the
    # active env (typically an sdist-only build). Default isolation installs torch
    # via [build-system].requires, so this branch shouldn't fire there.
    logger.warning(
        "torch not importable; skipping C++ extension build. "
        "Install torch first (e.g. tools/ci/install_torch.sh) for wheels that include _C."
    )


# Optional override for the accelerator family auto-detected from torch.version.
# Use case: force a CPU-only wheel from a GPU-torch env (e.g. CI cross-builds).
# Cuda/rocm overrides are sanity-checked against torch.version below so we can't
# silently produce a wheel tagged for the wrong torch C++ ABI.
_QUARK_ACCELERATOR = os.getenv("QUARK_ACCELERATOR", "").strip().lower()
if _QUARK_ACCELERATOR and _QUARK_ACCELERATOR not in ("cpu", "cuda", "rocm"):
    raise ValueError(
        f"QUARK_ACCELERATOR={_QUARK_ACCELERATOR!r}: must be one of cpu, cuda, rocm "
        "(or unset to auto-detect from torch.version)."
    )


def _accelerator_family() -> str:
    """Active accelerator family: ``QUARK_ACCELERATOR`` override (if set) else inferred from torch.

    Raises if the override asks for a GPU family the active torch can't provide
    (e.g. ``cuda`` requested but torch is a ROCm build) -- silently producing a
    wrong-ABI wheel is the bug we want to prevent.
    """
    if _QUARK_ACCELERATOR:
        if HAS_TORCH and _QUARK_ACCELERATOR == "cuda" and not torch.version.cuda:
            raise RuntimeError(
                "QUARK_ACCELERATOR=cuda but installed torch is not a CUDA build "
                f"(torch.version.cuda is None; torch.version.hip={torch.version.hip!r}). "
                "Install a CUDA torch first, or unset QUARK_ACCELERATOR for auto-detect."
            )
        if HAS_TORCH and _QUARK_ACCELERATOR == "rocm" and not torch.version.hip:
            raise RuntimeError(
                "QUARK_ACCELERATOR=rocm but installed torch is not a ROCm build "
                f"(torch.version.hip is None; torch.version.cuda={torch.version.cuda!r}). "
                "Install a ROCm torch first, or unset QUARK_ACCELERATOR for auto-detect."
            )
        return _QUARK_ACCELERATOR
    if HAS_TORCH and torch.version.hip:
        return "rocm"
    if HAS_TORCH and torch.version.cuda:
        return "cuda"
    return "cpu"

"""Quark source-tree builds. ``--no-build-isolation`` is the preferred path:
it reuses the active env's torch directly (no per-build re-download, and the
wheel links against the exact torch you've installed). Default isolation is
supported as a fallback -- ``torch`` is in ``[build-system].requires`` so pip
populates the isolated build env automatically. See CONTRIBUTING.md for the
source-build workflow and accelerator-variant selection (set
``PIP_EXTRA_INDEX_URL`` to override PyPI's default torch index for ROCm).

Build invocations::

    # Recommended (install torch first, then):
    pip install --no-build-isolation -e .
    python -m build --no-isolation --wheel

    # Fallback (default isolation; pip pulls torch into the build env):
    pip install .
    python -m build --wheel

Version naming (X.Y.Z read from ``quark/version.txt``)::

    default          -> X.Y.Z+<git_hash>
    QUARK_NIGHTLY=1  -> X.Y.Z.devYYYYMMDD+<accel>
    QUARK_RELEASE=1  -> X.Y.Z+<accel>

``<accel>`` is computed from ``torch.version.{hip,cuda}`` of the build-env
torch (the one ``_C`` is linked against), so the wheel tag and torch C++ ABI
can't disagree by construction. Override the auto-detected family with
``QUARK_ACCELERATOR=cpu|cuda|rocm`` to force a specific build type (e.g.
``QUARK_ACCELERATOR=cpu`` produces a CPU-only wheel even on a GPU-torch host).
Incoherent combinations (``cuda`` requested with a ROCm torch installed, or
vice versa) raise rather than silently produce a wrong-ABI wheel.
"""

def update_pyproject_toml_project_name(new_name: str) -> None:
    if not new_name.replace("-", "_").isidentifier():
        raise ValueError(f"Invalid python package name: {new_name}")

    with open('pyproject.toml', 'r') as f:
        lines = f.readlines()
    with open('pyproject.toml', 'w') as f:
        for line in lines:
            if line.replace(" ", "").startswith("name="):
                line = f'name = "{new_name}"\n'
            f.write(line)


def string_to_bool(s):
    s = s.lower()
    if s in ("true", "1", "yes"):
        return True
    elif s in ("false", "0", "no"):
        return False
    else:
        raise ValueError(f"Invalid boolean string: {s}")


package_name = os.getenv("QUARK_WHEEL_NAME", "amd-quark")
_version_txt = open("quark/version.txt", "r").read().strip()
is_nightly = string_to_bool(os.getenv('QUARK_NIGHTLY', 'false')) is True
is_release = string_to_bool(os.getenv('QUARK_RELEASE', 'false')) is True

if is_nightly:
    # Naming nightly packagse differently to allow easy pip install without conflicting with the release versions
    # E.g., pip install amd-quark-nightly --trusted-host artifactory.domain.com -i "https://artifactory.domain.com/artifactory/api/pypi/repository_name/simple"
    package_name += "-nightly"

# Update quark package name to support nightly build packages with -nightly appended to package_name
# local pyproject.toml needs to be updated to rename wheel package in additional to setup.py
# https://discuss.python.org/t/dynamic-project-names-and-pep-621/21359/13
update_pyproject_toml_project_name(package_name)


def os_path_join(*args, **kwargs):
    p = os.path.join(*args, **kwargs)
    p = os.path.normpath(p)
    return p


def os_path_exists(path):
    path = os.path.normpath(path)
    return os.path.exists(path)


def os_path_dirname(path):
    path = os.path.normpath(path)
    res_path = os.path.dirname(path)
    return os.path.normpath(res_path)


def os_path_abspath(path):
    path = os.path.normpath(path)
    res_path = os.path.abspath(path)
    return os.path.normpath(res_path)


def read_requirements():
    with open('requirements.txt', 'r') as f:
        requirements = f.read().splitlines()
    return requirements


def generate_proto(source_dir: Path, dest_dir: Path, mypy: bool) -> None:
    """
    Generate python files from proto source files.

    Args:
        source_dir (Path): Path to a directory containing .proto files
        dest_dir (Path): Path to a directory to store generated .py files
        mypy (bool): Whether to generate mypy type stubs

    Raises:
        RuntimeError: Raised if protoc fails to generate the files
    """
    proto_files = source_dir.glob("*.proto")
    for proto_file in proto_files:
        output = dest_dir / proto_file.name.replace(".proto", "_pb2.py")

        # skip generation if output is up-to-date
        if not os.path.exists(output) or os.path.getmtime(proto_file) > os.path.getmtime(output):
            protoc_command = ["protoc", f"-I{source_dir}", f"--python_out={dest_dir}"]
            if mypy:
                protoc_command.append(f"--mypy_out={dest_dir}")
            protoc_command.append(str(proto_file))
            retval = subprocess.run(protoc_command, capture_output=True)
            if retval.returncode != 0:
                raise RuntimeError(f"protoc failed for {proto_file}: {retval.stderr.decode('utf-8')}")


def get_extensions():
    if not HAS_TORCH:
        return []

    # Lazy: torch_utils imports torch at module load.
    from quark.common.torch_cpp_build_specs import (
        HW_EMULATION_C_MODULE,
        ONNX_CUSTOM_OPS_C_MODULE,
        ONNX_CUSTOM_OPS_CPU_C_MODULE,
        ORT_WINDOWS_DEFINE,
        TORCH_TARGET_VERSION_DEFINE,
    )
    from quark.common.torch_cpp_build_specs.onnx_ops import (
        onnx_ops_sources,
        onnx_ort_cpu_sources,
        ort_include_paths,
        stable_abi_include_paths,
    )
    from quark.common.torch_cpp_build_specs.torch_ops import torch_include_paths, torch_ops_sources
    from quark.common.torch_cpp_ext import make_setuptools_extension
    from quark.common.utils.torch_utils import torch_supports_stable_abi

    if not torch_supports_stable_abi():
        logger.warning(
            f"PyTorch {torch.__version__} lacks torch::stable headers; "
            "skipping C++ stable-ABI extension build (JIT fallback will be used at runtime)."
        )
        return []

    debug_mode = os.getenv("DEBUG", "0") == "1"
    # Build GPU extensions iff torch was built with GPU support, a real toolkit is on
    # disk, AND ``QUARK_ACCELERATOR`` doesn't force CPU (override case: GPU-torch env
    # + ``QUARK_ACCELERATOR=cpu`` produces a CPU-only wheel).
    use_cuda = (
        _accelerator_family() in ("cuda", "rocm")
        and torch.cuda.is_available()
        and (CUDA_HOME is not None or ROCM_HOME is not None)
    )
    if debug_mode:
        logger.info("Compiling in debug mode")
    is_windows = platform.system() == "Windows"

    # Multi-arch wheel: CUDAExtension reads PYTORCH_ROCM_ARCH /
    # TORCH_CUDA_ARCH_LIST from the build env; do not pin them here. The
    # caller can narrow the arch fan-out by exporting either var (see
    # _build_wheel.yml's gpu-architectures input for PR CI).
    config_lines = [
        "=" * 60,
        "  Quark C++ Extension Build Configuration",
        "=" * 60,
        f"  Platform        : {platform.system()}",
        f"  Compiler        : {'MSVC' if is_windows else 'GCC/Clang'}",
        f"  PyTorch version : {torch.__version__}",
        f"  CUDA available  : {use_cuda}",
        f"  Extension type  : {'CUDAExtension' if use_cuda else 'CppExtension'}",
        f"  Debug mode      : {debug_mode}",
    ]
    if use_cuda:
        config_lines.append(f"  CUDA_HOME       : {CUDA_HOME}")
        config_lines.append(f"  ROCM_HOME       : {ROCM_HOME}")
    config_lines.append("=" * 60)
    logger.info("\n".join(config_lines))

    # Shared per-artifact source builders prevent pre-compilation/JIT source drift.
    torch_hw_emulation_ext = make_setuptools_extension(
        name=HW_EMULATION_C_MODULE,
        sources=torch_ops_sources(use_cuda=use_cuda),
        include_paths=torch_include_paths(),
        extra_defines=[TORCH_TARGET_VERSION_DEFINE],
        use_cuda=use_cuda,
        debug=debug_mode,
    )
    onnx_custom_ops_ext = make_setuptools_extension(
        name=ONNX_CUSTOM_OPS_C_MODULE,
        sources=onnx_ops_sources(use_cuda=use_cuda),
        include_paths=stable_abi_include_paths(),
        # ORT headers stay out of include_dirs to skip hipify rewriting;
        # see ``_jit_compile_stable_abi`` for the same workaround.
        extra_isolated_includes=ort_include_paths(),
        extra_defines=[TORCH_TARGET_VERSION_DEFINE],
        extra_windows_defines=[ORT_WINDOWS_DEFINE],
        use_cuda=use_cuda,
        debug=debug_mode,
    )
    extensions = [torch_hw_emulation_ext, onnx_custom_ops_ext]

    # GPU builds bind ``_C``'s ORT custom ops to the GPU EP, leaving CPU
    # inference sessions unable to bind them, so ship a torch-free CPU-EP
    # companion. CPU wheels skip it: their ``_C`` is already CPU-EP.
    if use_cuda:
        extensions.append(
            make_setuptools_extension(
                name=ONNX_CUSTOM_OPS_CPU_C_MODULE,
                sources=onnx_ort_cpu_sources(),
                include_paths=stable_abi_include_paths(),
                extra_isolated_includes=ort_include_paths(),
                extra_defines=["-DQUARK_PYINIT_MODULE_NAME=_C_cpu"],
                extra_windows_defines=[ORT_WINDOWS_DEFINE],
                use_cuda=False,
                debug=debug_mode,
            )
        )

    # ORT custom-op visibility on Linux: the manylinux gcc-toolset defaults to
    # -fvisibility=hidden, which breaks the ONNX custom ops two ways:
    #   1. RegisterCustomOps / RegisterCustomOpsAltName go hidden, so ORT's
    #      dlsym("RegisterCustomOps") returns null (hipify also elides the
    #      source-level visibility attributes on rocm). --export-dynamic-symbol
    #      re-exports them at link time regardless.
    #   2. ORT dispatches each custom-op kernel to its node via RTTI (typeid)
    #      across the _C.so / libonnxruntime.so boundary; hidden visibility makes
    #      the kernels' typeinfo local, so the compare never matches and every
    #      com.amd.quark op fails NOT_IMPLEMENTED at run time. -fvisibility=default
    #      exports the typeinfo so dispatch resolves.
    # ubuntu-22.04's system gcc defaults to public visibility (flags are a no-op
    # there). The CPU-EP companion ships the same ops and needs the same
    # treatment; Windows MSVC exports via /EXPORT: (cpu-only here), so these
    # Linux-only flags don't apply.
    def _apply_ort_visibility_flags(ext, ext_use_cuda):
        # The rocm build hipifies all sources through hipcc (the "nvcc" arm), so
        # -fvisibility=default must reach that arm too or the kernels' typeinfo
        # stays hidden and only rocm fails. hipcc takes the flag bare; nvcc needs
        # -Xcompiler.
        ext.extra_compile_args["cxx"].append("-fvisibility=default")
        if ext_use_cuda:
            if ROCM_HOME is not None:
                ext.extra_compile_args["nvcc"].append("-fvisibility=default")
            else:
                ext.extra_compile_args["nvcc"].append("-Xcompiler=-fvisibility=default")
        ext.extra_link_args.extend([
            "-Wl,--export-dynamic-symbol=RegisterCustomOps",
            "-Wl,--export-dynamic-symbol=RegisterCustomOpsAltName",
        ])

    if platform.system() != "Windows":
        # The hw-emulation ext carries no ORT custom ops, so skip it; the companion
        # is always built cpu-only (use_cuda=False).
        _apply_ort_visibility_flags(onnx_custom_ops_ext, use_cuda)
        for ext in extensions[2:]:
            _apply_ort_visibility_flags(ext, False)

    return extensions


def get_package_data():
    package_data = {
        "quark": [
            "version.txt",
        ],
        "quark.torch.kernel.hw_emulation": [
            "csrc/**/*",
        ],
        "quark.torch": [
            "include/*.h",
        ],
        "quark.onnx.operators.custom_ops": [
            "src/**/*",
            "include/**/*",
        ],
    }
    return package_data


class BuildCommand(_build_py):
    def run(self):
        super().run()


if HAS_TORCH:

    class VerboseBuildExtension(BuildExtension):
        def build_extension(self, ext):
            header_lines = [
                "=" * 60,
                f"  Building C++ extension: {ext.name}",
                f"  Sources ({len(ext.sources)} files):",
                *(f"    - {src}" for src in ext.sources),
                "=" * 60,
            ]
            logger.info("\n".join(header_lines))
            super().build_extension(ext)
            logger.info(f"Finished building: {ext.name}\n{'=' * 60}")


def build_config_setup():
    cmdclass = {
        "build_py": BuildCommand,
    }
    if HAS_TORCH:
        cmdclass["build_ext"] = VerboseBuildExtension
    return cmdclass


def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _detected_torch_version() -> str:
    """Active-env torch version stripped of any local segment; ``""`` when torch isn't installed (sdist-only path)."""
    if HAS_TORCH:
        return torch.__version__.split("+")[0]
    return ""


def _get_torch_compile_tag(torch_version: str) -> str:
    if not torch_version:
        return ""
    major_minor = ".".join(torch_version.split(".")[:2])
    return f"torch{major_minor}"


def get_accelerator_suffix() -> str:
    """Build the wheel local-version segment, e.g. ``+cu128.torch2.10``.

    Encodes the accelerator family (``cuXY`` / ``rocmXY`` / ``cpu``) and torch
    major.minor, dot-separated per PEP 440. Family comes from
    :func:`_accelerator_family` (auto-detected from torch or forced via
    ``QUARK_ACCELERATOR``); GPU SDK version comes from ``torch.version.{hip,cuda}``
    of the build-env torch -- the same torch ``_C`` is linked against, so the
    wheel tag and torch C++ ABI can't disagree. CPU builds carry an explicit
    ``cpu`` fragment mirroring torch's own ``+cpu`` tag.

    Note: dot-separated is non-negotiable. Issue 4625's prose example uses
    ``+cu1281-torch2.10`` with a dash, but PEP 427 treats ``-`` as a build-tag
    separator requiring a leading digit, so that form fails
    ``packaging.utils.parse_wheel_filename``. The compile-kernels CI jobs that
    build and upload real wheels validate the dot form end-to-end.
    """
    torch_version = _detected_torch_version()
    family = _accelerator_family()

    parts: list[str] = []
    if family == "rocm" and HAS_TORCH and torch.version.hip:
        # torch.version.hip = "7.0.51831-d4f7afc1c" -> major.minor "7.0"
        rocm_ver = ".".join(torch.version.hip.split("-", 1)[0].split(".")[:2])
        parts.append("rocm" + rocm_ver.replace(".", ""))
    elif family == "cuda" and HAS_TORCH and torch.version.cuda:
        # torch.version.cuda is already in "X.Y" form for stable torch builds.
        parts.append("cu" + torch.version.cuda.replace(".", ""))
    elif family == "cpu" and HAS_TORCH:
        # Mirror torch's own ``+cpu`` local-version tag so CPU wheels are
        # explicitly marked rather than relying on the absence of a GPU
        # fragment. Gated on HAS_TORCH: a torch-less sdist build has no
        # compiled ABI to tag, so it stays bare (py3-none-any).
        parts.append("cpu")
    torch_tag = _get_torch_compile_tag(torch_version)
    if torch_tag:
        parts.append(torch_tag)
    if not parts:
        return ""
    return "+" + ".".join(parts)


def get_version(is_nightly=False, is_release=False):
    """Return the version of the Quark package

    Nightly builds will have a `.devYYYYMMDD+<git_hash>` suffix and release builds will not have any suffix.
    Non-nightly and non-release builds will have a `+<git-hash>` suffix for debugging purposes.

    Release/nightly builds also append :func:`get_accelerator_suffix`.

    Args:
        is_nightly (bool, optional): Whether the build is a nightly build. Defaults to False.
        is_release (bool, optional): Whether the build is a release build. Defaults to False.

    Returns:
        str: The version of the Quark package
    """
    assert not (is_nightly and is_release), "Quark build cannot be both nightly and release at the same time!"

    global _version_txt
    accel_suffix = get_accelerator_suffix()

    if is_release:
        return f"{_version_txt}{accel_suffix}"
    if is_nightly:
        dev_suffix = f".dev{datetime.now().strftime('%Y%m%d')}"
        return f"{_version_txt}{dev_suffix}{accel_suffix}"
    git_hash = get_git_hash()
    return f"{_version_txt}+{git_hash}"


cmdclass = build_config_setup()
install_requires = read_requirements()
cwd = os_path_dirname(os_path_abspath(__file__))
sha = get_git_hash()
pkg_version = get_version(is_nightly, is_release)
version_path = os_path_join(cwd, "quark", "version.py")
with open(version_path, "w") as f:
    f.write(f"__version__ = '{pkg_version}'\n")
    f.write(f"git_version = '{sha}'\n")

ext_modules = get_extensions()

summary_lines = [
    "=" * 60,
    "  Quark Setup Summary",
    "=" * 60,
    f"  HAS_TORCH       : {HAS_TORCH}",
    f"  C++ extensions  : {len(ext_modules)}",
]
for ext in ext_modules:
    summary_lines.append(f"    - {ext.name} ({len(ext.sources)} sources)")
if not ext_modules:
    summary_lines.append("  (no C++ extensions will be built)")
summary_lines.append("=" * 60)
logger.info("\n".join(summary_lines))

setup(
    name=package_name,
    version=pkg_version,
    description="The deep learning model compression toolkit.",
    author="Advanced Micro Devices, Inc.",
    author_email="help@amd.com",
    license="MIT",
    # In-package test directories (quark/contrib/*/test) must not ship in the wheel:
    # installed into site-packages they create a second copy of e.g.
    # quark.contrib.llm_eval.test.test_dataset_names, which collides with the
    # source-tree copy at collection time ("import file mismatch"). Contrib tests
    # always run from the source checkout, so excluding them from the wheel is safe.
    packages=find_packages(
        include=["quark", "quark.*"],
        exclude=["quark.contrib.dummy", "*.test", "*.test.*", "*.tests", "*.tests.*"],
    ),
    ext_modules=ext_modules,
    include_package_data=True,
    package_data=get_package_data(),
    cmdclass=cmdclass,
    install_requires=install_requires,
    # Keep in sync with ``pyproject.toml::[project].requires-python``.
    python_requires=">=3.10,<3.14",
)
