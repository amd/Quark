#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py
from distutils import core
from distutils.core import Distribution
from distutils.errors import DistutilsArgError

"""
  To make a wheel package:
    $ python -m build . --sdist --wheel --outdir $YOUR_TARGET
  To make a wheel package with specific python version and specific platform:
    $ python -m build . --sdist --wheel --outdir $YOUR_TARGET -C"--python-tag=py39" -C"--plat-name=linux_x86_64"

  By default, the generated version is X.Y.Z+git_commit_hash.
  For a nightly build, set the environment var QUARK_NIGHTLY=1 before building the wheel package.
    The resulting version will be X.Y.Z.devYYYYMMDD.
  For a release build, set the environment var QUARK_RELEASE=1 before building the wheel package.
    The resulting version will be X.Y.Z.

  The version X.Y.Z is read from version.txt which lives in the repository.
"""

def update_pyproject_toml_project_name(new_name: str) -> None:
    if not new_name.replace("-", "_").isidentifier():
        raise ValueError(f"Invalid python package name: {new_name}")

    with open('pyproject.toml', 'r') as f:
        lines = f.readlines()
    with open('pyproject.toml', 'w') as f:
        for line in lines:
            if line.replace(" ", "").startswith('name='):
                line = f'name = "{new_name}"\n'
            f.write(line)

def string_to_bool(s):
    s = s.lower()
    if s in ('true', '1', 'yes'):
        return True
    elif s in ('false', '0', 'no'):
        return False
    else:
        raise ValueError("Invalid boolean string: {}".format(s))

package_name = os.getenv("QUARK_WHEEL_NAME", "amd-quark")
_version_txt = open("quark/version.txt", "r").read().strip()
is_nightly = string_to_bool(os.getenv('QUARK_NIGHTLY', 'false')) is True
is_release = string_to_bool(os.getenv('QUARK_RELEASE', 'false')) is True

if is_nightly:
    # Naming nightly packagse differently to allow easy pip install without conflicting with the release versions
    # E.g., pip install amd-quark-nightly --trusted-host artifactory.domain.com -i "https://artifactory.domain.com/artifactory/api/pypi/repository_name/simple"
    package_name += '-nightly'

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
        if (not os.path.exists(output) or os.path.getmtime(proto_file) > os.path.getmtime(output)):
            protoc_command = [ "protoc", f"-I{source_dir}", f"--python_out={dest_dir}" ]
            if mypy:
                protoc_command.append(f"--mypy_out={dest_dir}")
            protoc_command.append(str(proto_file))
            retval = subprocess.run(protoc_command, capture_output=True)
            if retval.returncode != 0:
                raise RuntimeError(f"protoc failed for {proto_file}: {retval.stderr.decode('utf-8')}")

class BuildCommand(_build_py):
    def run(self):
        # Generate protobuf files before building
        source_dir = Path("quark/contrib/onnx_utils/src/onnx_custom_ops")
        dest_dir = Path("quark/contrib/onnx_utils/src/ryzenai_onnx_utils/proto")
        generate_proto(source_dir, dest_dir, mypy=True)

        # Continue with the standard build process
        super().run()


def build_config_setup():
    cmdclass={
        "build_py": BuildCommand,
    }
    return cmdclass


def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_version(is_nightly=False, is_release=False):
    """Return the version of the Quark package

    Nightly builds will have a `.devYYYYMMDD+<git_hash>` suffix and release builds will not have any suffix.
    Non-nightly and non-release builds will have a `+<git-hash>` suffix for debugging purposes.

    Args:
        is_nightly (bool, optional): Whether the build is a nightly build. Defaults to False.
        is_release (bool, optional): Whether the build is a release build. Defaults to False.

    Returns:
        str: The version of the Quark package
    """
    assert not (is_nightly and is_release), "Quark build cannot be both nightly and release at the same time!"

    global _version_txt
    if is_release:
        return f"{_version_txt}"
    if is_nightly:
        dev_suffix = f".dev{datetime.now().strftime('%Y%m%d')}"
        return f"{_version_txt}{dev_suffix}"
    git_hash = get_git_hash()
    return f"{_version_txt}+{git_hash}"


if __name__ == '__main__':
    dist = Distribution()
    dist.script_name = sys.argv[0]
    dist.script_args = sys.argv[1:]
    try:
       is_valid_args = dist.parse_command_line()
    except DistutilsArgError as msg:
       raise SystemExit(f"{core.gen_usage(dist.script_name)}\nerror:{msg}")

    if not is_valid_args:
       sys.exit()

    cmdclass = build_config_setup()
    install_requires = read_requirements()
    cwd = os_path_dirname(os_path_abspath(__file__))
    sha = get_git_hash()
    version_path = os_path_join(cwd, "quark", "version.py")
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{get_version(is_nightly, is_release)}'\n")
        f.write(f"git_version = '{sha}'\n")

    setup(name=package_name,
          version=get_version(is_nightly, is_release),
          description="The deep learning model compression toolkit.",
          author="Advanced Micro Devices, Inc.",
          author_email='help@amd.com',
          license="MIT",
          packages=find_packages(include=['quark', 'quark.*'],
                                 exclude=['quark.contrib.dummy']), # Only include folder 'quark'
          include_package_data=True,
          cmdclass=cmdclass,
          install_requires=install_requires,
          python_requires='>=3.9.0,<3.13',
          )
