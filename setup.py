#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import subprocess
import sys
import zipfile
import tempfile
from datetime import datetime
from setuptools import find_packages, setup
from distutils import core
from distutils.core import Distribution
from distutils.errors import DistutilsArgError
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

"""
  To make a wheel package:
    $ python setup.py sdist bdist_wheel -d $YOUR_TARGET
  To make a wheel package with specific python version and specific platform:
    $ python setup.py sdist bdist_wheel -d $YOUR_TARGET --python-tag py310 --plat-name=linux_x86_64

  By default, the generated version is X.Y.Z+git_commit_hash.
  For a nightly build, set the environment var QUARK_NIGHTLY=1 before building the wheel package.
    The resulting version will be X.Y.Z.devYYYYMMDD.
  For a release build, set the environment var QUARK_RELEASE=1 before building the wheel package.
    The resulting version will be X.Y.Z.

  The version X.Y.Z is read from version.txt which lives in the repository.
"""

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


class CustomBdistWheel(_bdist_wheel):
    def run(self):
        super().run()
        dist_dir = os.path.abspath(self.dist_dir)
        wheel_files = [f for f in os.listdir(dist_dir) if f.endswith('.whl')]
        wheel_file = os.path.join(dist_dir, wheel_files[0])

        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(wheel_file, 'r') as zf:
                zf.extractall(temp_dir)

            dist_info = os.path.join(temp_dir, f"{self.wheel_dist_name}.dist-info")
            record_path = os.path.join(dist_info, "RECORD")

            with open(record_path, 'a') as record_file:
                library_dir = "quark/onnx/operators/custom_ops/lib"
                library_files = [
                    "libcustom_ops.so",
                    "libcustom_ops_gpu.so",
                    "custom_ops.dll",
                    "custom_ops.pyd"
                ]
                for library_file in library_files:
                    library_path = os.path.join(library_dir, library_file)
                    record_file.write(f"{library_path},,\n")

            with zipfile.ZipFile(wheel_file, 'a') as zf:
                arcname = os.path.join(f"{self.wheel_dist_name}.dist-info", "RECORD")
                zf.write(record_path, arcname)


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


def build_config_setup():
    cmdclass={"bdist_wheel": CustomBdistWheel}
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
          packages=find_packages(include=['quark', 'quark.*']), # Only include folder 'quark'
          include_package_data=True,
          cmdclass=cmdclass,
          install_requires=install_requires,
          python_requires='>=3.10.0,<3.13',
          )
