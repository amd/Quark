# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import subprocess

mypy_command = ["mypy", "examples", "src", "tests", "tools"]

try:
    result = subprocess.run(mypy_command, capture_output=True, text=True, check=True)
    print("MyPy type check passed successfully!")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print("MyPy type check failed with errors:")
    print(e.stdout)
    print(e.stderr)
except FileNotFoundError:
    print("Error: MyPy command not found. Make sure MyPy is installed in your env: pip install mypy")
