## Contributing to Quark tests

### Avoid hard-coding PyTorch devices

Instead, we should make use of the `torch_device` variable defined at [quark/shares/utils/testing_utils.py](/quark/shares/utils/testing_utils.py#L16-L41). This one is by default initialized to "cpu" device, and in case a GPU is available, is set to "cuda" device. This is useful as tests as this one can be run independently of the device, and may support other devices in the future as well.

Tests that use `torch_device` but can only be run on GPU can be skipped on CPU with the `@require_torch_cuda` decorator as shown at [example](/test/test_for_torch/test_lsq_FakeQuantize.py#L34).

### Do not leave artifacts behind

For tests that require to serialize on disk some data, please use `tempfile.TemporaryFile()` or `tempfile.TemporaryDirectory()` to avoid having leftover files after running the tests locally or in the hosted CI.

Example:

```python
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    my_function_that_serializes_data(output_dir=tmpdir)
```

### Testing with external libraries

Some tests may require external libraries that are not in Quark `requirements.txt`. It is best to add a marker for these specific libraries in `pyproject.toml`, and mark such tests as `@pytest.mark.mylibrary_test`.

This allows to test Quark in the CI in a basic environment without many external libraries installed, and to run specific tests that are specific to an external library using `pytest test/ -s -vvvvvv -m "mylibrary_test"`

### Flaky tests

Tests that are flaky (sometime fail, but most of the time pass), for example due to non-deterministic GPU execution, can be decorated with:

```python
from quark.shares.utils.testing_utils import retry_flaky_test

@retry_flaky_test()
def test_that_is_flaky():
    ...
```

<!--
## License
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
-->
