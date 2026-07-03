# Testing Patterns

Guidelines for writing and running tests in the Quark codebase.

## Running Tests

### Run all tests for a backend

```bash
# PyTorch backend tests
pytest test/test_for_torch/

# ONNX backend tests
pytest test/test_for_onnx/
```

### Run a specific test file

```bash
pytest test/test_for_torch/test_quantization.py
```

### Run a specific test function

```bash
pytest test/test_for_torch/test_quantization.py::test_specific_function
```

### Run tests matching a pattern

```bash
# Run all tests with "awq" in the name
pytest test/ -k "awq"

# Run tests matching multiple patterns
pytest test/ -k "awq or gptq"

# Exclude tests matching a pattern
pytest test/ -k "not slow"
```

### Filter by pytest markers

```bash
# Run tests requiring accelerate library
pytest test/ -m "accelerate_test"

# Run tests requiring dual GPU
pytest test/ -m "require_dual_gpu"

# Run tensor parallel tests
pytest test/ -m "tensor_parallel"

# Exclude tests with a marker
pytest test/ -m "not require_dual_gpu"
```

Available markers are defined in `pyproject.toml` under `[tool.pytest.ini_options]`.

## Test Utilities

Use helpers from `quark/common/utils/testing_utils.py`:

- `torch_device` - Use instead of hardcoding "cuda" or "cpu"
- `require_torch_cuda` - Decorator to skip tests on CPU
- `retry_flaky_test` - Decorator for non-deterministic tests

## Temporary Files

Always use `tempfile.TemporaryDirectory()` to avoid leaving artifacts.

## Code Coverage

- Minimum **95% code coverage** required (strictly enforced in CI)
- Add tests for all new features and bug fixes

## Examples

See existing tests in `test/test_for_torch/` and `test/test_for_onnx/` for patterns.
