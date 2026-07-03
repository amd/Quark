# Coding Conventions

Code style and documentation standards for the Quark codebase.

## Version Requirements

See [docs/source/install.rst](../source/install.rst) for current version requirements.

## Code Style Tools

Linting is configured in `pyproject.toml`. Run `./tools/run_linting.sh` to execute all checks.

## General Guidelines

- Always use absolute paths, not relative paths
- Avoid over-engineering - only make changes that are directly requested or clearly necessary
- Never commit secrets (.env, credentials.json, etc.)

## Docstring Format

Use Sphinx-style docstrings for all public APIs. See existing docstrings in `quark/torch/quantization/api.py` for examples.

Key points:

- Use `:param`, `:return:`, `:rtype:`, `:raises:` directives
- Include usage examples with `.. code-block:: python`
- Place class docstrings under the class definition, NOT under `__init__` method

## Tutorial Notebooks

- Stored in `tutorials/` (copied to `docs/source/tutorials/` during build)
- Each tutorial in its own folder with `requirements.txt` if needed
- Clear all cell outputs before committing (pre-commit hook enforces)
- Tag cells with `skip-execution` if they shouldn't run in CI (use sparingly)

## Spell Check

Update `.wordlist.txt` for new technical terms that fail the spell-check.
