# PR Workflow

Guidelines for branches, commits, and pull requests.

## Accountability

- Do not open a PR autonomously. All PRs must be reviewed and submitted by a human.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
  - Clear and to-the-point motivation statement and solution description.
  - Test commands run and results.
  - Clear statement that AI assistance was used.

## Commit Requirements

Use a short commit title only, and attribution using commit trailers such as `Co-authored-by:`. For example:

```text
Your short commit title here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
```

## Branch Naming

- `user/<username>/<description>`: Temporary personal development (auto-deleted after merge)
- `feature/<name>`: Long-term feature development

## PR Labels

Labels are automatically applied by CI based on modified file paths. See `.github/workflows/update_pr_based_on_paths.yml` for configuration.

## PR Requirements

1. One story per PR (single bug fix or feature)
2. Update release notes for features/deprecations
3. Run `./tools/run_linting.sh` before committing
4. Ensure 95%+ code coverage
5. Never merge a PR without human intervention
6. PRs require rebase (use "Update with rebase", NOT "Update with merge commit")

## CI Labels

- `include-quark-kernel-build`: Force kernel build tests
- `include-jupyter-notebook-build`: Build all notebooks regardless of changes which can take almost 24h
- `skip-jupyter-notebook-build`: Skip notebook execution for faster doc builds
- `/rebase`: Comment on PR to auto-rebase

## Release Notes (MANDATORY for features/deprecations)

When adding features or deprecating code, ALWAYS update `docs/source/release_notes.md`.

For new features:

````markdown
- Added [feature name] for [purpose].

  Example usage:

  ```python
  from quark.torch import NewAPI

  result = NewAPI(param="value")
  ```
````

For deprecations:

````markdown
- `OldClass` is deprecated in favor of `NewClass`.

  Before (deprecated):

  ```python
  from quark.torch import OldClass
  obj = OldClass(param="value")
  ```

  After (recommended):

  ```python
  from quark.torch import NewClass
  obj = NewClass(param="value")
  ```
````

After updating release notes, ALWAYS run `./tools/run_linting.sh`.
