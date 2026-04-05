# Contributing

This repository is part of the [robot-code workspace](https://github.com/personalrobotics/robot-code).
All projects in the workspace share a common layout (`src/` + `tests/`), a single
package manager (`uv`), and a common CI setup.

## Development setup

```bash
git clone https://github.com/personalrobotics/robot-code
cd robot-code
./setup.sh
```

This clones every sibling repo (including this one) into a single uv workspace
and runs `uv sync`. You can then work in any sibling's directory.

## Running tests and linters

From inside this repo:

```bash
uv run pytest tests/ -v
uv run ruff check .
uv run ruff format --check .
```

From the workspace root, you can also run the cross-repo integration suite:

```bash
uv run pytest tests/integration -v
```

## Pull requests

- Branch from `main`. Open a PR with a clear summary and test plan.
- Per-repo CI (ruff + pytest) must pass.
- Cross-repo integration CI in robot-code runs automatically when this repo's
  `main` advances, and on scheduled nightly runs.
- Review is by [@siddhss5](https://github.com/siddhss5) (enforced via CODEOWNERS).

## Package manager: uv only

We use [uv](https://docs.astral.sh/uv/) exclusively. **Do not use pip.**
The workspace layout in robot-code relies on uv's workspace resolution.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License (see [LICENSE](LICENSE)).
