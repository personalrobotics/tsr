# Releasing sstsr

The published name is `sstsr`; the import name stays `tsr`.

## The version comes from the tag

`pyproject.toml` declares `dynamic = ["version"]` and hatch-vcs derives the version
from the git tag, writing `src/tsr/_version.py` at build time (git-ignored, shipped in
both artifacts so a wheel can be built from an unpacked sdist). `tsr.__version__` reads
that file, falling back to installed package metadata.

This is what makes the dry run real: **an rc tag necessarily builds an rc artifact**,
so TestPyPI only ever receives pre-releases. Before this (through 2.2.0), the version
was static in `pyproject.toml`, so tagging `v2.1.0rc1` published `2.1.0` to TestPyPI —
the rc was never exercised, and the final version number was consumed there.

Two consequences worth knowing:

- **A dirty tree or a commit after the tag builds a dev version** (`2.3.0.dev4`), not
  the tag's. That is deliberate: only a clean checkout of the tag can publish the
  release. Local segments (`+g<sha>`) are disabled, so a dry-run build is still
  uploadable to TestPyPI.
- **Two guards run before any publish job**: the built artifact's version must equal
  the tag, and an rc tag must build a pre-release (and a final tag must not).
- **In CI the tag is pinned as the version** (`SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SSTSR`).
  `git describe` has to choose when several tags point at one commit, which is the
  normal flow — `vX.Y.Zrc1` and `vX.Y.Z` usually sit on the same commit — and it is not
  guaranteed to choose the one being built. It resolved to the rc in CI while resolving
  to the final tag locally, so the version is taken from the ref instead of inferred.
  The guard above then still has something to check: that the built artifact really
  carries it.

## Steps

1. Land everything for the release. Move the changelog's `[Unreleased]` heading to
   `## [X.Y.Z] — YYYY-MM`, summarising highlights and anything now withheld.
   Do **not** edit a version number anywhere; there is none to edit.
2. Check the gate locally:
   ```bash
   uv run pytest tests/ -q
   uv run ruff check . && uv run ruff format --check .
   TSR_MATRIX_SCALE=10 uv run pytest tests/tsr/hands/test_grasp_matrix.py \
       tests/tsr/hands/test_grasp_mutation_gate.py
   ```
3. Tag the release candidate and push it — this publishes to **TestPyPI**:
   ```bash
   git tag -a vX.Y.Zrc1 -m "sstsr X.Y.Zrc1" && git push origin vX.Y.Zrc1
   ```
4. Verify from TestPyPI in a clean venv (its own index first, PyPI only for
   dependencies):
   ```bash
   uv venv /tmp/rc && VIRTUAL_ENV=/tmp/rc uv pip install \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     --index-strategy unsafe-best-match "sstsr==X.Y.Zrc1"
   /tmp/rc/bin/python -c "import tsr; print(tsr.__version__)"
   ```
   Exercise what the release actually changed, not just the import.
5. Tag the final release — this publishes to **PyPI** and creates the GitHub Release:
   ```bash
   git tag -a vX.Y.Z -m "sstsr X.Y.Z" && git push origin vX.Y.Z
   ```

Publishing uses Trusted Publishing (OIDC): there are no tokens, and the publish jobs
run in the `testpypi-release` / `release` GitHub environments.

## If an rc needs a fix

Land the fix and tag `rc2`. Never move a published tag: the artifact for a given
version can never be replaced on PyPI or TestPyPI.
