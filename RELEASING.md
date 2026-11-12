# Releasing SaturnCube

Publishing uses PyPI "trusted publishing": GitHub Actions proves its identity to the index, so no API token is stored
anywhere. There are two workflows:

| Where | Workflow file | GitHub environment | Starts when |
|---|---|---|---|
| TestPyPI (dry run) | `publish-testpypi.yml` | `testpypi` | you run it by hand (Actions tab) |
| PyPI (real) | `publish.yml` | `pypi` | you publish a GitHub release |

## One-time setup
1. `LICENSE` (MIT), `license`, `authors` and `[project.urls]` are already set in `pyproject.toml`.
2. In the GitHub repository settings, create two environments: `testpypi` and `pypi`. On `pypi`, add yourself as a
   required reviewer so a release cannot publish without a click.
3. TestPyPI is a separate site with its own account: register at https://test.pypi.org, then Account settings ->
   Publishing -> "Add a new pending publisher" with these values (all are case-sensitive and must match the workflow run):
   - PyPI Project Name: `saturncube`
   - Owner: `ChristopherDarien-3`, Repository name: `SaturnCube`
   - Workflow name: `publish-testpypi.yml`, Environment name: `testpypi`
4. Later, the same on https://pypi.org with Workflow name `publish.yml` and Environment name `pypi`.
   A pending publisher does not reserve the project name; until the first successful publish, anyone can take it.

## Every release
1. Bump `__version__` in `saturncube/__init__.py` and update `CHANGELOG.md`.
2. Run the tests: `pip install -e ".[dev]" && pytest` (the slow tests take about ten minutes: `pytest -m slow`).
3. Build and check locally:
   ```bash
   rm -rf dist && python -m build && twine check dist/*
   ```
4. Dry run: GitHub -> Actions -> "Publish to TestPyPI" -> Run workflow. Then, in a fresh environment:
   ```bash
   python -m venv /tmp/t && /tmp/t/bin/pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ saturncube
   /tmp/t/bin/python -c "import saturncube as sc; print(sc.__version__, len(sc.available()))"
   ```
   The extra index is needed because `torch` and `numpy` are not on TestPyPI.
5. Real release: `git tag vX.Y.Z && git push --tags`, then "Draft a new release" on GitHub and publish it.
   `publish.yml` uploads to PyPI.

## Notes
- A version number can never be re-uploaded to an index, so bump it for any fix. TestPyPI and PyPI are independent, so the
  same version can be on both.
- `invalid-publisher` in the publish step means the settings registered on the index do not match this workflow run
  exactly (owner, repository, workflow file name, or environment name), or the publisher was added on the wrong site.
- PyTorch is a dependency; users who want a CPU-only or a specific CUDA build should install `torch` first.
