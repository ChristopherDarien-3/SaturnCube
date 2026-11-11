# Releasing SaturnCube

One-time setup
1. Put your `LICENSE` file in the repository root, then in `pyproject.toml` uncomment `license` / `license-files`
   (use your SPDX id, e.g. `"MIT"`), and fill in `authors` and `[project.urls]`.
2. Create the project on PyPI and TestPyPI (https://pypi.org, https://test.pypi.org) and add a *trusted publisher*
   for this GitHub repository (workflow `publish.yml`, environment `pypi`). No API token is needed afterwards.
3. In the GitHub repository settings, create an environment named `pypi`.

Every release
1. Bump `__version__` in `saturncube/__init__.py` and add an entry to `CHANGELOG.md`.
2. Run the tests: `pip install -e ".[dev]" && pytest` (the slow tests take about ten minutes: `pytest -m slow`).
3. Build and check locally:
   ```bash
   rm -rf dist && python -m build && twine check dist/*
   ```
4. Try it on TestPyPI first:
   ```bash
   twine upload --repository testpypi dist/*
   python -m venv /tmp/t && /tmp/t/bin/pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ saturncube
   /tmp/t/bin/python -c "import saturncube as sc; print(sc.__version__)"
   ```
5. Tag and publish a GitHub release (`git tag v0.6.0 && git push --tags`, then "Draft a new release").
   The `publish.yml` workflow uploads to PyPI.

Notes
- A version number can never be re-uploaded to PyPI, so always bump it for any fix.
- PyTorch is a dependency; users who want a CPU-only or a specific CUDA build should install `torch` first.
