import pathlib

import pytest

import saturncube as sc
from saturncube import _docs
from saturncube.base import REGISTRY

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"


def test_readme_registry_block_matches_the_registry():
    if not README.exists():
        pytest.skip("README.md is not shipped with this checkout")
    text = README.read_text(encoding="utf-8")
    assert _docs.START in text and _docs.END in text, "README lost its registry-table markers"
    assert _docs.splice(text) == text, "README is out of date: run `python scripts/update_readme.py`"


def test_every_registered_class_has_its_own_docstring():
    # an inherited docstring would silently describe the wrong thing in the generated table
    bare = [n for n, c in REGISTRY.items() if not (c.__dict__.get("__doc__") or "").strip()]
    assert not bare, bare


def test_unit_tests_runner_is_hidden_but_still_creatable():
    assert "unit_tests" not in sc.available()
    assert "unit_tests" not in sc.available("verification")
    assert "unit_tests" in sc.available(include_hidden=True)
    assert sc.create("unit_tests", function=lambda x: x, cases=[((1,), 1)]) is not None


def test_generated_table_lists_every_visible_entry_once():
    md = _docs.registry_markdown()
    rows = [line for line in md.splitlines() if line.startswith("| `")]
    names = [line.split("`")[1] for line in rows]
    assert sorted(names) == sc.available() and len(set(names)) == len(names)
    assert f"{len(names)} registered entries" in md


def test_kinds_are_assigned_and_descriptions_are_not_empty():
    for n in sc.available():
        cls = REGISTRY[n]
        assert _docs.kind_of(cls) in {"model", "classical", "agent", "tool"}
        assert _docs.summary_of(cls), n
