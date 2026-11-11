"""Builds the README's list of registered entries from the registry, so the two cannot drift apart.

Run ``python scripts/update_readme.py`` after adding or changing a registered class; the test suite fails if the
README block no longer matches (``python scripts/update_readme.py --check`` does the same from the command line).
"""
from __future__ import annotations

import re
from collections import Counter

from .base import REGISTRY, BaseModel, available

START = "<!-- registry-table:start -->"
END = "<!-- registry-table:end -->"

# Input conventions, only for families where one sentence is accurate for every entry.
FAMILY_NOTES = {
    "vector": "Input: 2-D array `(n_samples, n_features)`.",
    "classical": "Pure NumPy; input: 2-D array `(n_samples, n_features)`.",
    "sequence": "Input: `(n, T, d)` floats, or integer tokens `(n, T)` with `vocab_size=`.",
    "vision": "Input: images as `(n, C, H, W)` or `(n, H, W)`.",
    "graph": "Input: `Graph` / `Tree` containers (dense adjacency, so small and medium graphs).",
    "metric": "Few-shot: `predict(X, support=(X_support, y_support))` works on unseen classes.",
}

FAMILY_ORDER = ["vector", "vision", "bio", "sequence", "memory", "metric", "graph", "unsupervised", "classical",
                "agent", "bandit", "search", "symbolic", "cognition", "verification", "codegen", "experiment"]


def kind_of(cls):
    """model: shared fit/predict/evaluate/save interface; classical: NumPy learner with fit/predict;
    agent: learns by interacting with an environment or bandit; tool: a plain Python helper class."""
    if issubclass(cls, BaseModel):
        return "model"
    if cls.family == "classical":
        return "classical"
    if cls.family in ("agent", "bandit"):
        return "agent"
    return "tool"


def summary_of(cls):
    """First sentence of the class's OWN docstring (never an inherited one)."""
    doc = (cls.__dict__.get("__doc__") or "").strip()
    if not doc:
        return ""
    para = " ".join(doc.split("\n\n")[0].split())
    m = re.search(r"(?<=[a-z0-9\)`])\. (?=[A-Z])", para)
    if m and m.start() >= 25:
        para = para[:m.start()]
    return para.rstrip(".").replace("|", "\\|")


def registry_markdown():
    names = available()
    kinds = Counter(kind_of(REGISTRY[n]) for n in names)
    out = ["%d registered entries (`sc.available()`): %d models, %d classical learners, %d agents and bandits, "
           "%d tools." % (len(names), kinds["model"], kinds["classical"], kinds["agent"], kinds["tool"]),
           "",
           "`model` entries share one `fit` / `predict` / `evaluate` / `save` / `load` interface and expose "
           "`.module`. `classical` entries have `fit` / `predict` only. `agent` entries learn from an "
           "environment or bandit instead of a dataset. `tool` entries are plain Python classes with their "
           "own methods. Build any of them with `sc.create(name, **kwargs)`.", ""]
    families = [f for f in FAMILY_ORDER if any(REGISTRY[n].family == f for n in names)]
    families += sorted({REGISTRY[n].family for n in names} - set(families))  # a new family still shows up
    for fam in families:
        members = [n for n in names if REGISTRY[n].family == fam]
        out.append("### %s (%d)" % (fam, len(members)))
        out.append("")
        if fam in FAMILY_NOTES:
            out += [FAMILY_NOTES[fam], ""]
        out += ["| Registry name | Class | Kind | Description |", "|---|---|---|---|"]
        for n in members:
            cls = REGISTRY[n]
            out.append("| `%s` | `%s` | %s | %s |" % (n, cls.__name__, kind_of(cls), summary_of(cls)))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_block():
    return START + "\n" + registry_markdown() + END


def splice(readme_text):
    """README text with the generated block replaced (or raises if the markers are missing)."""
    i, j = readme_text.find(START), readme_text.find(END)
    if i < 0 or j < i:
        raise ValueError("README is missing the %s ... %s markers" % (START, END))
    return readme_text[:i] + render_block() + readme_text[j + len(END):]
