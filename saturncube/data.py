"""Automatic preparation of messy datasets for SaturnCube models.

    data = sc.load_dataset("data.csv", target="Purchased")      # CSV path, DataFrame, dict, array, list of dicts
    print(data.report())                                        # what was detected and what will be done
    prepared = data.prepare(test_size=0.2, seed=0)              # clean, encode, split, scale (fit on train only)
    model = sc.MLP().fit(prepared.X_train, prepared.y_train)
    prepared.decode_target(model.predict(prepared.X_test))      # back to the original labels

Other model types: data.prepare_sequences(...) (windows for sequence models) and data.prepare_images(...)
(flat pixel columns -> images for the CNN). sc.auto_model(...) does the whole pipeline in one call.

Only NumPy and the standard library are needed; pandas DataFrames are accepted when pandas is installed.
Nothing is transformed silently: report() lists every decision and each one can be overridden."""
from __future__ import annotations

import csv
import datetime as _dt
import inspect
import json
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import REGISTRY

# ----------------------------------------------------------------------------- parsing ---
_MISSING = {"", "na", "n/a", "#n/a", "n.a.", "nan", "null", "none", "nil", "nat", "?", "-", "--", "missing", "unknown"}
_BOOL = {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0, "y": 1.0, "n": 0.0, "t": 1.0, "f": 0.0}
_CURRENCY = "$€£¥"
_NUM = re.compile(rf"^[\s{_CURRENCY}]*([-+]?)[\s{_CURRENCY}]*(\(?)([0-9][0-9,]*\.?[0-9]*|\.[0-9]+)(\)?)\s*(%?)\s*$")
_DATE = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?)?\s*(?:Z|[+-]\d{2}:?\d{2})?$")
_ID_NAME = re.compile(r"(^|[_\s.-])(id|idx|index|uuid|guid|key)$|^unnamed", re.I)


def _is_missing(v):
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().lower() in _MISSING
    try:
        return bool(v != v)                  # NaN / NaT
    except Exception:
        return False


def _to_number(v):
    """Float for numbers and number-like text ('$1,200.50', '12%', '(5)', '1e3'); None otherwise."""
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        return x if math.isfinite(x) else None
    if not isinstance(v, str):
        return None
    s = v.strip()
    try:
        x = float(s)
        return x if math.isfinite(x) else None
    except ValueError:
        pass
    m = _NUM.match(s)
    if not m:
        return None
    sign, lp, digits, rp, _pct = m.groups()
    if bool(lp) != bool(rp) or ("," in digits and not re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", digits)):
        return None
    x = float(digits.replace(",", ""))
    return -x if (sign == "-" or lp) else x


def _to_bool(v):
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    if isinstance(v, str):
        return _BOOL.get(v.strip().lower())
    return None


def _to_datetime(v):
    if isinstance(v, _dt.datetime):
        return v
    if hasattr(v, "to_pydatetime"):
        return v.to_pydatetime()
    if isinstance(v, _dt.date):
        return _dt.datetime(v.year, v.month, v.day)
    if isinstance(v, str):
        m = _DATE.match(v.strip())
        if m:
            y, mo, d, h, mi, sec = (int(g) if g else 0 for g in m.groups())
            try:
                return _dt.datetime(y, mo, d, h, mi, sec)
            except ValueError:
                return None
    return None


def _norm_cat(v):
    """Category text with case and whitespace differences removed: ' Male', 'MALE ' -> 'male'."""
    return re.sub(r"\s+", " ", str(v).strip()).lower()


_EPOCH = _dt.datetime(1970, 1, 1)


def _datetime_features(values, with_hour):
    n = len(values)
    out = {k: np.full(n, np.nan) for k in ("year", "month", "day", "dayofweek", "timestamp")}
    if with_hour:
        out["hour"] = np.full(n, np.nan)
    for i, v in enumerate(values):
        d = None if _is_missing(v) else _to_datetime(v)
        if d is not None:
            out["year"][i], out["month"][i], out["day"][i], out["dayofweek"][i] = d.year, d.month, d.day, d.weekday()
            out["timestamp"][i] = (d - _EPOCH).total_seconds() / 86400.0
            if with_hour:
                out["hour"][i] = d.hour + d.minute / 60.0
    return out


# --------------------------------------------------------------------------- profiling ---
@dataclass
class ColumnProfile:
    name: str
    kind: str                       # numeric | boolean | datetime | categorical | id | text | constant | empty
    n_rows: int
    n_missing: int
    n_unique: int
    notes: list = field(default_factory=list)
    example: str = ""

    @property
    def missing_frac(self):
        return self.n_missing / max(self.n_rows, 1)


def _profile_column(name, raw, force=None):
    n = len(raw)
    nonmiss = [v for v in raw if not _is_missing(v)]
    notes = []
    prof = lambda kind, uniq: ColumnProfile(name, kind, n, n - len(nonmiss), uniq, notes, str(nonmiss[0])[:20] if nonmiss else "")
    if not nonmiss:
        return prof("empty", 0)
    kind = force
    if kind is None:
        if all(isinstance(v, (bool, np.bool_)) for v in nonmiss):
            kind = "boolean"
        else:
            nums = [_to_number(v) for v in nonmiss]
            ok = sum(x is not None for x in nums)
            if ok == len(nonmiss) or ok / len(nonmiss) >= 0.95:
                kind = "numeric"
                if ok < len(nonmiss):
                    bad = next(v for v, x in zip(nonmiss, nums) if x is None)
                    notes.append(f"{len(nonmiss) - ok} non-numeric entries (e.g. {str(bad)[:15]!r}) treated as missing")
            elif all(isinstance(v, str) and v.strip().lower() in _BOOL for v in nonmiss):
                kind = "boolean"
            elif all(_to_datetime(v) is not None for v in nonmiss):
                kind = "datetime"
            else:
                kind = "categorical"
    if kind == "numeric":
        vals = [x for x in (_to_number(v) for v in nonmiss) if x is not None]
        uniq = len(set(vals))
        if uniq <= 1:
            return prof("constant", uniq)
        integer = all(float(x).is_integer() for x in vals)
        if n >= 20 and integer and uniq == len(vals) and (_ID_NAME.search(name) or sorted(vals)[1:] == [x + 1 for x in sorted(vals)[:-1]]):
            return prof("id", uniq)
        return prof("numeric", uniq)
    if kind == "boolean":
        uniq = len({_to_bool(v) for v in nonmiss if _to_bool(v) is not None})
        return prof("constant" if uniq <= 1 else "boolean", uniq)
    if kind == "datetime":
        uniq = len({str(v) for v in nonmiss})
        return prof("constant" if uniq <= 1 else "datetime", uniq)
    cats = [_norm_cat(v) for v in nonmiss]
    uniq, avg_len = len(set(cats)), float(np.mean([len(str(v)) for v in nonmiss]))
    raw_forms = len({str(v).strip() for v in nonmiss})
    if raw_forms > uniq:
        notes.append(f"{raw_forms - uniq} spelling variants merged (case / whitespace)")
    if uniq <= 1:
        return prof("constant", uniq)
    if force is None:
        if uniq == len(nonmiss) and n >= 20:
            return prof("text" if avg_len > 30 else "id", uniq)
        if avg_len > 40 and uniq / len(nonmiss) > 0.5:
            return prof("text", uniq)
    return prof("categorical", uniq)


# ---------------------------------------------------------------------- coercing sources ---
def _coerce_columns(source, header=True, delimiter=None, names=None, encoding="utf-8-sig"):
    """Return (columns: dict name -> list, notes: list[str], origin: str)."""
    notes = []
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if not os.path.exists(path):
            raise FileNotFoundError(f"no such file: {path}")
        if path.lower().endswith((".parquet", ".xlsx", ".xls", ".feather")):
            raise ValueError("this file type is not read directly; load it with pandas and pass the DataFrame to load_dataset")
        with open(path, newline="", encoding=encoding, errors="replace") as f:
            text = f.read()
        if delimiter is None:
            try:
                delimiter = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
            except csv.Error:
                delimiter = ","
        rows = [r for r in csv.reader(text.splitlines(), delimiter=delimiter) if any(c.strip() for c in r)]
        if not rows:
            raise ValueError(f"{path} is empty")
        head = [h.strip() for h in rows[0]] if header else [f"col{i}" for i in range(len(rows[0]))]
        body = rows[1:] if header else rows
        width, ragged = len(head), 0
        cols = {h: [] for h in _unique_names(head)}
        keys = list(cols)
        for r in body:
            if len(r) != width:
                ragged += 1
                r = (r + [""] * width)[:width]
            for k, v in zip(keys, r):
                cols[k].append(v.strip())
        if ragged:
            notes.append(f"{ragged} rows had the wrong number of fields (padded or truncated)")
        return cols, notes, path
    if hasattr(source, "columns") and hasattr(source, "to_dict"):          # pandas DataFrame
        return {str(c): source[c].tolist() for c in source.columns}, notes, "DataFrame"
    if isinstance(source, dict):
        return {str(k): list(v) for k, v in source.items()}, notes, "dict"
    if isinstance(source, (list, tuple)) and source and isinstance(source[0], dict):
        keys = list(dict.fromkeys(k for r in source for k in r))
        return {str(k): [r.get(k) for r in source] for k in keys}, notes, "records"
    arr = np.asarray(source, dtype=object) if not isinstance(source, np.ndarray) else source
    if arr.ndim == 2:
        names = list(names) if names is not None else [f"x{i}" for i in range(arr.shape[1])]
        if len(names) != arr.shape[1]:
            raise ValueError("names must have one entry per column")
        return {str(n): arr[:, i].tolist() for i, n in enumerate(names)}, notes, "array"
    raise ValueError("could not interpret the data source; use a CSV path, DataFrame, dict of columns, list of dicts or 2-D array")


def _unique_names(names):
    seen, out = {}, []
    for i, n in enumerate(names):
        n = n or f"col{i}"
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


# --------------------------------------------------------------------------- splitting ---
def split_indices(n, test_size=0.2, val_size=0.0, seed=0, y=None, stratify=True, chronological=False):
    """Row indices for (train, val, test). Stratified by class when y is given; chronological keeps order."""
    if not (0 <= test_size < 1 and 0 <= val_size < 1 and test_size + val_size < 1):
        raise ValueError("test_size and val_size must be in [0, 1) and sum to less than 1")
    n_test = max(1, round(n * test_size)) if test_size > 0 else 0
    n_val = max(1, round(n * val_size)) if val_size > 0 else 0
    if n - n_test - n_val < 1:
        raise ValueError(f"{n} rows are too few for test_size={test_size}, val_size={val_size}")
    idx = np.arange(n)
    if chronological:
        return idx[: n - n_test - n_val], idx[n - n_test - n_val: n - n_test], idx[n - n_test:]
    rng = np.random.default_rng(seed)
    if y is not None and stratify:
        tr, va, te = [], [], []
        for c in np.unique(y):
            members = rng.permutation(np.flatnonzero(y == c))
            cnt = len(members)
            t, v = int(round(cnt * test_size)), int(round(cnt * val_size))
            while cnt - t - v < 1 and (t or v):
                t, v = (t - 1, v) if t >= v else (t, v - 1)
            te += members[:t].tolist(); va += members[t:t + v].tolist(); tr += members[t + v:].tolist()
        return rng.permutation(tr), rng.permutation(va).astype(int), rng.permutation(te).astype(int)
    perm = rng.permutation(n)
    return perm[: n - n_test - n_val], perm[n - n_test - n_val: n - n_test], perm[n - n_test:]


# ------------------------------------------------------------------------ preprocessing ---
class TabularPreprocessor:
    """Fitted cleaning / encoding / scaling for the feature columns. fit() learns every statistic (medians,
    categories, scales) from the rows it is given; transform() applies exactly the same steps to any new
    raw data. Saved to and loaded from JSON."""

    def __init__(self, scale="standard", impute="median", missing_indicators="auto", max_categories=20, clip_outliers=None):
        if scale not in ("standard", "minmax", "robust", None):
            raise ValueError("scale must be 'standard', 'minmax', 'robust' or None")
        if impute not in ("median", "mean", "zero"):
            raise ValueError("impute must be 'median', 'mean' or 'zero'")
        self.scale, self.impute, self.indicators = scale, impute, missing_indicators
        self.max_categories, self.clip = max_categories, clip_outliers
        self.numeric, self.categorical, self.sources, self.feature_names_ = [], [], [], []

    # --- parsing shared by fit and transform
    def _numeric_parts(self, kind, values, with_hour=False):
        if kind == "numeric":
            return {None: np.array([np.nan if _is_missing(v) else (lambda x: np.nan if x is None else x)(_to_number(v)) for v in values], dtype=float)}
        if kind == "boolean":
            return {None: np.array([np.nan if _is_missing(v) or _to_bool(v) is None else _to_bool(v) for v in values], dtype=float)}
        return _datetime_features(values, with_hour)

    def fit(self, columns, kinds):
        """columns: name -> list of raw values (training rows); kinds: name -> numeric|boolean|datetime|categorical."""
        self.sources = list(kinds)
        self.numeric, self.categorical = [], []
        for name, kind in kinds.items():
            vals = columns[name]
            if kind == "categorical":
                cats = [None if _is_missing(v) else _norm_cat(v) for v in vals]
                counts = {}
                for c in cats:
                    if c is not None:
                        counts[c] = counts.get(c, 0) + 1
                ranked = sorted(counts, key=lambda c: (-counts[c], c))
                keep = ranked[: self.max_categories]
                has_other, has_missing = len(ranked) > len(keep), any(c is None for c in cats)
                self.categorical.append({"source": name, "categories": keep, "other": has_other, "missing": has_missing,
                                         "binary": len(keep) == 2 and not has_other and not has_missing})
                continue
            with_hour = kind == "datetime" and any(d is not None and (d.hour or d.minute) for d in (None if _is_missing(v) else _to_datetime(v) for v in vals))
            for part, x in self._numeric_parts(kind, vals, with_hour).items():
                feat = name if part is None else f"{name}__{part}"
                miss = int(np.isnan(x).sum())
                good = x[~np.isnan(x)]
                if kind == "boolean":
                    fill = float(np.round(good.mean())) if len(good) else 0.0
                elif len(good) == 0 or self.impute == "zero":
                    fill = 0.0
                else:
                    fill = float(np.median(good) if self.impute == "median" else good.mean())
                lo = hi = None
                if self.clip and len(good) > 1 and kind != "boolean":
                    lo, hi = (float(q) for q in np.quantile(good, [self.clip, 1 - self.clip]))
                xi = np.where(np.isnan(x), fill, x)
                if lo is not None:
                    xi = np.clip(xi, lo, hi)
                spec = {"source": name, "part": part, "feature": feat, "kind": kind, "fill": fill, "clip": [lo, hi] if lo is not None else None,
                        "indicator": bool(miss > 0 if self.indicators == "auto" else self.indicators), "with_hour": with_hour}
                q1, q3 = np.quantile(xi, [0.25, 0.75])
                spec["stats"] = {"mean": float(xi.mean()), "std": float(xi.std()), "min": float(xi.min()), "max": float(xi.max()),
                                 "median": float(np.median(xi)), "iqr": float(q3 - q1)}
                self.numeric.append(spec)
        self.feature_names_ = self._names()
        return self

    def _names(self):
        out = []
        for s in self.numeric:
            out.append(s["feature"])
            if s["indicator"]:
                out.append(s["feature"] + "__missing")
        for c in self.categorical:
            if c["binary"]:
                out.append(f"{c['source']}={c['categories'][0]}")
            else:
                out += [f"{c['source']}={k}" for k in c["categories"]]
                if c["other"]:
                    out.append(f"{c['source']}=__other__")
                if c["missing"]:
                    out.append(f"{c['source']}=__missing__")
        return out

    def _scale(self, x, st, kind):
        if kind == "boolean" or self.scale is None:
            return x
        if self.scale == "standard":
            return (x - st["mean"]) / (st["std"] or 1.0)
        if self.scale == "minmax":
            return (x - st["min"]) / ((st["max"] - st["min"]) or 1.0)
        return (x - st["median"]) / (st["iqr"] or 1.0)

    def transform(self, raw):
        """Apply the fitted steps to raw data (a dict of columns, DataFrame, list of dicts or CSV path)."""
        cols = raw if isinstance(raw, dict) and all(isinstance(v, list) for v in raw.values()) else _coerce_columns(raw)[0]
        missing = [s for s in self.sources if s not in cols]
        if missing:
            raise ValueError(f"the new data is missing column(s) {missing}; it has {list(cols)}")
        n = len(next(iter(cols.values())))
        feats = []
        parts_cache = {}
        for s in self.numeric:
            key = (s["source"], s["kind"], s["with_hour"])
            if key not in parts_cache:
                parts_cache[key] = self._numeric_parts(s["kind"], cols[s["source"]], s["with_hour"])
            x = parts_cache[key][s["part"]]
            isn = np.isnan(x)
            xi = np.where(isn, s["fill"], x)
            if s["clip"]:
                xi = np.clip(xi, *s["clip"])
            feats.append(self._scale(xi, s["stats"], s["kind"]))
            if s["indicator"]:
                feats.append(isn.astype(float))
        for c in self.categorical:
            cats = [None if _is_missing(v) else _norm_cat(v) for v in cols[c["source"]]]
            if c["binary"]:
                feats.append(np.array([1.0 if v == c["categories"][0] else 0.0 for v in cats]))
                continue
            index = {k: i for i, k in enumerate(c["categories"])}
            width = len(index) + int(c["other"]) + int(c["missing"])
            block = np.zeros((n, width))
            for r, v in enumerate(cats):
                if v is None:
                    if c["missing"]:
                        block[r, -1] = 1.0
                elif v in index:
                    block[r, index[v]] = 1.0
                elif c["other"]:
                    block[r, len(index)] = 1.0          # unseen category -> the 'other' bucket (or all zeros)
            feats.extend(block.T)
        return np.column_stack(feats).astype(np.float32) if feats else np.zeros((n, 0), np.float32)

    def to_dict(self):
        return {"scale": self.scale, "impute": self.impute, "indicators": self.indicators, "max_categories": self.max_categories,
                "clip": self.clip, "sources": self.sources, "numeric": self.numeric, "categorical": self.categorical}

    @classmethod
    def from_dict(cls, d):
        p = cls(d["scale"], d["impute"], d["indicators"], d["max_categories"], d["clip"])
        p.sources, p.numeric, p.categorical = d["sources"], d["numeric"], d["categorical"]
        p.feature_names_ = p._names()
        return p


# ------------------------------------------------------------------------------ report ---
class Report:
    """What was detected and what will be done. print(report) shows it; .columns / .warnings are data."""

    def __init__(self, header, columns, warnings_, footer=""):
        self.header, self.columns, self.warnings, self.footer = header, columns, list(warnings_), footer

    def __str__(self):
        w = max([len(c["column"]) for c in self.columns] + [6])
        lines = [self.header, "", f"{'column':<{w}}  {'kind':<11} {'missing':>7} {'unique':>7}  plan"]
        for c in self.columns:
            lines.append(f"{c['column']:<{w}}  {c['kind']:<11} {c['missing']:>7} {c['unique']:>7}  {c['plan']}")
        if self.warnings:
            lines += ["", "Warnings:"] + [f"  - {x}" for x in self.warnings]
        if self.footer:
            lines += ["", self.footer]
        return "\n".join(lines)

    __repr__ = __str__


# ----------------------------------------------------------------------------- prepared ---
@dataclass
class Pipeline:
    """Everything needed to turn new raw data into model input, and predictions back into labels."""
    kind: str                                   # tabular | sequence | image
    preprocessor: Optional[TabularPreprocessor]
    task: Optional[str] = None
    target: Optional[str] = None
    classes: Optional[list] = None
    target_mean: float = 0.0
    target_std: float = 1.0
    scale_target: bool = False
    window: Optional[int] = None
    image_shape: Optional[tuple] = None
    pixel_scale: Optional[dict] = None
    pixel_columns: Optional[list] = None

    def transform(self, raw):
        if self.kind == "image":
            cols = _coerce_columns(raw)[0] if not (isinstance(raw, dict) and all(isinstance(v, list) for v in raw.values())) else raw
            X = np.column_stack([np.array([0.0 if _to_number(v) is None else _to_number(v) for v in cols[c]]) for c in self.pixel_columns])
            return _scale_pixels(X, self.pixel_scale).reshape(-1, *self.image_shape).astype(np.float32)
        X = self.preprocessor.transform(raw)
        if self.kind == "sequence":      # windows ending at each row; rows must be in time order
            return np.stack([X[i - self.window + 1:i + 1] for i in range(self.window - 1, len(X))]) if len(X) >= self.window else np.zeros((0, self.window, X.shape[1]), np.float32)
        return X

    def decode_target(self, y):
        y = np.asarray(y)
        if self.task == "classification" and self.classes is not None:
            return np.array(self.classes, dtype=object)[y.astype(int)] if self.classes else y
        if self.task == "regression" and self.scale_target:
            return y * self.target_std + self.target_mean
        return y

    def to_dict(self):
        d = {k: getattr(self, k) for k in ("kind", "task", "target", "classes", "target_mean", "target_std", "scale_target", "window",
                                          "image_shape", "pixel_scale", "pixel_columns")}
        d["preprocessor"] = self.preprocessor.to_dict() if self.preprocessor else None
        return d

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, default=_json_default)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        pre = TabularPreprocessor.from_dict(d.pop("preprocessor")) if d.get("preprocessor") else None
        if d.get("image_shape"):
            d["image_shape"] = tuple(d["image_shape"])
        return cls(preprocessor=pre, **d)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


@dataclass
class Prepared:
    X_train: np.ndarray
    X_val: Optional[np.ndarray]
    X_test: Optional[np.ndarray]
    y_train: Optional[np.ndarray]
    y_val: Optional[np.ndarray]
    y_test: Optional[np.ndarray]
    pipeline: Pipeline
    feature_names: list
    text_report: Report
    kind: str = "tabular"

    @property
    def task(self):
        return self.pipeline.task

    @property
    def classes_(self):
        return None if self.pipeline.classes is None else np.array(self.pipeline.classes, dtype=object)

    @property
    def n_features(self):
        return self.X_train.shape[-1]

    def transform(self, raw):
        """Raw new data -> model input, using the fitted preprocessing."""
        return self.pipeline.transform(raw)

    def decode_target(self, y):
        """Model outputs -> original labels (classification) or original scale (regression)."""
        return self.pipeline.decode_target(y)

    def report(self):
        return self.text_report

    def save(self, path):
        """Save the fitted preprocessing (not the arrays) as JSON; reload with sc.load_pipeline(path)."""
        self.pipeline.save(path)

    def __repr__(self):
        sizes = {k: (None if v is None else tuple(v.shape)) for k, v in
                 (("train", self.X_train), ("val", self.X_val), ("test", self.X_test))}
        return f"Prepared(kind={self.kind!r}, task={self.task!r}, shapes={sizes})"


def load_pipeline(path):
    """Load a saved Pipeline (from Prepared.save) to prepare new data for inference."""
    return Pipeline.load(path)


def _scale_pixels(X, st):
    if st["mode"] == "none":
        return X
    if st["mode"] == "divide":
        return X / st["value"]
    return (X - st["min"]) / (st["max"] - st["min"] or 1.0)


# ------------------------------------------------------------------------------ dataset ---
@dataclass
class _Plan:
    profiles: dict
    features: dict            # name -> kind used by the preprocessor
    dropped: list             # (name, reason)
    warnings: list
    row_mask: np.ndarray      # rows kept (target present, duplicates removed)
    y: Optional[np.ndarray]   # encoded / parsed target for all rows (NaN where invalid)
    task: Optional[str]
    classes: Optional[list]
    header: str


class Dataset:
    """A raw table with inferred column types. See the module docstring for the workflow."""

    def __init__(self, columns, target=None, source="memory", notes=None):
        lengths = {len(v) for v in columns.values()}
        if len(lengths) != 1:
            raise ValueError("all columns must have the same length")
        self.columns, self.names, self.target, self.source = columns, list(columns), target, source
        self.n_rows, self.notes = lengths.pop(), list(notes or [])
        if target is not None and target not in columns:
            raise ValueError(f"target {target!r} is not a column; columns are {self.names}")

    def __repr__(self):
        return f"Dataset({self.n_rows} rows x {len(self.names)} columns, target={self.target!r}, source={self.source})"

    def head(self, n=5):
        return {k: v[:n] for k, v in self.columns.items()}

    # ------------------------------------------------------------------- planning ---
    def _plan(self, target, task, *, categorical=None, numeric=None, drop=None, keep=None, max_missing=0.8,
              drop_duplicates=False, exclude=(), encode_target=True):
        target = target or self.target
        categorical, numeric, drop, keep = (set(x or ()) for x in (categorical, numeric, drop, keep))
        for group in (categorical, numeric, drop, keep):
            bad = group - set(self.names)
            if bad:
                raise ValueError(f"unknown column(s) {sorted(bad)}; columns are {self.names}")
        warns = list(self.notes)
        profiles = {}
        for name in self.names:
            force = "categorical" if name in categorical else "numeric" if name in numeric else None
            profiles[name] = _profile_column(name, self.columns[name], force)
        row_mask = np.ones(self.n_rows, bool)
        y, classes, t_task = None, None, None
        if target is not None:
            y, classes, t_task, tnotes, valid = self._target(target, task, encode_target)
            warns += tnotes
            if (~valid).sum():
                warns.append(f"{int((~valid).sum())} rows have no usable target value and are removed")
            row_mask &= valid
        if drop_duplicates or True:
            keys = list(zip(*[[str(v).strip() for v in self.columns[c]] for c in self.names]))
            seen, dup = set(), np.zeros(self.n_rows, bool)
            for i, k in enumerate(keys):
                dup[i] = k in seen
                seen.add(k)
            if dup.sum():
                warns.append(f"{int(dup.sum())} duplicate rows" + (" removed" if drop_duplicates else " (kept; pass drop_duplicates=True to remove)"))
            if drop_duplicates:
                row_mask &= ~dup
        features, dropped = {}, []
        for name, p in profiles.items():
            if name == target or name in exclude:
                continue
            reason = None
            if name in drop:
                reason = "dropped by request"
            elif p.kind in ("empty", "constant", "id", "text") and name not in keep:
                reason = {"empty": "all values missing", "constant": "same value in every row",
                          "id": "looks like an identifier (all values unique)", "text": "free text (not supported yet)"}[p.kind]
            elif p.missing_frac > max_missing and name not in keep:
                reason = f"{p.missing_frac:.0%} missing (more than max_missing={max_missing:.0%})"
            if reason:
                dropped.append((name, reason))
            else:
                k = p.kind if p.kind in ("numeric", "boolean", "datetime", "categorical") else ("categorical" if p.kind in ("id", "text") else "numeric")
                features[name] = k
        for name, p in profiles.items():
            warns += [f"{name}: {x}" for x in p.notes if name in features or name == target]
        if target is not None and y is not None:
            warns += self._leak_warnings(features, profiles, y, t_task, row_mask)
        n_used = int(row_mask.sum())
        if n_used < 50:
            warns.append(f"only {n_used} usable rows; results will be noisy")
        elif features and n_used < 10 * len(features):
            warns.append(f"{n_used} rows for {len(features)} columns; consider fewer features or more data")
        head = f"Dataset: {self.n_rows} rows x {len(self.names)} columns (source: {self.source})"
        if target is not None:
            if t_task == "classification":
                counts = {c: int(((y == i) & row_mask).sum()) for i, c in enumerate(classes)}
                head += f"\nTarget: {target} -> classification, {len(classes)} classes: " + ", ".join(f"{k}={v}" for k, v in list(counts.items())[:8])
                small = min(counts.values())
                if len(classes) > 1 and small / max(sum(counts.values()), 1) < 0.1:
                    warns.append(f"imbalanced classes (smallest class is {small / max(sum(counts.values()), 1):.1%} of rows); accuracy can be misleading")
                if small < 2:
                    warns.append("a class has fewer than 2 rows; it cannot be split between train and test")
            else:
                head += f"\nTarget: {target} -> {t_task}"
        else:
            head += "\nTarget: none (unsupervised)"
        return _Plan(profiles, features, dropped, warns, row_mask, y, t_task, classes, head)

    def _target(self, target, task, encode_target):
        raw, n = self.columns[target], self.n_rows
        notes = []
        nums = [None if _is_missing(v) else _to_number(v) for v in raw]
        present = [v for v in raw if not _is_missing(v)]
        if not present:
            raise ValueError(f"target column {target!r} has no values")
        numeric = sum(x is not None for x in nums) / len(present) >= 0.95
        if task == "auto":
            vals = [x for x in nums if x is not None]
            task = "regression" if (numeric and (len(set(vals)) > 10 or not all(float(x).is_integer() for x in vals))) else "classification"
            notes.append(f"task inferred as {task} from the target values (override with task=...)")
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'auto', 'classification' or 'regression'")
        if task == "regression":
            if not numeric:
                raise ValueError(f"target {target!r} is not numeric, so it cannot be used for regression")
            y = np.array([np.nan if x is None else x for x in nums], dtype=float)
            return y, None, "regression", notes, ~np.isnan(y)
        keys = [None if _is_missing(v) else _norm_cat(v) for v in raw]
        display = {}
        for v, k in zip(raw, keys):
            if k is not None:
                display.setdefault(k, {}).setdefault(str(v).strip(), 0)
                display[k][str(v).strip()] += 1
        if numeric and all(float(x).is_integer() for x in nums if x is not None):
            order = sorted({int(x) for x in nums if x is not None})
            lookup = {str(k): i for i, k in enumerate(order)}
            y = np.array([np.nan if x is None else lookup[str(int(x))] for x in nums], dtype=float)
            classes = order
        else:
            order = sorted(display)
            lookup = {k: i for i, k in enumerate(order)}
            y = np.array([np.nan if k is None else lookup[k] for k in keys], dtype=float)
            classes = [max(display[k], key=display[k].get) for k in order]
            merged = sum(len(v) - 1 for v in display.values())
            if merged:
                notes.append(f"{merged} target spelling variants merged (case / whitespace)")
        return y, classes, "classification", notes, ~np.isnan(y)

    def _leak_warnings(self, features, profiles, y, task, mask):
        out = []
        ok = mask & ~np.isnan(y)
        for name, kind in features.items():
            vals = np.asarray(self.columns[name], dtype=object)[ok]
            if kind == "numeric" and task == "regression":
                x = np.array([np.nan if _is_missing(v) or _to_number(v) is None else _to_number(v) for v in vals])
                g = ~np.isnan(x)
                if g.sum() > 5 and np.std(x[g]) > 0 and np.std(y[ok][g]) > 0 and abs(np.corrcoef(x[g], y[ok][g])[0, 1]) > 0.999:
                    out.append(f"{name}: correlation with the target is {np.corrcoef(x[g], y[ok][g])[0, 1]:+.4f}; possible target leakage")
            elif kind in ("categorical", "boolean") and task == "classification" and 2 <= profiles[name].n_unique <= 50:
                groups = {}
                for v, t in zip(vals, y[ok]):
                    if not _is_missing(v):
                        groups.setdefault(_norm_cat(v), set()).add(t)
                if len(groups) >= 2 and all(len(s) == 1 for s in groups.values()) and len({next(iter(s)) for s in groups.values()}) > 1 and ok.sum() > 3 * len(groups):
                    out.append(f"{name}: every category maps to a single class; possible target leakage")
        return out

    def _column_plan(self, plan, o):
        rows = []
        for name in self.names:
            p = plan.profiles[name]
            if name == (o["target"]):
                rows.append(dict(column=name, kind="target", missing=f"{p.missing_frac:.1%}", unique=p.n_unique, plan=f"{plan.task}" + (", labels encoded as 0..k-1" if plan.task == "classification" and o["encode_target"] else "")))
                continue
            reason = next((r for n_, r in plan.dropped if n_ == name), None)
            if reason:
                rows.append(dict(column=name, kind=p.kind, missing=f"{p.missing_frac:.1%}", unique=p.n_unique, plan=f"DROPPED: {reason}"))
                continue
            if name not in plan.features:       # e.g. the time column, used for ordering only
                rows.append(dict(column=name, kind=p.kind, missing=f"{p.missing_frac:.1%}", unique=p.n_unique, plan="used to order rows, not a feature"))
                continue
            k, steps = plan.features[name], []
            if k == "numeric":
                if p.n_missing:
                    steps.append(f"{o['impute']}-impute" + (" + missing flag" if o["missing_indicators"] in ("auto", True) else ""))
                if o["clip_outliers"]:
                    steps.append(f"clip {o['clip_outliers']:.1%} tails")
                steps.append({"standard": "standardize", "minmax": "min-max scale", "robust": "robust scale", None: "no scaling"}[o["scale"]])
            elif k == "boolean":
                steps.append("map to 0/1" + (", impute mode" if p.n_missing else ""))
            elif k == "datetime":
                steps.append("expand to year, month, day, weekday, timestamp")
            else:
                steps.append(f"one-hot (top {o['max_categories']})" if p.n_unique > 2 else "binary 0/1")
                if p.n_missing:
                    steps.append("missing -> own category")
            rows.append(dict(column=name, kind=p.kind, missing=f"{p.missing_frac:.1%}", unique=p.n_unique, plan=" + ".join(steps)))
        return rows

    # ---------------------------------------------------------------------- report ---
    def report(self, target=None, task="auto", **options):
        """Describe the detected column types and the planned preparation without changing anything."""
        o = self._options(target, options)
        plan = self._plan(target, task, categorical=o["categorical"], numeric=o["numeric"], drop=o["drop"], keep=o["keep"],
                          max_missing=o["max_missing"], drop_duplicates=o["drop_duplicates"], encode_target=o["encode_target"])
        return Report(plan.header, self._column_plan(plan, o), plan.warnings)

    def _options(self, target, options):
        o = dict(target=target or self.target, scale="standard", impute="median", missing_indicators="auto", max_categories=20,
                 clip_outliers=None, categorical=None, numeric=None, drop=None, keep=None, max_missing=0.8,
                 drop_duplicates=False, encode_target=True)
        unknown = set(options) - set(o)
        if unknown:
            raise TypeError(f"unknown option(s): {sorted(unknown)}")
        o.update(options)
        return o

    # ------------------------------------------------------------------- tabular ---
    def prepare(self, target=None, *, task="auto", test_size=0.2, val_size=0.0, seed=0, split="random", stratify=True,
                scale_target=False, **options):
        """Clean, encode, split and scale into float32 arrays ready for any vector model.

        Options (all optional): scale ('standard' | 'minmax' | 'robust' | None), impute ('median' | 'mean' | 'zero'),
        missing_indicators ('auto' | True | False), max_categories (20), clip_outliers (e.g. 0.01), categorical=[...],
        numeric=[...], drop=[...], keep=[...], max_missing (0.8), drop_duplicates, encode_target (True),
        split ('random' | 'chronological'), scale_target (standardize a regression target)."""
        o = self._options(target, options)
        plan = self._plan(target, task, categorical=o["categorical"], numeric=o["numeric"], drop=o["drop"], keep=o["keep"],
                          max_missing=o["max_missing"], drop_duplicates=o["drop_duplicates"], encode_target=o["encode_target"])
        if not plan.features:
            raise ValueError("no usable feature columns remain; dropped: " + "; ".join(f"{n} ({r})" for n, r in plan.dropped))
        rows = np.flatnonzero(plan.row_mask)
        y_all = None if plan.y is None else plan.y[rows]
        tr, va, te = split_indices(len(rows), test_size, val_size, seed, y_all if plan.task == "classification" else None,
                                   stratify, chronological=(split == "chronological"))
        if split not in ("random", "chronological"):
            raise ValueError("split must be 'random' or 'chronological'")
        raw = {c: np.asarray(self.columns[c], dtype=object)[rows] for c in plan.features}
        sub = lambda idx: {c: raw[c][idx].tolist() for c in raw}
        pre = TabularPreprocessor(o["scale"], o["impute"], o["missing_indicators"], o["max_categories"], o["clip_outliers"]).fit(sub(tr), plan.features)
        X = {"train": pre.transform(sub(tr)),
             "val": pre.transform(sub(va)) if len(va) else None,
             "test": pre.transform(sub(te)) if len(te) else None}
        y = {"train": None, "val": None, "test": None}
        pipe = Pipeline("tabular", pre, plan.task, o["target"], plan.classes if plan.task == "classification" and o["encode_target"] else None)
        if y_all is not None:
            yt = y_all.astype(int) if plan.task == "classification" else y_all
            if plan.task == "regression" and scale_target:
                pipe.scale_target, pipe.target_mean, pipe.target_std = True, float(yt[tr].mean()), float(yt[tr].std() or 1.0)
                yt = (yt - pipe.target_mean) / pipe.target_std
            if plan.task == "classification" and not o["encode_target"]:
                labels = np.array(plan.classes, dtype=object)
                yt = labels[yt]
            for k, idx in (("train", tr), ("val", va), ("test", te)):
                y[k] = yt[idx] if len(idx) else None
            if plan.task == "classification":
                missing_cls = set(range(len(plan.classes))) - set(np.asarray(y_all[tr]).astype(int))
                if missing_cls:
                    plan.warnings.append(f"class(es) {[plan.classes[i] for i in sorted(missing_cls)]} are absent from the training split")
        cols = self._column_plan(plan, o)
        foot = f"Split ({split}, seed={seed}): train={len(tr)}, val={len(va)}, test={len(te)}.  Features: {len(pre.feature_names_)} " \
               f"({', '.join(pre.feature_names_[:6])}{', ...' if len(pre.feature_names_) > 6 else ''})"
        return Prepared(X["train"], X["val"], X["test"], y["train"], y["val"], y["test"], pipe, pre.feature_names_,
                        Report(plan.header, cols, plan.warnings, foot))

    # ----------------------------------------------------------------- sequences ---
    def prepare_sequences(self, window, target=None, *, horizon=1, task="auto", time_column=None, test_size=0.2, val_size=0.0,
                          use_target_as_feature=True, scale_target=False, **options):
        """Sliding windows for sequence models: X has shape (n, window, features) and y is the target `horizon`
        steps after the window. Rows are split chronologically (never shuffled) and preprocessing is fitted
        on the training rows only. Pass time_column to sort the rows by time first."""
        if window < 1 or horizon < 1:
            raise ValueError("window and horizon must be at least 1")
        target = target or self.target
        if target is None:
            raise ValueError("prepare_sequences needs a target column")
        ds = self
        if time_column is not None:
            if time_column not in self.columns:
                raise ValueError(f"time_column {time_column!r} is not a column")
            keys = [(_to_datetime(v) or _EPOCH + _dt.timedelta(days=(_to_number(v) if _to_number(v) is not None else float("nan")))) if not _is_missing(v) else None
                    for v in self.columns[time_column]]
            if any(k is None for k in keys):
                raise ValueError("time_column has missing or unparsable values")
            order = sorted(range(self.n_rows), key=lambda i: keys[i])
            ds = Dataset({c: [v[i] for i in order] for c, v in self.columns.items()}, target, self.source, self.notes)
        o = ds._options(target, options)
        exclude = (time_column,) if time_column else ()
        plan = ds._plan(target, task, categorical=o["categorical"], numeric=o["numeric"], drop=o["drop"], keep=o["keep"],
                        max_missing=o["max_missing"], exclude=exclude, encode_target=True)
        row_ok = np.ones(ds.n_rows, bool)
        feats = dict(plan.features)
        if not use_target_as_feature:
            feats.pop(target, None)
        else:
            p = plan.profiles[target]
            feats[target] = "numeric" if plan.task == "regression" else ("boolean" if p.kind == "boolean" else "categorical")
        if not feats:
            raise ValueError("no usable feature columns remain")
        n = ds.n_rows
        a, b = n - round(n * test_size) - round(n * val_size), n - round(n * test_size)
        if a < window + horizon:
            raise ValueError(f"the training part ({a} rows) is shorter than window + horizon ({window + horizon})")
        raw = {c: ds.columns[c] for c in feats}
        pre = TabularPreprocessor(o["scale"], o["impute"], o["missing_indicators"], o["max_categories"], o["clip_outliers"]).fit(
            {c: v[:a] for c, v in raw.items()}, feats)
        M = pre.transform(raw)
        y = plan.y.copy()
        pipe = Pipeline("sequence", pre, plan.task, target, plan.classes if plan.task == "classification" else None, window=window)
        if plan.task == "regression" and scale_target:
            good = y[:a][~np.isnan(y[:a])]
            pipe.scale_target, pipe.target_mean, pipe.target_std = True, float(good.mean()), float(good.std() or 1.0)
            y = (y - pipe.target_mean) / pipe.target_std

        def windows(lo, hi):
            xs, ys = [], []
            for t in range(max(lo, window + horizon - 1), hi):
                s = t - horizon
                if not np.isnan(y[t]):
                    xs.append(M[s - window + 1:s + 1]); ys.append(y[t])
            if not xs:
                return None, None
            return np.stack(xs).astype(np.float32), (np.array(ys).astype(int) if plan.task == "classification" else np.array(ys))

        parts = {"train": windows(0, a), "val": windows(a, b) if b > a else (None, None), "test": windows(b, n) if n > b else (None, None)}
        if parts["train"][0] is None:
            raise ValueError("no training windows could be built; the target may be missing")
        foot = f"Windows: window={window}, horizon={horizon}; chronological split train={0 if parts['train'][1] is None else len(parts['train'][1])}, " \
               f"val={0 if parts['val'][1] is None else len(parts['val'][1])}, test={0 if parts['test'][1] is None else len(parts['test'][1])}. " \
               f"Inputs may look back into earlier splits; targets never do. Features: {len(pre.feature_names_)}"
        cols = ds._column_plan(plan, {**o, "target": target})
        return Prepared(parts["train"][0], parts["val"][0], parts["test"][0], parts["train"][1], parts["val"][1], parts["test"][1], pipe,
                        pre.feature_names_, Report(plan.header, cols, plan.warnings, foot), kind="sequence")

    # -------------------------------------------------------------------- images ---
    def prepare_images(self, target=None, *, shape=None, task="classification", test_size=0.2, val_size=0.0, seed=0, stratify=True,
                       pixel_scale="auto", columns=None):
        """Flat pixel columns (one column per pixel, e.g. MNIST CSV) -> images of shape (n, C, H, W) for CNN.
        shape=(C, H, W) is inferred when the column count is a perfect square (grayscale) or 3 x a square (RGB)."""
        target = target or self.target
        pix = list(columns) if columns else [c for c in self.names if c != target]
        for c in pix:
            if c not in self.columns:
                raise ValueError(f"unknown column {c!r}")
        n_pix = len(pix)
        if shape is None:
            s = int(round(math.sqrt(n_pix)))
            s3 = int(round(math.sqrt(n_pix / 3)))
            shape = (1, s, s) if s * s == n_pix else (3, s3, s3) if 3 * s3 * s3 == n_pix else None
            if shape is None:
                raise ValueError(f"{n_pix} pixel columns do not form a square image; pass shape=(channels, height, width)")
        if int(np.prod(shape)) != n_pix:
            raise ValueError(f"shape {tuple(shape)} needs {int(np.prod(shape))} pixel columns but there are {n_pix}")
        notes = []
        mat = np.column_stack([np.array([np.nan if _is_missing(v) or _to_number(v) is None else _to_number(v) for v in self.columns[c]]) for c in pix])
        if np.isnan(mat).any():
            notes.append(f"{int(np.isnan(mat).sum())} missing pixel values set to 0")
        mat = np.nan_to_num(mat, nan=0.0)
        plan = self._plan(target, task, exclude=pix) if target else None
        rows = np.flatnonzero(plan.row_mask) if plan else np.arange(self.n_rows)
        y_all = None if plan is None else plan.y[rows]
        tr, va, te = split_indices(len(rows), test_size, val_size, seed, y_all if plan and plan.task == "classification" else None, stratify)
        mat = mat[rows]
        mx = float(mat[tr].max())
        if pixel_scale == "auto":
            st = {"mode": "divide", "value": 255.0} if 1.0 < mx <= 255.0 else {"mode": "none"} if mx <= 1.0 else {"mode": "minmax", "min": float(mat[tr].min()), "max": mx}
        elif pixel_scale in (None, "none"):
            st = {"mode": "none"}
        else:
            st = {"mode": "minmax", "min": float(mat[tr].min()), "max": mx}
        imgs = _scale_pixels(mat, st).reshape(-1, *shape).astype(np.float32)
        pipe = Pipeline("image", None, plan.task if plan else None, target, plan.classes if plan and plan.task == "classification" else None,
                        image_shape=tuple(shape), pixel_scale=st, pixel_columns=pix)
        y = {"train": None, "val": None, "test": None}
        if y_all is not None:
            yt = y_all.astype(int) if plan.task == "classification" else y_all
            for k, idx in (("train", tr), ("val", va), ("test", te)):
                y[k] = yt[idx] if len(idx) else None
        head = f"Images: {n_pix} pixel columns -> {tuple(shape)} (channels, height, width); pixel scaling: {st['mode']}"
        rep = Report(head + ("\n" + plan.header if plan else ""), [], (plan.warnings if plan else []) + notes,
                     f"Split (seed={seed}): train={len(tr)}, val={len(va)}, test={len(te)}")
        return Prepared(imgs[tr], imgs[va] if len(va) else None, imgs[te] if len(te) else None, y["train"], y["val"], y["test"], pipe,
                        [f"pixel{i}" for i in range(n_pix)], rep, kind="image")


def load_dataset(source, target=None, *, header=True, delimiter=None, names=None):
    """Load a CSV/TSV path, pandas DataFrame, dict of columns, list of dicts or 2-D array as a Dataset."""
    cols, notes, origin = _coerce_columns(source, header, delimiter, names)
    return Dataset(cols, target, origin, notes)


# ------------------------------------------------------------------------------ checks ---
def check_arrays(X, y=None, expect="tabular"):
    """Validate model input and raise an error that explains what to fix. Returns the arrays as float32 / ndarray."""
    from .base import to_float_array
    X = to_float_array(X)
    want = {"tabular": 2, "sequence": 3, "image": 4}[expect]
    if X.ndim != want:
        hint = {"tabular": "(n_samples, n_features)", "sequence": "(n_samples, time_steps, features)", "image": "(n_samples, channels, height, width)"}[expect]
        raise ValueError(f"{expect} models expect X with shape {hint}, got {X.shape}")
    if y is not None:
        y = np.asarray(y)
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        if y.dtype.kind == "f" and not np.isfinite(y).all():
            raise ValueError("y contains NaN or infinite values; drop those rows (Dataset.prepare does this)")
    return X, y


# --------------------------------------------------------------------------- auto_model ---
@dataclass
class AutoResult:
    model: object
    prepared: Prepared
    metrics: dict
    model_name: str

    def predict(self, raw):
        """Raw new data (same columns as the original) -> predictions in the original label / value scale."""
        return self.prepared.decode_target(self.model.predict(self.prepared.transform(raw)))

    def predict_proba(self, raw):
        return self.model.predict_proba(self.prepared.transform(raw))

    def report(self):
        return self.prepared.report()


_TABULAR = {"vector", "classical", "bio", "unsupervised"}


def auto_model(source, target=None, model="mlp", *, task="auto", prepare=None, fit=None, seed=0, window=None, image_shape=None, **model_kwargs):
    """Load, clean, split, train and evaluate in one call.

        result = sc.auto_model("data.csv", target="Purchased", model="mlp", hidden=(64, 32))
        result.metrics            # {'test': {'accuracy': ...}}
        result.predict(new_rows)  # original labels

    The model's family decides the preparation: vector / classical models get a tabular table; sequence models
    need window=...; the CNN needs flat pixel columns (image_shape=(C, H, W) if not square). `prepare` and `fit`
    are dicts of options for Dataset.prepare and model.fit; everything else goes to the model constructor."""
    if model not in REGISTRY:
        raise ValueError(f"unknown model {model!r}; see saturncube.available()")
    cls = REGISTRY[model]
    fam = getattr(cls, "family", None)
    data = source if isinstance(source, Dataset) else load_dataset(source, target)
    target = target or data.target
    prepare, fit = dict(prepare or {}), dict(fit or {})
    prepare.setdefault("seed", seed)
    prepare.setdefault("val_size", 0.0)
    if fam == "vision":
        prepared = data.prepare_images(target, shape=image_shape, task=prepare.pop("task", "classification"), **prepare)
    elif fam == "sequence":
        if window is None:
            raise ValueError(f"{model} is a sequence model: pass window=... (and optionally horizon=, time_column=) so rows can be turned into windows")
        prepare.pop("seed", None)
        prepared = data.prepare_sequences(window, target, task=task, **{k: v for k, v in prepare.items() if k not in ("stratify",)})
    elif fam in _TABULAR:
        prepared = data.prepare(target, task=task, **prepare)
    else:
        raise ValueError(f"{model} (family {fam!r}) needs data this helper cannot build automatically (pairs, graphs, environments, ...); "
                         "prepare the arrays yourself")
    params = inspect.signature(cls.__init__).parameters
    kwargs = dict(model_kwargs)
    if "seed" in params or any(p.kind == p.VAR_KEYWORD for p in params.values()):
        kwargs.setdefault("seed", seed)
    m = cls(**kwargs)
    if prepared.y_train is None:
        m.fit(prepared.X_train, **fit)
    else:
        m.fit(prepared.X_train, prepared.y_train, **fit)
    metrics = {}
    for name, X, y in (("val", prepared.X_val, prepared.y_val), ("test", prepared.X_test, prepared.y_test)):
        if X is not None and len(X):
            metrics[name] = m.evaluate(X, y) if y is not None else m.evaluate(X)
    return AutoResult(m, prepared, metrics, model)
