"""Raw text in, any model out.

    clf = sc.TextModel("naive_bayes").fit(train_docs, train_labels)   # TF-IDF features
    clf = sc.TextModel("lstm", hidden=32).fit(train_docs, train_labels)  # token ids, chosen automatically
    clf.predict(new_docs); clf.evaluate(test_docs, test_labels)

Text is turned into numbers by one of two small encoders that are fitted on the training documents only
(vocabulary, document frequencies and SVD directions never see validation or test text):

* TextVectorizer  words -> TF-IDF vectors (optionally reduced with a truncated SVD): for tabular models.
* TokenEncoder    words -> padded integer ids: for sequence / attention models (they take ``vocab_size=``).

Limits: no stemming, stop-word lists or subword tokenisation; TF-IDF matrices are dense, so keep
``max_features`` modest (a guard stops matrices above ~1.6 GB). Padding id 0 is an ordinary learned token
because the sequence models do not mask padding, so choose ``max_len`` close to your typical document length.
"""
from __future__ import annotations

import difflib
import inspect
import json
import re
from collections import Counter

import numpy as np

from .base import REGISTRY
from .sequence import SequenceModel

_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?")
_MAX_DENSE_ELEMENTS = 400_000_000


def tokenize(text, lowercase=True):
    """Words (letters and digits, with simple apostrophes); punctuation and underscores split tokens."""
    return _WORD.findall(text.lower() if lowercase else text)


def _check_texts(texts):
    if isinstance(texts, (str, bytes)):
        raise TypeError("pass a list of documents, not a single string")
    out = list(texts)
    for i, t in enumerate(out):
        if not isinstance(t, str):
            raise ValueError("document %d is %s, expected str (remove or replace missing text first)"
                             % (i, type(t).__name__))
    return out


def _randomized_svd(X, k, seed, n_iter=3, oversample=10):
    """Top-k right singular vectors of X (k, n_features), deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    q = min(k + oversample, min(X.shape))
    Y = X @ rng.standard_normal((X.shape[1], q)).astype(X.dtype)
    for _ in range(n_iter):
        Y, _ = np.linalg.qr(Y)
        Z, _ = np.linalg.qr(X.T @ Y)
        Y = X @ Z
    Q, _ = np.linalg.qr(Y)
    _, _, Vt = np.linalg.svd(Q.T @ X, full_matrices=False)
    return Vt[:k].astype(np.float32)


class TextVectorizer:
    """TF-IDF features, optionally reduced to ``svd`` dimensions. Fitted on the documents you pass to fit().

    max_features   keep the most frequent terms (by document frequency); min_df drops terms in fewer documents.
    ngram_range    e.g. (1, 2) adds word pairs. sublinear_tf uses 1 + log(tf). Rows are L2-normalised.
    svd            None, or the number of components of a truncated SVD (latent-semantic features).
    """

    def __init__(self, max_features=5000, min_df=2, ngram_range=(1, 1), lowercase=True, sublinear_tf=True,
                 svd=None, seed=0):
        lo, hi = ngram_range
        if not (1 <= lo <= hi):
            raise ValueError("ngram_range must satisfy 1 <= low <= high")
        if max_features < 1 or min_df < 1 or (svd is not None and svd < 1):
            raise ValueError("max_features, min_df and svd must be positive")
        self.max_features, self.min_df, self.ngram_range = max_features, min_df, (int(lo), int(hi))
        self.lowercase, self.sublinear_tf, self.svd, self.seed = lowercase, sublinear_tf, svd, seed
        self.vocabulary_ = self.idf_ = self.components_ = None

    def _terms(self, text):
        words = tokenize(text, self.lowercase)
        lo, hi = self.ngram_range
        out = []
        for n in range(lo, hi + 1):
            out += [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
        return out

    def _tfidf(self, docs):
        X = np.zeros((len(docs), len(self.vocabulary_)), dtype=np.float32)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                j = self._index.get(term)
                if j is not None:
                    X[i, j] = (1.0 + np.log(tf) if self.sublinear_tf else tf) * self.idf_[j]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norms > 0, norms, 1.0)

    def fit(self, texts):
        docs = [self._terms(t) for t in _check_texts(texts)]
        n = len(docs)
        if n < 2:
            raise ValueError("need at least 2 documents to fit a vectorizer")
        df = Counter()
        for d in docs:
            df.update(set(d))
        kept = sorted(((t, c) for t, c in df.items() if c >= self.min_df), key=lambda tc: (-tc[1], tc[0]))
        kept = kept[:self.max_features]
        if not kept:
            raise ValueError("no term appears in at least min_df=%d documents; lower min_df" % self.min_df)
        if n * len(kept) > _MAX_DENSE_ELEMENTS:
            raise ValueError("a dense %d x %d TF-IDF matrix is too large; lower max_features" % (n, len(kept)))
        self.vocabulary_ = [t for t, _ in kept]
        self._index = {t: j for j, t in enumerate(self.vocabulary_)}
        self.idf_ = (np.log((1.0 + n) / (1.0 + np.array([c for _, c in kept], dtype=np.float64))) + 1.0)
        self.components_ = None
        if self.svd:
            k = min(self.svd, len(self.vocabulary_), n)
            self.components_ = _randomized_svd(self._tfidf(docs), k, self.seed)
        return self

    def transform(self, texts):
        if self.vocabulary_ is None:
            raise RuntimeError("call fit() first")
        X = self._tfidf([self._terms(t) for t in _check_texts(texts)])
        return X if self.components_ is None else (X @ self.components_.T).astype(np.float32)

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

    @property
    def n_features_(self):
        if self.vocabulary_ is None:
            raise RuntimeError("call fit() first")
        return len(self.vocabulary_) if self.components_ is None else len(self.components_)

    def to_dict(self):
        if self.vocabulary_ is None:
            raise RuntimeError("call fit() first")
        return {"type": "TextVectorizer", "version": 1,
                "params": {"max_features": self.max_features, "min_df": self.min_df,
                           "ngram_range": list(self.ngram_range), "lowercase": self.lowercase,
                           "sublinear_tf": self.sublinear_tf, "svd": self.svd, "seed": self.seed},
                "vocabulary": self.vocabulary_, "idf": [float(v) for v in self.idf_],
                "components": None if self.components_ is None else self.components_.tolist()}

    @classmethod
    def from_dict(cls, d):
        p = dict(d["params"])
        p["ngram_range"] = tuple(p["ngram_range"])
        v = cls(**p)
        v.vocabulary_, v.idf_ = list(d["vocabulary"]), np.array(d["idf"], dtype=np.float64)
        v._index = {t: j for j, t in enumerate(v.vocabulary_)}
        v.components_ = None if d["components"] is None else np.array(d["components"], dtype=np.float32)
        return v

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class TokenEncoder:
    """Words -> integer ids of shape (n, max_len): 0 = padding, 1 = unknown word, 2.. = vocabulary.

    Documents longer than max_len keep their first max_len tokens. After fit(), ``vocab_size_`` is the value
    to give a sequence model as ``vocab_size=`` (TextModel does that for you).
    """

    PAD, UNK = 0, 1

    def __init__(self, vocab_size=5000, max_len=128, min_count=2, lowercase=True):
        if vocab_size < 3 or max_len < 1 or min_count < 1:
            raise ValueError("vocab_size must be >= 3 and max_len, min_count >= 1")
        self.vocab_size, self.max_len, self.min_count, self.lowercase = vocab_size, max_len, min_count, lowercase
        self.vocabulary_ = None

    def fit(self, texts):
        counts = Counter()
        for t in _check_texts(texts):
            counts.update(tokenize(t, self.lowercase))
        kept = sorted(((w, c) for w, c in counts.items() if c >= self.min_count), key=lambda wc: (-wc[1], wc[0]))
        self.vocabulary_ = [w for w, _ in kept[:self.vocab_size - 2]]
        self._index = {w: i + 2 for i, w in enumerate(self.vocabulary_)}
        return self

    @property
    def vocab_size_(self):
        if self.vocabulary_ is None:
            raise RuntimeError("call fit() first")
        return len(self.vocabulary_) + 2

    def transform(self, texts):
        if self.vocabulary_ is None:
            raise RuntimeError("call fit() first")
        docs = _check_texts(texts)
        out = np.zeros((len(docs), self.max_len), dtype=np.int64)
        for i, t in enumerate(docs):
            ids = [self._index.get(w, self.UNK) for w in tokenize(t, self.lowercase)[:self.max_len]]
            out[i, :len(ids)] = ids
        return out

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

    def to_dict(self):
        if self.vocabulary_ is None:
            raise RuntimeError("call fit() first")
        return {"type": "TokenEncoder", "version": 1,
                "params": {"vocab_size": self.vocab_size, "max_len": self.max_len,
                           "min_count": self.min_count, "lowercase": self.lowercase},
                "vocabulary": self.vocabulary_}

    @classmethod
    def from_dict(cls, d):
        e = cls(**d["params"])
        e.vocabulary_ = list(d["vocabulary"])
        e._index = {w: i + 2 for i, w in enumerate(e.vocabulary_)}
        return e

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


_ENCODERS = {"TextVectorizer": TextVectorizer, "TokenEncoder": TokenEncoder}


class TextModel:
    """Any registry model (or class) behind raw-text fit / predict / evaluate.

    model         registry name such as "mlp", "naive_bayes", "lstm", "transformer_encoder", or a model class.
    mode          "auto" (default): sequence / attention models get token ids, everything else TF-IDF features;
                  or force "tfidf" / "tokens".
    vectorizer    an unfitted TextVectorizer / TokenEncoder to control max_features, svd, max_len, ...
    model_kwargs  passed to the model's constructor (seed=, hidden=, epochs=, ...). For token mode,
                  vocab_size= is filled in from the fitted vocabulary.

    The encoder is fitted on the documents given to fit() only. ``.model`` is the underlying model (its
    ``.module`` is the raw PyTorch module for neural models) and ``.vectorizer`` the fitted encoder.
    """

    def __init__(self, model="mlp", *, vectorizer=None, mode="auto", **model_kwargs):
        if isinstance(model, str):
            key = model.lower()
            if key not in REGISTRY:
                close = difflib.get_close_matches(key, list(REGISTRY), n=3)
                raise ValueError("unknown model %r%s; see sc.available()" % (
                    model, " (did you mean %s?)" % ", ".join(map(repr, close)) if close else ""))
            cls = REGISTRY[key]
        elif isinstance(model, type):
            cls = model
        else:
            raise TypeError("model must be a registry name or a class, not %s" % type(model).__name__)
        if mode not in ("auto", "tfidf", "tokens"):
            raise ValueError("mode must be 'auto', 'tfidf' or 'tokens'")
        is_seq = issubclass(cls, SequenceModel)
        if mode == "auto":
            mode = "tokens" if is_seq else "tfidf"
        if mode == "tokens" and not is_seq:
            raise ValueError("token ids are for sequence / attention models; use mode='tfidf' for %s" % cls.__name__)
        if mode == "tfidf" and is_seq:
            raise ValueError("%s reads token sequences; use mode='tokens'" % cls.__name__)
        want = TokenEncoder if mode == "tokens" else TextVectorizer
        if vectorizer is None:
            vectorizer = want()
        elif not isinstance(vectorizer, want):
            raise TypeError("mode=%r needs a %s, got %s" % (mode, want.__name__, type(vectorizer).__name__))
        self._cls, self.mode, self.vectorizer, self._kwargs = cls, mode, vectorizer, model_kwargs
        self.model = None

    def _need_fit(self):
        if self.model is None:
            raise RuntimeError("call fit() first")

    def fit(self, texts, y, val_data=None, **fit_kwargs):
        """val_data=(val_texts, val_y) is encoded with the training vocabulary and passed on to the model."""
        texts, y = _check_texts(texts), np.asarray(y)
        if len(texts) != len(y):
            raise ValueError("%d documents but %d labels" % (len(texts), len(y)))
        X = self.vectorizer.fit_transform(texts)
        kw = dict(self._kwargs)
        if self.mode == "tokens":
            need = self.vectorizer.vocab_size_
            if kw.get("vocab_size", need) < need:
                raise ValueError("vocab_size=%d is smaller than the vocabulary (%d)" % (kw["vocab_size"], need))
            kw.setdefault("vocab_size", need)
        self.model = self._cls(**kw)
        if val_data is not None:
            if "val_data" not in inspect.signature(self.model.fit).parameters:
                raise TypeError("%s.fit() does not accept val_data" % self._cls.__name__)
            fit_kwargs["val_data"] = (self.vectorizer.transform(val_data[0]), np.asarray(val_data[1]))
        self.model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, texts):
        self._need_fit()
        return self.model.predict(self.vectorizer.transform(texts))

    def predict_proba(self, texts):
        self._need_fit()
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("%s has no predict_proba" % self._cls.__name__)
        return self.model.predict_proba(self.vectorizer.transform(texts))

    def evaluate(self, texts, y):
        self._need_fit()
        return self.model.evaluate(self.vectorizer.transform(texts), np.asarray(y))

    @property
    def classes_(self):
        self._need_fit()
        return self.model.classes_

    def save(self, path):
        """Writes the model to ``path`` and the encoder (plain JSON) and settings to ``path + '.text.json'``."""
        self._need_fit()
        name = getattr(self._cls, "name", None)
        if not hasattr(self.model, "save"):
            raise NotImplementedError("%s has no save(); only models with save() can be stored" % self._cls.__name__)
        if name not in REGISTRY or REGISTRY[name] is not self._cls:
            raise ValueError("only registered model classes can be saved and reloaded")
        self.model.save(path)
        with open(str(path) + ".text.json", "w", encoding="utf-8") as f:
            json.dump({"model": name, "mode": self.mode, "encoder": self.vectorizer.to_dict()}, f)

    @classmethod
    def load(cls, path):
        with open(str(path) + ".text.json", encoding="utf-8") as f:
            meta = json.load(f)
        model_cls = REGISTRY[meta["model"]]
        tm = cls(meta["model"], mode=meta["mode"])
        tm.vectorizer = _ENCODERS[meta["encoder"]["type"]].from_dict(meta["encoder"])
        tm.model = model_cls.load(path)
        return tm

    def __repr__(self):
        return "TextModel(%s, mode=%s, fitted=%s)" % (self._cls.__name__, self.mode, self.model is not None)
