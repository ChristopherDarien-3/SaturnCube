import json

import numpy as np
import pytest

import saturncube as sc
from saturncube import text as sct

TOPICS = {
    "sport": "goal match team coach league player score season referee stadium".split(),
    "tech": "software server network compiler kernel database cloud latency browser protocol".split(),
    "food": "recipe oven butter garlic simmer dough sauce pepper roast kitchen".split(),
}
FILLER = "the a of and to in is that it was for on with as very quite really".split()


def corpus(n_per_class=60, seed=0, length=10):
    r = np.random.default_rng(seed)
    docs, labels = [], []
    for label, words in TOPICS.items():
        for _ in range(n_per_class):
            doc = [r.choice(words) if r.random() < 0.6 else r.choice(FILLER) for _ in range(length)]
            docs.append(" ".join(doc)); labels.append(label)
    order = r.permutation(len(docs))
    return [docs[i] for i in order], np.array(labels)[order]


def split(docs, labels, frac=0.7):
    k = int(len(docs) * frac)
    return docs[:k], labels[:k], docs[k:], labels[k:]


# --------------------------------------------------------------------------- TextVectorizer ---
def test_tfidf_shape_norm_and_empty_document():
    docs, _ = corpus()
    v = sc.TextVectorizer(max_features=50, min_df=2)
    X = v.fit_transform(docs)
    assert X.shape == (len(docs), len(v.vocabulary_)) and len(v.vocabulary_) <= 50 and X.dtype == np.float32
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)
    Z = v.transform(["", "zzzz qqqq unseen words only"])
    assert np.all(Z == 0)  # nothing known -> zero vector, no NaN


def test_vocabulary_and_idf_come_from_training_documents_only():
    train = ["alpha beta", "alpha beta gamma", "alpha delta", "beta delta"]
    test = ["zzqx zzqx", "zzqx zzqx", "zzqx qqwe"]
    v = sc.TextVectorizer(min_df=2).fit(train)
    assert "zzqx" not in v.vocabulary_                       # test-only term never enters the vocabulary
    assert v.transform(test).shape[1] == len(v.vocabulary_)  # and is ignored at transform time


def test_rarer_terms_get_larger_idf_and_min_df_filters():
    docs = ["common rare1", "common x", "common y", "common rare1", "common z z"]
    v = sc.TextVectorizer(min_df=2).fit(docs)
    assert set(v.vocabulary_) == {"common", "rare1", "z"} - {"z"} | set()  # z appears in 1 document only
    idf = dict(zip(v.vocabulary_, v.idf_))
    assert idf["rare1"] > idf["common"]


def test_ngrams_and_determinism():
    docs, _ = corpus(20)
    a = sc.TextVectorizer(ngram_range=(1, 2), max_features=200).fit(docs)
    b = sc.TextVectorizer(ngram_range=(1, 2), max_features=200).fit(docs)
    assert any(" " in t for t in a.vocabulary_) and a.vocabulary_ == b.vocabulary_
    assert np.array_equal(a.transform(docs), b.transform(docs))


def test_svd_reduces_dimensions_and_keeps_topics_separable():
    docs, y = corpus()
    v = sc.TextVectorizer(svd=3, seed=1)
    Z = v.fit_transform(docs)
    assert Z.shape == (len(docs), 3) and v.n_features_ == 3
    centroids = {c: Z[y == c].mean(0) for c in np.unique(y)}
    pred = [min(centroids, key=lambda c: np.linalg.norm(z - centroids[c])) for z in Z]
    assert np.mean(np.array(pred) == y) > 0.9
    assert np.array_equal(Z, sc.TextVectorizer(svd=3, seed=1).fit_transform(docs))


def test_vectorizer_json_roundtrip(tmp_path):
    docs, _ = corpus(20)
    for v in (sc.TextVectorizer(max_features=60), sc.TextVectorizer(max_features=60, svd=4)):
        X = v.fit_transform(docs)
        p = tmp_path / "vec.json"
        v.save(p)
        json.loads(p.read_text())  # plain JSON, no pickle
        assert np.allclose(sc.TextVectorizer.load(p).transform(docs), X, atol=1e-6)


def test_input_validation():
    with pytest.raises(TypeError, match="not a single string"):
        sc.TextVectorizer().fit("one string")
    with pytest.raises(ValueError, match="document 1 is NoneType"):
        sc.TextVectorizer().fit(["ok", None])
    with pytest.raises(ValueError, match="document 0 is float"):
        sc.TokenEncoder().fit([float("nan")])
    with pytest.raises(RuntimeError, match="fit"):
        sc.TextVectorizer().transform(["x"])
    with pytest.raises(ValueError, match="min_df"):
        sc.TextVectorizer(min_df=5).fit(["a b", "c d"])
    with pytest.raises(ValueError, match="ngram_range"):
        sc.TextVectorizer(ngram_range=(2, 1))


# ----------------------------------------------------------------------------- TokenEncoder ---
def test_token_encoder_ids_padding_unknown_and_truncation():
    docs = ["red red red green green blue", "red green", "red blue blue"]
    e = sc.TokenEncoder(max_len=4, min_count=2).fit(docs)
    assert e.vocabulary_ == ["red", "blue", "green"]  # by count, ties alphabetical
    out = e.transform(["red green purple", "red red red green green blue", ""])
    assert out.shape == (3, 4) and out.dtype == np.int64
    assert out[0].tolist() == [2, 4, 1, 0]   # known, known, unknown, pad
    assert out[1].tolist() == [2, 2, 2, 4]   # truncated to the first 4 tokens
    assert out[2].tolist() == [0, 0, 0, 0]
    assert out.max() < e.vocab_size_


def test_token_encoder_vocab_comes_from_training_only_and_roundtrips(tmp_path):
    e = sc.TokenEncoder(min_count=2).fit(["a b", "a b", "a c"])
    assert "c" not in e.vocabulary_
    p = tmp_path / "enc.json"
    e.save(p)
    assert np.array_equal(sc.TokenEncoder.load(p).transform(["a b c"]), e.transform(["a b c"]))


# ------------------------------------------------------------------------------- TextModel ---
@pytest.mark.parametrize("name,kw", [("naive_bayes", {}), ("random_forest", {"seed": 0}),
                                     ("mlp", {"seed": 0, "epochs": 30})])
def test_tabular_models_classify_raw_text(name, kw):
    docs, y = corpus()
    trd, try_, ted, tey = split(docs, y)
    m = sc.TextModel(name, vectorizer=sc.TextVectorizer(max_features=100), **kw).fit(trd, try_)
    assert m.mode == "tfidf" and np.mean(m.predict(ted) == tey) > 0.9


@pytest.mark.parametrize("name", ["lstm", "transformer_encoder"])
def test_sequence_models_get_token_ids_automatically(name):
    docs, y = corpus(60)  # a last-state LSTM needs ~125 training documents to clear 0.85 on this task
    trd, try_, ted, tey = split(docs, y)
    m = sc.TextModel(name, hidden=16, epochs=60, seed=0, vectorizer=sc.TokenEncoder(max_len=10)).fit(trd, try_)
    assert m.mode == "tokens" and m.model.vocab_size == m.vectorizer.vocab_size_
    assert np.mean(m.predict(ted) == tey) > 0.85
    assert m.evaluate(ted, tey)["accuracy"] > 0.85


def test_mode_selection_and_errors():
    assert sc.TextModel("mlp").mode == "tfidf" and sc.TextModel("gru").mode == "tokens"
    assert sc.TextModel("ntm").mode == "tokens"  # memory-family sequence models too
    with pytest.raises(ValueError, match="mode='tfidf'"):
        sc.TextModel("mlp", mode="tokens")
    with pytest.raises(ValueError, match="mode='tokens'"):
        sc.TextModel("lstm", mode="tfidf")
    with pytest.raises(TypeError, match="needs a TokenEncoder"):
        sc.TextModel("lstm", vectorizer=sc.TextVectorizer())
    with pytest.raises(ValueError, match="did you mean 'naive_bayes'"):
        sc.TextModel("naive_bayse")
    with pytest.raises(TypeError, match="registry name or a class"):
        sc.TextModel(sc.create("mlp"))


def test_encoder_is_fitted_on_training_texts_only_even_inside_textmodel():
    docs, y = corpus(30)
    m = sc.TextModel("naive_bayes", vectorizer=sc.TextVectorizer(min_df=2)).fit(docs, y)
    held_out = ["leakword leakword leakword"] * 5
    m.predict(held_out)  # transform only; must not change the vocabulary
    assert "leakword" not in m.vectorizer.vocabulary_


def test_fit_validates_and_predict_requires_fit():
    docs, y = corpus(10)
    with pytest.raises(ValueError, match="documents but"):
        sc.TextModel("naive_bayes").fit(docs, y[:-1])
    with pytest.raises(RuntimeError, match="fit"):
        sc.TextModel("mlp").predict(docs)
    with pytest.raises(ValueError, match="smaller than the vocabulary"):
        sc.TextModel("lstm", vocab_size=3).fit(docs, y)


def test_val_data_is_encoded_and_forwarded_only_where_supported():
    docs, y = corpus(40)
    trd, try_, ted, tey = split(docs, y)
    sc.TextModel("mlp", epochs=5, seed=0).fit(trd, try_, val_data=(ted, tey))  # runs
    with pytest.raises(TypeError, match="does not accept val_data"):
        sc.TextModel("naive_bayes").fit(trd, try_, val_data=(ted, tey))


def test_predict_proba_and_float_targets():
    docs, y = corpus(30)
    m = sc.TextModel("mlp", epochs=5, seed=0).fit(docs, y)
    P = m.predict_proba(docs[:7])
    assert P.shape == (7, 3) and np.allclose(P.sum(1), 1, atol=1e-4) and list(m.classes_) == sorted(TOPICS)
    reg = sc.TextModel("mlp", epochs=5, seed=0).fit(docs, np.array([d.count("goal") for d in docs], dtype=float))
    assert reg.predict(docs[:4]).shape[0] == 4


@pytest.mark.parametrize("name,enc,hidden", [("mlp", sc.TextVectorizer(max_features=80, svd=6), (16,)),
                                             ("lstm", sc.TokenEncoder(max_len=10), 16)])
def test_save_load_roundtrip_gives_identical_predictions(tmp_path, name, enc, hidden):
    docs, y = corpus(30)
    m = sc.TextModel(name, hidden=hidden, epochs=8, seed=0, vectorizer=enc).fit(docs, y)
    p = str(tmp_path / "model.pt")
    m.save(p)
    json.loads(open(p + ".text.json").read())
    m2 = sc.TextModel.load(p)
    assert np.array_equal(m.predict(docs), m2.predict(docs))
    assert np.allclose(m.predict_proba(docs[:5]), m2.predict_proba(docs[:5]), atol=1e-5)


def test_models_without_save_say_so(tmp_path):
    docs, y = corpus(10)
    m = sc.TextModel("naive_bayes").fit(docs, y)
    with pytest.raises(NotImplementedError, match="no save"):
        m.save(str(tmp_path / "x.pt"))
