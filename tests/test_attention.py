import numpy as np
import pytest
import torch

import saturncube as sc
from saturncube.attention import MultiHeadAttention, LinearAttention, TransformerLayer, make_mask
from test_sequence import Xtr, ytr, Xte, yte, tokens, V, perm

ATTN = ["transformer_encoder", "transformer_decoder", "linear_attention", "local_attention", "sparse_attention",
        "multi_query", "grouped_query", "universal_transformer", "memory_transformer", "perceiver",
        "ssm_attention_hybrid", "recurrent_attention"]


@pytest.mark.parametrize("name", ATTN)
def test_attention_models_classify(name):
    m = sc.create(name, epochs=30, lr=5e-3, seed=0).fit(Xtr, ytr)
    assert m.evaluate(Xte, yte)["accuracy"] > 0.8, name


@pytest.mark.parametrize("name,kw", [("transformer_decoder", {}), ("universal_transformer", {"causal": True}),
                                     ("memory_transformer", {"causal": True}), ("ssm_attention_hybrid", {"causal": True}),
                                     ("linear_attention", {"causal": True}), ("local_attention", {"causal": True, "window": 2})])
def test_causal_language_models(name, kw):
    m = sc.create(name, vocab_size=V, epochs=50, lr=1e-2, seed=0, **kw).fit(tokens(300))
    assert m.evaluate(tokens(100, seed=5))["accuracy"] > 0.9, name


def test_head_variants_share_kv_heads():
    x = torch.randn(2, 5, 16)
    for kv in (4, 2, 1):
        mha = MultiHeadAttention(16, 4, kv)
        assert mha(x).shape == x.shape and mha.k.out_features == kv * 4


def test_cross_attention_uses_context():
    mha, x = MultiHeadAttention(16, 4), torch.randn(2, 3, 16)
    c1, c2 = torch.randn(2, 6, 16), torch.randn(2, 6, 16)
    assert not torch.allclose(mha(x, context=c1), mha(x, context=c2))


@pytest.mark.parametrize("make", [lambda: TransformerLayer(16, 4, causal=True),
                                  lambda: TransformerLayer(16, 4, attention="linear", causal=True)])
def test_causality(make):
    layer = make().eval()
    x = torch.randn(1, 8, 16)
    x2 = x.clone(); x2[:, 5:] = torch.randn(1, 3, 16)       # change only the future
    assert torch.allclose(layer(x)[:, :5], layer(x2)[:, :5], atol=1e-5)
    assert not torch.allclose(layer(x)[:, 5:], layer(x2)[:, 5:], atol=1e-3)   # the perturbation is visible


def test_local_window_limits_receptive_field():
    layer = TransformerLayer(16, 4, attention="local", window=1).eval()
    x = torch.randn(1, 8, 16)
    x2 = x.clone(); x2[:, 6:] = torch.randn(1, 2, 16)
    assert torch.allclose(layer(x)[:, :4], layer(x2)[:, :4], atol=1e-5)   # positions 0..3 never see 6..7
    assert not torch.allclose(layer(x)[:, 5], layer(x2)[:, 5], atol=1e-5)
    m = make_mask(8, window=1, stride=4)
    assert m[0, 4] and not m[0, 3]                                          # strided link outside the window


def test_encoder_decoder_reverses_sequences():
    r = np.random.default_rng(0)
    src = r.integers(6, size=(800, 5)); tgt = src[:, ::-1].copy()
    m = sc.EncoderDecoderTransformer(vocab_size=6, hidden=48, epochs=80, lr=3e-3, seed=0).fit(src[:700], tgt[:700])
    assert m.evaluate(src[700:], tgt[700:])["token_accuracy"] > 0.9


def test_retrieval_augmented_reads_knowledge_base():
    r = np.random.default_rng(0)
    keys = r.normal(size=(30, 8)); cls = r.integers(4, size=30)
    values = np.eye(4)[cls]                                # the labels live ONLY in the knowledge base
    idx = r.integers(30, size=600)
    X = (keys[idx][:, None, :] + 0.15 * r.normal(size=(600, 6, 8))).astype(np.float32)
    y = cls[idx]
    m = sc.RetrievalAugmented(knowledge=(keys, values), epochs=40, lr=5e-3, seed=0).fit(X[:500], y[:500])
    assert m.evaluate(X[500:], y[500:])["accuracy"] > 0.9
