"""Graph, tree and pointer networks (A5).

Node-level tasks: fit(Graph, y, train_mask=...). Graph-level tasks: fit([Graph, ...], y).
Graphs use dense adjacency (A[i, j] = 1 means an edge j -> i, i.e. j sends to i), which suits
graphs up to a few thousand nodes in total.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layers as L
from .base import BaseModel, register
from .optim import make_optimizer


@dataclass
class Graph:
    x: np.ndarray                              # (N, d) node features
    adj: np.ndarray                            # (N, N) adjacency
    rel_adj: Optional[np.ndarray] = None       # (R, N, N) one adjacency per edge type
    incidence: Optional[np.ndarray] = None     # (N, E) hyperedge membership

    @staticmethod
    def from_edges(x, edges, undirected=True, edge_types=None, n_relations=None):
        """edges: (E, 2) pairs (source, target). edge_types: optional (E,) integer types."""
        x, edges = np.asarray(x, dtype=np.float32), np.asarray(edges, dtype=int).reshape(-1, 2)
        N = len(x)
        adj = np.zeros((N, N), np.float32)
        adj[edges[:, 1], edges[:, 0]] = 1
        rel = None
        if edge_types is not None:
            R = n_relations or int(np.max(edge_types)) + 1
            rel = np.zeros((R, N, N), np.float32)
            rel[np.asarray(edge_types), edges[:, 1], edges[:, 0]] = 1
        if undirected:
            adj = np.maximum(adj, adj.T)
            if rel is not None:
                rel = np.maximum(rel, rel.transpose(0, 2, 1))
        return Graph(x, adj, rel)


class _Batch:
    """Block-diagonal batch of graphs as tensors."""

    def __init__(self, graphs, device):
        sizes = [g.x.shape[0] for g in graphs]
        N, G = sum(sizes), len(graphs)
        off = np.concatenate([[0], np.cumsum(sizes)])
        self.x = torch.as_tensor(np.concatenate([g.x for g in graphs]), dtype=torch.float32, device=device)
        adj = np.zeros((N, N), np.float32)
        R = max([g.rel_adj.shape[0] for g in graphs if g.rel_adj is not None], default=0)
        rel = np.zeros((R, N, N), np.float32) if R else None
        inc_cols, cols = [], 0
        for g, o, n in zip(graphs, off, sizes):
            adj[o:o + n, o:o + n] = g.adj
            if g.rel_adj is not None:
                rel[:g.rel_adj.shape[0], o:o + n, o:o + n] = g.rel_adj
            if g.incidence is not None:
                inc_cols.append((o, n, g.incidence)); cols += g.incidence.shape[1]
        self.adj = torch.as_tensor(adj, device=device)
        self.rel = None if rel is None else torch.as_tensor(rel, device=device)
        self.inc = None
        if inc_cols:
            inc = np.zeros((N, cols), np.float32); c = 0
            for o, n, H in inc_cols:
                inc[o:o + n, c:c + H.shape[1]] = H; c += H.shape[1]
            self.inc = torch.as_tensor(inc, device=device)
        self.gidx = torch.as_tensor(np.repeat(np.arange(G), sizes), device=device)
        self.G, self.N = G, N
        member = (self.gidx[None, :] == torch.arange(G, device=device)[:, None]).float()
        self.pool_sum, self.pool_mean = member, member / member.sum(1, keepdim=True)
        self.same = self.gidx[:, None] == self.gidx[None, :]
        src, dst = torch.nonzero(self.adj, as_tuple=True)[1], torch.nonzero(self.adj, as_tuple=True)[0]
        self.src, self.dst = src, dst                    # edge j -> i  (src=j, dst=i)
        self._gcn = None

    def with_x(self, x):
        self.x = x
        return self

    @property
    def gcn_adj(self):
        if self._gcn is None:
            A = self.adj + torch.eye(self.N, device=self.adj.device)
            d = A.sum(1).pow(-0.5)
            self._gcn = d[:, None] * A * d[None, :]
        return self._gcn


# ---------------------------------------------------------------------- layers ---
class GCNLayer(nn.Module):
    def __init__(self, h, **kw):
        super().__init__()
        self.W = nn.Linear(h, h)

    def forward(self, h, b):
        return b.gcn_adj @ self.W(h)


class GATLayer(nn.Module):
    def __init__(self, h, heads=2, **kw):
        super().__init__()
        self.H, self.dh = heads, h // heads
        self.W = nn.Linear(h, h, bias=False)
        self.a_src, self.a_dst = nn.Parameter(torch.randn(heads, self.dh) * 0.1), nn.Parameter(torch.randn(heads, self.dh) * 0.1)

    def forward(self, h, b):
        Wh = self.W(h).view(-1, self.H, self.dh)
        e = F.leaky_relu((Wh * self.a_src).sum(-1)[:, None, :] + (Wh * self.a_dst).sum(-1)[None, :, :], 0.2)  # (N,N,H)
        mask = (b.adj + torch.eye(b.N, device=h.device)) > 0
        alpha = F.softmax(e.masked_fill(~mask[:, :, None], float("-inf")), dim=1)
        return torch.einsum("ijh,jhd->ihd", alpha, Wh).reshape(-1, self.H * self.dh)


class SAGELayer(nn.Module):
    def __init__(self, h, **kw):
        super().__init__()
        self.self_w, self.nbr_w = nn.Linear(h, h), nn.Linear(h, h, bias=False)

    def forward(self, h, b):
        mean = (b.adj @ h) / b.adj.sum(1, keepdim=True).clamp(min=1)
        return self.self_w(h) + self.nbr_w(mean)


class GINLayer(nn.Module):
    def __init__(self, h, **kw):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))

    def forward(self, h, b):
        return self.mlp((1 + self.eps) * h + b.adj @ h)


class MPNNLayer(nn.Module):
    """Message passing with learned edge messages m_ij = MLP([h_i, h_j]) and a GRU update."""

    def __init__(self, h, **kw):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, h))
        self.upd = nn.GRUCell(h, h)

    def forward(self, h, b):
        m = self.msg(torch.cat([h[b.dst], h[b.src]], -1))
        agg = torch.zeros_like(h).index_add_(0, b.dst, m)
        return self.upd(agg, h)


class RGCNLayer(nn.Module):
    """Relational GCN: one weight matrix per edge type (mean-normalised) plus a self-loop."""

    def __init__(self, h, n_relations=1, **kw):
        super().__init__()
        self.self_w = nn.Linear(h, h)
        self.rel_w = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(n_relations)])

    def forward(self, h, b):
        A = b.rel if b.rel is not None else b.adj[None]
        out = self.self_w(h)
        for r, W in enumerate(self.rel_w):
            Ar = A[r] / A[r].sum(1, keepdim=True).clamp(min=1)
            out = out + Ar @ W(h)
        return out


class GraphTransformerLayer(nn.Module):
    """Attention among all nodes of the same graph; edges add a learned bias to the scores."""

    def __init__(self, h, heads=2, **kw):
        super().__init__()
        self.H, self.dh = heads, h // heads
        self.qkv, self.o = nn.Linear(h, 3 * h), nn.Linear(h, h)
        self.edge_bias = nn.Parameter(torch.full((heads,), 3.0))  # start by favouring graph neighbours
        self.ln1, self.ln2 = nn.LayerNorm(h), nn.LayerNorm(h)
        self.ff = nn.Sequential(nn.Linear(h, 2 * h), nn.GELU(), nn.Linear(2 * h, h))

    def forward(self, h, b):
        q, k, v = (t.view(-1, self.H, self.dh) for t in self.qkv(self.ln1(h)).chunk(3, -1))
        s = torch.einsum("ihd,jhd->hij", q, k) / math.sqrt(self.dh)
        s = s + self.edge_bias[:, None, None] * b.adj[None]
        s = s.masked_fill(~b.same[None], float("-inf"))
        h = h + self.o(torch.einsum("hij,jhd->ihd", F.softmax(s, dim=-1), v).reshape(-1, self.H * self.dh))
        return self.ff(self.ln2(h))  # the network wrapper adds the residual connection


class GatedGraphLayer(nn.Module):
    """Gated graph neural network: `steps` rounds of (message = A W h, h = GRU(message, h))
    with weights shared across rounds. With edge types, each type has its own message matrix."""

    def __init__(self, h, steps=3, n_relations=0, **kw):
        super().__init__()
        self.steps, self.typed = steps, n_relations > 0
        self.msg = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(max(n_relations, 1))])
        self.gru = nn.GRUCell(h, h)

    def forward(self, h, b):
        A = b.rel if (self.typed and b.rel is not None) else b.adj[None]
        for _ in range(self.steps):
            m = sum((A[r] / A[r].sum(1, keepdim=True).clamp(min=1)) @ W(h) for r, W in enumerate(self.msg))
            h = self.gru(m, h)
        return h


class RecurrentGNLayer(nn.Module):
    """Graph-network block with a global state (Battaglia et al., 2018), applied recurrently
    with shared weights: edge update -> aggregate -> node GRU update -> global update."""

    def __init__(self, h, steps=3, **kw):
        super().__init__()
        self.steps = steps
        self.edge = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, h))
        self.node = nn.GRUCell(2 * h, h)
        self.glob = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, h))

    def forward(self, h, b):
        u = b.pool_mean @ h
        for _ in range(self.steps):
            e = self.edge(torch.cat([h[b.dst], h[b.src], u[b.gidx[b.dst]]], -1))
            agg = torch.zeros_like(h).index_add_(0, b.dst, e)
            h = self.node(torch.cat([agg, u[b.gidx]], -1), h)
            u = self.glob(torch.cat([u, b.pool_mean @ h], -1))
        return h


class HypergraphLayer(nn.Module):
    """Hypergraph convolution: X' = Dv^-1/2 H De^-1 H^T Dv^-1/2 X Theta. Without explicit
    hyperedges, every node's closed neighbourhood is used as a hyperedge."""

    def __init__(self, h, **kw):
        super().__init__()
        self.W = nn.Linear(h, h)

    def forward(self, h, b):
        H = b.inc if b.inc is not None else ((b.adj + torch.eye(b.N, device=h.device)) > 0).float()
        dv, de = H.sum(1).clamp(min=1), H.sum(0).clamp(min=1)
        x = self.W(h) * dv.pow(-0.5)[:, None]
        return dv.pow(-0.5)[:, None] * (H @ ((H.T @ x) / de[:, None]))


class GNNNet(nn.Module):
    def __init__(self, d_in, hidden, d_out, layers, level, pool):
        super().__init__()
        self.embed, self.layers = nn.Linear(d_in, hidden), nn.ModuleList(layers)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in layers])
        self.head, self.level, self.pool = nn.Linear(hidden, d_out), level, pool

    def forward(self, b):
        h = F.relu(self.embed(b.x))
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + F.relu(layer(h, b)))
        if self.level == "graph":
            h = (b.pool_sum if self.pool == "sum" else b.pool_mean) @ h
        return self.head(h)


# ---------------------------------------------------------------------- models ---
class GraphModel(BaseModel):
    family, layer_cls = "graph", GCNLayer
    layer_kwargs: dict = {}

    def __init__(self, hidden=32, n_layers=2, pool="sum", heads=2, steps=3, **kw):
        super().__init__(**kw)
        self.hidden, self.n_layers, self.pool, self.heads, self.steps = hidden, n_layers, pool, heads, steps
        self.level, self.n_relations = None, 0

    def make_layer(self):
        return self.layer_cls(self.hidden, heads=self.heads, steps=self.steps, n_relations=self.n_relations)

    def build(self, d_in, d_out):
        return GNNNet(d_in, self.hidden, d_out, [self.make_layer() for _ in range(self.n_layers)], self.level, self.pool)

    def _graphs(self, graphs):
        return [graphs] if isinstance(graphs, Graph) else list(graphs)

    def _prep(self, graphs):
        gs = self._graphs(graphs)
        b = _Batch(gs, self.device)
        if self._mu is not None:
            b.x = (b.x - torch.as_tensor(self._mu, device=self.device)) / torch.as_tensor(self._sd, device=self.device)
        return b

    def fit(self, graphs, y, train_mask=None, epochs=None, val_data=None):
        epochs, gs = epochs or self.epochs, self._graphs(graphs)
        y = np.asarray(y)
        if self._module is None:
            self.level = "node" if isinstance(graphs, Graph) else "graph"
            self.task = "classification" if (self.task == "auto" and y.dtype.kind in "biuOUS") or self.task == "classification" else "regression"
            allx = np.concatenate([g.x for g in gs]).astype(np.float32)
            if self.normalize:
                self._mu, self._sd = allx.mean(0), allx.std(0)
                self._sd[self._sd < 1e-8] = 1.0
            self.input_dim = allx.shape[1]
            self.n_relations = max([g.rel_adj.shape[0] for g in gs if g.rel_adj is not None], default=0)
            if self.task == "classification":
                self.classes_ = self._declared_classes if self._declared_classes is not None else np.unique(y)
                self.output_dim = len(self.classes_)
            else:
                self.output_dim = 1 if y.ndim == 1 else y.shape[-1]
            self._module = self.build(self.input_dim, self.output_dim).to(self.device)
        m, b, yt = self._module, self._prep(graphs), self._encode(y)
        mask = None if train_mask is None else torch.as_tensor(np.asarray(train_mask), dtype=torch.bool, device=self.device)
        opt = make_optimizer(self.optimizer, m.parameters(), self.lr, self.weight_decay, module=m, **self.optimizer_kwargs)
        for _ in range(epochs):
            m.train(); opt.zero_grad()
            out = m(b)
            loss = self._loss(out if mask is None else out[mask], yt if mask is None else yt[mask])
            loss.backward(); opt.step()
            self.history_["loss"].append(loss.item())
        return self

    def _out(self, graphs):
        self._check_fitted()
        self._module.eval()
        with torch.no_grad():
            return self._module(self._prep(graphs))

    def predict_proba(self, graphs):
        return F.softmax(self._out(graphs), dim=-1).cpu().numpy()

    def predict(self, graphs):
        out = self._out(graphs)
        if self.task == "classification":
            return self.classes_[out.argmax(-1).cpu().numpy()]
        return out.squeeze(-1).cpu().numpy() if out.shape[-1] == 1 else out.cpu().numpy()

    def evaluate(self, graphs, y, mask=None):
        out, yt = self._out(graphs), self._encode(np.asarray(y))
        if mask is not None:
            m = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=self.device)
            out, yt = out[m], yt[m]
        res = {"loss": self._loss(out, yt).item()}
        if self.task == "classification":
            res["accuracy"] = (out.argmax(-1) == yt).float().mean().item()
        return res


@register("gcn", family="graph")
class GraphConvolutionalNetwork(GraphModel):
    """Graph Convolutional Network (Kipf & Welling): symmetric-normalised neighbourhood averaging."""
    layer_cls = GCNLayer


@register("gat", family="graph")
class GraphAttentionNetwork(GraphModel):
    """Graph Attention Network: learned attention weights over each node's neighbours."""
    layer_cls = GATLayer


@register("graphsage", family="graph")
class GraphSAGE(GraphModel):
    """GraphSAGE with mean aggregation (separate self and neighbour weights)."""
    layer_cls = SAGELayer


@register("gin", family="graph")
class GraphIsomorphismNetwork(GraphModel):
    """Graph Isomorphism Network: sum aggregation + MLP, as expressive as 1-WL."""
    layer_cls = GINLayer


@register("mpnn", family="graph")
class MessagePassingNetwork(GraphModel):
    """Message-passing neural network with learned edge messages and a GRU node update."""
    layer_cls = MPNNLayer


@register("rgcn", family="graph")
class RelationalGCN(GraphModel):
    """Relational GCN: separate weights per edge type (Graph.rel_adj)."""
    layer_cls = RGCNLayer


@register("graph_transformer", family="graph")
class GraphTransformer(GraphModel):
    layer_cls = GraphTransformerLayer


@register("ggnn", family="graph")
class GatedGraphNetwork(GraphModel):
    """Gated graph neural network (GRU updates, weights shared over propagation steps)."""
    layer_cls = GatedGraphLayer

    def __init__(self, n_layers=1, **kw):
        super().__init__(n_layers=n_layers, **kw)


@register("graph_net_recurrent", family="graph")
class RecurrentGraphNetwork(GraphModel):
    """Graph network with global state and recurrent (weight-shared) updates."""
    layer_cls = RecurrentGNLayer

    def __init__(self, n_layers=1, **kw):
        super().__init__(n_layers=n_layers, **kw)


@register("hypergraph", family="graph")
class HypergraphNeuralNetwork(GraphModel):
    """Hypergraph convolution; pass Graph(..., incidence=H) or use neighbourhoods as hyperedges."""
    layer_cls = HypergraphLayer


@register("program_graph", family="graph")
class ProgramGraphNetwork(GraphModel):
    """Neural program-graph network: a gated GNN with a separate message matrix per edge type
    (syntax child, next-token, data flow, control flow, ... and their reverses) for graphs
    built from source code. Give node types as features and typed edges via Graph.rel_adj."""
    layer_cls = GatedGraphLayer

    def __init__(self, n_layers=1, steps=4, **kw):
        super().__init__(n_layers=n_layers, steps=steps, **kw)


# ---------------------------------------------------------------------- trees ---
@dataclass
class Tree:
    x: np.ndarray                                   # (N, d) node features
    children: List[List[int]] = field(default_factory=list)  # children[i] = child node ids
    root: int = 0

    def order(self):
        out, stack = [], [(self.root, False)]
        while stack:  # post-order: children before parents
            n, seen = stack.pop()
            if seen:
                out.append(n)
            else:
                stack.append((n, True))
                stack.extend((c, False) for c in self.children[n])
        return out


class TreeLSTMCell(nn.Module):
    """Child-sum Tree-LSTM (Tai et al., 2015): one forget gate per child."""

    def __init__(self, d, h):
        super().__init__()
        self.iou, self.iou_h = nn.Linear(d, 3 * h), nn.Linear(h, 3 * h, bias=False)
        self.f, self.f_h = nn.Linear(d, h), nn.Linear(h, h, bias=False)

    def forward(self, x, hs, cs):
        h_sum = hs.sum(0) if len(hs) else torch.zeros_like(self.f.bias)
        i, o, u = (self.iou(x) + self.iou_h(h_sum)).chunk(3)
        c = torch.sigmoid(i) * torch.tanh(u)
        if len(hs):
            c = c + (torch.sigmoid(self.f(x) + self.f_h(hs)) * cs).sum(0)
        return torch.sigmoid(o) * torch.tanh(c), c


class RecursiveCell(nn.Module):
    """Plain recursive neural network node: h = tanh(W x + U sum(children) + b)."""

    def __init__(self, d, h):
        super().__init__()
        self.W, self.U = nn.Linear(d, h), nn.Linear(h, h, bias=False)

    def forward(self, x, hs, cs):
        s = hs.sum(0) if len(hs) else torch.zeros_like(self.W.bias)
        h = torch.tanh(self.W(x) + self.U(s))
        return h, h


class TreeNet(nn.Module):
    def __init__(self, d_in, hidden, d_out, kind):
        super().__init__()
        self.cell = (TreeLSTMCell if kind == "lstm" else RecursiveCell)(d_in, hidden)
        self.head = nn.Linear(hidden, d_out)

    def forward(self, trees):
        roots = []
        for x, children, order, root in trees:
            H, C = {}, {}
            for n in order:
                ch = children[n]
                hs = torch.stack([H[c] for c in ch]) if ch else x.new_zeros(0, self.head.in_features)
                cs = torch.stack([C[c] for c in ch]) if ch else x.new_zeros(0, self.head.in_features)
                H[n], C[n] = self.cell(x[n], hs, cs)
            roots.append(H[root])
        return self.head(torch.stack(roots))


class _TreeModel(BaseModel):
    family, kind = "graph", "lstm"

    def __init__(self, hidden=32, **kw):
        super().__init__(**kw)
        self.hidden = hidden

    def build(self, d_in, d_out):
        return TreeNet(d_in, self.hidden, d_out, self.kind)

    def _prep(self, trees):
        return [(torch.as_tensor((np.asarray(t.x, np.float32) - (self._mu if self._mu is not None else 0)) /
                                 (self._sd if self._sd is not None else 1), dtype=torch.float32, device=self.device),
                 t.children, t.order(), t.root) for t in trees]

    def fit(self, trees, y, epochs=None, val_data=None):
        y = np.asarray(y)
        if self._module is None:
            allx = np.concatenate([np.asarray(t.x, np.float32) for t in trees])
            if self.normalize:
                self._mu, self._sd = allx.mean(0), allx.std(0)
                self._sd[self._sd < 1e-8] = 1.0
            self.task = "classification" if y.dtype.kind in "biuOUS" else "regression"
            self.input_dim = allx.shape[1]
            if self.task == "classification":
                self.classes_ = np.unique(y); self.output_dim = len(self.classes_)
            else:
                self.output_dim = 1 if y.ndim == 1 else y.shape[-1]
            self._module = self.build(self.input_dim, self.output_dim).to(self.device)
        m, data, yt = self._module, self._prep(trees), self._encode(y)
        opt = make_optimizer(self.optimizer, m.parameters(), self.lr, self.weight_decay, module=m)
        for _ in range(epochs or self.epochs):
            m.train(); total = 0.0
            for idx in torch.randperm(len(data)).split(self.batch_size):
                opt.zero_grad()
                loss = self._loss(m([data[i] for i in idx]), yt[idx])
                loss.backward(); opt.step(); total += loss.item() * len(idx)
            self.history_["loss"].append(total / len(data))
        return self

    def _out(self, trees):
        self._check_fitted(); self._module.eval()
        with torch.no_grad():
            return self._module(self._prep(trees))

    def predict(self, trees):
        out = self._out(trees)
        return self.classes_[out.argmax(-1).cpu().numpy()] if self.task == "classification" else out.squeeze(-1).cpu().numpy()

    def evaluate(self, trees, y):
        out, yt = self._out(trees), self._encode(np.asarray(y))
        res = {"loss": self._loss(out, yt).item()}
        if self.task == "classification":
            res["accuracy"] = (out.argmax(-1) == yt).float().mean().item()
        return res


@register("tree_lstm", family="graph")
class TreeLSTM(_TreeModel):
    kind = "lstm"


@register("tree_nn", family="graph")
class TreeStructuredNetwork(_TreeModel):
    """Recursive (tree-structured) neural network."""
    kind = "recursive"


# ------------------------------------------------------------- pointer networks ---
class PointerNet(nn.Module):
    """Encoder-decoder whose 'output vocabulary' is the set of input positions: each decoding
    step points at one input element (Vinyals et al., 2015)."""

    def __init__(self, d_in, hidden):
        super().__init__()
        self.enc, self.dec = nn.LSTM(d_in, hidden, batch_first=True), nn.LSTMCell(d_in, hidden)
        self.W1, self.W2, self.v = nn.Linear(hidden, hidden, bias=False), nn.Linear(hidden, hidden, bias=False), nn.Linear(hidden, 1, bias=False)
        self.start = nn.Parameter(torch.zeros(d_in))
        self.hidden = hidden

    def forward(self, x, tgt=None):
        B, T, _ = x.shape
        e, (h, c) = self.enc(x)
        h, c = h[0], c[0]
        inp, visited, logits, picks = self.start.expand(B, -1), torch.zeros(B, T, dtype=torch.bool, device=x.device), [], []
        for t in range(T):
            h, c = self.dec(inp, (h, c))
            u = self.v(torch.tanh(self.W1(e) + self.W2(h).unsqueeze(1))).squeeze(-1)
            logits.append(u)
            idx = tgt[:, t] if tgt is not None else u.masked_fill(visited, float("-inf")).argmax(-1)
            visited = visited | F.one_hot(idx, T).bool()
            picks.append(idx)
            inp = x[torch.arange(B), idx]
        return torch.stack(logits, 1), torch.stack(picks, 1)


@register("pointer_network", family="graph")
class PointerNetwork(BaseModel):
    """fit(X, y): X is (n, T, d) set/sequence elements, y is (n, T) the order to point at
    (e.g. argsort for sorting). predict returns pointer indices decoded without repeats."""
    family = "graph"

    def __init__(self, hidden=64, **kw):
        super().__init__(**kw)
        self.hidden = hidden

    def build(self, d_in, d_out):
        return PointerNet(d_in, self.hidden)

    def fit(self, X, y, epochs=None, val_data=None):
        self.task, self._declared_classes = "classification", np.arange(np.asarray(X).shape[1])
        return super().fit(X, y, epochs=epochs)

    def _as_x(self, X):
        return np.asarray(X, dtype=np.float32)

    def _train_batch(self, m, opt, xb, yb):
        opt.zero_grad()
        loss = self._loss(m(xb, yb)[0], yb)
        loss.backward(); opt.step()
        return loss.item()

    def predict(self, X):
        self._check_fitted(); self._module.eval()
        with torch.no_grad():
            return self._module(self._norm(self._as_x(X)))[1].cpu().numpy()

    def evaluate(self, X, y):
        pred, y = self.predict(X), np.asarray(y)
        return {"accuracy": float((pred == y).mean()), "sequence_accuracy": float((pred == y).all(1).mean())}


class PointerGenNet(nn.Module):
    """Pointer-generator (See et al., 2017): each output token mixes a generation
    distribution over the vocabulary with a copy distribution over the source tokens."""

    def __init__(self, vocab, hidden):
        super().__init__()
        self.V, self.hidden = vocab, hidden
        self.emb = nn.Embedding(vocab + 1, hidden)                  # +1 for the start token
        self.enc = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * hidden, hidden)
        self.cell = nn.GRUCell(hidden + hidden, hidden)
        self.att_k, self.att_q, self.att_v = nn.Linear(hidden, hidden, bias=False), nn.Linear(hidden, hidden, bias=False), nn.Linear(hidden, 1, bias=False)
        self.out = nn.Linear(2 * hidden, vocab)
        self.gate = nn.Linear(3 * hidden, 1)

    def forward(self, src, length, tgt=None):
        B, S = src.shape
        enc = self.proj(self.enc(self.emb(src))[0])                  # (B, S, h)
        h, prev, probs, outs = enc.mean(1), torch.full((B,), self.V, device=src.device), [], []
        for t in range(length):
            e = self.emb(prev)
            att = F.softmax(self.att_v(torch.tanh(self.att_k(enc) + self.att_q(h).unsqueeze(1))).squeeze(-1), dim=1)
            ctx = (att.unsqueeze(-1) * enc).sum(1)
            h = self.cell(torch.cat([e, ctx], -1), h)
            p_vocab = F.softmax(self.out(torch.cat([h, ctx], -1)), dim=-1)
            p_gen = torch.sigmoid(self.gate(torch.cat([h, ctx, e], -1)))
            p_copy = torch.zeros(B, self.V, device=src.device).scatter_add(1, src, att)
            p = p_gen * p_vocab + (1 - p_gen) * p_copy
            probs.append(p)
            prev = tgt[:, t] if tgt is not None else p.argmax(-1)
            outs.append(prev)
        return torch.stack(probs, 1), torch.stack(outs, 1)


@register("pointer_generator", family="graph")
class PointerGenerator(BaseModel):
    """Sequence-to-sequence over integer tokens with copying: fit(src, tgt), predict(src)."""
    family = "graph"

    def __init__(self, vocab_size, hidden=64, **kw):
        kw["normalize"] = False
        super().__init__(**kw)
        self.vocab_size, self.hidden, self.tgt_len = vocab_size, hidden, None

    def build(self, d_in, d_out):
        return PointerGenNet(self.vocab_size, self.hidden)

    def _as_x(self, X):
        return np.asarray(X, dtype=np.int64)

    def _norm(self, X):
        return torch.as_tensor(X).long().to(self.device)

    def fit(self, X, y, epochs=None, val_data=None):
        y = np.asarray(y)
        self.tgt_len, self.task, self._declared_classes = y.shape[1], "classification", np.arange(self.vocab_size)
        return super().fit(X, y, epochs=epochs)

    def _train_batch(self, m, opt, xb, yb):
        opt.zero_grad()
        probs = m(xb, yb.shape[1], yb)[0]
        loss = F.nll_loss(probs.clamp(min=1e-9).log().reshape(-1, self.vocab_size), yb.reshape(-1))
        loss.backward(); opt.step()
        return loss.item()

    def predict(self, X):
        self._check_fitted(); self._module.eval()
        with torch.no_grad():
            return self._module(self._norm(self._as_x(X)), self.tgt_len)[1].cpu().numpy()

    def evaluate(self, X, y):
        pred, y = self.predict(X), np.asarray(y)
        return {"token_accuracy": float((pred == y).mean()), "sequence_accuracy": float((pred == y).all(1).mean())}
