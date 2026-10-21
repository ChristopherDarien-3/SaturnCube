"""Deep reinforcement learning (B5): value-based, policy-gradient, actor-critic, trust-region,
maximum-entropy, model-based and preference-based methods.

Environment protocol: reset() -> obs (float vector); step(a) -> (obs, reward, done).
Discrete envs expose n_actions and obs_dim; continuous envs expose action_dim, action_high and
obs_dim. Wrap integer-state envs with saturncube.OneHotObs.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from . import layers as L
from .base import register


# ----------------------------------------------------------------------- utilities ---
class ReplayBuffer:
    """Ring buffer; prioritized=True samples proportionally to |TD error|^alpha with
    importance-sampling weights."""

    def __init__(self, capacity, obs_dim, act_dim=None, prioritized=False, alpha=0.6):
        self.o, self.o2 = np.zeros((capacity, obs_dim), np.float32), np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros((capacity,) if act_dim is None else (capacity, act_dim), np.float32)
        self.r, self.d, self.p = np.zeros(capacity, np.float32), np.zeros(capacity, np.float32), np.ones(capacity)
        self.cap, self.n, self.i, self.prioritized, self.alpha = capacity, 0, 0, prioritized, alpha

    def add(self, o, a, r, o2, d):
        i = self.i
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.p[i] = self.p[:self.n].max() if self.n else 1.0
        self.i, self.n = (i + 1) % self.cap, min(self.n + 1, self.cap)

    def sample(self, bs, rng, beta=0.4):
        if self.prioritized:
            pr = self.p[:self.n] ** self.alpha
            pr = pr / pr.sum()
            idx = rng.choice(self.n, bs, p=pr)
            w = (self.n * pr[idx]) ** -beta
            w = w / w.max()
        else:
            idx, w = rng.integers(self.n, size=bs), np.ones(bs)
        return idx, (self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx], w.astype(np.float32))

    def update(self, idx, td):
        self.p[idx] = np.abs(td) + 1e-3


def gae(rew, val, done, last_val, gamma, lam):
    adv, last = np.zeros(len(rew), np.float32), 0.0
    for t in reversed(range(len(rew))):
        nxt = last_val if t == len(rew) - 1 else val[t + 1]
        delta = rew[t] + gamma * nxt * (1 - done[t]) - val[t]
        last = delta + gamma * lam * (1 - done[t]) * last
        adv[t] = last
    return adv, adv + np.asarray(val, np.float32)


class DeepAgent:
    family = "agent"

    def __init__(self, hidden=(64, 64), gamma=0.99, lr=1e-3, seed=None, device=None):
        self.hidden, self.gamma, self.lr = tuple(hidden), gamma, lr
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.returns_ = []

    def _spec(self, env):
        self.discrete, self.obs_dim = hasattr(env, "n_actions"), env.obs_dim
        if self.discrete:
            self.n_actions = env.n_actions
        else:
            self.act_dim, self.act_high = env.action_dim, env.action_high

    def _t(self, x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    def evaluate(self, env, episodes=10):
        total = 0.0
        for _ in range(episodes):
            o, done = env.reset(), False
            while not done:
                o, r, done = env.step(self.act(o, greedy=True))
                total += r
        return total / episodes


# ------------------------------------------------------------------------------ DQN ---
class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden, dueling):
        super().__init__()
        self.dueling = dueling
        if dueling:  # shared trunk, separate value and advantage streams
            self.trunk = L.MLPNet(obs_dim, hidden[-1], hidden[:-1], "relu")
            self.v, self.a = nn.Linear(hidden[-1], 1), nn.Linear(hidden[-1], n_actions)
        else:
            self.net = L.MLPNet(obs_dim, n_actions, hidden, "relu")

    def forward(self, x):
        if not self.dueling:
            return self.net(x)
        h = F.relu(self.trunk(x))
        a = self.a(h)
        return self.v(h) + a - a.mean(1, keepdim=True)


@register("dqn", family="agent")
class DQN(DeepAgent):
    """Deep Q-Network (replay + target network). Flags: double=True (Double DQN: online net
    selects, target net evaluates), dueling=True (Dueling DQN), prioritized=True (PER)."""
    double = dueling = prioritized = False

    def __init__(self, double=None, dueling=None, prioritized=None, buffer_size=20000, batch_size=64,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=1500, target_update=100, learning_starts=200,
                 lr=1e-3, **kw):
        super().__init__(lr=lr, **kw)
        for k, v in (("double", double), ("dueling", dueling), ("prioritized", prioritized)):
            if v is not None:
                setattr(self, k, v)
        self.buffer_size, self.batch_size, self.target_update, self.learning_starts = buffer_size, batch_size, target_update, learning_starts
        self.eps_start, self.eps_end, self.eps_decay, self.steps = eps_start, eps_end, eps_decay_steps, 0

    def _eps(self):
        return max(self.eps_end, self.eps_start - (self.eps_start - self.eps_end) * self.steps / self.eps_decay)

    def act(self, obs, greedy=False):
        if not greedy and self.rng.random() < self._eps():
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            return int(self.q(self._t(obs).unsqueeze(0)).argmax(1))

    def _learn(self):
        idx, (o, a, r, o2, d, w) = self.buf.sample(self.batch_size, self.rng)
        o, o2, r, d, w, a = map(self._t, (o, o2, r, d, w, a))
        q = self.q(o).gather(1, a.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nxt = self.q(o2).argmax(1, keepdim=True) if self.double else self.target(o2).argmax(1, keepdim=True)
            y = r + self.gamma * (1 - d) * self.target(o2).gather(1, nxt).squeeze(1)
        td = y - q
        loss = (w * F.smooth_l1_loss(q, y, reduction="none")).mean()
        self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.q.parameters(), 10.0); self.opt.step()
        if self.prioritized:
            self.buf.update(idx, td.detach().cpu().numpy())

    def train(self, env, episodes=200):
        self._spec(env)
        if not hasattr(self, "q"):
            self.q = QNet(self.obs_dim, self.n_actions, self.hidden, self.dueling).to(self.device)
            self.target = copy.deepcopy(self.q)
            self.opt = torch.optim.Adam(self.q.parameters(), lr=self.lr)
            self.buf = ReplayBuffer(self.buffer_size, self.obs_dim, None, self.prioritized)
        for _ in range(episodes):
            o, done, G = env.reset(), False, 0.0
            while not done:
                a = self.act(o)
                o2, r, done = env.step(a)
                self.buf.add(o, a, r, o2, float(done))
                o, G, self.steps = o2, G + r, self.steps + 1
                if self.buf.n >= self.learning_starts:
                    self._learn()
                if self.steps % self.target_update == 0:
                    self.target.load_state_dict(self.q.state_dict())
            self.returns_.append(G)
        return self.returns_


@register("double_dqn", family="agent")
class DoubleDQN(DQN):
    double = True


@register("dueling_dqn", family="agent")
class DuelingDQN(DQN):
    dueling = True


@register("prioritized_dqn", family="agent")
class PrioritizedDQN(DQN):
    prioritized = True


# ------------------------------------------------------------------- policy gradient ---
class ActorNet(nn.Module):
    def __init__(self, obs_dim, hidden, discrete, n_out):
        super().__init__()
        self.body, self.discrete = L.MLPNet(obs_dim, n_out, hidden, "tanh"), discrete
        if not discrete:
            self.log_std = nn.Parameter(torch.full((n_out,), -0.5))

    def dist(self, obs):
        out = self.body(obs)
        return Categorical(logits=out) if self.discrete else Normal(out, self.log_std.exp())

    def logp(self, obs, act):
        d = self.dist(obs)
        lp = d.log_prob(act)
        return lp if self.discrete else lp.sum(-1)

    def entropy(self, obs):
        e = self.dist(obs).entropy()
        return e if self.discrete else e.sum(-1)


class _PolicyAgent(DeepAgent):
    def _build(self, env):
        self._spec(env)
        n_out = self.n_actions if self.discrete else self.act_dim
        self.actor = ActorNet(self.obs_dim, self.hidden, self.discrete, n_out).to(self.device)
        self.critic = L.MLPNet(self.obs_dim, 1, self.hidden, "tanh").to(self.device)

    def _sample(self, obs):
        with torch.no_grad():
            d = self.actor.dist(self._t(obs).unsqueeze(0))
            a = d.sample()[0]
        return a

    def _env_action(self, a):
        a = a.cpu().numpy()
        return int(a) if self.discrete else np.clip(a, -self.act_high, self.act_high)

    def act(self, obs, greedy=False):
        with torch.no_grad():
            d = self.actor.dist(self._t(obs).unsqueeze(0))
            a = (d.probs.argmax(-1) if self.discrete else d.mean)[0] if greedy else d.sample()[0]
        return self._env_action(a)

    def _value(self, obs):
        with torch.no_grad():
            return self.critic(self._t(obs).unsqueeze(0)).item()

    def _reset(self, env):
        if not hasattr(self, "_obs"):
            self._obs, self._G = env.reset(), 0.0

    def _rollout(self, env, n):
        O, A, R, D, V, LP = [], [], [], [], [], []
        for _ in range(n):
            a = self._sample(self._obs)
            with torch.no_grad():
                lp = self.actor.logp(self._t(self._obs).unsqueeze(0), a.unsqueeze(0))[0].item()
            o2, r, done = env.step(self._env_action(a))
            O.append(self._obs); A.append(a.cpu().numpy()); R.append(r); D.append(float(done))
            V.append(self._value(self._obs)); LP.append(lp)
            self._G += r
            if done:
                self.returns_.append(self._G)
                o2, self._G = env.reset(), 0.0
            self._obs = o2
        adv, ret = gae(np.array(R), np.array(V), np.array(D), self._value(self._obs), self.gamma, self.lam)
        at = torch.long if self.discrete else torch.float32
        return (self._t(np.array(O)), self._t(np.array(A), at), self._t(adv), self._t(ret), self._t(LP))


@register("reinforce", family="agent")
class REINFORCE(_PolicyAgent):
    """Monte Carlo policy gradient with a mean-return baseline (discrete or continuous actions)."""

    def __init__(self, lr=3e-3, **kw):
        super().__init__(lr=lr, **kw)

    def train(self, env, episodes=500):
        self._build(env)
        opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        for _ in range(episodes):
            o, done, O, A, R = env.reset(), False, [], [], []
            while not done:
                a = self._sample(o)
                o2, r, done = env.step(self._env_action(a))
                O.append(o); A.append(a.cpu().numpy()); R.append(r); o = o2
            G, ret = 0.0, []
            for r in reversed(R):
                G = r + self.gamma * G
                ret.append(G)
            ret = self._t(ret[::-1])
            ret = (ret - ret.mean()) / (ret.std() + 1e-8)  # baseline + scaling
            at = torch.long if self.discrete else torch.float32
            logp = self.actor.logp(self._t(np.array(O)), self._t(np.array(A), at))
            loss = -(logp * ret).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            self.returns_.append(sum(R))
        return self.returns_


@register("actor_critic", family="agent")
class ActorCritic(_PolicyAgent):
    """One-step online actor-critic: the critic's TD error is the actor's learning signal."""
    lam = 0.0

    def __init__(self, lr=3e-3, **kw):
        super().__init__(lr=lr, **kw)

    def train(self, env, episodes=500):
        self._build(env)
        opt = torch.optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=self.lr)
        for _ in range(episodes):
            o, done, G = env.reset(), False, 0.0
            while not done:
                ot = self._t(o).unsqueeze(0)
                a = self._sample(o)
                o2, r, done = env.step(self._env_action(a))
                v = self.critic(ot)[0, 0]
                with torch.no_grad():
                    target = r + (0.0 if done else self.gamma * self.critic(self._t(o2).unsqueeze(0))[0, 0])
                delta = target - v
                loss = -self.actor.logp(ot, a.unsqueeze(0))[0] * delta.detach() + 0.5 * delta ** 2 \
                    - 0.01 * self.actor.entropy(ot)[0]
                opt.zero_grad(); loss.backward(); opt.step()
                o, G = o2, G + r
            self.returns_.append(G)
        return self.returns_


@register("a2c", family="agent")
class A2C(_PolicyAgent):
    """Advantage actor-critic: n-step rollouts, GAE advantages, entropy bonus."""

    def __init__(self, n_steps=16, lam=0.95, entropy_coef=0.01, value_coef=0.5, lr=3e-3, **kw):
        super().__init__(lr=lr, **kw)
        self.n_steps, self.lam, self.ent, self.vc = n_steps, lam, entropy_coef, value_coef

    def train(self, env, total_steps=20000):
        self._build(env)
        self._reset(env)
        opt = torch.optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=self.lr)
        for _ in range(total_steps // self.n_steps):
            O, A, adv, ret, _ = self._rollout(env, self.n_steps)
            loss = -(self.actor.logp(O, A) * adv).mean() + self.vc * F.mse_loss(self.critic(O).squeeze(-1), ret) \
                - self.ent * self.actor.entropy(O).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); opt.step()
        return self.returns_


@register("ppo", family="agent")
class PPO(_PolicyAgent):
    """Proximal Policy Optimization with the clipped surrogate objective."""

    def __init__(self, n_steps=256, epochs=4, minibatch=64, clip=0.2, lam=0.95, entropy_coef=0.01, lr=3e-3, **kw):
        super().__init__(lr=lr, **kw)
        self.n_steps, self.epochs, self.mb, self.clip, self.lam, self.ent = n_steps, epochs, minibatch, clip, lam, entropy_coef

    def train(self, env, total_steps=20000):
        self._build(env)
        self._reset(env)
        opt = torch.optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=self.lr)
        for _ in range(total_steps // self.n_steps):
            O, A, adv, ret, old = self._rollout(env, self.n_steps)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            for _ in range(self.epochs):
                for idx in torch.randperm(len(O)).split(self.mb):
                    ratio = (self.actor.logp(O[idx], A[idx]) - old[idx]).exp()
                    surr = torch.min(ratio * adv[idx], ratio.clamp(1 - self.clip, 1 + self.clip) * adv[idx])
                    loss = -surr.mean() + 0.5 * F.mse_loss(self.critic(O[idx]).squeeze(-1), ret[idx]) \
                        - self.ent * self.actor.entropy(O[idx]).mean()
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5); opt.step()
        return self.returns_


@register("trpo", family="agent")
class TRPO(_PolicyAgent):
    """Trust Region Policy Optimization: natural-gradient step (conjugate gradient with
    Fisher-vector products) sized to a KL limit, followed by a backtracking line search."""

    def __init__(self, n_steps=512, max_kl=0.02, damping=0.1, cg_iters=10, lam=0.95, lr=3e-3, vf_epochs=10, **kw):
        super().__init__(lr=lr, **kw)
        self.n_steps, self.max_kl, self.damping, self.cg_iters, self.lam, self.vf_epochs = n_steps, max_kl, damping, cg_iters, lam, vf_epochs

    def _kl(self, O, old):
        new = self.actor.dist(O)
        return torch.distributions.kl_divergence(old, new).sum(-1).mean() if not self.discrete \
            else torch.distributions.kl_divergence(old, new).mean()

    def train(self, env, total_steps=20000):
        self._build(env)
        self._reset(env)
        vopt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        params = list(self.actor.parameters())
        flat = lambda gs: torch.cat([g.reshape(-1) for g in gs])
        for _ in range(total_steps // self.n_steps):
            O, A, adv, ret, old_lp = self._rollout(env, self.n_steps)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            with torch.no_grad():
                d0 = self.actor.dist(O)
                old = Categorical(logits=d0.logits) if self.discrete else Normal(d0.loc.clone(), d0.scale.clone())
            surr = lambda: (torch.exp(self.actor.logp(O, A) - old_lp) * adv).mean()
            g = flat(torch.autograd.grad(surr(), params))

            def fvp(v):
                kl_grad = flat(torch.autograd.grad(self._kl(O, old), params, create_graph=True))
                return flat(torch.autograd.grad((kl_grad * v).sum(), params)) + self.damping * v

            x, r, p, rs = torch.zeros_like(g), g.clone(), g.clone(), g @ g  # conjugate gradient: solve F x = g
            for _ in range(self.cg_iters):
                Fp = fvp(p)
                alpha = rs / (p @ Fp + 1e-10)
                x, r = x + alpha * p, r - alpha * Fp
                rs_new = r @ r
                p, rs = r + (rs_new / rs) * p, rs_new
            step = torch.sqrt(2 * self.max_kl / (x @ fvp(x) + 1e-10)) * x
            theta0, s0 = nn.utils.parameters_to_vector(params).detach().clone(), surr().item()
            for frac in 0.5 ** np.arange(10):  # backtracking line search
                nn.utils.vector_to_parameters(theta0 + frac * step, params)
                with torch.no_grad():
                    if surr().item() > s0 and self._kl(O, old).item() <= 1.5 * self.max_kl:
                        break
            else:
                nn.utils.vector_to_parameters(theta0, params)
            for _ in range(self.vf_epochs):
                vopt.zero_grad(); F.mse_loss(self.critic(O).squeeze(-1), ret).backward(); vopt.step()
        return self.returns_


# ------------------------------------------------------------------------------ SAC ---
@register("sac", family="agent")
class SAC(DeepAgent):
    """Soft Actor-Critic for continuous actions: squashed-Gaussian policy, twin critics,
    target networks and an automatically tuned entropy temperature."""

    def __init__(self, tau=0.01, batch_size=64, buffer_size=20000, learning_starts=300, lr=3e-3, **kw):
        super().__init__(lr=lr, **kw)
        self.tau, self.batch_size, self.buffer_size, self.learning_starts = tau, batch_size, buffer_size, learning_starts

    def _pi(self, obs, deterministic=False):
        mu, log_std = self.actor(obs).chunk(2, dim=-1)
        std = log_std.clamp(-5, 2).exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = Normal(mu, std).log_prob(u).sum(-1) - torch.log(1 - a ** 2 + 1e-6).sum(-1)
        return a, logp

    def act(self, obs, greedy=False):
        with torch.no_grad():
            a, _ = self._pi(self._t(obs).unsqueeze(0), deterministic=greedy)
        return a[0].cpu().numpy() * self.act_high

    def train(self, env, total_steps=4000):
        self._spec(env)
        if self.discrete:
            raise ValueError("SAC here supports continuous actions only")
        if not hasattr(self, "actor"):
            d, a = self.obs_dim, self.act_dim
            self.actor = L.MLPNet(d, 2 * a, self.hidden, "relu").to(self.device)
            self.q1, self.q2 = (L.MLPNet(d + a, 1, self.hidden, "relu").to(self.device) for _ in range(2))
            self.t1, self.t2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
            self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
            self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
            self.opt_q = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=self.lr)
            self.opt_al = torch.optim.Adam([self.log_alpha], lr=self.lr)
            self.buf, self.target_entropy, self.steps = ReplayBuffer(self.buffer_size, d, a), -float(a), 0
        o, G = env.reset(), 0.0
        for _ in range(total_steps):
            a = self.rng.uniform(-1, 1, self.act_dim) * self.act_high if self.steps < self.learning_starts else self.act(o)
            o2, r, done = env.step(a)
            self.buf.add(o, a / self.act_high, r, o2, float(done))
            o, G, self.steps = o2, G + r, self.steps + 1
            if done:
                self.returns_.append(G)
                o, G = env.reset(), 0.0
            if self.buf.n >= self.learning_starts:
                self._update()
        return self.returns_

    def _update(self):
        _, (o, a, r, o2, d, _) = self.buf.sample(self.batch_size, self.rng)
        o, a, r, o2, d = map(self._t, (o, a, r, o2, d))
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, lp2 = self._pi(o2)
            x2 = torch.cat([o2, a2], 1)
            y = r + self.gamma * (1 - d) * (torch.min(self.t1(x2), self.t2(x2)).squeeze(-1) - alpha * lp2)
        x = torch.cat([o, a], 1)
        lq = F.mse_loss(self.q1(x).squeeze(-1), y) + F.mse_loss(self.q2(x).squeeze(-1), y)
        self.opt_q.zero_grad(); lq.backward(); self.opt_q.step()
        an, lp = self._pi(o)
        xn = torch.cat([o, an], 1)
        la = (alpha * lp - torch.min(self.q1(xn), self.q2(xn)).squeeze(-1)).mean()
        self.opt_a.zero_grad(); la.backward(); self.opt_a.step()
        lal = -(self.log_alpha * (lp.detach() + self.target_entropy)).mean()
        self.opt_al.zero_grad(); lal.backward(); self.opt_al.step()
        with torch.no_grad():
            for q, t in ((self.q1, self.t1), (self.q2, self.t2)):
                for p, pt in zip(q.parameters(), t.parameters()):
                    pt.mul_(1 - self.tau).add_(p, alpha=self.tau)


# ------------------------------------------------------------- model-based RL (MPC) ---
@register("model_based_mpc", family="agent")
class ModelBasedMPC(DeepAgent):
    """Learns a neural dynamics + reward model from experience and plans with random-shooting
    model-predictive control (re-planning at every step). Works for discrete and continuous
    actions."""

    def __init__(self, horizon=8, n_candidates=300, model_epochs=40, random_episodes=8, iterations=5,
                 episodes_per_iter=3, lr=2e-3, project=None, **kw):
        super().__init__(lr=lr, **kw)
        self.project = project  # optional map that snaps predicted observations back onto valid ones
        self.horizon, self.n_cand, self.model_epochs = horizon, n_candidates, model_epochs
        self.random_episodes, self.iterations, self.episodes_per_iter = random_episodes, iterations, episodes_per_iter
        self.data = []

    def _enc_a(self, a):  # actions as model input
        return F.one_hot(a.long(), self.n_actions).float() if self.discrete else a

    def _fit_model(self):
        O, A, R, O2, D = (np.array(x) for x in zip(*self.data))
        A = self._t(A, torch.long if self.discrete else torch.float32)
        X = torch.cat([self._t(O), self._enc_a(A)], 1)
        Y = torch.cat([self._t(O2 - O), self._t(R).unsqueeze(1), self._t(D).unsqueeze(1)], 1)
        self.xm, self.xs, self.ym, self.ys = X.mean(0), X.std(0) + 1e-6, Y.mean(0), Y.std(0) + 1e-6
        Xn, Yn = (X - self.xm) / self.xs, (Y - self.ym) / self.ys
        for _ in range(self.model_epochs):
            for idx in torch.randperm(len(Xn)).split(128):
                loss = F.mse_loss(self.model(Xn[idx]), Yn[idx])
                self.opt.zero_grad(); loss.backward(); self.opt.step()

    def _predict(self, o, a):
        x = (torch.cat([o, self._enc_a(a)], 1) - self.xm) / self.xs
        y = self.model(x) * self.ys + self.ym
        nxt = o + y[:, :-2]
        return (self.project(nxt) if self.project else nxt), y[:, -2], y[:, -1]  # next obs, reward, done

    def act(self, obs, greedy=True):
        o = self._t(obs).unsqueeze(0).expand(self.n_cand, -1)
        shape = (self.n_cand, self.horizon)
        if self.discrete:
            acts = torch.randint(self.n_actions, shape, device=self.device)
        else:
            acts = (torch.rand(*shape, self.act_dim, device=self.device) * 2 - 1) * self.act_high
        total = torch.zeros(self.n_cand, device=self.device)
        alive = torch.ones(self.n_cand, device=self.device)
        with torch.no_grad():
            for t in range(self.horizon):
                o, r, done = self._predict(o, acts[:, t])
                total += alive * self.gamma ** t * r
                alive = alive * (done < 0.5).float()  # nothing is earned after a predicted episode end
        best = acts[total.argmax(), 0]
        return int(best) if self.discrete else best.cpu().numpy()

    def _collect(self, env, episodes, random):
        for _ in range(episodes):
            o, done, G = env.reset(), False, 0.0
            while not done:
                if random:
                    a = int(self.rng.integers(self.n_actions)) if self.discrete else self.rng.uniform(-1, 1, self.act_dim) * self.act_high
                else:
                    a = self.act(o)
                o2, r, done = env.step(a)
                self.data.append((o, a, r, o2, float(done)))
                o, G = o2, G + r
            self.returns_.append(G)

    def train(self, env):
        self._spec(env)
        a_in = self.n_actions if self.discrete else self.act_dim
        self.model = L.MLPNet(self.obs_dim + a_in, self.obs_dim + 2, self.hidden, "relu").to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self._collect(env, self.random_episodes, random=True)
        for _ in range(self.iterations):
            self._fit_model()
            self._collect(env, self.episodes_per_iter, random=False)
        self._fit_model()
        return self.returns_


# ------------------------------------------------------------ preference-based learning ---
class PreferenceRewardModel:
    """Learn a reward function from pairwise comparisons of trajectory segments (Bradley-Terry
    model, as in RLHF): P(A preferred over B) = sigmoid(R(A) - R(B)), R(segment) = sum of
    per-step rewards. Segments are arrays of shape (n, T, feature_dim)."""

    def __init__(self, hidden=(32, 32), lr=3e-3, epochs=150, seed=None):
        self.hidden, self.lr, self.epochs, self.net = hidden, lr, epochs, None
        if seed is not None:
            torch.manual_seed(seed)

    def _R(self, seg):
        return self.net(seg).squeeze(-1).sum(-1)

    def fit(self, seg_a, seg_b, prefs):
        """prefs[i] = 1 if segment A was preferred, 0 if B (soft labels in [0, 1] allowed)."""
        A, B = (torch.as_tensor(np.asarray(s), dtype=torch.float32) for s in (seg_a, seg_b))
        y = torch.as_tensor(np.asarray(prefs), dtype=torch.float32)
        self.mu, self.sd = torch.cat([A, B]).reshape(-1, A.shape[-1]).mean(0), torch.cat([A, B]).reshape(-1, A.shape[-1]).std(0) + 1e-6
        A, B = (A - self.mu) / self.sd, (B - self.mu) / self.sd
        self.net = L.MLPNet(A.shape[-1], 1, self.hidden, "tanh")
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.history_ = []
        for _ in range(self.epochs):
            loss = F.binary_cross_entropy_with_logits(self._R(A) - self._R(B), y)
            opt.zero_grad(); loss.backward(); opt.step()
            self.history_.append(loss.item())
        return self

    def reward(self, x):
        x = (torch.as_tensor(np.asarray(x), dtype=torch.float32) - self.mu) / self.sd
        with torch.no_grad():
            return self.net(x).squeeze(-1).numpy()

    def accuracy(self, seg_a, seg_b, prefs):
        A, B = (torch.as_tensor(np.asarray(s), dtype=torch.float32) for s in (seg_a, seg_b))
        with torch.no_grad():
            pred = (self._R((A - self.mu) / self.sd) > self._R((B - self.mu) / self.sd)).float().numpy()
        return float((pred == (np.asarray(prefs) > 0.5)).mean())


class LearnedRewardEnv:
    """Replace an environment's reward with a learned reward model (applied to the next observation)."""

    def __init__(self, env, reward_model):
        self.env, self.rm = env, reward_model

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        return self.env.reset()

    def step(self, a):
        o, _, done = self.env.step(a)
        return o, float(self.rm.reward(o[None])[0]), done
