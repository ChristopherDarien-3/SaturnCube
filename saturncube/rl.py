"""Reinforcement learning and bandits (B5, tabular and contextual-linear)."""
from __future__ import annotations

import numpy as np

from .base import register


# ----------------------------------------------------------------- environments ---
class GridWorld:
    """size x size grid, start top-left, goal bottom-right. Reward +1 at the goal, -0.01 per step.
    States are integers; actions are 0=up, 1=down, 2=left, 3=right."""

    n_actions = 4

    def __init__(self, size=4, max_steps=60):
        self.size, self.max_steps, self.n_states = size, max_steps, size * size

    def reset(self):
        self.r = self.c = self.t = 0
        return 0

    def step(self, a):
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][a]
        self.r = min(max(self.r + dr, 0), self.size - 1)
        self.c = min(max(self.c + dc, 0), self.size - 1)
        self.t += 1
        goal = self.r == self.size - 1 and self.c == self.size - 1
        return self.r * self.size + self.c, (1.0 if goal else -0.01), goal or self.t >= self.max_steps


class PotentialShaping:
    """Reward shaping that keeps the optimal policy: r' = r + gamma*phi(s') - phi(s)."""

    def __init__(self, env, potential, gamma=0.99):
        self.env, self.phi, self.gamma = env, potential, gamma

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        self.s = self.env.reset()
        return self.s

    def step(self, a):
        s2, r, done = self.env.step(a)
        r = r + (0.0 if done else self.gamma * self.phi(s2)) - self.phi(self.s)
        self.s = s2
        return s2, r, done


# ------------------------------------------------------------------ tabular agents ---
class TabularAgent:
    family = "agent"

    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.3, epsilon_decay=0.995, epsilon_min=0.02, seed=None):
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.decay, self.eps_min = epsilon, epsilon_decay, epsilon_min
        self.rng, self.Q = np.random.default_rng(seed), None

    def act(self, s, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.Q.shape[1]))
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))  # random tie-break

    def policy(self):
        return self.Q.argmax(1)

    def _episode(self, env):
        raise NotImplementedError

    def train(self, env, episodes=500):
        if self.Q is None:
            self.Q = np.zeros((env.n_states, env.n_actions))
        returns = []
        for _ in range(episodes):
            returns.append(self._episode(env))
            self.epsilon = max(self.eps_min, self.epsilon * self.decay)
        return returns

    def evaluate(self, env, episodes=20):
        total = 0.0
        for _ in range(episodes):
            s, done = env.reset(), False
            while not done:
                s, r, done = env.step(self.act(s, greedy=True))
                total += r
        return total / episodes


class _TDAgent(TabularAgent):
    def _target(self, r, s2, a2, done):
        raise NotImplementedError

    def _episode(self, env):
        s, done, G = env.reset(), False, 0.0
        a = self.act(s)
        while not done:
            s2, r, done = env.step(a)
            a2 = self.act(s2)
            self.Q[s, a] += self.alpha * (self._target(r, s2, a2, done) - self.Q[s, a])
            s, a, G = s2, a2, G + r
        return G


@register("q_learning", family="agent")
class QLearning(_TDAgent):
    def _target(self, r, s2, a2, done):
        return r + (0.0 if done else self.gamma * self.Q[s2].max())


@register("sarsa", family="agent")
class SARSA(_TDAgent):
    def _target(self, r, s2, a2, done):
        return r + (0.0 if done else self.gamma * self.Q[s2, a2])


@register("expected_sarsa", family="agent")
class ExpectedSARSA(_TDAgent):
    def _target(self, r, s2, a2, done):
        if done:
            return r
        q = self.Q[s2]
        expected = (1 - self.epsilon) * q.max() + self.epsilon * q.mean()  # epsilon-greedy policy
        return r + self.gamma * expected


@register("monte_carlo", family="agent")
class MonteCarloControl(TabularAgent):
    """First-visit Monte Carlo control with an epsilon-soft policy."""

    def _episode(self, env):
        s, done, traj = env.reset(), False, []
        while not done:
            a = self.act(s)
            s2, r, done = env.step(a)
            traj.append((s, a, r))
            s = s2
        G, seen, total = 0.0, set(), sum(r for _, _, r in traj)
        returns = []
        for s, a, r in reversed(traj):
            G = r + self.gamma * G
            returns.append((s, a, G))
        for s, a, G in reversed(returns):  # first visit
            if (s, a) not in seen:
                seen.add((s, a))
                self.Q[s, a] += self.alpha * (G - self.Q[s, a])
        return total


# ------------------------------------------------------------------------ bandits ---
class BernoulliBandit:
    def __init__(self, probs, seed=None):
        self.probs, self.rng = np.asarray(probs), np.random.default_rng(seed)
        self.n_arms = len(self.probs)

    def pull(self, arm):
        return float(self.rng.random() < self.probs[arm])


class _Bandit:
    family = "bandit"

    def __init__(self, n_arms, seed=None):
        self.n_arms, self.rng = n_arms, np.random.default_rng(seed)
        self.counts, self.values = np.zeros(n_arms), np.zeros(n_arms)

    def update(self, arm, reward, context=None):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

    def _argmax(self, v):
        return int(self.rng.choice(np.flatnonzero(v == v.max())))


@register("epsilon_greedy", family="bandit")
class EpsilonGreedyBandit(_Bandit):
    def __init__(self, n_arms, epsilon=0.1, seed=None):
        super().__init__(n_arms, seed)
        self.epsilon = epsilon

    def select(self, context=None):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        return self._argmax(self.values)


@register("ucb1", family="bandit")
class UCB1(_Bandit):
    def __init__(self, n_arms, c=2.0, seed=None):
        super().__init__(n_arms, seed)
        self.c = c

    def select(self, context=None):
        if (self.counts == 0).any():
            return int(np.flatnonzero(self.counts == 0)[0])
        bonus = np.sqrt(self.c * np.log(self.counts.sum()) / self.counts)
        return self._argmax(self.values + bonus)


@register("thompson", family="bandit")
class ThompsonBandit(_Bandit):
    """Beta-Bernoulli Thompson sampling (rewards in [0, 1])."""

    def __init__(self, n_arms, seed=None):
        super().__init__(n_arms, seed)
        self.a, self.b = np.ones(n_arms), np.ones(n_arms)

    def select(self, context=None):
        return self._argmax(self.rng.beta(self.a, self.b))

    def update(self, arm, reward, context=None):
        super().update(arm, reward)
        self.a[arm] += reward
        self.b[arm] += 1 - reward


@register("linucb", family="bandit")
class LinUCB(_Bandit):
    """Contextual bandit with a linear reward model per arm (disjoint LinUCB)."""

    def __init__(self, n_arms, n_features, alpha=1.0, seed=None):
        super().__init__(n_arms, seed)
        self.alpha = alpha
        self.A = np.stack([np.eye(n_features) for _ in range(n_arms)])
        self.b = np.zeros((n_arms, n_features))

    def select(self, context):
        x, scores = np.asarray(context, dtype=float), np.empty(self.n_arms)
        for a in range(self.n_arms):
            Ainv = np.linalg.inv(self.A[a])
            scores[a] = (Ainv @ self.b[a]) @ x + self.alpha * np.sqrt(x @ Ainv @ x)
        return self._argmax(scores)

    def update(self, arm, reward, context):
        x = np.asarray(context, dtype=float)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.counts[arm] += 1


def run_bandit(bandit, env, steps=1000):
    """Play a Bernoulli bandit; returns the reward at each step."""
    rewards = np.empty(steps)
    for t in range(steps):
        arm = bandit.select()
        rewards[t] = env.pull(arm)
        bandit.update(arm, rewards[t])
    return rewards


# ======================================================== environments for deep RL ===
class OneHotObs:
    """Present an integer-state environment (e.g. GridWorld) as one-hot float vectors."""

    def __init__(self, env):
        self.env, self.obs_dim, self.n_actions = env, env.n_states, env.n_actions

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _oh(self, s):
        v = np.zeros(self.obs_dim, dtype=np.float32)
        v[s] = 1.0
        return v

    def reset(self):
        return self._oh(self.env.reset())

    def step(self, a):
        s, r, d = self.env.step(a)
        return self._oh(s), r, d


class PointMass:
    """Continuous control: push a point mass at x to the origin. obs = (x, v), action in [-1, 1].
    Reward = -x^2 - 0.01 a^2 per step."""
    obs_dim, action_dim, action_high = 2, 1, 1.0

    def __init__(self, max_steps=30, seed=None):
        self.max_steps, self.rng = max_steps, np.random.default_rng(seed)

    def reset(self):
        self.x, self.v, self.t = self.rng.uniform(-2, 2), 0.0, 0
        return np.array([self.x, self.v], dtype=np.float32)

    def step(self, a):
        a = float(np.clip(np.asarray(a).reshape(-1)[0], -1, 1))
        self.v = float(np.clip(0.95 * self.v + 0.3 * a, -2, 2))
        self.x += 0.3 * self.v
        self.t += 1
        return np.array([self.x, self.v], dtype=np.float32), -(self.x ** 2) - 0.01 * a ** 2, self.t >= self.max_steps


def _transitions(self):
    """Deterministic transition tensor P[s, a, s'] with the goal state absorbing."""
    S, A = self.n_states, self.n_actions
    P = np.zeros((S, A, S))
    for s in range(S):
        r0, c0 = divmod(s, self.size)
        for a, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            if s == S - 1:
                P[s, a, s] = 1.0
                continue
            r, c = min(max(r0 + dr, 0), self.size - 1), min(max(c0 + dc, 0), self.size - 1)
            P[s, a, r * self.size + c] = 1.0
    return P


GridWorld.transitions = _transitions


# =========================================================== more tabular agents ===
@register("sarsa_lambda", family="agent")
class SARSALambda(TabularAgent):
    """SARSA(lambda) with accumulating eligibility traces."""

    def __init__(self, lam=0.8, **kw):
        super().__init__(**kw)
        self.lam = lam

    def _episode(self, env):
        s, done, G, E = env.reset(), False, 0.0, np.zeros_like(self.Q)
        a = self.act(s)
        while not done:
            s2, r, done = env.step(a)
            a2 = self.act(s2)
            delta = r + (0.0 if done else self.gamma * self.Q[s2, a2]) - self.Q[s, a]
            E[s, a] += 1.0
            self.Q += self.alpha * delta * E
            E *= self.gamma * self.lam
            s, a, G = s2, a2, G + r
        return G


@register("dyna_q", family="agent")
class DynaQ(TabularAgent):
    """Dyna-Q: Q-learning plus `planning_steps` simulated updates per real step, replayed from
    a learned model of the environment (model-based RL, tabular)."""

    def __init__(self, planning_steps=10, **kw):
        super().__init__(**kw)
        self.planning_steps, self.model = planning_steps, {}

    def _update(self, s, a, r, s2, done):
        self.Q[s, a] += self.alpha * (r + (0.0 if done else self.gamma * self.Q[s2].max()) - self.Q[s, a])

    def _episode(self, env):
        s, done, G = env.reset(), False, 0.0
        while not done:
            a = self.act(s)
            s2, r, done = env.step(a)
            self._update(s, a, r, s2, done)
            self.model[(s, a)] = (r, s2, done)
            keys = list(self.model)
            for i in self.rng.integers(len(keys), size=self.planning_steps):
                ps, pa = keys[i]
                pr, ps2, pd = self.model[(ps, pa)]
                self._update(ps, pa, pr, ps2, pd)
            s, G = s2, G + r
        return G


# ================================================================ inverse RL (tabular) ===
class MaxEntIRL:
    """Maximum-entropy inverse RL (Ziebart et al., 2008) with a per-state reward: finds the
    reward under which the expert's state-visitation frequencies are most likely. Needs the
    environment's transition tensor (env.transitions()) and expert state trajectories."""

    def __init__(self, gamma=0.95, horizon=20, lr=0.2, iterations=300, l2=1e-3):
        self.gamma, self.horizon, self.lr, self.iterations, self.l2 = gamma, horizon, lr, iterations, l2

    def fit(self, env, demos, start_state=0):
        P = env.transitions()
        S, A, _ = P.shape
        T = self.horizon
        mu_expert = np.zeros(S)
        for traj in demos:  # pad with the final (absorbing) state to a fixed horizon
            traj = list(traj)[:T] + [traj[-1]] * max(0, T - len(traj))
            for s in traj:
                mu_expert[s] += 1.0 / len(demos)
        theta = np.zeros(S)
        for _ in range(self.iterations):
            V = np.zeros(S)
            for _ in range(T):  # soft value iteration
                Q = theta[:, None] + self.gamma * (P @ V)
                V = np.log(np.exp(Q - Q.max(1, keepdims=True)).sum(1)) + Q.max(1)
            pi = np.exp(Q - V[:, None])
            D, svf = np.zeros(S), np.zeros(S)
            D[start_state] = 1.0
            for _ in range(T):  # expected state visitation under the soft-optimal policy
                svf += D
                D = np.einsum("s,sa,sat->t", D, pi, P)
            theta += self.lr * (mu_expert - svf - self.l2 * theta)
        self.reward_, self.policy_, self.Q_ = theta, Q.argmax(1), Q
        return self

    def evaluate(self, env, episodes=5):
        """Mean true return of the policy that is optimal for the recovered reward."""
        total = 0.0
        for _ in range(episodes):
            s, done = env.reset(), False
            while not done:
                s, r, done = env.step(int(self.policy_[s]))
                total += r
        return total / episodes
