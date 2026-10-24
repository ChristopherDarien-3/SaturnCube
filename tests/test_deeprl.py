import numpy as np
import pytest
import torch
import torch.nn.functional as F

import saturncube as sc

pytestmark = pytest.mark.slow

grid = lambda: sc.OneHotObs(sc.GridWorld(size=3, max_steps=30))


@pytest.mark.parametrize("name,train", [
    ("dqn", dict(episodes=150)), ("double_dqn", dict(episodes=150)), ("dueling_dqn", dict(episodes=150)),
    ("prioritized_dqn", dict(episodes=150)), ("reinforce", dict(episodes=300)), ("actor_critic", dict(episodes=300)),
    ("a2c", dict(total_steps=8000)), ("ppo", dict(total_steps=8000)), ("trpo", dict(total_steps=8000))])
def test_discrete_agents_solve_gridworld(name, train):
    agent = sc.create(name, seed=0)
    agent.train(grid(), **train)
    assert agent.evaluate(grid(), 5) > 0.9, name           # optimal return is 0.97


def do_nothing_return(n=30):
    tot = 0.0
    for i in range(n):
        e = sc.PointMass(seed=i); e.reset()
        tot += sum(e.step(0.0)[1] for _ in range(30))
    return tot / n


@pytest.mark.parametrize("name,train", [("sac", dict(total_steps=4000)), ("ppo", dict(total_steps=12000)),
                                         ("trpo", dict(total_steps=12000)), ("a2c", dict(total_steps=12000)),
                                         ("reinforce", dict(episodes=400))])
def test_continuous_agents_control_point_mass(name, train):
    agent = sc.create(name, seed=0)
    agent.train(sc.PointMass(seed=1), **train)
    assert agent.evaluate(sc.PointMass(seed=7), 10) > -15, name   # doing nothing scores about -36


def test_sac_rejects_discrete_envs():
    with pytest.raises(ValueError):
        sc.SAC().train(grid(), total_steps=10)


def test_model_based_mpc_continuous_and_discrete():
    a = sc.ModelBasedMPC(seed=0)
    a.train(sc.PointMass(seed=1))
    assert a.evaluate(sc.PointMass(seed=7), 10) > -15
    proj = lambda o: F.one_hot(o.argmax(-1), o.shape[-1]).float()   # keep predicted one-hot states valid
    g = sc.ModelBasedMPC(seed=0, horizon=6, n_candidates=300, iterations=3, random_episodes=20, project=proj)
    g.train(grid())
    assert g.evaluate(grid(), 5) > 0.9


@pytest.mark.parametrize("cls", [sc.SARSALambda, sc.DynaQ])
def test_tabular_traces_and_planning(cls):
    env = sc.GridWorld(size=4)
    agent = cls(alpha=0.2, epsilon=0.5, epsilon_decay=0.99, seed=0)
    agent.train(env, episodes=150)
    assert agent.evaluate(env) > 0.85


def test_max_ent_irl_recovers_goal_reward():
    env = sc.GridWorld(size=4)
    q = sc.QLearning(alpha=0.2, epsilon=0.5, epsilon_decay=0.99, seed=0)
    q.train(env, episodes=500)
    e, s, traj, done = sc.GridWorld(size=4), 0, [0], False
    s = e.reset(); traj = [s]
    while not done:
        s, _, done = e.step(int(q.policy()[s])); traj.append(s)
    irl = sc.MaxEntIRL(horizon=15).fit(env, [traj] * 5)
    assert irl.reward_.argmax() == 15                                   # the goal
    assert irl.evaluate(env) > 0.9                                      # policy optimal for the recovered reward


def segments(n, T=10, seed=0):
    r = np.random.default_rng(seed)
    states = np.stack([r.uniform(-2, 2, size=(n, T)), r.uniform(-1, 1, size=(n, T))], -1).astype(np.float32)
    return states, -(states[..., 0] ** 2).sum(1)                        # hidden true reward: -x^2


def test_preference_reward_model_recovers_ranking():
    (A, ra), (B, rb) = segments(600, seed=0), segments(600, seed=1)
    prefs = (ra > rb).astype(float)
    rm = sc.PreferenceRewardModel(epochs=200, seed=0).fit(A[:500], B[:500], prefs[:500])
    assert rm.accuracy(A[500:], B[500:], prefs[500:]) > 0.9
    x = np.random.default_rng(3).uniform(-2, 2, size=(300, 2)).astype(np.float32)
    assert np.corrcoef(rm.reward(x), -(x[:, 0] ** 2))[0, 1] > 0.9


def test_agent_trained_on_learned_reward_solves_true_task():
    (A, ra), (B, rb) = segments(600, seed=0), segments(600, seed=1)
    rm = sc.PreferenceRewardModel(epochs=200, seed=0).fit(A, B, (ra > rb).astype(float))
    agent = sc.PPO(seed=0)
    agent.train(sc.LearnedRewardEnv(sc.PointMass(seed=1), rm), total_steps=12000)   # never sees the true reward
    assert agent.evaluate(sc.PointMass(seed=7), 10) > -15


def test_prioritized_buffer_prefers_high_td_error():
    buf = sc.ReplayBuffer(100, 2, None, prioritized=True)
    for i in range(100):
        buf.add(np.zeros(2), 0, 0.0, np.zeros(2), 0.0)
    buf.update(np.arange(100), np.where(np.arange(100) < 10, 5.0, 0.0))
    idx, _ = buf.sample(2000, np.random.default_rng(0))
    assert (idx < 10).mean() > 0.5
