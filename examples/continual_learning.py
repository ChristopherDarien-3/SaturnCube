"""Two tasks in sequence (classes {0,1} then {2,3}); how much of task 1 survives?"""
import numpy as np
import saturncube as sc

rng = np.random.default_rng(0)
centers = rng.normal(size=(4, 6)) * 2.0
y = rng.integers(4, size=1200)
X = (centers[y] + rng.normal(size=(1200, 6))).astype(np.float32)
t1, t2 = y < 2, y >= 2

setups = {"naive": [], "replay": [sc.Replay(200)], "prioritized replay": [sc.Replay(200, prioritized=True)],
          "EWC": [sc.EWC(2000)], "LwF": [sc.LwF(2.0)]}
print(f"{'strategy':<20} {'task1 acc':>10} {'task2 acc':>10}")
for name, strategies in setups.items():
    m = sc.MLP(hidden=(32,), epochs=80, lr=1e-2, classes=[0, 1, 2, 3], strategies=strategies, seed=0)
    m.learn_task(X[t1], y[t1]).learn_task(X[t2], y[t2])
    print(f"{name:<20} {m.evaluate(X[t1], y[t1])['accuracy']:>10.2f} {m.evaluate(X[t2], y[t2])['accuracy']:>10.2f}")
