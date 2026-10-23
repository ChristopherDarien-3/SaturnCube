# SaturnCube

One `fit` / `predict` / `evaluate` interface across many neural-network architectures, built on PyTorch.

```python
import saturncube as sc

model = sc.MLP(hidden=(64, 32))
model.fit(X_train, y_train)
print(model.evaluate(X_test, y_test))   # {'loss': ..., 'accuracy': ...}
preds = model.predict(X_test)
```

Switching architecture is a one-word change, so you can compare models in the same experiment loop:

```python
for name in ["mlp", "resnet", "kan", "deq"]:
    m = sc.create(name, seed=0).fit(X_train, y_train)
    print(name, m.evaluate(X_test, y_test)["accuracy"])
```

> **Status: v0.5, early development.** Implemented: vector/tabular, sequence, attention, memory, metric/few-shot, graph/tree/pointer, unsupervised and self-supervised models; alternative learning rules; many optimizers; continual and meta-learning; tabular and deep RL. plus three techniques from each of the search, symbolic, memory, cognition, verification, code-generation, experiment-design and classical-ML wheels (below). **Not built yet:** the spiking / biologically inspired family (SNN, STDP, Hodgkin-Huxley, NCA, SOM, ART, ...). Image models (CNN) are not included. The API may change.

## Why

Libraries like Keras, scikit-learn and PyCaret already make common models short to write. SaturnCube aims at a different goal: a **consistent interface for architectures that normally live in different codebases and need different training procedures**, with transparent defaults and full access to the underlying PyTorch objects.

## Install

Not on PyPI yet. From a clone of this repository:

```bash
pip install -e .
```

Requires Python 3.9+, `torch` and `numpy`.

## Usage

Every model supports the same methods:

| Method | What it does |
|---|---|
| `fit(X, y, epochs=None, val_data=None)` | Train. Calling it again continues training. |
| `predict(X)` | Class labels (classification) or values (regression). |
| `predict_proba(X)` | Class probabilities (classification only). |
| `evaluate(X, y)` | Accuracy and loss, or MSE, RMSE and R². |
| `save(path)` / `Model.load(path)` | Persist and restore weights, preprocessing and settings. |
| `.module` | The raw `torch.nn.Module`, for full control. |

Defaults that do the repetitive work:

- **Task detection:** integer, bool or string labels mean classification; float targets mean regression. Override with `task="classification"` or `task="regression"`.
- **Dimensions:** `input_dim` and `output_dim` are inferred from the data (you may pass them explicitly).
- **Preprocessing:** inputs are standardized using training statistics (`normalize=False` to disable) and the same transform is applied at prediction time.
- **Training:** Adam, mini-batches, optional validation data and early stopping.

```python
model = sc.ResNet(width=128, depth=4, lr=1e-3, epochs=100, patience=10, seed=0)
model.fit(X_train, y_train, val_data=(X_val, y_val))
model.save("resnet.pt")

model = sc.ResNet.load("resnet.pt")
model.module            # plain torch.nn.Module
```

Shared training options (all models): `lr`, `epochs`, `batch_size`, `weight_decay`, `normalize`, `patience`, `device`, `seed`, `verbose`.

## More families (v0.3)

### Unsupervised and self-supervised

`fit(X)` takes no labels; `transform(X)` returns the learned representation.

| Registry name | Class | Notes |
|---|---|---|
| `autoencoder` | `Autoencoder` | Also `reconstruct(X)`. |
| `denoising_ae` | `DenoisingAutoencoder` | Reconstructs clean input from noisy input. |
| `vae` | `VAE` | Variational autoencoder; also `sample(n)`. |
| `masked` | `MaskedPrediction` | Hide random features and predict them. |
| `contrastive` | `Contrastive` | SimCLR-style, InfoNCE (NT-Xent) objective. |
| `byol` | `BYOL` | Bootstrap latent learning, no negative pairs. |
| `deep_clustering` | `DeepClustering` | Autoencoder + k-means + DEC refinement; `predict(X)` gives cluster ids. |
| `hebbian` | `HebbianLearner` | Local rules, no gradients: `hebbian`, `oja`, `sanger`, `bcm`, `competitive`. |

```python
enc = sc.Contrastive(latent_dim=8, epochs=100).fit(X)
Z = enc.transform(X)

pca_like = sc.HebbianLearner(rule="sanger", n_components=2).fit(X)
```

Augmentations for the contrastive models are Gaussian noise plus feature dropout, which suits tabular data.

### Continual learning, replay, distillation, active learning

Strategies plug into any supervised model. For class-incremental learning, declare every class up front with `classes=`.

```python
model = sc.MLP(classes=[0, 1, 2, 3], strategies=[sc.Replay(capacity=200)])
model.learn_task(X_task1, y_task1)
model.learn_task(X_task2, y_task2)       # task 1 is not forgotten

student = sc.distill(teacher, sc.MLP(hidden=(8,)), X)       # knowledge distillation
idx = sc.query(model, X_pool, n=20, strategy="entropy")     # active learning: what to label next
model.partial_fit(X_new, y_new)                             # one epoch of online learning
```

Available: `Replay` (optionally `prioritized=True`), `EWC`, `LwF`, `distill`, `query` (`entropy`, `margin`, `least_confident`, `random`).
`examples/continual_learning.py` compares them. On the toy task, replay keeps task 1 at about 0.97 accuracy, EWC at about 0.66, and no strategy at about 0.43. LwF barely helps in this class-incremental setting, which is a known weakness of the method.

### Reinforcement learning and bandits

These use an environment (`reset()`, `step(a)`) instead of `fit(X, y)`.

```python
env = sc.GridWorld(size=4)
agent = sc.QLearning(alpha=0.2, epsilon=0.5, seed=0)
agent.train(env, episodes=800)
print(agent.evaluate(env))

bandit = sc.UCB1(n_arms=4)
rewards = sc.run_bandit(bandit, sc.BernoulliBandit([0.1, 0.2, 0.8, 0.3]), steps=2000)
```

Agents: `QLearning`, `SARSA`, `ExpectedSARSA`, `MonteCarloControl` (tabular, integer states). Bandits: `EpsilonGreedyBandit`, `UCB1`, `ThompsonBandit`, `LinUCB` (contextual). `PotentialShaping` wraps an environment with policy-preserving reward shaping.

## v0.4 families (overview)

Every model below uses the same `create(name)` / `fit` / `predict` / `evaluate` pattern unless noted. `sc.available(family)` lists names.

| Family | Models (registry names) |
|---|---|
| Learning rules (vector) | `forward_gradient`, `local_error`, `synthetic_gradients`, `equilibrium_propagation`, `contrastive_hebbian`, `predictive_coding`, plus `fa`, `dfa` |
| Sequence (X is `(n, T, d)` or integer tokens with `vocab_size=`) | `rnn`, `elman`, `jordan`, `lstm`, `gru`, `birnn`, `bilstm`, `bigru`, `ctrnn`, `neural_ssm`, `s4`, `mamba`, `plastic_rnn`, `esn`, `lsm`, `rtrl_rnn`; truncated BPTT via `tbptt=k`; next-token prediction via `fit(tokens)` |
| Attention | `transformer_encoder`, `transformer_decoder`, `linear_attention`, `local_attention`, `sparse_attention`, `multi_query`, `grouped_query`, `universal_transformer`, `memory_transformer`, `perceiver`, `ssm_attention_hybrid`, `recurrent_attention`, `retrieval_augmented`, `encoder_decoder`; layers `MultiHeadAttention` (MHA/MQA/GQA, cross-attention), `LinearAttention`, `TransformerLayer` |
| Memory | `hopfield`, `modern_hopfield`, `associative_memory`, `content_addressable_memory`, `product_key_memory`, `memory_network`, `kv_memory_network`, `ntm`, `dnc` |
| Metric / few-shot | `siamese`, `triplet`, `twin_tower`, `contrastive_embedding`, `prototypical`, `relation`, `matching` (few-shot: `predict(X, support=(Xs, ys))` on unseen classes) |
| Graph / tree / pointer | `gcn`, `gat`, `graphsage`, `gin`, `mpnn`, `rgcn`, `graph_transformer`, `ggnn`, `graph_net_recurrent`, `hypergraph`, `program_graph`, `tree_lstm`, `tree_nn`, `pointer_network`, `pointer_generator` (graphs: `sc.Graph`, `sc.Tree`) |
| Unsupervised (added) | `temporal_contrastive`, `latent_prediction`, `energy_based` |
| Deep RL (`train(env)`) | `dqn`, `double_dqn`, `dueling_dqn`, `prioritized_dqn`, `reinforce`, `actor_critic`, `a2c`, `ppo`, `trpo`, `sac`, `model_based_mpc`; `MaxEntIRL`, `PreferenceRewardModel` + `LearnedRewardEnv`; tabular `sarsa_lambda`, `dyna_q` |
| Meta / transfer | `MAML`, `Reptile` (with `SineTasks`, `GaussianTasks`), `transfer`, `fit_curriculum`, `fit_self_paced`, `GenerativeReplay` |
| Optimizers (added) | `cg`, `coordinate_descent`, `shampoo`, `kfac`, `natural_gradient` |

Notes: the alternative learning rules (`forward_gradient`, `local_error`, `synthetic_gradients`, `equilibrium_propagation`, `contrastive_hebbian`, `predictive_coding`) are research-style methods. They learn the easy blob task in `tests/test_rules.py` but are not reliable with default settings on harder problems such as XOR-like data; expect to tune them. Environments need `obs_dim` and either `n_actions` or `action_dim`/`action_high` (see `sc.OneHotObs`, `sc.PointMass`). `ModelBasedMPC` on one-hot discrete observations needs `project=` to keep predicted states valid. Graphs use dense adjacency, so they suit up to a few thousand nodes in total.

## Wheels C-J (v0.5): tools around the models

Three commonly used techniques for each wheel of the project plan. They are plain Python classes (not
`fit(X, y)` models) that work together on **Python code** by default; `examples/python_pipeline.py` wires several
of them into one loop. Optional C support (a mini-C parser/interpreter and a gcc sandbox) is included for anyone who
wants it; it is not needed for anything else in the library.

| Wheel | Techniques |
|---|---|
| C. Search | `GeneticAlgorithm` (typed genes), `SuccessiveHalving` (+ `RandomSearch` baseline), `SimulatedAnnealing` |
| D. Symbolic | `EnumerativeSynthesizer` (programs from examples, emits Python or C); `Datalog` forward chaining with `python_facts` (call graph, assignments, reads from any Python source); `check_python` / `run_py` restricted step-limited evaluator for small generated functions. Optional C: `parse` / `run_c` / `to_c` (recursive-descent parser, AST, int32 interpreter) |
| E. Memory | `BM25Index`, `VectorStore` (cosine k-NN, optional LSH) with `HashingEmbedder`, `CaseLibrary` (case-based memory with forgetting and consolidation) |
| F. Cognition | `SM2Scheduler` (spaced repetition), `InterleavedPractice`, `RevisionLoop` (predict-act-observe-revise) with `MetacognitiveMonitor` (calibration, abstention) |
| G. Verification | `UnitTests` (backends `python`, `python-subprocess`, and optional C `interp` / `gcc`), `PythonSandbox` (child process with CPU / memory / time limits), `check_property` / `differential_test` with shrinking. Optional C: `Sandbox` (gcc compile + limited run) |
| H. Generation | `GrammarGenerator` with `PY_GRAMMAR` (and optional `C_GRAMMAR`; supports guidance), `TemplateFiller` (sketch synthesis, `language="python"` or `"c"`), `BPETokenizer` |
| I. Experiment design | `RandomArchitectureGenerator` (typed module graphs, seed-reproducible, budgeted) with `ablate_module` / `ablate_edge` / `execute`; `latin_hypercube`, `full_factorial`; `bootstrap_ci`, `permutation_test`, `holm_correction` |
| J. Classical ML | `RandomForest`, `DecisionTree`, `KMeans`, `NaiveBayes` (same `fit` / `predict` / `evaluate` as the neural models) |

```python
import saturncube as sc

examples = [((a, b), max(a, b)) for a in range(-3, 4) for b in range(-3, 4)]
synth = sc.EnumerativeSynthesizer(["x", "y"], max_size=7)
src = synth.to_function(synth.synthesize(examples))        # "def f(x, y):\n    return (y if x < y else x)\n"

sc.UnitTests("f", examples).run(src).ok                    # True: checked in the restricted evaluator
sc.UnitTests("f", examples, backend="python-subprocess").run(src).ok   # or in a resource-limited child process

best, fitness = sc.GeneticAlgorithm(lambda g: -sum(x * x for x in g), [("float", -5, 5)] * 4, seed=0).run()
```

**Safety.** `run_py` only accepts a small side-effect-free subset of Python (no imports, attributes, comprehensions,
exponentiation, ...) and has a step limit, but it is a convenience for checking small generated functions, **not** a
security boundary. `PythonSandbox` (and the optional gcc `Sandbox`) add process-level CPU, memory, file-size and time
limits and kill runaway processes, but provide no network or filesystem isolation. Run untrusted code in a container or VM.

## Optimizers

Any model can be trained with any of these, with `optimizer="..."`:

`sgd`, `momentum`, `nesterov`, `adagrad`, `rmsprop`, `adadelta`, `adam` (default), `adamw`, `adamax`, `amsgrad`, `radam`, `adafactor`, `lamb`, `lion`, `lbfgs`

```python
model = sc.MLP(optimizer="lion", lr=3e-4)
model = sc.MLP(optimizer="sgd", optimizer_kwargs={"momentum": 0.9})
model = sc.MLP(optimizer="lbfgs", batch_size=len(X_train))   # full-batch works best
```

If `lr` is not given, each optimizer uses its own starting learning rate (`saturncube.optim.DEFAULT_LR`). Most are the `torch.optim` implementations; Lion and LAMB are implemented here.

## Models (vector / tabular family)

Input is a 2-D array of shape `(n_samples, n_features)`.

| Registry name | Class | Notes |
|---|---|---|
| `mlp` | `MLP` | Multilayer perceptron; optional dropout and batch norm. |
| `deep_ffn` | `DeepFeedforward` | MLP with deeper defaults, batch norm and dropout. |
| `resnet` | `ResNet` | Pre-norm residual blocks. |
| `densenet` | `DenseNet` | Every layer sees all earlier features. |
| `highway` | `Highway` | Learned gates mix transformed and carried features. |
| `senet` | `SqueezeExcitation` | Residual blocks with a squeeze-and-excitation feature gate. |
| `moe` | `MixtureOfExperts` | Softmax-routed experts (dense, or `top_k`). |
| `gated_moe` | `GatedMixtureOfExperts` | Noisy top-k routing plus a load-balancing loss. |
| `hypernet` | `Hypernetwork` | A small network generates a per-sample layer. |
| `implicit` | `ImplicitNetwork` | Output is an equilibrium `z = relu(Az + Bx)`; unrolled solve. |
| `deq` | `DeepEquilibrium` | Deep Equilibrium Model with an implicit-function-theorem backward pass. |
| `kan` | `KAN` | Kolmogorov-Arnold Network with B-spline edge functions. |
| `rbf` | `RBF` | Radial basis function network; centers initialized from data. |
| `elm` | `ExtremeLearningMachine` | Random frozen hidden layer; readout solved in closed form (no epochs). |
| `random_nn` | `RandomNeuralNetwork` | Gelenbe random neural network with a linear readout. |
| `fa` | `FeedbackAlignment` | MLP trained with Feedback Alignment (fixed random backward weights). |
| `dfa` | `DirectFeedbackAlignment` | MLP trained with Direct Feedback Alignment (output error sent straight to each hidden layer). |

List what is available with `sc.available()`, and build by name with `sc.create("kan", hidden=(16,))`.

## Example results

`examples/compare_optimizers.py` trains the same MLP with each optimizer, plus `fa` and `dfa`, on the same synthetic task. Results vary a lot with learning rate and seed, so treat it as a smoke test, not a ranking.

`examples/compare_all.py` runs every model with default settings on a synthetic 3-class nonlinear problem (1000 samples, 8 features, seed 0):

```text
model        params  accuracy
deep_ffn     335107     0.836
densenet      16899     0.916
deq            4867     0.932
elm            6147     0.640
gated_moe      6312     0.888
highway      133635     0.908
hypernet      11219     0.860
implicit       4891     0.932
kan            1584     0.908
mlp            9603     0.912
moe            3120     0.892
random_nn      9795     0.952
rbf            1412     0.832
resnet       134915     0.940
senet        168323     0.916
```

This is a single small run on toy data, meant to show the interface working, not to rank the architectures.

## Design notes

- **A common interface, not a common implementation.** Each model keeps its own requirements. ELM has no epochs, DEQ differentiates through a fixed point, and MoE adds an auxiliary loss, all behind the same `fit()`.
- **Escape hatch.** `.module` always returns the underlying `nn.Module`.
- **Registry.** Models register by name, which makes experiment loops and future model families easy to add.

## Known limitations

- Vector/tabular input only. Sequence, image, graph and other families are not implemented yet.
- No GPU testing yet, and no handling of categorical features or missing values.
- Regression targets are not standardized, so very badly scaled targets may train slowly.
- `MoE` evaluates all experts and masks them, so it is correct but not sparse-efficient.
- `deep_ffn` is deliberately large for toy problems; shrink `hidden` for small datasets.

## Development

```bash
pip install -e . pytest
pytest tests
python examples/compare_all.py
```

The test suite trains every model on classification and regression, and checks save/load round trips.

## Roadmap

- Deep RL (DQN family, REINFORCE, actor-critic, PPO, TRPO, SAC), model-based RL, inverse RL, preference learning
- Meta-learning (MAML, Reptile), few-shot and transfer learning, curriculum and self-paced learning, generative replay
- Three-factor rules, STDP and predictive coding (these need the spiking and recurrent families)
- Remaining optimizers (conjugate gradient, coordinate descent, natural gradient, K-FAC, Shampoo)
- sklearn wrappers (random forest, logistic regression) in the same interface
- Sequence models (RNN, LSTM, GRU, state-space models) and attention layers
- Metric-learning models, then graph and spiking networks as optional extras

## License

Add a license before publishing (for example MIT).
