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

> **Status: v0.7, alpha.** Implemented: vector/tabular, image (CNN), sequence, attention, memory, metric/few-shot, graph/tree/pointer, spiking and biologically inspired, unsupervised and self-supervised models; alternative learning rules; many optimizers; continual and meta-learning; tabular and deep RL; plus search, program-synthesis, retrieval, verification, code-generation, experiment-design and classical-ML tools (see below). The API may change between releases.

## Why

Libraries like Keras, scikit-learn and PyCaret already make common models short to write. SaturnCube aims at a different goal: a **consistent interface for architectures that normally live in different codebases and need different training procedures**, with transparent defaults and full access to the underlying PyTorch objects.

## Install

```bash
pip install saturncube
```

Requires Python 3.9+, `numpy` and `torch` (installed automatically). For a CPU-only or a specific CUDA build of
PyTorch, install `torch` first following https://pytorch.org, then install SaturnCube.

From source:

```bash
git clone <this repository>
cd saturncube
pip install -e ".[dev]"
```

## Preparing your own data (v0.7)

Real datasets are messy. SaturnCube can load a CSV (or a DataFrame, dict, list of dicts or array), work out what each
column is, clean it, split it and hand back arrays in the shape your model expects. Nothing happens silently:
`report()` lists every decision and each one can be overridden.

```python
import saturncube as sc

data = sc.load_dataset("customers.csv", target="Purchased")
print(data.report())                 # detected column types, what will be done, and warnings

prepared = data.prepare(test_size=0.2, val_size=0.1, seed=0)
model = sc.MLP(hidden=(64, 32)).fit(prepared.X_train, prepared.y_train)
print(model.evaluate(prepared.X_test, prepared.y_test))
labels = prepared.decode_target(model.predict(prepared.X_test))        # back to the original labels

prepared.save("pipeline.json")       # later: sc.load_pipeline("pipeline.json").transform(new_rows)
```

What it handles by default:

| Problem | Behaviour |
|---|---|
| Delimiter, header, ragged rows | `;` `,` tab and `\|` are detected; short or long rows are padded or truncated and reported |
| Number-like text | `"$52,000"`, `"12%"`, `"(5)"`, `"1e3"` become numbers; stray text in a numeric column counts as missing |
| Missing values | `""`, `N/A`, `?`, `null`, `-`, ... are missing. Numbers: median (or mean/zero) plus an optional missing-flag column. Categories: their own category |
| Categories | Case and whitespace variants are merged; one-hot of the 20 most common values; unseen values at inference map to "other" |
| Booleans and dates | `yes/no/y/n/true/false` become 0/1; dates become year, month, day, weekday and timestamp |
| Identifier, free-text, constant and mostly-empty columns | Dropped, and the report says why (`keep=[...]` overrides) |
| Target | Task is inferred (override with `task=`); rows without a target are removed; labels are encoded and `decode_target` restores them |
| Leakage | All statistics are fitted on the training rows only; a stratified split keeps class balance; suspiciously perfect features are flagged |

Other model types:

```python
seq = data.prepare_sequences(window=24, horizon=1, target="demand", time_column="date")   # (n, 24, features), chronological split
img = data.prepare_images(target="label")                    # flat pixel columns (e.g. MNIST CSV) -> (n, 1, 28, 28)

result = sc.auto_model("customers.csv", target="Purchased", model="random_forest")        # load, clean, split, train, evaluate
result.metrics; result.predict(new_rows)
```

`auto_model` picks the preparation from the model's family (tabular for vector and classical models; `window=` for sequence
models; pixel columns for the CNN) and explains what to do for models it cannot prepare for automatically (graphs, pairs, ...).
Models also give a clear error if you pass text or NaN values directly. The detection is heuristic, so read the report:
a numeric code such as a zip code can be forced to categorical with `categorical=["zip"]`. Free-text columns are dropped here;
for documents use `sc.TextModel` (next sections). Graphs and multi-table data are not handled yet. Install pandas (`pip install saturncube[data]`) to pass DataFrames.

## Comparing models fairly

`sc.benchmark` runs several models on the same splits, repeats that over seeds, adds baselines and prints one table.

```python
import saturncube as sc

result = sc.benchmark(["mlp", "random_forest", "kan"], X, y, groups=source_ids, seeds=5)
print(result.to_rows()[0])      # raw numbers; also result.scores[model][metric] per seed, result.to_dataframe()
```

- **Grouped splits.** `groups` holds one id per row (image, patient, document, ...). Whole groups go to train, validation
  or test, so rows from one source never sit on both sides. Without `groups` the rows are split at random (stratified by
  class), and the table says so, because that can inflate scores when rows share a source.
- **Seeds.** Each seed changes the split and the model initialisation. The table reports mean ± std over seeds and, for
  each model, the paired difference to the best baseline; `~` marks a difference smaller than its own seed noise.
- **Baselines.** A majority-class guess (train mean for regression) and a standardised scikit-learn linear model
  (logistic or ridge regression) are added automatically; `baselines=False` turns them off. scikit-learn is optional
  (`pip install saturncube[bench]`); without it only the first baseline runs and the table says so.
- **Same data for everyone.** Every model and baseline trains on the same training rows, and the test rows are never
  passed to `fit()`. The validation part goes to `fit(val_data=...)` for models that accept it, so `patience=` (including the regression default) works inside `sc.benchmark` using that split. For regression, models
  are trained on a standardised target and scored on the original scale.
- **Failures are recorded, not hidden.** A model that raises on a seed shows up in the table's notes; a misspelled name
  raises immediately. Models can be given as registry names, `(name, kwargs)`, classes, `{label: ...}` dicts, or a
  function `seed -> fresh model`.

Output on a synthetic problem built so that labels cannot be learned across groups (40 groups of 25 rows, each group
a tight cluster, labels unrelated to the clusters). A random-row split of the same data gives `random_forest`
1.00 accuracy because it memorises the clusters; the grouped split gives 0.50, which is the honest number:

```text
SaturnCube benchmark: classification, 1000 rows, 3 seeds
split: group-disjoint over 40 groups; per seed about train 600 / val 150 / test 250 rows
largest class = 50.0% of all rows

model                                       accuracy  balanced_accuracy  vs best baseline  fit s   ok
-------------------------------------  -------------  -----------------  ----------------  -----  ---
mlp                                    0.636 ± 0.200      0.636 ± 0.200  +0.033 ± 0.344 ~   1.10  3/3
scaled logistic regression (baseline)  0.603 ± 0.176      0.603 ± 0.176                 -   0.00  3/3
naive_bayes                            0.581 ± 0.155      0.581 ± 0.155  -0.021 ± 0.156 ~   0.00  3/3
majority class (baseline)              0.500 ± 0.000      0.500 ± 0.000                 -   0.00  3/3
random_forest                          0.497 ± 0.096      0.497 ± 0.096  -0.105 ± 0.200 ~   0.14  3/3

std is over seeds (each seed changes the split and the model initialisation)
~ = difference within seed noise (|mean| <= std of the paired difference)
```

## Working with text

Models take numbers, so documents have to be encoded first. `sc.TextModel` does that and then behaves like any other model:

```python
import saturncube as sc

clf = sc.TextModel("naive_bayes").fit(train_docs, train_labels)            # TF-IDF features
clf = sc.TextModel("mlp", vectorizer=sc.TextVectorizer(ngram_range=(1, 2), svd=100)).fit(train_docs, train_labels)
clf = sc.TextModel("transformer_encoder", hidden=32,
                   vectorizer=sc.TokenEncoder(max_len=64)).fit(train_docs, train_labels)   # integer token ids

clf.predict(new_docs); clf.predict_proba(new_docs); clf.evaluate(test_docs, test_labels)
clf.save("model.pt")                     # model + model.pt.text.json (the encoder, plain JSON)
clf = sc.TextModel.load("model.pt")
```

- **Routing.** Sequence and attention models (`lstm`, `gru`, `mamba`, `transformer_encoder`, ...) get padded integer ids
  and `vocab_size=` is filled in for you; every other model gets TF-IDF vectors, optionally reduced by a truncated SVD
  (`svd=k`). Override with `mode="tfidf"` or `mode="tokens"`.
- **No leakage.** The vocabulary, document frequencies and SVD directions are fitted on the documents given to `fit()`
  only; `val_data=(texts, labels)` is encoded with the training vocabulary and passed on to models that accept it.
- **The pieces are usable alone.** `TextVectorizer` and `TokenEncoder` have `fit` / `transform` / `save` / `load`
  (plain JSON, no pickle), so you can feed their output to anything, including `sc.benchmark`, though a vectorizer
  fitted on all documents before benchmarking lets test text influence the vocabulary and idf weights.
- **Not covered.** `bm25` and `bpe_tokenizer` are a retrieval index and a byte-level tokenizer, not classifiers, so
  `TextModel` does not use them. There is no stemming, stop-word list or subword tokenisation. TF-IDF matrices are dense,
  so keep `max_features` modest. The sequence models do not mask padding (id 0 is an ordinary learned token), so set
  `max_len` close to your typical document length. `naive_bayes` is Gaussian, which is not the usual choice for TF-IDF.
- **Tested on small synthetic corpora only,** not on real text collections yet.

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

Shared training options (all models): `lr`, `epochs`, `batch_size`, `weight_decay`, `normalize`, `patience`, `device`, `seed`, `verbose`, `val_split`.

`patience` is early stopping: training stops when the validation loss has not improved for that many epochs, and the best
weights are restored. The default, `patience="auto"`, means **10 for regression and off for classification**; `patience=None`
turns it off and an integer sets it yourself. If you do not pass `val_data=`, the model holds out `val_split` (default 0.1,
stratified for classification) of the training rows for this. The automatic default only applies when that holdout has at least
20 rows (about 200 training rows); below that a validation loss is mostly noise and early stopping stays off, silently. An
explicit `patience=k` always applies, and with `val_split=0` it warns that it has nothing to monitor. After `fit`, `n_val_`,
`best_epoch_` and `stopped_epoch_` say what happened. `partial_fit` never holds rows out.

Why regression only: on small noisy regression sets the models overfit badly without it (`diabetes` r²: `mlp` 0.16 -> 0.42,
`highway` -0.14 -> 0.39), while on noisy classification early stopping cost 0.04 to 0.05 accuracy. It is not free on
regression either: it takes 10% of the rows away from training and gave small losses on some smoother sets (up to -0.065 for
`rbf`). Numbers and caveats are in [docs/TUNING.md](https://github.com/ChristopherDarien-3/SaturnCube/blob/HEAD/docs/TUNING.md). Fits that used to train for the full `epochs` on
regression data now may stop early; pass `patience=None` to get the old behaviour. No default learning rate or epoch count was
changed.

The first fit in a process is slow (about a second here, 10-20 s on a cold disk) because PyTorch imports its optimizer machinery
lazily. Call `sc.warmup()` once before timing anything; `sc.benchmark` does it for you.

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

## Images and biologically inspired models (v0.6)

| Registry name | Class | Notes |
|---|---|---|
| `cnn` | `CNN` | Compact ResNet-style convolutional network (residual stages, batch norm, global pooling). Images as `(n, C, H, W)` or `(n, H, W)`; classification or regression. |
| `snn` | `SpikingNetwork` | Leaky integrate-and-fire neurons trained with surrogate gradients. `encoding="direct"` or `"rate"`; `firing_rates(X)` reports sparsity. |
| `som` | `SelfOrganizingMap` | Kohonen map: unsupervised, topology-preserving. `transform`, `u_matrix`, `hit_map`, `quantization_error`, `topographic_error`, and `label_units` + `classify` for classification. |

```python
import saturncube as sc

cnn = sc.CNN(width=16, epochs=25).fit(images, labels)          # images: (n, 1, 16, 16)
snn = sc.SpikingNetwork(hidden=(64,), n_steps=16, encoding="rate").fit(X, y)
print(snn.firing_rates(X))                                     # e.g. [0.2]: neurons fire on ~20% of time steps

som = sc.SelfOrganizingMap(grid=(10, 10)).fit(X)
som.transform(X)                                               # (row, col) of each sample's best-matching unit
som.label_units(X, y); som.classify(X_new)
```

In the test suite the CNN reaches 100% on a task where the pattern's position is random while a flat MLP reaches 60%,
and the SOM unfolds into an ordered grid on uniform data (topographic error near 0).

## Search, symbolic, memory and other tools (v0.5)

Besides the `fit` / `predict` models, SaturnCube includes a set of commonly used building blocks. They are plain
Python classes that work together on **Python code**; `examples/python_pipeline.py` wires several of them into one loop
(retrieve a similar solved case, synthesize a program from examples, verify it, store it).

| Area | Techniques |
|---|---|
| Search and optimization | `GeneticAlgorithm` (typed genes), `SuccessiveHalving` (+ `RandomSearch` baseline), `SimulatedAnnealing` |
| Program synthesis and symbolic reasoning | `EnumerativeSynthesizer` (programs from input/output examples); `Datalog` forward chaining with `python_facts` (call graph, assignments and reads from any Python source); `check_python` / `run_py` (restricted, step-limited evaluator for small generated functions) |
| Memory and retrieval | `BM25Index`, `VectorStore` (cosine k-NN, optional LSH) with `HashingEmbedder`, `CaseLibrary` (case-based memory with forgetting and consolidation) |
| Learning strategies | `SM2Scheduler` (spaced repetition), `InterleavedPractice`, `RevisionLoop` (predict-act-observe-revise) with `MetacognitiveMonitor` (calibration, abstention) |
| Verification and testing | `UnitTests` (backends `python` and `python-subprocess`), `PythonSandbox` (child process with CPU / memory / time limits), `check_property` / `differential_test` with shrinking |
| Code generation | `GrammarGenerator` with `PY_GRAMMAR` (supports guidance), `TemplateFiller` (sketch synthesis), `BPETokenizer` |
| Experiment design and statistics | `RandomArchitectureGenerator` (typed module graphs, seed-reproducible, budgeted) with `ablate_module` / `ablate_edge` / `execute`; `latin_hypercube`, `full_factorial`; `bootstrap_ci`, `permutation_test`, `holm_correction` |
| Classical ML | `RandomForest`, `DecisionTree`, `KMeans`, `NaiveBayes` (same `fit` / `predict` / `evaluate` as the neural models) |

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
security boundary. `PythonSandbox` adds process-level CPU, memory, file-size and time limits and kills runaway
processes, but provides no network or filesystem isolation. Run untrusted code in a container or VM.

## Optimizers

Any model can be trained with any of these, with `optimizer="..."`:

`sgd`, `momentum`, `nesterov`, `adagrad`, `rmsprop`, `adadelta`, `adam` (default), `adamw`, `adamax`, `amsgrad`, `radam`, `adafactor`, `lamb`, `lion`, `lbfgs`

```python
model = sc.MLP(optimizer="lion", lr=3e-4)
model = sc.MLP(optimizer="sgd", optimizer_kwargs={"momentum": 0.9})
model = sc.MLP(optimizer="lbfgs", batch_size=len(X_train))   # full-batch works best
```

If `lr` is not given, each optimizer uses its own starting learning rate (`saturncube.optim.DEFAULT_LR`). Most are the `torch.optim` implementations; Lion and LAMB are implemented here.

## Models

The table below is generated from the registry by `python scripts/update_readme.py`, so it always matches `sc.available()`. Narrative sections for each family are further up.

<!-- registry-table:start -->
139 registered entries (`sc.available()`): 94 models, 4 classical learners, 21 agents and bandits, 20 tools.

`model` entries share one `fit` / `predict` / `evaluate` / `save` / `load` interface and expose `.module`. `classical` entries have `fit` / `predict` only. `agent` entries learn from an environment or bandit instead of a dataset. `tool` entries are plain Python classes with their own methods. Build any of them with `sc.create(name, **kwargs)`.

### vector (24)

Input: 2-D array `(n_samples, n_features)`.

| Registry name | Class | Kind | Description |
|---|---|---|---|
| `contrastive_hebbian` | `ContrastiveHebbian` | model | Contrastive Hebbian Learning: clamped-phase minus free-phase Hebbian statistics |
| `deep_ffn` | `DeepFeedforward` | model | Deep feedforward net: an MLP with deeper defaults, batch norm and dropout |
| `densenet` | `DenseNet` | model | Densely connected MLP: every layer receives all earlier features |
| `deq` | `DeepEquilibrium` | model | Deep Equilibrium Model with implicit-function-theorem backward pass |
| `dfa` | `DirectFeedbackAlignment` | model | MLP trained with Direct Feedback Alignment (output error sent to every hidden layer) |
| `elm` | `ExtremeLearningMachine` | model | Random frozen hidden layer; readout solved by ridge regression (no epochs) |
| `equilibrium_propagation` | `EquilibriumPropagation` | model | Equilibrium Propagation (Scellier & Bengio, 2017): a free phase, then a weakly nudged phase; the weight update is the contrast of the two equilibria divided by beta |
| `fa` | `FeedbackAlignment` | model | MLP trained with Feedback Alignment (random fixed backward weights) |
| `forward_gradient` | `ForwardGradient` | model | Forward-mode differentiation: the gradient is estimated from a single forward pass with a random tangent, g = (dL . v) v (Baydin et al., 2022) |
| `gated_moe` | `GatedMixtureOfExperts` | model | Sparsely-gated MoE: noisy top-k routing plus a load-balancing loss |
| `highway` | `Highway` | model | Highway network: learned gates mix transformed and carried features |
| `hypernet` | `Hypernetwork` | model | A hypernetwork generates a per-sample layer; an ordinary head reads it out |
| `implicit` | `ImplicitNetwork` | model | Implicit network: output defined by an equilibrium z = relu(Az + Bx) |
| `kan` | `KAN` | model | Kolmogorov-Arnold Network with B-spline edge functions |
| `local_error` | `LocalErrorLearning` | model | Local-gradient learning: every hidden layer has its own small classifier and is trained on its own loss only; nothing is backpropagated between layers |
| `mlp` | `MLP` | model | Multilayer perceptron |
| `moe` | `MixtureOfExperts` | model | Mixture of experts with a softmax router (dense by default, optional top_k) |
| `predictive_coding` | `PredictiveCoding` | model | Predictive-coding network (hierarchical when it has several hidden layers) |
| `product_key_memory` | `ProductKeyMemoryNet` | model | MLP with a large sparse product-key memory layer between input and head |
| `random_nn` | `RandomNeuralNetwork` | model | Gelenbe random neural network (spiking-queue neurons, positive weights) with a linear readout on the firing probabilities |
| `rbf` | `RBF` | model | Radial basis function network (centers/widths initialised from data, then trained) |
| `resnet` | `ResNet` | model | Residual MLP with pre-norm blocks |
| `senet` | `SqueezeExcitation` | model | Residual MLP whose blocks re-weight features with a squeeze-and-excitation gate |
| `synthetic_gradients` | `SyntheticGradients` | model | Decoupled training (Jaderberg et al., 2017): each layer is updated immediately using a gradient *predicted* from its activations; predictors learn from the true gradients that arrive from the layer above |

### vision (1)

Input: images as `(n, C, H, W)` or `(n, H, W)`.

| Registry name | Class | Kind | Description |
|---|---|---|---|
| `cnn` | `CNN` | model | Compact ResNet-style convolutional network for image classification or regression |

### bio (2)

| Registry name | Class | Kind | Description |
|---|---|---|---|
| `snn` | `SpikingNetwork` | model | Spiking neural network of leaky integrate-and-fire neurons, trained end to end with surrogate gradients |
| `som` | `SelfOrganizingMap` | model | Self-Organizing Map (Kohonen): a grid of units whose weight vectors move toward the data so that nearby units respond to similar inputs, giving a topology-preserving 2-D map without labels |

### sequence (30)

Input: `(n, T, d)` floats, or integer tokens `(n, T)` with `vocab_size=`.

| Registry name | Class | Kind | Description |
|---|---|---|---|
| `bigru` | `BidirectionalGRU` | model | Bidirectional GRU |
| `bilstm` | `BidirectionalLSTM` | model | Bidirectional LSTM |
| `birnn` | `BidirectionalRNN` | model | Bidirectional vanilla recurrent network |
| `ctrnn` | `ContinuousTimeRNN` | model | Continuous-time RNN integrated with Euler steps (`dt`, `substeps`) |
| `elman` | `ElmanNetwork` | model | Elman network: the classic simple recurrent net (hidden state fed back as context) |
| `encoder_decoder` | `EncoderDecoderTransformer` | model | Sequence-to-sequence Transformer over integer tokens: fit(src, tgt), predict(src) |
| `esn` | `EchoStateNetwork` | model | Echo state network: fixed random recurrent reservoir, ridge-regression readout |
| `grouped_query` | `GroupedQueryTransformer` | model | Transformer with grouped-query attention: query heads share `n_kv_heads` key/value heads (2 by default) |
| `gru` | `GRU` | model | Gated recurrent unit network |
| `jordan` | `JordanNetwork` | model | Jordan network: the previous output, not the hidden state, is fed back |
| `linear_attention` | `LinearAttentionTransformer` | model | Transformer with linear attention (positive feature map instead of softmax), linear in sequence length |
| `local_attention` | `LocalAttentionTransformer` | model | Transformer whose attention is restricted to a local window (`window`) |
| `lsm` | `LiquidStateMachine` | model | Liquid state machine: spiking (LIF) reservoir + ridge readout |
| `lstm` | `LSTM` | model | Long short-term memory network |
| `mamba` | `Mamba` | model | Mamba-style selective state-space model |
| `memory_transformer` | `MemoryAugmentedTransformer` | model | Memory-augmented Transformer: learned memory slots (`n_memory`) take part in attention |
| `multi_query` | `MultiQueryTransformer` | model | Transformer with multi-query attention: all query heads share one key/value head |
| `neural_ssm` | `NeuralStateSpace` | model | Neural state-space model: stable latent linear dynamics with a nonlinear emission network |
| `perceiver` | `Perceiver` | model | Perceiver: a small latent array (`n_latents`) cross-attends to the input |
| `plastic_rnn` | `PlasticRNN` | model | Differentiable plasticity: an RNN with learned Hebbian fast weights |
| `recurrent_attention` | `RecurrentAttention` | model | Recurrent attention: a GRU controller repeatedly queries the whole input (`n_glimpses`) |
| `retrieval_augmented` | `RetrievalAugmented` | model | Retrieval-augmented model. `knowledge=(keys, values)`: keys (M, d_in) live in the same space as the (unnormalised) input features, values (M, d_v) are what the model may read |
| `rnn` | `SimpleRNN` | model | Vanilla recurrent network |
| `rtrl_rnn` | `RTRLRNN` | model | Simple RNN trained with exact RTRL (no backward pass through time) |
| `s4` | `S4` | model | Structured state-space sequence model (diagonal S4D variant, FFT convolution) |
| `sparse_attention` | `SparseAttentionTransformer` | model | Transformer with sparse attention: a local window plus every `stride`-th position |
| `ssm_attention_hybrid` | `SSMAttentionHybrid` | model | Hybrid of Mamba-style state-space blocks and attention layers |
| `transformer_decoder` | `TransformerDecoder` | model | Causal Transformer; use fit(tokens) with vocab_size for language modelling |
| `transformer_encoder` | `TransformerEncoder` | model | Transformer encoder: bidirectional self-attention with a mean-pooled readout over the sequence |
| `universal_transformer` | `UniversalTransformer` | model | Universal Transformer: one shared layer applied repeatedly (`n_steps`) |

### memory (11)

| Registry name | Class | Kind | Description |
|---|---|---|---|
| `associative_memory` | `AssociativeMemory` | tool | Hetero-associative memory (key -> value) with Hebbian outer-product storage, or pseudo-inverse storage (rule='pinv') which is exact for linearly independent keys |
| `bm25` | `BM25Index` | tool | Okapi BM25 ranking over a growing list of documents |
| `case_library` | `CaseLibrary` | tool | Episodic memory of solved problems |
| `content_addressable_memory` | `ContentAddressableMemory` | tool | Key-value store read by cosine similarity: top-k entries, softmax-weighted |
| `dnc` | `DifferentiableNeuralComputer` | model | Differentiable Neural Computer: external memory with usage-based allocation and a temporal link matrix |
| `hopfield` | `HopfieldNetwork` | tool | Classical Hopfield network: binary (+1/-1) patterns stored with the Hebbian rule and recalled by iterating sign(W x) |
| `kv_memory_network` | `KeyValueMemoryNetwork` | model | Like MemoryNetwork, but each row is [key (d_key dims) \| value]; the query row uses its first d_key dims |
| `memory_network` | `MemoryNetwork` | model | X is (n, S+1, d): S memory slots, then the query as the last row |
| `modern_hopfield` | `ModernHopfield` | tool | Modern (dense-associative) Hopfield network: retrieval is softmax attention over the stored patterns, xi <- X^T softmax(beta X xi) |
| `ntm` | `NeuralTuringMachine` | model | Neural Turing Machine: an LSTM controller with an external memory addressed by content and location |
| `vector_store` | `VectorStore` | tool | Cosine-similarity k-NN store |

### metric (7)

Few-shot: `predict(X, support=(X_support, y_support))` works on unseen classes.

| Registry name | Class | Kind | Description |
|---|---|---|---|
| `contrastive_embedding` | `ContrastiveEmbedding` | model | Supervised contrastive embedding (SupCon): samples of the same class are pulled together, others pushed apart. fit(X, y); transform(X); predict by nearest class centroid |
| `matching` | `MatchingNetwork` | model | Attention over the support set (cosine similarity) votes for each class |
| `prototypical` | `PrototypicalNetwork` | model | Classify by distance to class prototypes (mean embeddings); few-shot and one-shot (k_shot=1) learning of new classes via predict(X, support=...) |
| `relation` | `RelationNetwork` | model | A learned relation module scores each (query, prototype) pair |
| `siamese` | `Siamese` | model | Siamese network (shared encoder) trained with the contrastive loss on pairs |
| `triplet` | `TripletNetwork` | model | Triplet network: X is (n, 3, d) = (anchor, positive, negative) |
| `twin_tower` | `TwinTower` | model | Two separate encoders (e.g. query and item) trained so that matching pairs score highest among the batch (in-batch softmax) |

### graph (15)

Input: `Graph` / `Tree` containers (dense adjacency, so small and medium graphs).

| Registry name | Class | Kind | Description |
|---|---|---|---|
| `gat` | `GraphAttentionNetwork` | model | Graph Attention Network: learned attention weights over each node's neighbours |
| `gcn` | `GraphConvolutionalNetwork` | model | Graph Convolutional Network (Kipf & Welling): symmetric-normalised neighbourhood averaging |
| `ggnn` | `GatedGraphNetwork` | model | Gated graph neural network (GRU updates, weights shared over propagation steps) |
| `gin` | `GraphIsomorphismNetwork` | model | Graph Isomorphism Network: sum aggregation + MLP, as expressive as 1-WL |
| `graph_net_recurrent` | `RecurrentGraphNetwork` | model | Graph network with global state and recurrent (weight-shared) updates |
| `graph_transformer` | `GraphTransformer` | model | Graph Transformer: attention among all nodes of a graph, with edges adding a learned bias to the scores |
| `graphsage` | `GraphSAGE` | model | GraphSAGE with mean aggregation (separate self and neighbour weights) |
| `hypergraph` | `HypergraphNeuralNetwork` | model | Hypergraph convolution; pass Graph(..., incidence=H) or use neighbourhoods as hyperedges |
| `mpnn` | `MessagePassingNetwork` | model | Message-passing neural network with learned edge messages and a GRU node update |
| `pointer_generator` | `PointerGenerator` | model | Sequence-to-sequence over integer tokens with copying: fit(src, tgt), predict(src) |
| `pointer_network` | `PointerNetwork` | model | fit(X, y): X is (n, T, d) set/sequence elements, y is (n, T) the order to point at (e.g. argsort for sorting). predict returns pointer indices decoded without repeats |
| `program_graph` | `ProgramGraphNetwork` | model | Neural program-graph network: a gated GNN with a separate message matrix per edge type (syntax child, next-token, data flow, control flow, ... and their reverses) for graphs built from source code |
| `rgcn` | `RelationalGCN` | model | Relational GCN: separate weights per edge type (Graph.rel_adj) |
| `tree_lstm` | `TreeLSTM` | model | Tree-LSTM over tree-structured inputs |
| `tree_nn` | `TreeStructuredNetwork` | model | Recursive (tree-structured) neural network |

### unsupervised (11)

| Registry name | Class | Kind | Description |
|---|---|---|---|
| `autoencoder` | `Autoencoder` | model | Autoencoder: compresses inputs to a latent code and reconstructs them |
| `byol` | `BYOL` | model | BYOL-style bootstrap latent learning (no negative pairs) |
| `contrastive` | `Contrastive` | model | SimCLR-style contrastive learning with the InfoNCE / NT-Xent objective |
| `deep_clustering` | `DeepClustering` | model | Deep Embedded Clustering: autoencoder pretraining, k-means on the latent space, then refinement (`n_clusters`) |
| `denoising_ae` | `DenoisingAutoencoder` | model | Denoising autoencoder: reconstructs clean inputs from noise-corrupted ones (`noise_std`) |
| `energy_based` | `EnergyBasedModel` | model | Energy-based model of the data distribution. transform(X) returns the energy of each sample (low = likely); sample(n) draws samples by Langevin dynamics |
| `hebbian` | `HebbianLearner` | model | Unsupervised feature learner trained with a local learning rule |
| `latent_prediction` | `LatentPrediction` | model | JEPA-style latent prediction: predicts a slowly moving target encoder's embedding of the full input from a masked view |
| `masked` | `MaskedPrediction` | model | Masked prediction: hides random features (`mask_ratio`) and predicts them from the rest |
| `temporal_contrastive` | `TemporalContrastive` | model | Temporal contrastive / predictive representation learning on sequences (n, T, d) |
| `vae` | `VAE` | model | Variational autoencoder with a `beta`-weighted KL term; `sample()` draws new points |

### classical (4)

Pure NumPy; input: 2-D array `(n_samples, n_features)`.

| Registry name | Class | Kind | Description |
|---|---|---|---|
| `decision_tree` | `DecisionTree` | classical | CART tree: Gini impurity (classification) or squared error (regression), exhaustive best splits |
| `kmeans` | `KMeans` | classical | k-means with k-means++ seeding and several restarts |
| `naive_bayes` | `NaiveBayes` | classical | Gaussian naive Bayes |
| `random_forest` | `RandomForest` | classical | Bagged decision trees with random feature subsets (sqrt(d) for classification, d/3 for regression) |

### agent (17)

| Registry name | Class | Kind | Description |
|---|---|---|---|
| `a2c` | `A2C` | agent | Advantage actor-critic: n-step rollouts, GAE advantages, entropy bonus |
| `actor_critic` | `ActorCritic` | agent | One-step online actor-critic: the critic's TD error is the actor's learning signal |
| `double_dqn` | `DoubleDQN` | agent | DQN with double Q-learning targets |
| `dqn` | `DQN` | agent | Deep Q-Network (replay + target network) |
| `dueling_dqn` | `DuelingDQN` | agent | DQN with a dueling value/advantage head |
| `dyna_q` | `DynaQ` | agent | Dyna-Q: Q-learning plus `planning_steps` simulated updates per real step, replayed from a learned model of the environment (model-based RL, tabular) |
| `expected_sarsa` | `ExpectedSARSA` | agent | Tabular Expected SARSA: updates towards the expected value under an epsilon-greedy policy |
| `model_based_mpc` | `ModelBasedMPC` | agent | Learns a neural dynamics + reward model from experience and plans with random-shooting model-predictive control (re-planning at every step) |
| `monte_carlo` | `MonteCarloControl` | agent | First-visit Monte Carlo control with an epsilon-soft policy |
| `ppo` | `PPO` | agent | Proximal Policy Optimization with the clipped surrogate objective |
| `prioritized_dqn` | `PrioritizedDQN` | agent | DQN with prioritized experience replay |
| `q_learning` | `QLearning` | agent | Tabular Q-learning (off-policy temporal-difference control) |
| `reinforce` | `REINFORCE` | agent | Monte Carlo policy gradient with a mean-return baseline (discrete or continuous actions) |
| `sac` | `SAC` | agent | Soft Actor-Critic for continuous actions: squashed-Gaussian policy, twin critics, target networks and an automatically tuned entropy temperature |
| `sarsa` | `SARSA` | agent | Tabular SARSA (on-policy temporal-difference control) |
| `sarsa_lambda` | `SARSALambda` | agent | SARSA(lambda) with accumulating eligibility traces |
| `trpo` | `TRPO` | agent | Trust Region Policy Optimization: natural-gradient step (conjugate gradient with Fisher-vector products) sized to a KL limit, followed by a backtracking line search |

### bandit (4)

| Registry name | Class | Kind | Description |
|---|---|---|---|
| `epsilon_greedy` | `EpsilonGreedyBandit` | agent | Epsilon-greedy multi-armed bandit |
| `linucb` | `LinUCB` | agent | Contextual bandit with a linear reward model per arm (disjoint LinUCB) |
| `thompson` | `ThompsonBandit` | agent | Beta-Bernoulli Thompson sampling (rewards in [0, 1]) |
| `ucb1` | `UCB1` | agent | UCB1 multi-armed bandit (upper-confidence-bound exploration) |

### search (3)

| Registry name | Class | Kind | Description |
|---|---|---|---|
| `genetic_algorithm` | `GeneticAlgorithm` | tool | Genetic algorithm over a list of typed genes |
| `simulated_annealing` | `SimulatedAnnealing` | tool | Simulated annealing (minimises energy). neighbor(state, rng) -> new state |
| `successive_halving` | `SuccessiveHalving` | tool | Successive halving: evaluate many configurations cheaply, keep the best 1/eta, give the survivors eta times more budget, repeat. sampler(rng) -> config; evaluate(config, budget) -> score (higher is better), where budget could be epochs, data size, or search depth |

### symbolic (2)

| Registry name | Class | Kind | Description |
|---|---|---|---|
| `datalog` | `Datalog` | tool | Bottom-up (forward-chaining) evaluation of Horn rules to a fixed point |
| `enumerative_synthesis` | `EnumerativeSynthesizer` | tool | Bottom-up enumerative synthesis of an integer expression from input/output examples |

### cognition (4)

| Registry name | Class | Kind | Description |
|---|---|---|---|
| `interleaved_practice` | `InterleavedPractice` | tool | Orders practice items from several topics. 'blocked' finishes one topic at a time; 'interleaved' mixes topics at random while avoiding runs longer than max_run (when other topics remain) |
| `metacognitive_monitor` | `MetacognitiveMonitor` | tool | Tracks (confidence, correct) pairs to measure calibration and to choose an abstention threshold (selective prediction): answer only when confidence is high enough to reach a target accuracy |
| `revision_loop` | `RevisionLoop` | tool | Predict -> act -> observe -> revise. propose(task, attempts) returns a candidate given the earlier attempts and their feedback (so it can re-plan instead of repeating itself); verify(candidate) returns (ok, feedback) |
| `spaced_repetition` | `SM2Scheduler` | tool | SuperMemo-2 spaced repetition. review(item, quality 0-5, day) schedules the next review; quality < 3 resets the item. due(day) lists items to practise |

### codegen (3)

| Registry name | Class | Kind | Description |
|---|---|---|---|
| `bpe_tokenizer` | `BPETokenizer` | tool | Byte-level byte-pair-encoding tokenizer with code-aware pre-splitting (identifiers, numbers, whitespace, punctuation) |
| `grammar_generator` | `GrammarGenerator` | tool | Generate programs by choosing grammar productions |
| `template_filler` | `TemplateFiller` | tool | Slot-based code construction: a template with {{slots}} is completed from candidate fragments |

### experiment (1)

| Registry name | Class | Kind | Description |
|---|---|---|---|
| `random_architecture` | `RandomArchitectureGenerator` | tool | Randomly assembles module graphs. connectivity: 'erdos_renyi' (each valid forward edge with probability p), 'small_world' (ring-like local edges with random rewiring) or 'sparse' (few edges) |
<!-- registry-table:end -->

Build by name with `sc.create("kan", hidden=(16,))`; filter with `sc.available("sequence")`.

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

- Image support is a single compact CNN, and the spiking / biologically inspired family has two models (LIF network, SOM); neither is a replacement for a dedicated vision or neuromorphic library.
- Most models are tested on small synthetic problems only. There is no GPU testing, and no handling of categorical features or missing values.
- Regression targets are not standardized inside models, so very badly scaled targets may train slowly (`sc.benchmark` and `prepare(scale_target=True)` standardize them for you).
- `sc.benchmark` compares models on one numeric array (it does not take raw text yet); its linear baseline is deliberately simple, so beating it is a low bar. Group sizes make the actual split fractions approximate, and with few seeds the std itself is noisy.
- The alternative learning rules (`forward_gradient`, `local_error`, `synthetic_gradients`, `equilibrium_propagation`, `contrastive_hebbian`, `predictive_coding`) need tuning on harder problems.
- `MoE` evaluates all experts and masks them, so it is correct but not sparse-efficient. `deep_ffn` is deliberately large for toy problems; shrink `hidden` for small datasets.
- Graph models use dense adjacency, so they suit up to a few thousand nodes in total. Deep RL is tested on tiny environments.
- `LwF` helps little in class-incremental settings, which is a known weakness of the method.
- Only a few techniques are implemented in each of the tool areas above (for example, search has three algorithms).

## Development

```bash
pip install -e . pytest
pytest tests                      # the full suite takes several minutes
python examples/compare_all.py
python examples/python_pipeline.py
```

The tests train and check each model family on small synthetic tasks and cover save/load round trips, so a regression in the shared trainer shows up across many families at once.

## Roadmap

- More biologically inspired models: STDP and reward-modulated STDP, liquid/LTC networks, neural cellular automata, ART, Hodgkin-Huxley neurons
- More search and planning: CMA-ES, NEAT, MAP-Elites, Bayesian optimization, A*, Monte Carlo tree search, SAT/SMT-based synthesis
- Program repair and mutation testing on top of the verification tools
- Text input inside `sc.benchmark`, subword tokenisation for `TextModel`; data preparation for graph data; larger image architectures and GPU benchmarking
