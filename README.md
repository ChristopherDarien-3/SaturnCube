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

> **Status: v0.1, early development.** Only the vector/tabular family is implemented (see below). The API may change.

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

List what is available with `sc.available()`, and build by name with `sc.create("kan", hidden=(16,))`.

## Example results

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

- sklearn wrappers (random forest, logistic regression) in the same interface
- Sequence models (RNN, LSTM, GRU, state-space models) and attention layers
- Metric-learning models, then graph and spiking networks as optional extras

## License

Add a license before publishing (for example MIT).
