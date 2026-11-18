# Changelog

## Unreleased
- **Probabilities and calibration:** `sc.log_loss`, `brier_score`, `expected_calibration_error`, `reliability_table` and
  `format_reliability`; `sc.PriorShift` (class-frequency correction estimated from labels), `sc.TemperatureScaler`, and
  `sc.Calibrated` (fits both on validation rows only, wraps any classifier). `class_weight=` (`"balanced"`, dict or list) for
  models that train with the standard cross-entropy; models with their own loss warn.
- **`sc.benchmark`:** `log_loss`, `brier` and `ece` as metrics (a `log_loss` column appears for classifiers with `predict_proba`),
  `result.reliability(model)`, stored test probabilities (`keep_predictions`). **Breaking:** the classification baseline is now
  labelled `class prior` (it was `majority class`) and gives the training class frequencies as probabilities, so its log loss is
  meaningful. `test_size` / `val_size` now default to `None` (0.25 / 0.15, or 0.1 / 0.1 for walk-forward). Failure notes say
  "runs", not "seeds".
- **Walk-forward validation:** `benchmark(..., split="walk_forward", n_folds, purge, embargo, window, time)`, `result.fold_table()`
  and `sc.walk_forward_splits`. With walk-forward the table's mean ± std is over folds (seeds are averaged within a fold).
- **`sc.SeedEnsemble`:** n seeds of one model averaged, with member probabilities and a disagreement measure.
- **`cnn1d` and `tcn`** in the sequence family (`(n, T, d)` or tokens), with `readout="max"` available for these two. `tcn` also does
  next-token prediction; `cnn1d` refuses it because it is not causal.
- New registry family `wrapper` (`calibrated`, `seed_ensemble`), shown in the generated README table. `base.resolve_model` accepts a
  registry name or a class.
- docs/CONV_SEQUENCE.md and scripts/bench_conv_vs_rnn.py: the benchmark behind the `cnn1d` / `tcn` claims. The scripts now run from
  a clone without installing the package.
- **Behaviour change:** `patience` now defaults to `"auto"`: 10 for regression, off for classification. Regression fits that used to
  run all `epochs` now hold out `val_split` (10%) of the training rows and may stop early with the best weights restored. The
  automatic default needs at least 20 held-out rows (`AUTO_MIN_VAL_ROWS`) and never warns; `patience=None` restores the old behaviour.
- `patience` now always has data to monitor: without `val_data=`, models hold out `val_split` (default 0.1, stratified for
  classification) of the training rows; `val_split=0` disables it and `patience` warns. New attributes `n_val_`, `best_epoch_`,
  `stopped_epoch_`. `partial_fit` never holds rows out. Early stopping is still off by default.
- `sc.warmup()` pays PyTorch's one-time optimizer start-up cost; `sc.benchmark` calls it so the first model's timing is fair.
- docs/TUNING.md and scripts/tune_defaults.py, scripts/tune_followup.py: measured evidence for the training defaults. No default
  learning rate or epoch count was changed.
- `sc.benchmark(models, X, y, groups=None, seeds=5, baselines=True)`: fair model comparison with group-disjoint splits, mean ± std
  over seeds, automatic majority and scaled scikit-learn baselines, and a printed table. Returns a `BenchmarkResult`.
- New optional extra `bench` (scikit-learn); `dev` now includes it.
- Text input: `TextVectorizer` (TF-IDF, optional SVD), `TokenEncoder` (padded token ids) and `TextModel`, which puts any
  registry model behind raw-text `fit` / `predict` / `evaluate` / `save` / `load`. Encoders are fitted on training text only
  and saved as JSON.
- The README model table is generated from the registry (`python scripts/update_readme.py`); a test fails if it drifts.
- Every registered class now has its own docstring (38 were missing); a test enforces it.
- `sc.available()` no longer lists `unit_tests` (not a model). `register(..., hidden=True)` and
  `available(include_hidden=True)` keep such helpers creatable by name.

## 0.7.0
- Automatic data preparation: `load_dataset`, `Dataset.report/prepare/prepare_sequences/prepare_images`, a saved-and-reloadable
  `Pipeline`, `auto_model`, `check_arrays`. Handles messy CSVs (formats, missing values, categories, dates), splits without leakage.
- Models now raise a clear error for text or NaN input instead of failing obscurely.

## 0.6.0
- Added `CNN`, a compact ResNet-style convolutional network for images (classification and regression).
- Added `SpikingNetwork`, a leaky integrate-and-fire network trained with surrogate gradients (direct or rate encoding).
- Added `SelfOrganizingMap`, with quantization / topographic error, U-matrix, hit map and unit labelling.
- Removed the experimental C-language helpers; program tools are Python-only.
- Packaging: full project metadata, `slow` test marker, CI and publish workflows.

## 0.5.0
- Search (genetic algorithm, successive halving, simulated annealing), program synthesis and Datalog,
  retrieval and case memory, learning strategies, verification and property testing, code generation,
  experiment design and statistics, and classical ML (random forest, k-means, naive Bayes).

## 0.4.0
- Sequence, attention, memory, metric / few-shot, graph, tree and pointer models; deep RL; meta-learning;
  more learning rules and optimizers.

## 0.1.0 - 0.3.0
- Vector/tabular networks, optimizers, unsupervised models, Hebbian rules, continual learning, tabular RL.
