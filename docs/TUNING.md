# Training defaults: what was measured

Run on 2026-11-07 with saturncube 0.7.0 + this change, torch 2.15.0, CPU only (1 core). Every number is the mean ± std over 5 seeds (each seed changes the split and the initialisation) from `sc.benchmark`, on random stratified splits of small datasets. `scripts/tune_defaults.py` and `scripts/tune_followup.py` reproduce everything here (they need scikit-learn). These are small problems and few seeds: treat differences below roughly 0.02 accuracy as noise.

Datasets: `synth3` (1000 rows, 8 features, XOR-style, the README example), `wine`, `breast_cancer`, `hard4` (800 rows, 4 classes, label noise, scaled logistic regression reaches only 0.395), `digits1000` (1000 of the 1797 digits), and `diabetes` (regression, r², target standardised by the harness). The follow-up adds `friedman1_n500` and `linear_noisy_n300` (regression).

## 1. Defaults vs the fixed `epochs=80` setup vs early stopping (`patience=10`, epochs doubled)

| dataset | model | default | epochs=80 | early stopping |
|---|---|---|---|---|
| synth3 | mlp | 0.854 ± 0.028 | 0.846 ± 0.028 | 0.837 ± 0.032 |
| synth3 | kan | 0.878 ± 0.018 | 0.885 ± 0.023 | 0.884 ± 0.019 |
| synth3 | rbf | 0.758 ± 0.032 | 0.730 ± 0.035 | 0.762 ± 0.026 |
| synth3 | random_nn | 0.902 ± 0.009 | 0.896 ± 0.018 | 0.891 ± 0.018 |
| wine | mlp | 0.964 ± 0.025 | 0.964 ± 0.025 | 0.969 ± 0.030 |
| wine | kan | 0.964 ± 0.012 | 0.969 ± 0.012 | 0.969 ± 0.012 |
| wine | rbf | 0.973 ± 0.019 | 0.978 ± 0.016 | 0.964 ± 0.012 |
| wine | random_nn | 0.987 ± 0.012 | 0.987 ± 0.012 | 0.987 ± 0.012 |
| breast_cancer | mlp | 0.976 ± 0.008 | 0.976 ± 0.008 | 0.972 ± 0.014 |
| breast_cancer | kan | 0.972 ± 0.011 | 0.972 ± 0.011 | 0.968 ± 0.015 |
| breast_cancer | rbf | 0.970 ± 0.008 | 0.966 ± 0.012 | 0.972 ± 0.009 |
| breast_cancer | random_nn | 0.966 ± 0.013 | 0.962 ± 0.015 | 0.972 ± 0.015 |
| hard4 | mlp | 0.475 ± 0.031 | 0.473 ± 0.034 | 0.423 ± 0.032 |
| hard4 | kan | 0.448 ± 0.008 | 0.439 ± 0.013 | 0.403 ± 0.030 |
| hard4 | rbf | 0.465 ± 0.024 | 0.447 ± 0.032 | 0.425 ± 0.030 |
| hard4 | random_nn | 0.467 ± 0.031 | 0.465 ± 0.019 | 0.417 ± 0.028 |
| digits1000 | mlp | 0.960 ± 0.008 | 0.960 ± 0.008 | 0.959 ± 0.005 |
| digits1000 | kan | 0.959 ± 0.010 | 0.959 ± 0.010 | 0.958 ± 0.013 |
| digits1000 | rbf | 0.947 ± 0.006 | 0.951 ± 0.007 | 0.949 ± 0.007 |
| digits1000 | random_nn | 0.954 ± 0.010 | 0.953 ± 0.012 | 0.952 ± 0.017 |
| diabetes(reg) | mlp | 0.209 ± 0.103 | 0.128 ± 0.101 | 0.410 ± 0.066 |
| diabetes(reg) | kan | 0.415 ± 0.042 | 0.388 ± 0.049 | 0.432 ± 0.076 |
| diabetes(reg) | rbf | 0.379 ± 0.057 | 0.413 ± 0.059 | 0.431 ± 0.050 |
| diabetes(reg) | random_nn | 0.241 ± 0.121 | 0.430 ± 0.052 | 0.443 ± 0.063 |

Baselines (mean): synth3: majority class 0.484, scaled logistic regression 0.477; wine: majority class 0.400, scaled logistic regression 0.987; breast_cancer: majority class 0.627, scaled logistic regression 0.968; hard4: majority class 0.255, scaled logistic regression 0.395; digits1000: majority class 0.112, scaled logistic regression 0.951; diabetes(reg): train mean -0.004, scaled ridge regression 0.441

Note: inside `sc.benchmark` the early-stopping runs used the harness's own validation split (15% of the data), so these rows measure early stopping with a supplied validation set; the automatic holdout (`val_split`, default 10% of the training rows) behaves the same way but also removes those rows from training.

## 2. Learning rate (model's default epochs)

| dataset | model | 1e-3 | 3e-3 | 1e-2 | 3e-2 | default lr |
|---|---|---|---|---|---|---|
| synth3 | mlp | 0.835 | 0.854 | 0.882 | 0.880 | 3e-3 |
| synth3 | kan | 0.738 | 0.878 | 0.880 | 0.885 | 3e-3 |
| synth3 | rbf | 0.586 | 0.705 | 0.758 | 0.774 | 1e-2 |
| synth3 | random_nn | 0.572 | 0.874 | 0.902 | 0.915 | 1e-2 |
| wine | mlp | 0.969 | 0.964 | 0.973 | 0.960 | 3e-3 |
| wine | kan | 0.951 | 0.964 | 0.969 | 0.973 | 3e-3 |
| wine | rbf | 0.973 | 0.978 | 0.973 | 0.969 | 1e-2 |
| wine | random_nn | 0.956 | 0.987 | 0.987 | 0.982 | 1e-2 |
| breast_cancer | mlp | 0.979 | 0.976 | 0.973 | 0.976 | 3e-3 |
| breast_cancer | kan | 0.969 | 0.972 | 0.973 | 0.969 | 3e-3 |
| breast_cancer | rbf | 0.962 | 0.962 | 0.970 | 0.966 | 1e-2 |
| breast_cancer | random_nn | 0.970 | 0.966 | 0.966 | 0.966 | 1e-2 |
| hard4 | mlp | 0.471 | 0.475 | 0.475 | 0.481 | 3e-3 |
| hard4 | kan | 0.421 | 0.448 | 0.425 | 0.410 | 3e-3 |
| hard4 | rbf | 0.410 | 0.436 | 0.465 | 0.475 | 1e-2 |
| hard4 | random_nn | 0.420 | 0.446 | 0.467 | 0.452 | 1e-2 |
| digits1000 | mlp | 0.960 | 0.960 | 0.956 | 0.949 | 3e-3 |
| digits1000 | kan | 0.957 | 0.959 | 0.957 | 0.956 | 3e-3 |
| digits1000 | rbf | 0.910 | 0.935 | 0.947 | 0.946 | 1e-2 |
| digits1000 | random_nn | 0.944 | 0.954 | 0.954 | 0.950 | 1e-2 |
| diabetes(reg) | mlp | 0.376 | 0.209 | 0.113 | 0.109 | 3e-3 |
| diabetes(reg) | kan | 0.441 | 0.415 | 0.244 | 0.081 | 3e-3 |
| diabetes(reg) | rbf | 0.443 | 0.440 | 0.379 | 0.235 | 1e-2 |
| diabetes(reg) | random_nn | 0.441 | 0.450 | 0.241 | 0.034 | 1e-2 |

## 3. Regression follow-up (r², 5 seeds; scaled ridge regression is the baseline)

| dataset | model | default | lr=1e-3 | lr=3e-3 | early stopping |
|---|---|---|---|---|---|
| diabetes | mlp | 0.209 ± 0.103 | 0.376 ± 0.088 | 0.209 ± 0.103 | 0.410 ± 0.066 |
| diabetes | kan | 0.415 ± 0.042 | 0.441 ± 0.065 | 0.415 ± 0.042 | 0.432 ± 0.076 |
| diabetes | rbf | 0.379 ± 0.057 | 0.443 ± 0.052 | 0.440 ± 0.049 | 0.431 ± 0.050 |
| diabetes | random_nn | 0.241 ± 0.121 | 0.441 ± 0.059 | 0.450 ± 0.059 | 0.443 ± 0.063 |
| diabetes | *scaled ridge* | 0.441 | | | |
| friedman1_n500 | mlp | 0.863 ± 0.027 | 0.856 ± 0.031 | 0.863 ± 0.027 | 0.865 ± 0.027 |
| friedman1_n500 | kan | 0.883 ± 0.023 | 0.864 ± 0.021 | 0.883 ± 0.023 | 0.896 ± 0.029 |
| friedman1_n500 | rbf | 0.831 ± 0.025 | 0.690 ± 0.048 | 0.741 ± 0.036 | 0.744 ± 0.096 |
| friedman1_n500 | random_nn | 0.870 ± 0.033 | 0.709 ± 0.036 | 0.861 ± 0.022 | 0.876 ± 0.034 |
| friedman1_n500 | *scaled ridge* | 0.676 | | | |
| linear_noisy_n300 | mlp | 0.929 ± 0.005 | 0.928 ± 0.005 | 0.929 ± 0.005 | 0.910 ± 0.032 |
| linear_noisy_n300 | kan | 0.933 ± 0.015 | 0.908 ± 0.026 | 0.933 ± 0.015 | 0.933 ± 0.017 |
| linear_noisy_n300 | rbf | 0.951 ± 0.012 | 0.804 ± 0.044 | 0.950 ± 0.008 | 0.953 ± 0.007 |
| linear_noisy_n300 | random_nn | 0.924 ± 0.010 | 0.907 ± 0.031 | 0.931 ± 0.013 | 0.908 ± 0.030 |
| linear_noisy_n300 | *scaled ridge* | 0.965 | | | |

## 4. Where does the seed spread come from? (`hard4`, accuracy, 8 runs each)

| model | vary initialisation only (one split) | vary the split only (one initialisation) |
|---|---|---|
| mlp | 0.491 ± 0.021 | 0.473 ± 0.026 |
| kan | 0.483 ± 0.016 | 0.458 ± 0.035 |
| rbf | 0.504 ± 0.025 | 0.478 ± 0.028 |
| random_nn | 0.483 ± 0.030 | 0.458 ± 0.027 |
## What follows from this

1. **Early stopping stays opt-in (`patience=None` by default).** On the noisy four-class problem it cost every model accuracy (mlp -0.052, kan -0.045, rbf -0.040, random_nn -0.050 against the defaults), with small losses on `synth3` for mlp (-0.017) and random_nn (-0.011). On small noisy regression it helped a lot: `diabetes` r² mlp 0.209 -> 0.410, random_nn 0.241 -> 0.443, rbf 0.379 -> 0.431, kan 0.415 -> 0.432. On the two other regression sets it was neutral to slightly negative (mlp -0.019 and random_nn -0.016 on `linear_noisy_n300`, rbf -0.087 on `friedman1_n500`). Averaged over the three regression sets: mlp +0.062, random_nn +0.064, kan +0.010, rbf -0.011. So `patience=10` is worth trying for mlp or random_nn on small, noisy regression data, and not a safe default.
2. **No learning-rate default changed.** On the five classification sets the default learning rate is within 0.03 accuracy of the best of the four rates tried for every model (largest gaps: mlp 0.029 on `synth3`, rbf 0.016, random_nn 0.013, kan 0.009), which is about one seed std. `random_nn` at 3e-3 gains 0.069 r² on average over the three regression sets, but nearly all of that is `diabetes`, and it loses about 0.03 on `synth3`; one small dataset is not enough to move a default for both task types. For `rbf`, lowering the rate on regression hurts (1e-3: -0.074 on average).
3. **Keep `rbf` and `random_nn` at their default 150 epochs.** They are designed for `lr=1e-2` and 150 epochs; forcing `epochs=80` cost `rbf` 0.027 on `synth3` and 0.018 on `hard4`. A fixed `epochs=80` for every model is part of the spread in a comparison, not a fair setup.
4. **The `kan` / `rbf` spread you saw was not reproduced here.** On `hard4`, varying only the initialisation gives a std of 0.016 (kan) to 0.030 (random_nn); varying only the split gives 0.026 to 0.035. `kan` had the smallest spread of the four. Ranges as wide as 0.43 to 0.61 would need a much noisier test set; with few groups per split (theorem families, source images) the test composition changes a lot between seeds, which `sc.benchmark(groups=...)` will show as a large std. That is a hypothesis about your data, not something measured here.
5. **The slow first `mlp` fit is PyTorch, not SaturnCube.** The first `torch.optim` optimizer built in a process makes PyTorch import about 900 modules lazily: roughly 1.1 to 1.3 s on a warm machine here, 15 s on a cold disk, and 0.05 ms afterwards. It happens for Adam, AdamW, SGD and RMSprop alike and `foreach=False` does not avoid it. `sc.warmup()` pays it on demand and `sc.benchmark` calls it before timing, so the first model's "fit s" is not inflated.
