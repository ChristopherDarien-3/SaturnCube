# 1-D convolutional models: what was measured

`cnn1d` and `tcn` (the sequence family, input `(n, T, d)`) were compared with `gru` and `lstm` on synthetic windows with
`sc.benchmark` (random stratified splits, 40 epochs and hidden size 32 for every model, `scripts/bench_conv_vs_rnn.py`
reproduces everything here; the logistic-regression baseline row needs scikit-learn). Every cell is mean ± std over
seeds (5 seeds, 3 for tasks C and D); each seed changes the split and the initialisation. These are small synthetic
problems: nothing here says how the models behave on real data.

Tasks: **A** noise windows (1,200 rows, T = 48, 2 channels); positives hold a 5-step motif at a random position and a
quarter of the negatives hold the time-reversed motif as a distractor. **E** is A with only 300 rows. **B** the label is
the XOR of the sign of channel 0 at the first and last step. **C** the label is the sign of channel 0 at the first step
(T = 96). **D** is C with the middle step.

## A. Local motif, 1,200 rows

| model | accuracy | log loss | fit s |
|---|---|---|---|
| cnn1d (max pool) | 0.954 ± 0.011 | 0.171 ± 0.054 | 4.8 |
| gru | 0.950 ± 0.011 | 0.191 ± 0.020 | 4.4 |
| tcn (max pool) | 0.945 ± 0.006 | 0.234 ± 0.047 | 9.0 |
| cnn1d (mean pool, the default) | 0.939 ± 0.014 | 0.216 ± 0.043 | 4.9 |
| tcn (last step, the default) | 0.913 ± 0.011 | 0.452 ± 0.062 | 9.1 |
| lstm | 0.769 ± 0.224 | 0.397 ± 0.268 | 2.1 |
| scaled logistic regression *(baseline)* | 0.659 ± 0.014 | 0.690 ± 0.024 | 0.0 |
| class prior *(baseline)* | 0.507 ± 0.000 | 0.693 ± 0.000 | 0.0 |

## E. Local motif, 300 rows

| model | accuracy | log loss | fit s |
|---|---|---|---|
| tcn (max pool) | 0.927 ± 0.025 | 0.254 ± 0.067 | 2.6 |
| cnn1d (max pool) | 0.922 ± 0.015 | 0.263 ± 0.044 | 1.2 |
| gru | 0.916 ± 0.035 | 0.266 ± 0.105 | 1.2 |
| cnn1d (mean pool, the default) | 0.830 ± 0.035 | 0.407 ± 0.064 | 1.3 |
| lstm | 0.795 ± 0.143 | 0.474 ± 0.119 | 0.5 |
| scaled logistic regression *(baseline)* | 0.616 ± 0.065 | 1.341 ± 0.314 | 0.0 |
| tcn (last step, the default) | 0.562 ± 0.026 | 1.937 ± 0.282 | 2.5 |
| class prior *(baseline)* | 0.514 ± 0.000 | 0.693 ± 0.000 | 0.0 |

## B. XOR of the two ends (no model learned it)

| model | accuracy | log loss | fit s |
|---|---|---|---|
| cnn1d (max pool) | 0.537 ± 0.048 | 1.228 ± 0.153 | 5.1 |
| tcn (last step) | 0.519 ± 0.023 | 2.723 ± 0.257 | 9.1 |
| tcn 5 layers (last step) | 0.513 ± 0.030 | 2.922 ± 0.258 | 11.7 |
| lstm | 0.513 ± 0.030 | 1.434 ± 0.137 | 2.0 |
| gru | 0.510 ± 0.022 | 1.435 ± 0.181 | 4.6 |
| scaled logistic regression *(baseline)* | 0.505 ± 0.013 | 0.769 ± 0.025 | 0.0 |
| class prior *(baseline)* | 0.500 ± 0.000 | 0.693 ± 0.000 | 0.0 |
| tcn (max pool) | 0.490 ± 0.029 | 1.195 ± 0.161 | 8.9 |

## C. The sign of the first step, window of 96

| model | accuracy | log loss | fit s |
|---|---|---|---|
| cnn1d (max pool) | 0.989 ± 0.005 | 0.032 ± 0.005 | 8.3 |
| scaled logistic regression *(baseline)* | 0.899 ± 0.019 | 0.269 ± 0.016 | 0.0 |
| tcn (last step, field 61) | 0.513 ± 0.021 | 2.740 ± 0.153 | 15.9 |
| tcn 5 layers (last step, field 125) | 0.503 ± 0.046 | 2.798 ± 0.085 | 21.7 |
| class prior *(baseline)* | 0.500 ± 0.000 | 0.693 ± 0.000 | 0.0 |
| lstm | 0.497 ± 0.015 | 1.498 ± 0.119 | 3.4 |
| gru | 0.491 ± 0.023 | 1.490 ± 0.222 | 8.1 |

## D. The sign of the middle step, window of 96

| model | accuracy | log loss | fit s |
|---|---|---|---|
| scaled logistic regression *(baseline)* | 0.893 ± 0.017 | 0.226 ± 0.021 | 0.0 |
| tcn (last step, field 61) | 0.819 ± 0.100 | 0.890 ± 0.573 | 15.8 |
| gru | 0.522 ± 0.013 | 1.378 ± 0.111 | 8.5 |
| class prior *(baseline)* | 0.517 ± 0.000 | 0.693 ± 0.000 | 0.0 |
| cnn1d (max pool) | 0.510 ± 0.065 | 1.224 ± 0.150 | 9.4 |

## What follows from this

1. **On the motif tasks the convolutional models match `gru`; they do not beat it.** With max pooling, `cnn1d` minus `gru`
   is +0.004 ± 0.015 accuracy at 1,200 rows and +0.005 ± 0.037 at 300 rows (paired by seed); `tcn` minus `gru` is -0.005 ± 0.015
   and +0.011 ± 0.029. All within noise. At 1,200 rows `tcn` had a worse log loss than `gru` in all 5 seeds (+0.042 ± 0.044), and
   it takes about twice as long to fit as `cnn1d` or `gru`. `lstm` was unstable here (0.769 ± 0.224 on A).
2. **The readout decides a pattern-anywhere task, and the defaults are the worse choice here.** With the default
   readouts (`cnn1d` mean pooling, `tcn` last step) accuracy was 0.939 and 0.913 on A and 0.830 and 0.562 on E, against
   0.954 and 0.945 (A) and 0.922 and 0.927 (E) with `readout="max"`. The defaults were not tuned on this task and a mean
   or last-step readout suits other problems; for "is this pattern anywhere in the window" use `readout="max"`.
3. **`cnn1d`'s score on C is not memory.** It reached 0.989 on C but 0.510 (chance) on D, where the only change is that
   the label bit sits in the middle of the window. A convolution with zero padding can tell the first position apart because
   its neighbourhood includes padding, and global max pooling can then pick that position out; the middle has no such mark.
   This is a known effect of zero padding. The practical consequence is that `cnn1d` is translation-invariant away from the
   edges of the window (tested), not at them.
4. **A receptive field is necessary, not sufficient.** `tcn` (field 61) could not see the first step on C and scored 0.513;
   on D the middle step is inside its field and it reached 0.819 ± 0.100 (high variance), while `gru` stayed at chance (0.522).
   But `tcn` with 5 layers (field 125, which covers the whole window of C) also stayed at chance on C (0.503).
5. **Failed designs, kept for transparency.** On B no model learned the XOR (all between 0.49 and 0.54), and their log
   losses (1.2 to 2.9) were far worse than the class prior's 0.693, which is what over-fit noise looks like. On C `gru`,
   `lstm` and both `tcn` models stayed at chance within 40 epochs and 720 training rows, and the scaled logistic regression
   scored 0.899 because the label is a single input feature. That shows these recurrent models did not learn the task in
   this budget, not that they cannot; longer training was not tried.

Not tested: real data, other sizes (only hidden = 32), more epochs, other pooling for `tcn`, seeds beyond 5.
