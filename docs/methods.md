# Methods

## Question and evaluation protocol

We compare three model–representation pipelines on unseen putEMG participants:
the authors' strongest published mean-accuracy pipeline, **SVM + RMS**; their
**LDA + Du** comparator; and **TabPFN-3.5-Plus + Du**. The LDA/TabPFN pair holds
the representation fixed. SVM/RMS versus either Du pipeline changes both the
classifier and the representation.

The dataset contains 44 participants, two recording days per participant, and
three gesture trajectories per day (`sequential`, `repeats_long`, `repeats_short`):
**264 recordings**. We train one pooled model per pipeline on 40 participants
(240 recordings), holding out **08, 24, 34, 39** (24 recordings). Both days from
each person belong to the same side. There is no target-user calibration or
test-dependent scaling.

The four held-out IDs were selected with seed 42 before evaluation, excluding
development users 03/04 from selection. Those two users remain in training because
pilot results informed the feature choices. This is one fixed evaluation, not a
repeated split or an estimate of population-wide superiority. The authors'
published results use within-person/day evaluation, a different protocol.

The authoritative participant lists, parameters, and upstream pins are in
[`src/emgbench/comparison.json`](../src/emgbench/comparison.json).

## Signal processing and labels

- All **24 EMG channels** are used. Nominal acquisition rate: **5120 Hz**.
- Filtering preserves the authors' harmonic subtraction and fifth-order,
  zero-phase **20–700 Hz** bandpass, including their hardcoded filtering rate of
  **5124.07211903 Hz**. The pinned source is loaded without modification.
- Windows contain **2,500 samples**, with a **1,250-sample hop**: approximately
  488 ms and 244 ms at the nominal rate. These are the released XML's sample
  counts; the paper's stated counts and durations are inconsistent.
- Targets are **`TRAJ_GT` at the window endpoint**. Video, timestamps, subject IDs,
  trajectory names, and other metadata are never predictors.
- Labels are `0, 1, 2, 3, 6, 7, 8, 9`: idle, fist, flexion, extension, and
  index/middle/ring/small-finger pinch. **`-1` is excluded/unconstrained**, not idle.
- The authors' transition mask is applied **independently per recording**, before
  pooling. At a change to a positive label, the preceding and current feature
  rows are rejected; `-1` intervals and four following rows are rejected. The
  released implementation's semantics are tested against the pinned source.
  This mask does not guarantee that every retained window is label-pure.

After masking, every pipeline sees exactly **215,009 training windows** and
**20,725 held-out windows**, with matching window IDs and labels. Feature arrays
are checked for finite values. Across the original preprocessing, **14 of
918,720** harmonic optimizer calls reported non-success; their returned solutions
were retained, as in the authors' algorithm. Diagnostics are preserved per channel.

## Features

**RMS** gives 24 columns, one per channel. **Du** gives 144 columns: WL, ZC, SSC,
IAV, VAR, and WAMP for each of the 24 channels. Columns are feature-major, then
channel 1–24. With window values `x` and consecutive differences `d = diff(x)`:

| Feature | Released implementation used here |
| --- | --- |
| RMS | `sqrt(mean(x ** 2))` |
| WL | `sum(d)` — signed, without absolute values |
| ZC | Sign changes after removing samples with `abs(x) <= 30` |
| SSC | Count of `d[:-1] * d[1:] <= -16` |
| IAV | `sum(abs(x))` |
| VAR | Population variance, `var(x)` |
| WAMP | Count of `d >= 10` — signed, not `abs(d) >= 10` |

The signed WL/WAMP definitions intentionally reproduce the released code, even
where conventional feature definitions differ. Parity tests cover the selected
features, exact thresholds, transition mask, and filtering behavior.

The preserved historical feature tables also contain unused columns. They are
kept byte-for-byte to preserve their hashes; only the explicit RMS/Du column lists
enter a model. Fresh preprocessing emits the seven required feature families.

## Models

Each pipeline fits `StandardScaler` **only on retained training rows**. Its means,
scales, and ordered column names are retained and verified on cached reproduction.

| Pipeline | Settings |
| --- | --- |
| SVM + RMS | RBF SVC; `C=50`, `gamma="auto"`, shrinking enabled, tolerance 0.001, no class weights, `probability=False`; original fit kernel cache 200 MB |
| LDA + Du | SVD solver, empirical priors, no shrinkage, tolerance `1e-4` |
| TabPFN + Du | Hosted **TabPFN-3.5-Plus**, version `v3.5`, `model_path="v3.5_default"`, 8 estimators, seed 42; Thinking disabled |

TabPFN receives all 215,009 standardized training rows and 144 Du columns, then
predicts the held-out rows in one batch. Its original prediction quote was
**78,720 tokens**. Saved API model metadata records the actual model path and
fitted context ID. This experiment uses the Plus/default model, not Fast.

The snapshot includes fitted local SVM/LDA pipelines. Because the original
comparison saved predictions but not classical weights, these pipelines were
reconstructed on the same full training pool. Their labels matched **all 20,725
original held-out predictions exactly**. SVM reconstruction used a 2048 MB kernel
cache as a memory/runtime optimization; learning settings stayed fixed. Both
the original comparison and reconstruction provenance are retained.

TabPFN's saved model JSON is a hosted-context reference, not model weights. The
committed predictions are sufficient for cached reproduction and playback;
predicting new inputs requires the service and authentication.

## Metrics and results

Accuracy and fixed-eight-class macro precision/recall/F1 are computed **per held-out
participant**, pooling both days within that person. The headline scores are the
unweighted mean over the four people. Undefined class precision/recall contributes
zero, matching `zero_division=0`.

The [comparison report](../results/comparison.md) includes mean and per-person
results. [JSON results](../results/comparison.json) additionally contain
confusion matrices, per-gesture metrics, per-recording scores, and pooled-window
metrics. Pooled-window scores are distinct from the primary equal-participant mean.

Reported fit/prediction times come from the original runs. Classical models ran
locally; TabPFN times include API/network overhead. Common preprocessing is
excluded, and LDA's additional probability calculation is recorded separately.

## Training examples and video demo

The demo contains one first-day `sequential` recording for each held-out user and
training users **03, 04, 05, 06**. The four selected training recordings contain
**3,026 retained windows**. They reuse the same pooled models as the held-out
comparison; their predictions and metrics live in a separate training-replay cache.

Their mean recording-level macro-F1 is SVM **0.6877**, LDA **0.5235**, and TabPFN
**1.0000**. These are **in-sample examples**, not a full-training-set evaluation or
held-out evidence. TabPFN had these examples and labels in its training context.
The four training recordings are excluded from every held-out result above.

The demo replays 576p hand video, with predictions at window endpoints aligned by
`VIDEO_STAMP` in video seconds. Negative timestamps can occur before video zero.
Display traces are min/max bins of 64 filtered samples; this binning does not
alter model inputs. All channels appear in an **8-row × 3-column** layout.

The video displays the left **85%** of the source frame, retaining its full
height. A default-off toggle displays model-specific class probabilities in the
video. Unscored intervals keep eight zero-valued placeholder bars. SVM has no
probabilities under its original configuration, so it also uses zero placeholders.
Score cards describe the selected recording, and training/held-out roles are
explicitly labeled.

This is **offline replay**: zero-phase filtering is noncausal and uses future
samples. Playback is not a measurement of real-time inference latency.

## Provenance

[`results/provenance.json`](../results/provenance.json) records source URLs and
hashes, original experiment IDs, ordered input columns, training-only scalers,
environments, and prediction hashes. The snapshot preserves original artifact
bytes and identities rather than relabeling old results as newly fitted models.
Raw-source sidecars are retained even though raw signals are unnecessary for
cached reproduction.

Pinned reference revisions:

| Repository | Commit |
| --- | --- |
| `putemg_examples` | `48554520b69584cad48530e9339cec37a457555b` |
| `putemg_features` | `c2ada7c13b468771f93c0ccf96fd0091ed6eb195` |
| `biolab_utilities` | `c032e37535e696c028ae74bc4fdaea01af0d8f81` |
| `pyeeg` | `e5d34f8e8dfd976b3c52e6c58f80306028275798` |

Dataset/paper: Kaczmarek, P.; Mańkowski, T.; Tomczyński, J.
*putEMG—A Surface Electromyography Hand Gesture Recognition Dataset.*
Sensors 2019, 19(16), 3548. [doi:10.3390/s19163548](https://doi.org/10.3390/s19163548).
