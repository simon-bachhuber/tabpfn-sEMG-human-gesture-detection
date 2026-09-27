# Three-way putEMG comparison: fixed 40/4 participant split

Generated 2026-09-26T09:17:11.490989+00:00.

## Protocol

- Train on all retained windows from **40 participants / 80 sessions / 240 recordings**.
- Hold out **08, 24, 34, 39**, including both days: **24 test recordings**.
- **215,009 training windows**, **20,725 test windows**, zero target-user calibration.
- One pooled fit per pipeline. All three pipelines use identical window IDs, labels and exclusions.
- Primary scores average the four participant-level metrics equally, pooling both days within each person.
- SVM + RMS is the authors' highest-mean-accuracy published pipeline; LDA + Du is their strong Du comparator. Their paper used within-person/day evaluation. These results use the stated unseen-user split.

## Results

| Pipeline | Features | Mean accuracy | Macro-F1 | Macro recall | Fit | Predict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SVM + RMS | 24 | 77.80% | 0.6219 | 0.6038 | 2222.63s | 71.24s |
| LDA + Du | 144 | 82.06% | 0.6844 | 0.6952 | 4.20s | 0.02s |
| TabPFN + Du | 144 | 84.69% | 0.7374 | 0.7308 | 23.98s | 179.50s |

Classical models run locally. TabPFN times include API/network overhead. Common signal preprocessing is excluded; extra LDA probability calculation is recorded separately in JSON.

| Held-out user | SVM + RMS accuracy | LDA + Du accuracy | TabPFN + Du accuracy |
| --- | ---: | ---: | ---: |
| 08 | 80.06% | 81.56% | 86.29% |
| 24 | 85.79% | 91.38% | 93.76% |
| 34 | 73.89% | 77.90% | 85.88% |
| 39 | 71.46% | 77.41% | 72.84% |

This compares model–representation pipelines. The LDA/TabPFN pair holds Du features fixed; SVM uses RMS. Four held-out people give one fixed evaluation, not a population-wide superiority estimate. Per-gesture metrics, confusion matrices and per-recording scores are in `comparison.json`.

## Reproducibility

The package configuration records participant IDs, all model parameters and upstream source pins. `provenance.json` records source URLs/hashes, fitted scalers, input column order and prediction hashes. Cached results retain their original experiment identity rather than being relabeled as new fits.

The authors-compatible filter uses harmonic subtraction and zero-phase 20–700 Hz filtering. Windows contain 2,500 samples with a 1,250-sample hop. Targets are endpoint `TRAJ_GT`; the original transition mask is applied per recording. The released signed WL/WAMP definitions are retained. Idle is class 0; excluded/unconstrained -1 labels are not learned as idle.

Across preprocessing, 14 of 918,720 harmonic fits reported non-success. Their returned solutions are retained exactly as in the source algorithm; model-input features are finite.

## Demo

The demo shows the four held-out users plus four explicitly labeled training users and exactly these three pipelines. Training replay scores are in-sample and excluded from this report. Score cards describe the selected recording. It replays recorded video with `VIDEO_STAMP` alignment and cached predictions; offline filtering is noncausal. SVM preserves the authors' `probability=False` setting and therefore does not provide probabilities.

Restore the committed cache with `emgbench snapshot restore`, then run `emgbench demo`. See [the reproduction guide](../docs/reproduction.md) for verification, training-replay preparation, and complete regeneration commands.
