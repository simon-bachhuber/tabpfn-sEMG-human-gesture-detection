# Fresh-clone verification

**Passed on 2026-09-26**, using source/assets from commit
`951697513f67dfbc2bea3bf3275c00bc343e77c7`.
The follow-up documentation commit records this check without changing those
runtime files or the snapshot. Machine-readable evidence is in
[`results/clone-verification.json`](../results/clone-verification.json).

## Isolation

- Cloned the committed repository with `git clone --no-local`; no shared Git
  object store, copied runtime directories, or copied virtual environment.
- Installed a new Python **3.14.4** environment using `uv sync --locked` with
  fresh package downloads into the clone's own package cache.
- Confirmed the interpreter and imported `emgbench` package belonged to the clone.
- No `.env`, `API_TOKEN`, or `TABPFN_TOKEN`; no raw HDF5 recordings or filtered
  channel checkpoints.
- Obtained the pinned reference checkout and browser during setup. During cached
  scientific verification, Python external connections/DNS were blocked; there
  were **zero attempted external requests**. Loopback access allowed browser checks.

## Results

| Check | Result |
| --- | --- |
| Snapshot restore and re-verification | All **853 files** passed SHA-256 verification |
| Three cached comparison pipelines | Correct split, feature hashes, rows, parameters and training-only scalers; **0 pending API tokens** |
| Cached training replays and local videos | Four training users verified; **0 pending API tokens** |
| Recomputed report | Exact committed contents except `generated_at` |
| Recomputed source/model provenance | Exact match |
| Restored SVM inference | All **20,725 held-out** and **3,026 training-replay** labels exactly matched |
| Restored LDA inference | All **20,725 held-out** and **3,026 training-replay** labels exactly matched; held-out probabilities matched to numerical tolerance |
| Unit/reference tests | **37 passed**, no skips |
| Ruff and installed dependency checks | Passed |
| Browser replay | All eight users and three models; **88 timestamp** and **144 distribution** checks passed |
| Browser errors | **0** |

The browser also exercised actual decoded-video playback, seeking, shared fit
identities, training/held-out labels, the 24-channel 8×3 trace layout, 85%-width
video crop, probability toggle and zero states, and mobile layout. All videos
were the committed 576p versions. The temporary server was stopped afterward.

The combined snapshot/report/model/browser verification took **148.12 seconds**
after environment/browser setup. This checked cached reproduction and inference
from the saved classical models; it did not repeat the hours-long raw preprocessing
or purchase new hosted TabPFN inference.

Snapshot identity:

```text
26b22b1de1717a3670356e8f4fee70544e4a6fde08a9fe609443d4415d0111ba
```

Follow the [reproduction guide](reproduction.md) to repeat the normal restore,
comparison, reporting, test, and browser checks.
