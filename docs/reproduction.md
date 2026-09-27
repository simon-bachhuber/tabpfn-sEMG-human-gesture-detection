# Reproduction

The [fresh-clone verification record](verification.md) documents the successful
isolated restore, result/model checks, full test suite, and eight-user browser run.

## Fresh clone: restore results and run the demo

Use Git, [uv](https://docs.astral.sh/uv/), and Python **3.14.4** for the verified
environment. `uv` can obtain that Python version. No GPU is required. Allow about
**6 GB** for Git objects, the checkout, restored assets, dependencies, and browser.
The snapshot is stored as ordinary Git files: no Git LFS setup or asset download
service is involved.

```bash
git clone https://github.com/simon-bachhuber/tabpfn-sEMG-human-gesture-detection.git
cd tabpfn-sEMG-human-gesture-detection
uv sync --locked --python 3.14.4 --extra dev
.venv/bin/emgbench snapshot restore
.venv/bin/emgbench compare
.venv/bin/emgbench report
.venv/bin/emgbench demo
```

Open **http://127.0.0.1:8765**. Stop the foreground server with `Ctrl-C`.
For LAN access, use `emgbench demo --host 0.0.0.0 --port 8765`.

After installation and restoration, the comparison, report, training-replay cache,
and demo run **without external requests or an API key**. `compare` verifies the
saved data/model identities and reports **0 pending TabPFN tokens**. `report`
recomputes metrics from predictions, checks the shared test rows and training-only
scalers, and writes `results/comparison.{md,json}` and `results/provenance.json`.
It updates the report timestamp; original fit times/identities stay preserved.

The expected mean results are:

| Pipeline | Accuracy | Macro-F1 | Macro recall |
| --- | ---: | ---: | ---: |
| SVM + RMS | 77.80% | 0.6219 | 0.6038 |
| LDA + Du | 82.06% | 0.6844 | 0.6952 |
| TabPFN + Du | 84.69% | 0.7374 | 0.7308 |

## What the snapshot contains

[`snapshot/manifest.json`](../snapshot/manifest.json) inventories **853 restored
files**, with original-file SHA-256 hashes, sizes, ordered chunk references, and
compressed-chunk hashes. It is tied to the frozen comparison configuration.
The payload is **1,047.7 MB compressed / 1,164.5 MB restored** (decimal units).
Each original file is split into chunks of at most 32 MiB before gzip compression,
keeping every Git blob comfortably below GitHub's per-file limit.

- All **264 feature tables** and their filtering/feature provenance.
- Original manifests and held-out predictions for all three pipelines.
- Fitted SVM/LDA pipelines, verified against the original held-out labels.
- TabPFN's saved context metadata and original predictions/probabilities.
- Separate predictions for the **four in-sample training replays**.
- Eight ready-to-serve replay bundles, all 24 display traces, and the eight
  original **576p videos** with their download metadata.
- Public source catalog, cost-quote provenance, and raw-recording hash/URL sidecars.

Restoration verifies compressed chunks and original-file hashes before installing
each file atomically. It reuses matching files and rejects conflicting local work;
use a fresh workspace when a cache intentionally belongs to a different experiment.

```bash
.venv/bin/emgbench snapshot verify
.venv/bin/emgbench demo-training
.venv/bin/emgbench demo --export-only
```

`snapshot verify` checks both the committed payload and restored files.
`demo-training` verifies/reuses all four training recordings and their local videos.
`demo --export-only` validates the prepared replay and input hashes. Normal demo
startup uses that same cache path, so raw HDF5 and filtered-channel files are not
needed. The original ~10.5 GB raw signals and ~71 GB filter checkpoints can instead
be rebuilt using the full workflow below.

Workspaces default to the current directory. From the checkout, restore elsewhere
with `emgbench --workspace /path/to/workspace snapshot restore --directory snapshot`,
then use the same `--workspace` for `compare`, `report`, and `demo`.
The snapshot directory is separate from the installed Python package; retain the
checkout or explicitly pass its path to `--directory`.

## Tests and browser verification

Obtain the small pinned authors' checkout for feature/filter parity tests; this
does not download the signal dataset or run full preprocessing:

```bash
.venv/bin/emgbench reference
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
uv pip check --python .venv/bin/python
```

The test suite covers split integrity, per-recording masks, feature parity,
aggregation, fitted-model caching, API-free cache reuse, HTTP range serving, and
snapshot round-tripping/corruption/path handling. Reference tests explicitly skip
if the reference checkout is missing; obtain it to run the complete suite.

For browser checks:

```bash
uv sync --locked --python 3.14.4 --extra dev --extra browser
PLAYWRIGHT_BROWSERS_PATH=artifacts/browser-binaries .venv/bin/python -m playwright install chromium --only-shell
.venv/bin/emgbench demo
```

In another terminal in the checkout:

```bash
PLAYWRIGHT_BROWSERS_PATH=artifacts/browser-binaries .venv/bin/emgbench check-demo
```

This exercises all **eight recordings** and **three models**, role labels, shared
fitted-model identities, timestamp lookups, real MP4 playback, 24-channel traces,
probability overlays, cropping, seeking, and mobile layout. Results/screenshots
are saved to `artifacts/browser-check/`. On a minimal Linux installation,
Playwright may also need its documented OS browser dependencies.

## Full regeneration from raw recordings

This workflow computes new artifacts in a separate workspace. It requires Git,
`ffmpeg`/`ffprobe`, network access, roughly **100 GB** of working disk, and a Prior
Labs API token for new TabPFN predictions. Original preprocessing took about
**155 minutes with six workers**; the original pooled SVM fit took about 37 minutes.

Create an empty work directory, then run these commands from the installed checkout:

```bash
mkdir -p data/from-raw
.venv/bin/emgbench --workspace data/from-raw prepare --workers 6 --download-workers 3
.venv/bin/emgbench --workspace data/from-raw compare --models svm-rms lda-du
```

`prepare` acquires all 264 HDF5 recordings and eight 576p replay videos, validates
the source schema, and builds resumable per-channel filter and feature caches.
It uses the public Nextcloud/WebDAV source linked by putEMG. Published checksum
manifests were unavailable when checked; source sizes/ETags and locally computed
SHA-256 values are retained.

For TabPFN, supply `API_TOKEN` in the environment, or create
`data/from-raw/.env` using the name shown in [`.env.example`](../.env.example).
Quotes send dimensions/settings; the budget ceiling is checked before prediction.

```bash
.venv/bin/emgbench --workspace data/from-raw compare --models tabpfn-du --quote
.venv/bin/emgbench --workspace data/from-raw compare --models tabpfn-du --max-estimated-tokens 100000
.venv/bin/emgbench --workspace data/from-raw report
.venv/bin/emgbench --workspace data/from-raw demo-training --quote
.venv/bin/emgbench --workspace data/from-raw demo-training --max-estimated-tokens 100000
.venv/bin/emgbench --workspace data/from-raw demo --export-only --rebuild
.venv/bin/emgbench --workspace data/from-raw demo
```

The original held-out prediction quote was **78,720 tokens**; the additional
training replays were **69,262 tokens**. Check current quotes. New hosted fits
receive new context IDs and depend on the available service version. The committed
prediction snapshot preserves the original experiment independently of that service.
Fresh numerical-compatible feature tables can have different byte hashes from
the historical tables because the latter retain extra unused feature columns.

For durable long runs, append `--background` to `prepare`, `compare`, or
`demo-training`. Status/logs are under the selected workspace's `artifacts/jobs/`.
Wait for completion before starting the next stage.

## Updating the committed snapshot

After intentionally completing and verifying a new set of compatible caches:

```bash
.venv/bin/emgbench demo --export-only --rebuild
.venv/bin/emgbench snapshot build --directory snapshot-new
.venv/bin/emgbench snapshot verify --directory snapshot-new
```

The builder uses an explicit runtime-file allowlist. Review the new manifest and
its provenance before replacing the tracked snapshot. Compressed chunks are
content-addressed and deterministic; unchanged file chunks retain their identities.
Keep generated working copies under ignored `data/` and `artifacts/`.

For a release check, commit the intended source, documentation and snapshot, then
`git clone --no-local /path/to/repository /path/to/fresh-clone`. Install its own
environment and repeat the restore/check commands above without copying `.env`
or runtime directories. This ensures the clone stands on its committed contents.

## Layout

```text
readme.md                    Project overview and headline results
docs/                        Methods and reproduction instructions
pyproject.toml, uv.lock      Package metadata and locked dependency resolution
src/emgbench/                CLI, scientific workflow, snapshot tools, web demo
tests/                       Protocol, parity, caching, and serving tests
results/                     Comparison report and source/model provenance
snapshot/                    Tracked, hash-verified reproduction payload
data/, artifacts/            Ignored restored/generated runtime files
```

Project-code and dataset licensing are separate; see [LICENCE.md](../LICENCE.md)
and the [snapshot attribution](../snapshot/README.md).
