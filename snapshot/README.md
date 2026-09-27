# Reproduction snapshot

This directory contains the complete cached comparison and eight-recording demo
payload as ordinary Git blobs. Restore it with:

```bash
.venv/bin/emgbench snapshot restore
```

`manifest.json` records the comparison-configuration identity, each original
file's SHA-256/size and ordered chunks, and each compressed blob's SHA-256/size.
Files are split into at most 32 MiB chunks, then gzip-compressed. Blob names are
their compressed-content hashes. The CLI validates the manifest and payload,
restores original bytes atomically under `data/` and `artifacts/`, and rejects
conflicting files rather than overwriting another experiment.

See [reproduction instructions](../docs/reproduction.md) and
[methods](../docs/methods.md) for contents, verification, and provenance.

## Attribution and licence

The putEMG recordings and videos are by **Piotr Kaczmarek, Tomasz Mańkowski, and
Jakub Tomczyński**, Biomedical Engineering and Biocybernetics Team, Poznan
University of Technology. Source: <https://biolab.put.poznan.pl/putemg-dataset/>.

Kaczmarek, P.; Mańkowski, T.; Tomczyński, J.
*putEMG—A Surface Electromyography Hand Gesture Recognition Dataset.*
Sensors 2019, 19(16), 3548. <https://doi.org/10.3390/s19163548>.

Dataset material in this snapshot, including derived feature tables, replay
traces, and prediction/label tables, is distributed under **Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**:
<https://creativecommons.org/licenses/by-nc/4.0/>.

The original videos are byte-preserved. Derived assets apply the authors-compatible
filtering/feature extraction, transition masking, classification, and display-only
trace binning described in the methods. Metadata records source URLs, hashes,
upstream code pins, and transformations. The Apache-2.0 project-code licence does
not relicense this dataset material. No TabPFN weights are redistributed; the
snapshot preserves predictions and hosted-context metadata.
