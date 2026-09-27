"""Load the pinned authors' implementation without editing the checkout."""

import importlib
import os
import subprocess
import sys
from functools import lru_cache

from .common import ROOT, settings

REFERENCE_ROOT = ROOT / "data/reference/putemg_examples"


def ensure_reference():
    """Download the exact reference revisions as an ignored runtime dependency."""
    if not REFERENCE_ROOT.exists():
        REFERENCE_ROOT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "git", "clone", "--no-checkout",
            "https://github.com/biolab-put/putemg_examples.git", str(REFERENCE_ROOT),
        ], check=True)
        subprocess.run([
            "git", "-C", str(REFERENCE_ROOT), "checkout", settings()["upstream"]["putemg_examples"],
        ], check=True)
        subprocess.run([
            "git", "-C", str(REFERENCE_ROOT), "submodule", "update", "--init", "--recursive",
        ], check=True)
    upstream()


@lru_cache(maxsize=1)
def upstream():
    expected = settings()["upstream"]
    paths = {
        "putemg_examples": REFERENCE_ROOT,
        "putemg_features": REFERENCE_ROOT / "putemg_features",
        "biolab_utilities": REFERENCE_ROOT / "putemg_features/biolab_utilities",
        "pyeeg": REFERENCE_ROOT / "putemg_features/pyeeg",
    }
    for name, path in paths.items():
        if not path.exists():
            raise RuntimeError("Missing reference checkout; run emgbench reference")
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected[name]:
            raise RuntimeError(f"Unexpected {name} revision: {actual}")
        subprocess.run(["git", "-C", str(path), "diff", "--exit-code", "HEAD", "--"], check=True, capture_output=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/runtime/matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REFERENCE_ROOT))
    features = importlib.import_module("putemg_features.features")
    utilities = importlib.import_module("putemg_features.biolab_utilities.putemg_utilities")
    filtering = importlib.import_module("putemg_features.biolab_utilities.filtering")
    return features, utilities, filtering
