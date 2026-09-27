import numpy as np
import pandas as pd
import pytest
from scipy.signal import butter, filtfilt

from emgbench.processing import FEATURE_ORDER, extract_features
from emgbench.protocol import masked_labels
from emgbench.reference import REFERENCE_ROOT, upstream

pytestmark = pytest.mark.skipif(not REFERENCE_ROOT.exists(), reason="Run emgbench reference to obtain the pinned reference")


@pytest.mark.parametrize("feature", FEATURE_ORDER)
def test_selected_features_match_the_authors(feature):
    reference, _, _ = upstream()
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({f"EMG_{i}": rng.normal(0, 60, 10000) for i in range(1, 25)}, index=np.arange(10000) / 5124.07211903)
    frame["TRAJ_GT"] = 0
    frame["TRAJ_1"] = 0
    actual = extract_features(frame)
    assert len([c for c in actual if c.split("_")[0] in FEATURE_ORDER]) == 168
    args = {"window": 2500, "step": 1250}
    if feature in {"ZC", "SSC", "WAMP"}:
        args["threshold"] = {"ZC": 30, "SSC": 16, "WAMP": 10}[feature]
    for channel in (1, 9, 24):
        expected = getattr(reference, "feature_" + feature.lower())(frame[f"EMG_{channel}"], **args)
        np.testing.assert_array_equal(actual.index, expected.index)
        np.testing.assert_allclose(actual[f"{feature}_{channel}"], expected, rtol=1e-13, atol=1e-12)


def test_legacy_signed_features_and_exact_thresholds():
    frame = pd.DataFrame({"EMG_1": [31.0, -31, 0, 30, -30, 40], "TRAJ_GT": [0] * 6, "TRAJ_1": [0] * 6})
    actual = extract_features(frame, window=6, hop=3).iloc[0]
    assert actual.WL_1 == 9
    assert actual.WAMP_1 == 3
    assert actual.ZC_1 == 2


def test_mask_matches_reference_on_varied_label_runs():
    _, utilities, _ = upstream()
    rng = np.random.default_rng(3)
    for _ in range(20):
        labels = np.repeat(rng.choice([-1, 0, 1, 2, 6, 9], size=20), rng.integers(1, 8, size=20))
        expected = utilities.filter_transitions(labels, start_before=2, start_after=1, pause_after=4)
        np.testing.assert_array_equal(masked_labels(labels), expected)


def test_short_harmonic_tail_receives_original_bandpass_only():
    _, _, filtering = upstream()
    signal = pd.Series(np.random.default_rng(8).normal(size=10000), index=np.arange(10000) / 5124.07211903)
    coefficients = butter(5, [20 / (5124.07211903 / 2), 700 / (5124.07211903 / 2)], btype="band")
    np.testing.assert_array_equal(filtering.pre_process(signal), filtfilt(*coefficients, signal))
