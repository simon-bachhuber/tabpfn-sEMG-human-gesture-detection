import numpy as np
import pandas as pd
import pytest

from emgbench.common import settings
from emgbench.metrics import score, summarize
from emgbench.protocol import masked_labels, prepare_side, split_recordings, subjects_of


def stems():
    config = settings()
    return [
        f"emg_gestures-{subject}-{trajectory}-{date}-11-05-00-695"
        for subject in config["train_subjects"] + config["test_subjects"]
        for date in ("2018-05-11", "2018-06-14")
        for trajectory in ("repeats_long", "repeats_short", "sequential")
    ]


def test_frozen_split_has_complete_disjoint_participants():
    split = split_recordings(stems())
    assert len(split["train"]) == 240 and len(split["test"]) == 24
    assert len(subjects_of(split["train"])) == 40
    assert subjects_of(split["test"]) == ["08", "24", "34", "39"]
    assert set(subjects_of(split["train"])).isdisjoint(subjects_of(split["test"]))
    assert split_recordings(list(reversed(stems()))) == split
    eligible = sorted(set(settings()["train_subjects"] + settings()["test_subjects"]) - {"03", "04"})
    assert sorted(np.random.default_rng(42).choice(eligible, 4, replace=False).tolist()) == split["test_subjects"]


def test_partial_cohorts_and_duplicate_recordings_are_rejected():
    with pytest.raises(ValueError, match="264"):
        split_recordings(stems()[:-1])
    with pytest.raises(ValueError, match="264"):
        split_recordings(stems()[:-1] + [stems()[0]])


def test_mask_direction_and_negative_label_handling():
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, -1, 0, 0, 0, 0, 0])
    assert masked_labels(labels).tolist() == [0, 0, -5, -5, 1, 1, 0, 0, -6, -6, -6, -6, -6, 0]
    with pytest.raises(ValueError):
        score([-1, 0], [0, 0])


def test_masks_do_not_cross_recordings_and_metadata_cannot_become_features():
    frames = {}
    for name, labels in (("a", [0, 0, 0, 0, -1]), ("b", [0] * 6)):
        columns = {f"{f}_{c}": np.arange(len(labels), dtype=float) for f in settings()["representations"]["Du"] for c in range(1, 25)}
        frames[name] = pd.DataFrame({**columns, "TRAJ_GT": labels, "TRAJ_1": labels, "subject": 8, "VIDEO_STAMP": 1, "window_id": [f"{name}:{i}" for i in range(len(labels))]})
    X, _, kept, excluded = prepare_side(frames, ["a", "b"], "Du")
    assert X.shape[1] == 144
    assert "b:0" in set(kept.window_id)
    assert excluded == 1
    assert not any("subject" in column or "TRAJ" in column or "VIDEO" in column for column in X)


def test_subject_metrics_weight_people_equally():
    frames = []
    for subject in settings()["test_subjects"]:
        labels = [0, 1] if subject in {"08", "24"} else [0] * 10
        frames.append(pd.DataFrame({"subject": subject, "recording": subject, "TRAJ_GT": labels, "prediction": 0}))
    result = summarize(pd.concat(frames, ignore_index=True))
    assert result["participant_mean"]["accuracy"] == .75
    assert result["pooled"]["accuracy"] == 22 / 24


def test_exact_comparison_scope_and_published_svm_parameters():
    comparisons = settings()["comparisons"]
    assert set(comparisons) == {"svm-rms", "lda-du", "tabpfn-du"}
    assert comparisons["svm-rms"]["representation"] == "RMS"
    assert comparisons["svm-rms"]["parameters"]["C"] == 50
    assert comparisons["svm-rms"]["parameters"]["gamma"] == "auto"
    assert comparisons["svm-rms"]["parameters"]["probability"] is False
