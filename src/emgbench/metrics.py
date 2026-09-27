"""Fixed-eight-class metrics with equal weighting of the four test people."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .common import GESTURES, LABELS, settings

METRICS = ("accuracy", "macro_f1", "macro_recall", "macro_precision")


def score(truth, predictions):
    truth, predictions = np.asarray(truth, dtype=int), np.asarray(predictions, dtype=int)
    if not len(truth) or truth.shape != predictions.shape:
        raise ValueError("Expected equally sized, nonempty truth and predictions")
    if not np.isin(truth, LABELS).all() or not np.isin(predictions, LABELS).all():
        raise ValueError("Excluded labels cannot be scored as gesture classes")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=LABELS, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1.mean()), "macro_recall": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=LABELS).tolist(),
        "per_gesture": [
            {"label": label, "name": name, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(n)}
            for label, name, p, r, f, n in zip(LABELS, GESTURES, precision, recall, f1, support)
        ],
    }


def summarize(predictions):
    participants = {
        str(subject): score(rows.TRAJ_GT, rows.prediction)
        for subject, rows in predictions.groupby("subject", sort=True)
    }
    if sorted(participants) != settings()["test_subjects"]:
        raise ValueError("Metrics must cover exactly the four frozen held-out users")
    means = {
        metric: float(np.mean([values[metric] for values in participants.values()]))
        for metric in METRICS
    }
    return {
        "participant_mean": means, "participants": participants,
        "pooled": score(predictions.TRAJ_GT, predictions.prediction),
        "per_recording": {
            str(name): score(rows.TRAJ_GT, rows.prediction)
            for name, rows in predictions.groupby("recording", sort=True)
        },
        "per_gesture": [
            {
                "label": label, "name": name,
                **{metric: float(np.mean([v["per_gesture"][i][metric] for v in participants.values()])) for metric in ("precision", "recall", "f1")},
                "support": sum(v["per_gesture"][i]["support"] for v in participants.values()),
            }
            for i, (label, name) in enumerate(zip(LABELS, GESTURES))
        ],
    }
