from types import SimpleNamespace

import pandas as pd
import pytest

import emgbench.comparison as comparison
from emgbench.cli import background_command


def fake_dataset():
    X = pd.DataFrame([[1.0], [2.0]])
    side = (X, pd.Series([0, 1]), pd.DataFrame(), 0)
    return SimpleNamespace(tables=lambda _: (side, side))


def test_cached_tabpfn_results_never_trigger_authentication_or_billing(tmp_path, monkeypatch):
    directory = tmp_path / "cached"
    directory.mkdir()
    result, predictions = directory / "result.json", directory / "prediction.parquet"
    result.touch()
    predictions.touch()
    monkeypatch.setattr(comparison, "load_dataset", fake_dataset)
    monkeypatch.setattr(comparison, "paths_for", lambda key: (directory, result, predictions))
    monkeypatch.setattr(comparison, "verify_cached_inputs", lambda *args: {"cached": True})
    monkeypatch.setattr(comparison, "authentication", lambda: pytest.fail("Cached comparison must not authenticate"))
    monkeypatch.setattr(comparison, "quote_prediction", lambda *args: pytest.fail("Cached comparison must not request a quote"))
    assert comparison.compare(["tabpfn-du"])["tabpfn-du"]["cached"]


def test_quote_and_budget_checks_precede_any_billable_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "load_dataset", fake_dataset)
    monkeypatch.setattr(comparison, "paths_for", lambda key: (tmp_path / "missing", tmp_path / "result.json", tmp_path / "predictions.parquet"))
    monkeypatch.setattr(comparison, "quote_prediction", lambda *args: {"estimated_cost": 100})
    monkeypatch.setattr(comparison, "fit_comparison", lambda *args: pytest.fail("A quote or rejected budget must not fit"))
    assert comparison.compare(["tabpfn-du"], quote_only=True)["estimated_tokens"] == 100
    with pytest.raises(ValueError, match="budget"):
        comparison.compare(["tabpfn-du"], max_estimated_tokens=99)


def test_unsupported_comparisons_are_rejected_before_loading_data(monkeypatch):
    monkeypatch.setattr(comparison, "load_dataset", lambda: pytest.fail("Invalid selection must fail first"))
    with pytest.raises(ValueError, match="Choose only"):
        comparison.compare(["svm-du"])


def test_background_jobs_keep_the_resolved_workspace(tmp_path):
    command = background_command(
        ["--workspace", "relative", "compare", "--models", "svm-rms", "--background"], tmp_path,
    )
    assert command[3:] == ["--workspace", str(tmp_path), "compare", "--models", "svm-rms"]
