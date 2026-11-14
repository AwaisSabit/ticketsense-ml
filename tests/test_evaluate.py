"""Tests for src/ticketsense/evaluate.py — metrics and report."""

from pathlib import Path

import pytest

from ticketsense.data import load_tickets
from ticketsense.evaluate import (
    evaluate_models,
    evaluate_pipeline,
    write_report,
)
from ticketsense.train import build_pipeline, split_tickets, train_baseline

DATA = Path(__file__).resolve().parents[1] / "data" / "demo_tickets.csv"


def _trained_results(tmp_path):
    """Train both pipelines into tmp_path and evaluate them there."""
    category_out = tmp_path / "category_pipeline.joblib"
    priority_out = tmp_path / "priority_pipeline.joblib"
    train_baseline(
        data_path=DATA, category_out=category_out, priority_out=priority_out
    )
    return evaluate_models(
        category_path=category_out, priority_path=priority_out, data_path=DATA
    )


def test_evaluate_pipeline_reports_sane_metrics():
    df = load_tickets(DATA)
    train, test = split_tickets(df)
    pipeline = build_pipeline()
    pipeline.fit(train["text"], train["category"])
    result = evaluate_pipeline(
        pipeline,
        test["text"],
        test["category"],
        ("billing", "technical", "account", "delivery", "other"),
    )
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["macro_precision"] <= 1.0
    assert 0.0 <= result["macro_recall"] <= 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0
    # The confusion matrix covers exactly the 40 test tickets.
    assert sum(sum(row) for row in result["confusion_matrix"]) == 40
    assert sum(row["support"] for row in result["per_class"].values()) == 40
    assert result["labels"] == [
        "billing",
        "technical",
        "account",
        "delivery",
        "other",
    ]
    for name, row in result["per_class"].items():
        assert 0.0 <= row["precision"] <= 1.0
        assert 0.0 <= row["recall"] <= 1.0
        assert 0.0 <= row["f1"] <= 1.0
        assert row["support"] == 8


def test_evaluate_models_reproduces_the_training_split(tmp_path):
    results = _trained_results(tmp_path)
    assert results["n_train"] == 160
    assert results["n_test"] == 40
    assert results["seed"] == 42
    assert results["test_category_counts"] == {
        "billing": 8,
        "technical": 8,
        "account": 8,
        "delivery": 8,
        "other": 8,
    }


def test_evaluate_models_reports_the_real_baseline_numbers(tmp_path):
    """Exact numbers for the pinned demo set, seed and library versions.

    These are sanity values for a 40-ticket test set, not a
    performance claim — if the data or the seed changes, regenerate
    the report and update these numbers together.
    """
    results = _trained_results(tmp_path)
    assert results["category"]["accuracy"] == 0.85
    assert results["category"]["macro_f1"] == 0.8449
    assert results["priority"]["accuracy"] == 0.525
    assert results["priority"]["macro_f1"] == 0.4535
    # Priority is the weak half of the baseline: most high-priority
    # tickets are missed (see the confusion matrix in the report).
    assert results["priority"]["per_class"]["high"]["recall"] < 0.2


def test_write_report_covers_split_metrics_and_limits(tmp_path):
    results = _trained_results(tmp_path)
    report_path = tmp_path / "evaluation.md"
    written = write_report(results, report_path)
    assert written == report_path
    text = report_path.read_text(encoding="utf-8")
    assert "160" in text and "40" in text
    assert "seed 42" in text
    assert "0.8500" in text  # category accuracy
    assert "0.5250" in text  # priority accuracy
    assert "billing" in text and "high" in text
    assert "Confusion matrix" in text
    assert "Limitations" in text
    assert "Only 40 test tickets" in text


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_models(
            category_path=tmp_path / "no-such-model.joblib",
            priority_path=tmp_path / "no-such-model.joblib",
            data_path=DATA,
        )
