"""Tests for src/ticketsense/errors.py — error analysis and comparison."""

from pathlib import Path

import pandas as pd
import pytest

from ticketsense.errors import (
    build_naive_bayes_pipeline,
    compare_naive_bayes,
    confusion_pairs,
    error_analysis,
    make_charts,
    misclassified,
    weak_spots,
    write_report,
)
from ticketsense.train import train_baseline

DATA = Path(__file__).resolve().parents[1] / "data" / "demo_tickets.csv"


class _StubModel:
    """A fake pipeline whose predictions are fixed in advance."""

    def __init__(self, predictions):
        self._predictions = list(predictions)

    def predict(self, texts):
        return self._predictions


def _small_frame():
    """Three hand-made tickets with known labels."""
    return pd.DataFrame(
        {
            "ticket_id": ["TS-001", "TS-002", "TS-003"],
            "text": [
                "My card was charged twice.",
                "The app crashes on login.",
                "Thank you for the help.",
            ],
            "category": ["billing", "technical", "other"],
            "priority": ["high", "high", "low"],
        }
    )


def _trained_analysis(tmp_path):
    """Train both pipelines into tmp_path and run the full analysis."""
    category_out = tmp_path / "category_pipeline.joblib"
    priority_out = tmp_path / "priority_pipeline.joblib"
    train_baseline(
        data_path=DATA, category_out=category_out, priority_out=priority_out
    )
    return error_analysis(
        category_path=category_out, priority_path=priority_out, data_path=DATA
    )


def test_misclassified_lists_the_wrong_tickets():
    df = _small_frame()
    category_model = _StubModel(["billing", "other", "other"])
    priority_model = _StubModel(["medium", "high", "low"])
    errors = misclassified(df, category_model, priority_model)

    # Category: only TS-002 is wrong (technical called other).
    assert len(errors["category"]) == 1
    wrong = errors["category"][0]
    assert wrong["ticket_id"] == "TS-002"
    assert wrong["text"] == "The app crashes on login."
    assert wrong["true_label"] == "technical"
    assert wrong["predicted_label"] == "other"

    # Priority: only TS-001 is wrong (high called medium).
    assert len(errors["priority"]) == 1
    wrong = errors["priority"][0]
    assert wrong["ticket_id"] == "TS-001"
    assert wrong["true_label"] == "high"
    assert wrong["predicted_label"] == "medium"


def test_misclassified_is_empty_for_perfect_predictions():
    df = _small_frame()
    category_model = _StubModel(["billing", "technical", "other"])
    priority_model = _StubModel(["high", "high", "low"])
    errors = misclassified(df, category_model, priority_model)
    assert errors == {"category": [], "priority": []}


def test_confusion_pairs_counts_off_diagonal_only():
    matrix = [
        [5, 1, 0],
        [0, 6, 2],
        [1, 0, 8],
    ]
    labels = ("billing", "technical", "other")
    pairs = confusion_pairs(matrix, labels)
    # Diagonal counts (5, 6, 8) are correct predictions, not listed.
    assert pairs == [
        {"true_label": "technical", "predicted_label": "other", "count": 2},
        {"true_label": "billing", "predicted_label": "technical", "count": 1},
        {"true_label": "other", "predicted_label": "billing", "count": 1},
    ]


def test_confusion_pairs_empty_when_nothing_is_confused():
    assert confusion_pairs([[4, 0], [0, 3]], ("a", "b")) == []


def test_weak_spots_flags_only_low_recall_classes():
    per_class = {
        "billing": {"precision": 1.0, "recall": 0.625, "f1": 0.7692, "support": 8},
        "high": {"precision": 1.0, "recall": 0.0714, "f1": 0.1333, "support": 14},
        "low": {"precision": 0.75, "recall": 0.7059, "f1": 0.7273, "support": 17},
    }
    spots = weak_spots(per_class)
    assert [spot["label"] for spot in spots] == ["high"]
    assert spots[0]["recall"] == 0.0714


def test_error_analysis_reports_the_real_error_counts(tmp_path):
    """Exact counts for the pinned demo set, seed and library versions.

    These are sanity values for a 40-ticket test set, not a
    performance claim — if the data or the seed changes, regenerate
    the report and update these numbers together.
    """
    analysis = _trained_analysis(tmp_path)
    assert analysis["n_train"] == 160
    assert analysis["n_test"] == 40
    assert analysis["seed"] == 42
    assert len(analysis["misclassified"]["category"]) == 6
    assert len(analysis["misclassified"]["priority"]) == 19
    # High priority is the weak spot: 10 of the 13 missed high
    # tickets were called medium, 3 were called low.
    assert analysis["priority_confusion_pairs"][0] == {
        "true_label": "high",
        "predicted_label": "medium",
        "count": 10,
    }
    assert analysis["priority_weak_spots"][0]["label"] == "high"
    # No category weak spot: every category recall is 0.5 or above.
    assert analysis["category_weak_spots"] == []


def test_naive_bayes_comparison_runs_and_scores_in_range(tmp_path):
    comparison = compare_naive_bayes(data_path=DATA)
    assert comparison["n_train"] == 160
    assert comparison["n_test"] == 40
    assert comparison["seed"] == 42
    for task in ("category", "priority"):
        result = comparison[task]
        assert 0.0 <= result["accuracy"] <= 1.0
        assert 0.0 <= result["macro_f1"] <= 1.0
        assert sum(sum(row) for row in result["confusion_matrix"]) == 40


def test_naive_bayes_comparison_reports_the_real_numbers():
    """Exact comparison numbers for the pinned demo set and seed."""
    comparison = compare_naive_bayes(data_path=DATA)
    assert comparison["category"]["accuracy"] == 0.825
    assert comparison["priority"]["accuracy"] == 0.5
    # Naive Bayes is worse than Logistic Regression on both tasks
    # here, so the baseline stays Logistic Regression.
    assert comparison["category"]["accuracy"] < 0.85
    assert comparison["priority"]["accuracy"] < 0.525


def test_naive_bayes_pipeline_predicts_valid_labels():
    df = _small_frame()
    pipeline = build_naive_bayes_pipeline()
    pipeline.fit(df["text"], df["category"])
    predicted = pipeline.predict(df["text"])
    assert set(predicted) <= {"billing", "technical", "other"}


def test_write_report_covers_errors_pairs_comparison_and_reading(tmp_path):
    analysis = _trained_analysis(tmp_path)
    report_path = tmp_path / "error_analysis.md"
    written = write_report(analysis, report_path)
    assert written == report_path
    text = report_path.read_text(encoding="utf-8")
    assert "How many were wrong" in text
    assert "6 of 40" in text  # category misclassified
    assert "19 of 40" in text  # priority misclassified
    assert "Misclassified tickets" in text
    assert "TS-066" in text  # a real misclassified ticket id
    assert "Confusion pairs" in text
    assert "Weak spots" in text
    assert "Second baseline" in text
    assert "Naive Bayes" in text
    assert "0.8250" in text  # Naive Bayes category accuracy
    assert "What the errors mean" in text
    assert "Logistic Regression stays the baseline" in text


def test_make_charts_writes_both_confusion_pngs(tmp_path):
    analysis = _trained_analysis(tmp_path)
    reports_dir = tmp_path / "reports"
    written = make_charts(analysis, reports_dir)
    assert written == [
        reports_dir / "category_confusion.png",
        reports_dir / "priority_confusion.png",
    ]
    for chart in written:
        assert chart.exists()
        assert chart.stat().st_size > 1000


def test_missing_model_file_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        error_analysis(
            category_path=tmp_path / "no-such-model.joblib",
            priority_path=tmp_path / "no-such-model.joblib",
            data_path=DATA,
        )
