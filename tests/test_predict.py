"""Tests for src/ticketsense/predict.py — prediction with confidence."""

import sys

import joblib
import pytest

from ticketsense.predict import (
    LOW_CONFIDENCE_CATEGORY,
    LOW_CONFIDENCE_PRIORITY,
    format_prediction,
    load_models,
    predict_label,
    predict_ticket,
)
from ticketsense.train import DEFAULT_CATEGORY_MODEL, DEFAULT_PRIORITY_MODEL, build_pipeline

CATEGORIES = {"billing", "technical", "account", "delivery", "other"}
PRIORITIES = {"low", "medium", "high"}


@pytest.fixture(scope="module")
def saved_models():
    """The two pipelines trained on Day 4, loaded once per test run."""
    return joblib.load(DEFAULT_CATEGORY_MODEL), joblib.load(
        DEFAULT_PRIORITY_MODEL
    )


def _tiny_models():
    """Two tiny pipelines trained on a hand-made frame (fast, no disk)."""
    texts = [
        "charged twice refund payment invoice",
        "charged twice refund payment invoice",
        "app crashes error login broken",
        "app crashes error login broken",
        "parcel delivery tracking address courier",
        "parcel delivery tracking address courier",
    ]
    categories = ["billing", "billing", "technical", "technical", "delivery", "delivery"]
    priorities = ["high", "high", "high", "high", "medium", "medium"]
    category_model = build_pipeline()
    category_model.fit(texts, categories)
    priority_model = build_pipeline()
    priority_model.fit(texts, priorities)
    return category_model, priority_model


def test_load_models_returns_both_saved_pipelines():
    category_model, priority_model = load_models()
    assert hasattr(category_model, "predict")
    assert hasattr(priority_model, "predict")
    assert hasattr(category_model, "predict_proba")


def test_load_models_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "no_such_model.joblib"
    with pytest.raises(FileNotFoundError, match="Train the baseline first"):
        load_models(category_path=missing, priority_path=DEFAULT_PRIORITY_MODEL)


def test_predict_ticket_returns_labels_and_confidences(saved_models):
    category_model, priority_model = saved_models
    result = predict_ticket(
        "My card was charged twice, please refund the extra payment",
        category_model=category_model,
        priority_model=priority_model,
    )
    assert set(result) == {
        "text",
        "category",
        "category_confidence",
        "priority",
        "priority_confidence",
        "low_confidence",
        "warning",
    }
    assert result["category"] in CATEGORIES
    assert result["priority"] in PRIORITIES
    assert 0.0 < result["category_confidence"] <= 1.0
    assert 0.0 < result["priority_confidence"] <= 1.0
    assert result["text"] == (
        "my card was charged twice, please refund the extra payment"
    )


def test_clear_billing_ticket_is_billing(saved_models):
    category_model, priority_model = saved_models
    result = predict_ticket(
        "My card was charged twice for the same order, please refund it",
        category_model=category_model,
        priority_model=priority_model,
    )
    assert result["category"] == "billing"
    assert result["category_confidence"] >= LOW_CONFIDENCE_CATEGORY


def test_clear_technical_ticket_is_technical(saved_models):
    category_model, priority_model = saved_models
    result = predict_ticket(
        "The app crashes every time I try to open it",
        category_model=category_model,
        priority_model=priority_model,
    )
    assert result["category"] == "technical"
    assert result["category_confidence"] >= LOW_CONFIDENCE_CATEGORY


def test_vague_ticket_triggers_low_confidence_warning(saved_models):
    category_model, priority_model = saved_models
    result = predict_ticket(
        "hello I need some help please",
        category_model=category_model,
        priority_model=priority_model,
    )
    assert result["low_confidence"] is True
    assert result["warning"] is not None
    assert "check this ticket by hand" in result["warning"]
    # Both scores are below their thresholds for this vague text.
    assert "category" in result["warning"]
    assert "priority" in result["warning"]


def test_clear_thanks_ticket_has_no_warning(saved_models):
    category_model, priority_model = saved_models
    result = predict_ticket(
        "Thank you for the quick reply, everything works great now",
        category_model=category_model,
        priority_model=priority_model,
    )
    assert result["category"] == "other"
    assert result["priority"] == "low"
    assert result["low_confidence"] is False
    assert result["warning"] is None


def test_empty_and_blank_text_raise(saved_models):
    category_model, priority_model = saved_models
    for bad in ("", "   ", "\n\t "):
        with pytest.raises(ValueError, match="must not be empty"):
            predict_ticket(
                bad,
                category_model=category_model,
                priority_model=priority_model,
            )


def test_non_string_text_raises(saved_models):
    category_model, priority_model = saved_models
    with pytest.raises(ValueError, match="must be a string"):
        predict_ticket(
            42, category_model=category_model, priority_model=priority_model
        )


def test_missing_model_files_raise_clear_error(tmp_path):
    missing = tmp_path / "no_such_model.joblib"
    with pytest.raises(FileNotFoundError, match="Train the baseline first"):
        predict_ticket("some ticket text", category_path=missing)


def test_thresholds_control_the_warning():
    category_model, priority_model = _tiny_models()
    result = predict_ticket(
        "charged twice refund payment invoice",
        category_model=category_model,
        priority_model=priority_model,
        category_threshold=0.0,
        priority_threshold=0.0,
    )
    assert result["low_confidence"] is False
    assert result["warning"] is None

    result = predict_ticket(
        "charged twice refund payment invoice",
        category_model=category_model,
        priority_model=priority_model,
        category_threshold=1.0,
        priority_threshold=1.0,
    )
    assert result["low_confidence"] is True
    assert "category" in result["warning"]
    assert "priority" in result["warning"]


def test_predict_label_returns_valid_label_and_confidence():
    category_model, _priority_model = _tiny_models()
    label, confidence = predict_label(
        category_model, "charged twice refund payment invoice"
    )
    assert label in CATEGORIES
    assert 0.0 < confidence <= 1.0


def test_format_prediction_includes_warning_when_low():
    result = {
        "text": "hello help please",
        "category": "other",
        "category_confidence": 0.27,
        "priority": "high",
        "priority_confidence": 0.37,
        "low_confidence": True,
        "warning": "Low confidence on category (0.27) — please check this ticket by hand.",
    }
    text = format_prediction(result)
    assert "Category: other (confidence 0.27)" in text
    assert "Priority: high (confidence 0.37)" in text
    assert "Warning: Low confidence" in text


def test_format_prediction_omits_warning_when_confident():
    result = {
        "text": "thank you everything works",
        "category": "other",
        "category_confidence": 0.35,
        "priority": "low",
        "priority_confidence": 0.50,
        "low_confidence": False,
        "warning": None,
    }
    text = format_prediction(result)
    assert "Warning" not in text
    assert "Category: other (confidence 0.35)" in text


def test_cli_predicts_a_ticket(monkeypatch, capsys):
    from ticketsense.predict import main

    monkeypatch.setattr(
        sys,
        "argv",
        ["predict", "The app crashes every time I try to open it"],
    )
    main()
    out = capsys.readouterr().out
    assert "Category: technical" in out
    assert "Priority:" in out
    assert "confidence" in out


def test_cli_empty_text_exits_with_error(monkeypatch, capsys):
    from ticketsense.predict import main

    monkeypatch.setattr(sys, "argv", ["predict", "   "])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Ticket text must not be empty" in err


def test_cli_missing_model_exits_with_error(monkeypatch, capsys, tmp_path):
    from ticketsense.predict import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict",
            "--category-model",
            str(tmp_path / "no_such_model.joblib"),
            "some ticket text",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Train the baseline first" in err
