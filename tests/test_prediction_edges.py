"""Edge-case tests for prediction behaviour in predict.py that the
earlier test files do not cover.

The threshold boundary is the important one: a confidence exactly
at its threshold is *not* low (the check is a strict ``<``), so
the tests below drive ``predict_ticket`` with a small fake
pipeline whose probabilities are fixed, to sit exactly on and
just below each threshold (category 0.30, priority 0.45). The
other tests pin the exact warning wording, confidence rounding,
the missing-model error when each file is absent, and the
non-string / empty / very long inputs.
"""

import sys

import pytest

from ticketsense.predict import (
    LOW_CONFIDENCE_CATEGORY,
    LOW_CONFIDENCE_PRIORITY,
    format_prediction,
    load_models,
    predict_label,
    predict_text,
    predict_ticket,
)
from ticketsense.train import (
    DEFAULT_CATEGORY_MODEL,
    DEFAULT_PRIORITY_MODEL,
    build_pipeline,
)


class _FixedPipeline:
    """A pipeline that always returns one label and one fixed
    probability distribution, so a test can sit exactly on a
    threshold instead of hoping a trained model lands there."""

    def __init__(self, label, probabilities):
        self.label = label
        self.probabilities = probabilities

    def predict(self, texts):
        return [self.label] * len(texts)

    def predict_proba(self, texts):
        return [list(self.probabilities)] * len(texts)


def _models(category_probabilities, priority_probabilities):
    return (
        _FixedPipeline("billing", category_probabilities),
        _FixedPipeline("high", priority_probabilities),
    )


def test_confidence_exactly_at_category_threshold_is_not_low():
    category_model, priority_model = _models(
        [0.30, 0.25, 0.25, 0.20], [0.60, 0.25, 0.15]
    )
    result = predict_ticket(
        "some ticket text",
        category_model=category_model,
        priority_model=priority_model,
    )
    assert result["category_confidence"] == 0.30
    assert result["low_confidence"] is False
    assert result["warning"] is None


def test_confidence_exactly_at_priority_threshold_is_not_low():
    category_model, priority_model = _models(
        [0.40, 0.30, 0.20, 0.10], [0.45, 0.30, 0.25]
    )
    result = predict_ticket(
        "some ticket text",
        category_model=category_model,
        priority_model=priority_model,
    )
    assert result["priority_confidence"] == 0.45
    assert result["low_confidence"] is False
    assert result["warning"] is None


def test_category_just_below_threshold_warns_on_category_only():
    category_model, priority_model = _models(
        [0.29, 0.26, 0.25, 0.20], [0.60, 0.25, 0.15]
    )
    result = predict_ticket(
        "some ticket text",
        category_model=category_model,
        priority_model=priority_model,
    )
    assert result["low_confidence"] is True
    assert result["warning"] == (
        "Low confidence on category (0.29) — please check this "
        "ticket by hand instead of trusting the suggestion."
    )
    assert "priority" not in result["warning"]


def test_priority_just_below_threshold_warns_on_priority_only():
    category_model, priority_model = _models(
        [0.40, 0.30, 0.20, 0.10], [0.44, 0.31, 0.25]
    )
    result = predict_ticket(
        "some ticket text",
        category_model=category_model,
        priority_model=priority_model,
    )
    assert result["low_confidence"] is True
    assert result["warning"] == (
        "Low confidence on priority (0.44) — please check this "
        "ticket by hand instead of trusting the suggestion."
    )
    assert "category" not in result["warning"]


def test_both_below_threshold_names_both_in_one_warning():
    category_model, priority_model = _models(
        [0.29, 0.26, 0.25, 0.20], [0.44, 0.31, 0.25]
    )
    result = predict_ticket(
        "some ticket text",
        category_model=category_model,
        priority_model=priority_model,
    )
    assert result["warning"] == (
        "Low confidence on category (0.29) and priority (0.44) — "
        "please check this ticket by hand instead of trusting "
        "the suggestion."
    )


def test_threshold_constants_are_the_measured_values():
    assert LOW_CONFIDENCE_CATEGORY == 0.30
    assert LOW_CONFIDENCE_PRIORITY == 0.45


def test_predict_label_rounds_confidence_to_four_decimals():
    pipeline = _FixedPipeline("billing", [0.876544, 0.123456])
    label, confidence = predict_label(pipeline, "some ticket text")
    assert label == "billing"
    assert confidence == 0.8765


def test_predict_label_confidence_is_the_top_probability():
    # The label comes from predict() and the confidence from the
    # largest probability — even when they point at different
    # classes, the confidence is still the top probability.
    pipeline = _FixedPipeline("billing", [0.20, 0.60, 0.20])
    label, confidence = predict_label(pipeline, "some ticket text")
    assert label == "billing"
    assert confidence == 0.60


def test_predict_label_rejects_empty_and_non_string_directly():
    pipeline = _FixedPipeline("billing", [0.8, 0.2])
    with pytest.raises(ValueError, match="must not be empty"):
        predict_label(pipeline, "   ")
    with pytest.raises(ValueError, match="must be a string"):
        predict_label(pipeline, 42)


def test_predict_ticket_rejects_every_non_string_form():
    category_model, priority_model = _models([0.8, 0.2], [0.7, 0.3])
    for bad in (None, 42, 3.5, ["some text"], {"text": "some text"}, b"some text"):
        with pytest.raises(ValueError, match="must be a string"):
            predict_ticket(
                bad,  # type: ignore[arg-type]
                category_model=category_model,
                priority_model=priority_model,
            )


def test_predict_ticket_returns_cleaned_text_for_messy_input():
    category_model, priority_model = _models([0.8, 0.2], [0.7, 0.3])
    result = predict_ticket(
        "  Charged   TWICE\nfor my Order ",
        category_model=category_model,
        priority_model=priority_model,
    )
    assert result["text"] == "charged twice for my order"


def test_load_models_names_both_missing_files():
    missing_category = "/tmp/does-not-exist/category.joblib"
    missing_priority = "/tmp/does-not-exist/priority.joblib"
    with pytest.raises(FileNotFoundError) as exc:
        load_models(
            category_path=missing_category, priority_path=missing_priority
        )
    assert missing_category in str(exc.value)
    assert missing_priority in str(exc.value)
    assert "Train the baseline first" in str(exc.value)


def test_load_models_names_only_the_missing_priority_file(tmp_path):
    missing = tmp_path / "no_such_priority.joblib"
    with pytest.raises(FileNotFoundError) as exc:
        load_models(
            category_path=DEFAULT_CATEGORY_MODEL, priority_path=missing
        )
    assert str(missing) in str(exc.value)
    assert "category_pipeline" not in str(exc.value)


def test_predict_ticket_loads_only_the_missing_model_from_disk():
    # Passing one loaded pipeline still works: the other half is
    # loaded from its saved file, not retrained or skipped.
    category_model, _priority_model = _models([0.8, 0.2], [0.7, 0.3])
    result = predict_ticket(
        "My card was charged twice, please refund the extra payment",
        category_model=category_model,
    )
    assert result["category"] == "billing"
    assert result["priority"] in ("low", "medium", "high")
    assert 0.0 < result["priority_confidence"] <= 1.0


def test_predict_text_alias_returns_the_same_result():
    category_model, priority_model = _models([0.8, 0.2], [0.7, 0.3])
    kwargs = {
        "category_model": category_model,
        "priority_model": priority_model,
    }
    assert predict_text("some ticket text", **kwargs) == predict_ticket(
        "some ticket text", **kwargs
    )


def test_format_prediction_shows_two_decimal_confidences():
    result = {
        "text": "charged twice please refund",
        "category": "billing",
        "category_confidence": 0.3,
        "priority": "medium",
        "priority_confidence": 0.45,
        "low_confidence": False,
        "warning": None,
    }
    text = format_prediction(result)
    assert "Category: billing (confidence 0.30)" in text
    assert "Priority: medium (confidence 0.45)" in text


def test_punctuation_only_text_still_predicts():
    """An all-punctuation ticket is an all-zero TF-IDF row; the
    model still answers with a label instead of crashing."""
    texts = [
        "charged twice refund payment invoice",
        "charged twice refund payment invoice",
        "app crashes error login broken",
        "app crashes error login broken",
    ]
    category_model = build_pipeline()
    category_model.fit(texts, ["billing", "billing", "technical", "technical"])
    priority_model = build_pipeline()
    priority_model.fit(texts, ["high", "high", "medium", "medium"])
    result = predict_ticket(
        "!!! ??? ...",
        category_model=category_model,
        priority_model=priority_model,
    )
    assert result["category"] in ("billing", "technical")
    assert result["priority"] in ("high", "medium")
    assert 0.0 <= result["category_confidence"] <= 1.0


def test_very_long_text_still_predicts():
    texts = [
        "charged twice refund payment invoice",
        "charged twice refund payment invoice",
        "app crashes error login broken",
        "app crashes error login broken",
    ]
    category_model = build_pipeline()
    category_model.fit(texts, ["billing", "billing", "technical", "technical"])
    priority_model = build_pipeline()
    priority_model.fit(texts, ["high", "high", "medium", "medium"])
    long_text = "my card was charged twice " * 200  # about 5,000 characters
    result = predict_ticket(
        long_text,
        category_model=category_model,
        priority_model=priority_model,
    )
    assert result["category"] == "billing"
    assert result["text"].startswith("my card was charged twice")


def test_cli_threshold_flags_control_the_warning(monkeypatch, capsys):
    from ticketsense.predict import main

    # Force a warning on a clear ticket by raising both thresholds.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict",
            "--category-threshold",
            "1.0",
            "--priority-threshold",
            "1.0",
            "My card was charged twice, please refund me",
        ],
    )
    main()
    out = capsys.readouterr().out
    assert "Warning: Low confidence on category" in out
    assert "priority" in out

    # Drop both thresholds to zero and the same ticket warns never.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict",
            "--category-threshold",
            "0.0",
            "--priority-threshold",
            "0.0",
            "hello I need some help please",
        ],
    )
    main()
    out = capsys.readouterr().out
    assert "Warning" not in out


def test_cli_missing_priority_model_exits_with_error(
    monkeypatch, capsys, tmp_path
):
    from ticketsense.predict import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict",
            "--priority-model",
            str(tmp_path / "no_such_model.joblib"),
            "some ticket text",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Train the baseline first" in err
