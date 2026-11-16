"""Tests for the Day 1 text-cleaning helper in features.py."""

import pytest

from ticketsense.features import clean_text
from ticketsense.predict import predict_text


def test_clean_text_lowercases_and_collapses_whitespace():
    assert clean_text("  Charged   TWICE  for  my plan ") == "charged twice for my plan"


def test_clean_text_rejects_blank_and_non_string():
    with pytest.raises(ValueError):
        clean_text("   ")
    with pytest.raises(ValueError):
        clean_text(123)  # type: ignore[arg-type]


def test_predict_text_now_predicts_with_the_saved_model():
    result = predict_text("My parcel has not arrived")
    assert result["text"] == "my parcel has not arrived"
    # A model now exists, so a real prediction is returned.
    assert result["category"] == "delivery"
    assert result["priority"] in ("low", "medium", "high")
    assert 0.0 < result["category_confidence"] <= 1.0
    assert 0.0 < result["priority_confidence"] <= 1.0
    with pytest.raises(ValueError):
        predict_text("")
