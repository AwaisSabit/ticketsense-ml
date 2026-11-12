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


def test_predict_text_validates_but_does_not_invent_a_prediction():
    result = predict_text("My parcel has not arrived")
    assert result["text"] == "my parcel has not arrived"
    # No model exists on Day 1, so no category/priority is returned.
    assert result["category"] is None
    assert result["priority"] is None
    assert result["confidence"] is None
    with pytest.raises(ValueError):
        predict_text("")
