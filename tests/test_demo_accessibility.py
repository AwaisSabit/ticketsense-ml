"""Tests for the demo page accessibility and plain-English output.

Covers the form labels and accessible names, the live results
region, the alert roles on warnings and errors, the one-line
summary sentence on the result page and in the command-line
output, and that the JSON endpoint is unchanged by the
readability work (no new fields).
"""

import re
import sys

import pytest
from fastapi.testclient import TestClient

from ticketsense.app import app
from ticketsense.predict import format_prediction, summary_sentence
from ticketsense.samples import SAMPLE_TICKETS

CLEAR_BILLING = "My card was charged twice for the same order, please refund it"
VAGUE = "hello I need some help please"
THANK_YOU = "Thank you for the quick reply, everything works great now"


@pytest.fixture(scope="module")
def client():
    """A test client for the demo app, using the saved models."""
    return TestClient(app)


def test_home_page_declares_language_and_title(client):
    response = client.get("/")
    assert response.status_code == 200
    assert '<html lang="en">' in response.text
    assert "<title>TicketSense" in response.text


def test_ticket_text_box_has_an_associated_label(client):
    response = client.get("/")
    assert response.status_code == 200
    # The label points at the text box by id, and the box itself
    # carries that id and the matching form field name.
    assert '<label for="text">Ticket text</label>' in response.text
    assert '<textarea id="text" name="text">' in response.text


def test_predict_button_has_a_clear_accessible_name(client):
    response = client.get("/")
    assert response.status_code == 200
    assert '<button type="submit">Predict</button>' in response.text


def test_sample_buttons_have_accessible_names(client):
    response = client.get("/")
    assert response.status_code == 200
    # Every sample button's visible text is its accessible
    # name, so each label must sit inside a button element.
    for sample in SAMPLE_TICKETS:
        pattern = (
            r"<button[^>]*>" + re.escape(sample["label"]) + r"</button>"
        )
        assert re.search(pattern, response.text), sample["label"]


def test_result_region_uses_live_announcement(client):
    response = client.post("/predict", data={"text": CLEAR_BILLING})
    assert response.status_code == 200
    assert '<div class="result" aria-live="polite">' in response.text


def test_result_shows_a_plain_english_summary_sentence(client):
    response = client.post("/predict", data={"text": CLEAR_BILLING})
    assert response.status_code == 200
    assert "This looks like a billing ticket, medium priority." in response.text
    response = client.post("/predict", data={"text": THANK_YOU})
    assert response.status_code == 200
    assert "This looks like an other ticket, low priority." in response.text


def test_confidence_bars_are_paired_with_visible_percentages(client):
    response = client.post("/predict", data={"text": CLEAR_BILLING})
    assert response.status_code == 200
    # The percentage is visible text next to each bar, and the
    # bar itself carries the same percentage as a label.
    assert re.search(r"confidence 0\.\d\d, \d+%", response.text)
    assert re.search(
        r'<div class="bar" role="img" aria-label="Category confidence \d+%">',
        response.text,
    )
    assert re.search(
        r'<div class="bar" role="img" aria-label="Priority confidence \d+%">',
        response.text,
    )


def test_low_confidence_warning_uses_role_alert(client):
    response = client.post("/predict", data={"text": VAGUE})
    assert response.status_code == 200
    assert '<p class="warning" role="alert">' in response.text
    assert "check this ticket by hand" in response.text
    # A clear ticket shows no warning and no alert paragraph.
    response = client.post("/predict", data={"text": THANK_YOU})
    assert response.status_code == 200
    assert 'role="alert"' not in response.text


def test_empty_text_error_uses_role_alert(client):
    for bad in ("", "   "):
        response = client.post("/predict", data={"text": bad})
        assert response.status_code == 400
        assert '<p class="error" role="alert">' in response.text
        assert "must not be empty" in response.text
        assert "<form" in response.text  # the form is still there


def test_summary_sentence_reads_in_plain_english():
    result = {
        "category": "billing",
        "priority": "medium",
    }
    assert summary_sentence(result) == (
        "This looks like a billing ticket, medium priority."
    )


def test_formatted_prediction_starts_with_the_summary_sentence():
    result = {
        "text": "charged twice please refund",
        "category": "billing",
        "category_confidence": 0.44,
        "priority": "medium",
        "priority_confidence": 0.42,
        "low_confidence": True,
        "warning": "Low confidence on priority (0.42) — please check this ticket by hand.",
    }
    text = format_prediction(result)
    lines = text.splitlines()
    assert lines[0] == "This looks like a billing ticket, medium priority."
    assert "Category: billing (confidence 0.44)" in text
    assert "Category confidence 44%" in text
    assert "Priority: medium (confidence 0.42)" in text
    assert "Priority confidence 42%" in text
    assert "Warning: Low confidence" in text


def test_cli_output_has_labelled_lines_and_the_summary(monkeypatch, capsys):
    from ticketsense.predict import main

    monkeypatch.setattr(sys, "argv", ["predict", CLEAR_BILLING])
    main()
    out = capsys.readouterr().out
    assert out.splitlines()[0] == (
        "This looks like a billing ticket, medium priority."
    )
    assert "Ticket: " in out
    assert "Category: billing" in out
    assert "Priority: medium" in out
    assert "Category confidence " in out
    assert "Priority confidence " in out
    assert "Warning: " in out  # the priority half is low here


def test_json_endpoint_is_unchanged_by_the_readability_work(client):
    response = client.get("/api/predict", params={"text": CLEAR_BILLING})
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {
        "text",
        "category",
        "category_confidence",
        "priority",
        "priority_confidence",
        "low_confidence",
        "warning",
        "category_threshold",
        "priority_threshold",
        "category_confidence_note",
        "priority_confidence_note",
    }
    # No summary field sneaks into the JSON: the plain-English
    # sentence is for the page and the command line only.
    assert "summary" not in result
