"""Malformed-input tests for the TicketSense demo.

Empty, blank, missing and absurdly long inputs must never crash
the demo or leak a traceback: the form page answers with a
clear error message (and the form still there), and the JSON
endpoint answers with a 400. Non-string input to
``predict_ticket`` must raise a clear ``ValueError``. The
sample tickets from ``ticketsense.samples`` must all predict
successfully through the API.
"""

import time

import pytest
from fastapi.testclient import TestClient

from ticketsense.app import app
from ticketsense.predict import predict_ticket
from ticketsense.samples import SAMPLE_TICKETS

CATEGORIES = {"billing", "technical", "account", "delivery", "other"}
PRIORITIES = {"low", "medium", "high"}

# 11,000 characters of real ticket words — long enough to be
# absurd, short of anything a web form would reasonably reject.
VERY_LONG_TEXT = "please refund my order " * 500


@pytest.fixture(scope="module")
def client():
    """A test client for the demo app, using the saved models."""
    return TestClient(app)


# --- Empty / blank / missing input on the form ----------------


@pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
def test_form_empty_or_whitespace_text_returns_400_with_message(client, bad):
    response = client.post("/predict", data={"text": bad})
    assert response.status_code == 400
    assert "must not be empty" in response.text
    assert "<form" in response.text  # the form is still there to retry
    assert "Traceback" not in response.text


def test_form_missing_text_field_returns_400_with_message(client):
    response = client.post("/predict", data={})
    assert response.status_code == 400
    assert "must not be empty" in response.text
    assert "<form" in response.text
    assert "Traceback" not in response.text


# --- Empty / blank / missing input on the JSON endpoint -------


@pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
def test_api_empty_or_whitespace_text_returns_400(client, bad):
    response = client.get("/api/predict", params={"text": bad})
    assert response.status_code == 400
    assert "must not be empty" in response.json()["error"]
    assert "Traceback" not in response.text


def test_api_missing_text_param_returns_400(client):
    response = client.get("/api/predict")
    assert response.status_code == 400
    assert "must not be empty" in response.json()["error"]


# --- Very long input must not crash or hang -------------------


def test_api_very_long_text_returns_a_prediction(client):
    start = time.monotonic()
    response = client.get("/api/predict", params={"text": VERY_LONG_TEXT})
    elapsed = time.monotonic() - start
    assert response.status_code == 200
    result = response.json()
    assert result["category"] in CATEGORIES
    assert result["priority"] in PRIORITIES
    assert elapsed < 10  # a single ticket must answer in seconds


def test_form_very_long_text_returns_a_result_page(client):
    start = time.monotonic()
    response = client.post("/predict", data={"text": VERY_LONG_TEXT})
    elapsed = time.monotonic() - start
    assert response.status_code == 200
    assert "Category:" in response.text
    assert "Priority:" in response.text
    assert "Traceback" not in response.text
    assert elapsed < 10


def test_punctuation_only_text_does_not_crash(client):
    # No real words at all: the TF-IDF vector is all zeros, but
    # the pipelines still answer from their intercepts.
    response = client.get("/api/predict", params={"text": "!!! ??? ..."})
    assert response.status_code == 200
    assert response.json()["category"] in CATEGORIES


# --- Non-string input to predict_ticket -----------------------


@pytest.mark.parametrize(
    "bad", [None, 42, 3.5, ["my card was charged twice"], {"text": "refund"}]
)
def test_predict_ticket_rejects_non_string_input(bad):
    with pytest.raises(ValueError, match="must be a string"):
        predict_ticket(bad)


# --- Sample tickets through the API ---------------------------


@pytest.mark.parametrize("sample", SAMPLE_TICKETS, ids=lambda s: s["label"])
def test_sample_ticket_predicts_successfully_through_api(client, sample):
    response = client.get("/api/predict", params={"text": sample["text"]})
    assert response.status_code == 200
    result = response.json()
    assert result["category"] in CATEGORIES
    assert result["priority"] in PRIORITIES
    assert result["low_confidence"] == sample["should_warn"]
    if sample["expected_category"] is not None:
        assert result["category"] == sample["expected_category"]
    if sample["should_warn"]:
        assert result["warning"] is not None
        assert "check this ticket by hand" in result["warning"]
    else:
        assert result["warning"] is None


def test_home_page_shows_the_sample_tickets(client):
    response = client.get("/")
    assert response.status_code == 200
    for sample in SAMPLE_TICKETS:
        assert sample["label"] in response.text
    # The buttons fill the form's text box with the sample text.
    assert "getElementById('text').value=" in response.text
