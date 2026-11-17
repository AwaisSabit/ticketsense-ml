"""Tests for src/ticketsense/app.py — the FastAPI demo interface."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import ticketsense.app as app_module
from ticketsense.app import app

CLEAR_BILLING = "My card was charged twice for the same order, please refund it"
CLEAR_TECHNICAL = "The app crashes every time I try to open it"
VAGUE = "hello I need some help please"
THANK_YOU = "Thank you for the quick reply, everything works great now"


@pytest.fixture(scope="module")
def client():
    """A test client for the demo app, using the saved models."""
    return TestClient(app)


def test_home_page_loads_with_form(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<form" in response.text
    assert 'action="/predict"' in response.text
    assert 'name="text"' in response.text
    assert "Predict" in response.text


def test_form_prediction_shows_category_priority_and_confidences(client):
    response = client.post("/predict", data={"text": CLEAR_BILLING})
    assert response.status_code == 200
    assert "billing" in response.text
    assert "Category:" in response.text
    assert "Priority:" in response.text
    # Both confidence scores are shown as two-decimal numbers.
    assert "confidence 0." in response.text
    # The submitted text is shown back.
    assert "charged twice" in response.text


def test_form_prediction_technical_ticket(client):
    response = client.post("/predict", data={"text": CLEAR_TECHNICAL})
    assert response.status_code == 200
    assert "technical" in response.text


def test_vague_ticket_shows_low_confidence_warning(client):
    response = client.post("/predict", data={"text": VAGUE})
    assert response.status_code == 200
    assert "check this ticket by hand" in response.text
    assert "Warning:" in response.text


def test_clear_thank_you_ticket_shows_no_warning(client):
    response = client.post("/predict", data={"text": THANK_YOU})
    assert response.status_code == 200
    assert "other" in response.text
    assert "low" in response.text
    assert "Warning:" not in response.text


def test_empty_text_shows_error_not_a_crash(client):
    for bad in ("", "   "):
        response = client.post("/predict", data={"text": bad})
        assert response.status_code == 400
        assert "must not be empty" in response.text
        assert "<form" in response.text  # the form is still there


def test_missing_models_show_train_first_error_on_form(monkeypatch, tmp_path):
    missing = tmp_path / "no_such_model.joblib"
    monkeypatch.setattr(app_module, "CATEGORY_MODEL_PATH", missing)
    monkeypatch.setattr(app_module, "PRIORITY_MODEL_PATH", missing)
    with TestClient(app) as client:
        response = client.post("/predict", data={"text": CLEAR_BILLING})
    assert response.status_code == 503
    assert "Train the baseline first" in response.text
    assert "Traceback" not in response.text


def test_json_endpoint_returns_labels_confidences_and_warning(client):
    response = client.get("/api/predict", params={"text": VAGUE})
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
    }
    assert result["category"] in {
        "billing",
        "technical",
        "account",
        "delivery",
        "other",
    }
    assert result["priority"] in {"low", "medium", "high"}
    assert 0.0 < result["category_confidence"] <= 1.0
    assert 0.0 < result["priority_confidence"] <= 1.0
    assert result["low_confidence"] is True
    assert "check this ticket by hand" in result["warning"]


def test_json_endpoint_clear_ticket_has_no_warning(client):
    response = client.get("/api/predict", params={"text": THANK_YOU})
    assert response.status_code == 200
    result = response.json()
    assert result["category"] == "other"
    assert result["priority"] == "low"
    assert result["low_confidence"] is False
    assert result["warning"] is None


def test_json_endpoint_empty_text_returns_400(client):
    for bad in ("", "   "):
        response = client.get("/api/predict", params={"text": bad})
        assert response.status_code == 400
        assert "must not be empty" in response.json()["error"]


def test_json_endpoint_missing_models_return_503(monkeypatch, tmp_path):
    missing = tmp_path / "no_such_model.joblib"
    monkeypatch.setattr(app_module, "CATEGORY_MODEL_PATH", missing)
    monkeypatch.setattr(app_module, "PRIORITY_MODEL_PATH", missing)
    with TestClient(app) as client:
        response = client.get("/api/predict", params={"text": CLEAR_BILLING})
    assert response.status_code == 503
    assert "Train the baseline first" in response.json()["error"]


def test_json_endpoint_missing_text_param_returns_400(client):
    response = client.get("/api/predict")
    assert response.status_code == 400
    assert "must not be empty" in response.json()["error"]


def test_user_text_is_escaped_on_the_result_page(client):
    response = client.post(
        "/predict", data={"text": "my card was charged twice <script>alert(1)</script>"}
    )
    assert response.status_code == 200
    assert "<script>" not in response.text
    assert "&lt;script&gt;" in response.text
