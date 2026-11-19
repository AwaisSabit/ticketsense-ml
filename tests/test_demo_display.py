"""Tests for the TicketSense demo's confidence display and flow.

The result page must show each confidence as a percentage and
a simple visual bar, with the threshold it is judged against
in plain words — and the low-confidence warning must still
appear when flagged. The result page must also offer a way
onward (a "Try another ticket" link and the six sample tickets
as clickable buttons), and the JSON endpoint must carry the
same threshold context. The model and thresholds are unchanged:
a clear thank-you ticket scores above both thresholds, and the
vague ticket scores below both.
"""

import pytest
from fastapi.testclient import TestClient

from ticketsense.app import app
from ticketsense.samples import SAMPLE_TICKETS

# Thank-you ticket: other 0.35 / low 0.50 — above both
# thresholds, so no warning and both notes say "above the bar".
CLEAR_THANK_YOU = "Thank you for the quick reply, everything works great now"
# Vague ticket: other 0.27 / below the priority bar too — below
# both thresholds, so the warning fires on both halves.
VAGUE = "hello I need some help please"


@pytest.fixture(scope="module")
def client():
    """A test client for the demo app, using the saved models."""
    return TestClient(app)


# --- Confidence display on the result page --------------------


def test_result_page_shows_percentage_and_above_threshold_wording(client):
    response = client.post("/predict", data={"text": CLEAR_THANK_YOU})
    assert response.status_code == 200
    # Each confidence as a percentage and as the raw score, with
    # the threshold it is judged against in plain words.
    assert "35%" in response.text
    assert "50%" in response.text
    assert "confidence 0." in response.text
    assert "Category confidence 35% — above the 30% bar" in response.text
    assert "Priority confidence 50% — above the 45% bar" in response.text
    assert "looks fairly sure" in response.text
    # The labels and the submitted text are still on the page.
    assert "other" in response.text
    assert "quick reply" in response.text
    # A clear ticket has no warning.
    assert "Warning:" not in response.text


def test_result_page_shows_below_threshold_wording_for_vague_ticket(client):
    response = client.post("/predict", data={"text": VAGUE})
    assert response.status_code == 200
    assert "27%" in response.text
    assert "Category confidence 27% — below the 30% bar" in response.text
    assert "below the 45% bar" in response.text
    assert "treat this with care" in response.text


def test_result_page_shows_a_visual_confidence_bar(client):
    response = client.post("/predict", data={"text": CLEAR_THANK_YOU})
    assert response.status_code == 200
    # Two bars, one per task, filled to the confidence percentage.
    assert response.text.count('class="bar"') == 2
    assert 'style="width: 35%"' in response.text
    assert 'style="width: 50%"' in response.text


def test_low_confidence_warning_still_appears_when_flagged(client):
    response = client.post("/predict", data={"text": VAGUE})
    assert response.status_code == 200
    assert "Warning:" in response.text
    assert "check this ticket by hand" in response.text


# --- Demo flow: onward from the result page -------------------


def test_result_page_has_try_another_link_and_sample_buttons(client):
    response = client.post("/predict", data={"text": CLEAR_THANK_YOU})
    assert response.status_code == 200
    # A clear way back to a fresh form.
    assert "Try another ticket" in response.text
    assert 'href="/"' in response.text
    # The six sample tickets are clickable on the result page
    # too, so another sample is one click away.
    for sample in SAMPLE_TICKETS:
        assert sample["label"] in response.text
    assert "getElementById('text').value=" in response.text
    # The form itself is still there, with the submitted text.
    assert "<form" in response.text
    assert "quick reply" in response.text


def test_home_page_still_has_form_and_samples(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<form" in response.text
    assert "Try another ticket" not in response.text
    for sample in SAMPLE_TICKETS:
        assert sample["label"] in response.text


# --- JSON endpoint: threshold context -------------------------


def test_json_response_carries_threshold_fields_and_notes(client):
    response = client.get("/api/predict", params={"text": CLEAR_THANK_YOU})
    assert response.status_code == 200
    result = response.json()
    # The original fields are unchanged.
    assert result["category"] == "other"
    assert result["priority"] == "low"
    assert result["category_confidence"] == pytest.approx(0.35, abs=0.01)
    assert result["priority_confidence"] == pytest.approx(0.50, abs=0.01)
    assert result["low_confidence"] is False
    assert result["warning"] is None
    # New threshold context, matching the result page wording.
    assert result["category_threshold"] == 0.30
    assert result["priority_threshold"] == 0.45
    assert (
        result["category_confidence_note"]
        == "Category confidence 35% — above the 30% bar, "
        "so this looks fairly sure"
    )
    assert (
        result["priority_confidence_note"]
        == "Priority confidence 50% — above the 45% bar, "
        "so this looks fairly sure"
    )


def test_json_response_notes_say_below_the_bar_for_vague_ticket(client):
    response = client.get("/api/predict", params={"text": VAGUE})
    assert response.status_code == 200
    result = response.json()
    assert result["category_threshold"] == 0.30
    assert result["priority_threshold"] == 0.45
    assert "below the 30% bar" in result["category_confidence_note"]
    assert "below the 45% bar" in result["priority_confidence_note"]
    assert "treat this with care" in result["category_confidence_note"]
    # The original fields still carry the warning as before.
    assert result["low_confidence"] is True
    assert "check this ticket by hand" in result["warning"]
