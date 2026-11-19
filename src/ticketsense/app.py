"""Demo web interface for TicketSense (FastAPI).

A small web page that shows a ticket prediction with its
confidence scores and the plain-English low-confidence warning.
The prediction itself lives in ``ticketsense.predict`` — this
module only serves the page and calls it, so the model logic
is not duplicated here.

Endpoints
---------
``GET /``
    A page with a form: one text box for the ticket text and a
    Predict button, plus clickable sample tickets (from
    ``ticketsense.samples``) that fill the text box.
``POST /predict``
    The form post: predicts the category and priority for the
    submitted text and shows a result page with both labels,
    both confidence scores as percentages with a visual bar and
    the threshold each score is judged against in plain words,
    the low-confidence warning when it applies, the submitted
    text back, a "Try another ticket" link back to the form, and
    the six sample tickets as clickable buttons.
``GET /api/predict?text=...``
    The same prediction as JSON (labels, confidences,
    ``low_confidence`` and ``warning``), plus the threshold each
    confidence is judged against and a plain-English confidence
    note for each task.

Errors are shown on the page instead of a traceback: empty or
blank text shows a clear error message, and missing model files
show a clear message telling the user to train the baseline
first. The JSON endpoint answers the same two cases with a
400 and a 503 error respectively.

Run it with::

    PYTHONPATH=src python -m ticketsense.app

then open http://127.0.0.1:8000 in a browser.
"""

from __future__ import annotations

import html
import json

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .predict import (
    LOW_CONFIDENCE_CATEGORY,
    LOW_CONFIDENCE_PRIORITY,
    predict_ticket,
)
from .samples import SAMPLE_TICKETS
from .train import DEFAULT_CATEGORY_MODEL, DEFAULT_PRIORITY_MODEL

# Model file locations, kept as module-level names so tests can
# point them at missing or tiny files without touching disk.
CATEGORY_MODEL_PATH = DEFAULT_CATEGORY_MODEL
PRIORITY_MODEL_PATH = DEFAULT_PRIORITY_MODEL

EMPTY_TEXT_MESSAGE = "Ticket text must not be empty — type a ticket first."
MISSING_MODEL_MESSAGE = (
    "No trained model found. Train the baseline first with: "
    "PYTHONPATH=src python -m ticketsense.train"
)

app = FastAPI(title="TicketSense demo")

_PAGE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TicketSense — ticket prediction demo</title>
<style>
body { font-family: sans-serif; max-width: 640px; margin: 2em auto;
       padding: 0 1em; color: #222; }
h1 { font-size: 1.5em; }
textarea { width: 100%; min-height: 6em; font: inherit; }
button { font: inherit; padding: 0.4em 1.2em; margin-top: 0.5em; }
.result { border: 1px solid #999; padding: 1em; margin-top: 1em; }
.bar { background: #e0e0e0; height: 0.9em; margin: 0.2em 0; }
.bar span { display: block; height: 100%; background: #2a7d4f; }
.warning { color: #8a4b00; font-weight: bold; }
.error { color: #a00; font-weight: bold; }
blockquote { background: #f4f4f4; padding: 0.5em 1em; }
code { background: #f4f4f4; padding: 0 0.3em; }
</style>
</head>
<body>
<h1>TicketSense — ticket prediction demo</h1>
<p>Type a support ticket below. The demo suggests a category
and a priority, each with a confidence score. The scores are a
ranking, not a probability of being right — when either score
is low, the page says so plainly and a person should check the
ticket by hand.</p>
"""

_PAGE_TAIL = """
<p><a href="/api/predict?text=hello%20I%20need%20some%20help%20please">Try the JSON version</a>
(<code>GET /api/predict?text=...</code>)</p>
</body>
</html>
"""

_FORM = """
<form action="/predict" method="post">
<label for="text">Ticket text</label><br>
<textarea id="text" name="text">{text}</textarea><br>
<button type="submit">Predict</button>
</form>
"""


def _page(body: str) -> str:
    """Wrap a body fragment in the demo page shell."""
    return _PAGE_HEAD + body + _PAGE_TAIL


def _samples_html() -> str:
    """Clickable sample tickets that fill the form's text box."""
    buttons = []
    for sample in SAMPLE_TICKETS:
        # json.dumps gives a safely quoted JavaScript string; the
        # label is HTML-escaped like everything else on the page.
        buttons.append(
            "<button type=\"button\" onclick=\"document.getElementById"
            f"('text').value={json.dumps(sample['text'])};\">"
            f"{html.escape(sample['label'])}</button>"
        )
    return (
        "<p>Try a sample ticket (click one to fill the box):</p>\n"
        + "\n".join(buttons)
        + "\n"
    )


def _form_page(text: str = "", message: str = "", message_class: str = "") -> str:
    """The form page, optionally with an error message above it."""
    message_html = ""
    if message:
        message_html = (
            f'<p class="{message_class}">{html.escape(message)}</p>\n'
        )
    return _page(
        message_html + _samples_html() + _FORM.format(text=html.escape(text))
    )


def confidence_note(task: str, confidence: float, threshold: float) -> str:
    """Plain-English note comparing one confidence with its threshold.

    Args:
        task: "Category" or "Priority" — the first word of the note.
        confidence: The confidence score, 0 to 1.
        threshold: The bar this task's confidence is judged against.

    Returns:
        A sentence like "Category confidence 44% — above the 30%
        bar, so this looks fairly sure", or the below-the-bar
        version when the score is under the threshold.
    """
    percent = round(confidence * 100)
    bar = f"{round(threshold * 100)}%"
    if confidence >= threshold:
        return (
            f"{task} confidence {percent}% — above the {bar} bar, "
            "so this looks fairly sure"
        )
    return (
        f"{task} confidence {percent}% — below the {bar} bar, "
        "so treat this with care"
    )


def _confidence_html(
    task: str, label: str, confidence: float, threshold: float
) -> str:
    """One prediction as label, percentage, bar and threshold note.

    The raw two-decimal score is kept next to the percentage so
    the numbers stay comparable with the command-line output.
    The bar is plain HTML/CSS (a filled span inside a grey
    track) — no new dependency.
    """
    percent = round(confidence * 100)
    return (
        f"<p>{task}: <strong>{html.escape(label)}</strong> "
        f"(confidence {confidence:.2f}, {percent}%)</p>\n"
        f'<div class="bar"><span style="width: {percent}%"></span></div>\n'
        f"<p>{html.escape(confidence_note(task, confidence, threshold))}</p>\n"
    )


def _result_page(result: dict) -> str:
    """The form page plus the prediction result for one ticket."""
    warning_html = ""
    if result["warning"]:
        warning_html = (
            f'<p class="warning">Warning: {html.escape(result["warning"])}</p>\n'
        )
    body = (
        '<p><a href="/">Try another ticket</a></p>\n'
        + _samples_html()
        + _FORM.format(text=html.escape(result["text"]))
        + '<div class="result">\n'
        f"<p>Ticket: <blockquote>{html.escape(result['text'])}</blockquote></p>\n"
        + _confidence_html(
            "Category",
            result["category"],
            result["category_confidence"],
            LOW_CONFIDENCE_CATEGORY,
        )
        + _confidence_html(
            "Priority",
            result["priority"],
            result["priority_confidence"],
            LOW_CONFIDENCE_PRIORITY,
        )
        + warning_html
        + "</div>\n"
    )
    return _page(body)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Serve the demo page with its ticket form."""
    return _form_page()


@app.post("/predict", response_class=HTMLResponse)
def predict_form(request: Request, text: str = Form("")) -> HTMLResponse:
    """Predict from the submitted form and show the result page."""
    if not text or not text.strip():
        return HTMLResponse(
            _form_page(text=text, message=EMPTY_TEXT_MESSAGE, message_class="error"),
            status_code=400,
        )
    try:
        result = predict_ticket(
            text,
            category_path=CATEGORY_MODEL_PATH,
            priority_path=PRIORITY_MODEL_PATH,
        )
    except FileNotFoundError:
        return HTMLResponse(
            _form_page(
                text=text,
                message=MISSING_MODEL_MESSAGE,
                message_class="error",
            ),
            status_code=503,
        )
    return HTMLResponse(_result_page(result))


@app.get("/api/predict")
def predict_api(text: str = "") -> JSONResponse:
    """Predict one ticket and return the result as JSON."""
    if not text or not text.strip():
        return JSONResponse(
            {"error": EMPTY_TEXT_MESSAGE}, status_code=400
        )
    try:
        result = predict_ticket(
            text,
            category_path=CATEGORY_MODEL_PATH,
            priority_path=PRIORITY_MODEL_PATH,
        )
    except FileNotFoundError:
        return JSONResponse({"error": MISSING_MODEL_MESSAGE}, status_code=503)
    # Threshold context, same as the result page shows: the bar
    # each confidence is judged against, and the plain-English
    # note that says whether it cleared that bar. The original
    # fields are unchanged.
    result["category_threshold"] = LOW_CONFIDENCE_CATEGORY
    result["priority_threshold"] = LOW_CONFIDENCE_PRIORITY
    result["category_confidence_note"] = confidence_note(
        "Category", result["category_confidence"], LOW_CONFIDENCE_CATEGORY
    )
    result["priority_confidence_note"] = confidence_note(
        "Priority", result["priority_confidence"], LOW_CONFIDENCE_PRIORITY
    )
    return JSONResponse(result)


def main() -> None:
    """Start the demo server on http://127.0.0.1:8000."""
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
