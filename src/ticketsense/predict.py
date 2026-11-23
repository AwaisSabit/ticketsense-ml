"""Prediction for TicketSense: category and priority with confidence.

Loads the two saved pipelines trained by ``ticketsense.train``
(TF-IDF + Logistic Regression, one for category and one for
priority) and predicts both labels for a single ticket text,
together with a confidence score for each prediction.

The confidence is the probability the pipeline assigns to its
top label (``predict_proba``). A prediction is flagged as
low-confidence when its score is below the threshold for that
task. The warning is a suggestion to check the ticket by hand,
not a second prediction — the labels are returned unchanged.

Why two thresholds, and why these values
------------------------------------------
The first guess was a single threshold of 0.6. Measuring the
saved model on the 40 held-out test tickets showed that would
be wrong: this model's probabilities are compressed — category
confidence only ranges 0.22–0.47 (median 0.31) and priority
0.34–0.61 (median 0.46) — so 0.6 would flag every single
prediction and the warning would carry no information. The
scores still carry signal (correct predictions score higher
than wrong ones on both tasks), so the thresholds below were
picked from that held-out distribution instead:

- Category 0.30: on the test set, predictions at or above 0.30
  were right 95% of the time (20 of 21), while those below it
  were right 74% (14 of 19).
- Priority 0.45: predictions at or above 0.45 were right 70%
  of the time (16 of 23), while those below it were right only
  29% (5 of 17) — a flagged priority prediction is wrong more
  often than it is right, which is exactly when a human should
  check.

These numbers come from only 40 test tickets, so they are a
calibration check, not a guarantee — and the priority half is
still the weak half of the baseline (accuracy 0.525, and it
found only 1 of 14 high-priority test tickets). A larger,
properly licensed dataset and a recalibrated model must replace
this baseline before any real claim is made.

Run it with::

    PYTHONPATH=src python -m ticketsense.predict "my card was charged twice"

after ``python -m ticketsense.train`` has saved the pipelines.
Missing model files produce a clear error telling the user to
train first; empty or blank text is rejected by the same
``clean_text`` validation the rest of the project uses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

from .features import clean_text
from .train import DEFAULT_CATEGORY_MODEL, DEFAULT_PRIORITY_MODEL

# Confidence below these values counts as low, per task. See the
# module docstring for how they were measured on the held-out
# test set (a single 0.6 threshold would flag every prediction,
# because this model's probabilities are compressed).
LOW_CONFIDENCE_CATEGORY = 0.30
LOW_CONFIDENCE_PRIORITY = 0.45


def load_models(
    category_path: str | Path = DEFAULT_CATEGORY_MODEL,
    priority_path: str | Path = DEFAULT_PRIORITY_MODEL,
) -> tuple:
    """Load the two saved pipelines.

    Args:
        category_path: Saved category pipeline (joblib file).
        priority_path: Saved priority pipeline (joblib file).

    Returns:
        ``(category_model, priority_model)`` — fitted pipelines.

    Raises:
        FileNotFoundError: If either model file does not exist,
            with a message telling the user to train first.
    """
    category_path = Path(category_path)
    priority_path = Path(priority_path)
    missing = [p for p in (category_path, priority_path) if not p.exists()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Model file(s) not found: {names}. "
            "Train the baseline first with: "
            "PYTHONPATH=src python -m ticketsense.train"
        )
    return joblib.load(category_path), joblib.load(priority_path)


def predict_label(pipeline, text: str) -> tuple[str, float]:
    """Predict one label and its confidence from a fitted pipeline.

    Args:
        pipeline: A fitted scikit-learn pipeline with
            ``predict`` and ``predict_proba``.
        text: Raw ticket text (cleaned inside, so the same text
            cleaning is used as in training).

    Returns:
        ``(label, confidence)`` where confidence is the
        probability assigned to the predicted label, rounded to
        4 decimal places.
    """
    cleaned = clean_text(text)
    label = str(pipeline.predict([cleaned])[0])
    probabilities = pipeline.predict_proba([cleaned])[0]
    confidence = round(float(max(probabilities)), 4)
    return label, confidence


def predict_ticket(
    text: str,
    category_model=None,
    priority_model=None,
    category_path: str | Path = DEFAULT_CATEGORY_MODEL,
    priority_path: str | Path = DEFAULT_PRIORITY_MODEL,
    category_threshold: float = LOW_CONFIDENCE_CATEGORY,
    priority_threshold: float = LOW_CONFIDENCE_PRIORITY,
) -> dict:
    """Predict category and priority for one ticket, with confidence.

    The models are loaded from disk unless already-loaded
    pipelines are passed in (tests pass them in directly, which
    also keeps the tests fast).

    Args:
        text: Raw ticket text.
        category_model: Already-loaded category pipeline, or
            ``None`` to load it from ``category_path``.
        priority_model: Already-loaded priority pipeline, or
            ``None`` to load it from ``priority_path``.
        category_path: Where the saved category pipeline lives.
        priority_path: Where the saved priority pipeline lives.
        category_threshold: Category confidence below this
            counts as low.
        priority_threshold: Priority confidence below this
            counts as low.

    Returns:
        A plain dict with the cleaned text, the predicted
        category and priority, a confidence score for each
        (0 to 1), ``low_confidence`` (True when either score is
        below its threshold), and ``warning`` — a plain-English
        message when confidence is low, otherwise ``None``.

    Raises:
        ValueError: If ``text`` is not a string or is blank.
        FileNotFoundError: If a model file is missing and no
            loaded pipeline was passed in.
    """
    cleaned = clean_text(text)
    if category_model is None or priority_model is None:
        loaded_category, loaded_priority = load_models(
            category_path, priority_path
        )
        category_model = category_model or loaded_category
        priority_model = priority_model or loaded_priority

    category, category_confidence = predict_label(category_model, cleaned)
    priority, priority_confidence = predict_label(priority_model, cleaned)

    low_confidence = (
        category_confidence < category_threshold
        or priority_confidence < priority_threshold
    )
    warning = None
    if low_confidence:
        weak = []
        if category_confidence < category_threshold:
            weak.append(f"category ({category_confidence:.2f})")
        if priority_confidence < priority_threshold:
            weak.append(f"priority ({priority_confidence:.2f})")
        warning = (
            "Low confidence on " + " and ".join(weak) + " — "
            "please check this ticket by hand instead of trusting "
            "the suggestion."
        )

    return {
        "text": cleaned,
        "category": category,
        "category_confidence": category_confidence,
        "priority": priority,
        "priority_confidence": priority_confidence,
        "low_confidence": low_confidence,
        "warning": warning,
    }


def predict_text(text: str, **kwargs) -> dict:
    """Backwards-compatible alias for :func:`predict_ticket`.

    The Day 1 skeleton exposed this name; it now returns a real
    prediction from the saved models, like ``predict_ticket``.
    """
    return predict_ticket(text, **kwargs)


def summary_sentence(result: dict) -> str:
    """One plain-English sentence summarising a prediction.

    Returns e.g. "This looks like a billing ticket, medium
    priority." — the first thing a non-technical reader sees,
    on the demo result page and in the command-line output.
    """
    article = "an" if result["category"][0] in "aeiou" else "a"
    return (
        f"This looks like {article} {result['category']} ticket, "
        f"{result['priority']} priority."
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


def format_prediction(result: dict) -> str:
    """Render a prediction result as plain-English lines.

    The first line is a one-sentence summary a non-technical
    reader can follow, then clearly labelled lines for the
    category, the priority, each confidence as a percentage
    with the bar it is judged against, and the warning when
    confidence is low.
    """
    lines = [
        summary_sentence(result),
        f"Ticket: {result['text']}",
        f"Category: {result['category']} "
        f"(confidence {result['category_confidence']:.2f})",
        confidence_note(
            "Category",
            result["category_confidence"],
            LOW_CONFIDENCE_CATEGORY,
        ),
        f"Priority: {result['priority']} "
        f"(confidence {result['priority_confidence']:.2f})",
        confidence_note(
            "Priority",
            result["priority_confidence"],
            LOW_CONFIDENCE_PRIORITY,
        ),
    ]
    if result["warning"]:
        lines.append(f"Warning: {result['warning']}")
    return "\n".join(lines)


def main() -> None:
    """Predict one ticket from the command line."""
    parser = argparse.ArgumentParser(
        description="Predict a ticket category and priority."
    )
    parser.add_argument("text", help="Ticket text to classify.")
    parser.add_argument(
        "--category-model",
        default=str(DEFAULT_CATEGORY_MODEL),
        help="Saved category pipeline to use.",
    )
    parser.add_argument(
        "--priority-model",
        default=str(DEFAULT_PRIORITY_MODEL),
        help="Saved priority pipeline to use.",
    )
    parser.add_argument(
        "--category-threshold",
        type=float,
        default=LOW_CONFIDENCE_CATEGORY,
        help="Category confidence below this counts as low (default 0.30).",
    )
    parser.add_argument(
        "--priority-threshold",
        type=float,
        default=LOW_CONFIDENCE_PRIORITY,
        help="Priority confidence below this counts as low (default 0.45).",
    )
    args = parser.parse_args()

    try:
        result = predict_ticket(
            args.text,
            category_path=args.category_model,
            priority_path=args.priority_model,
            category_threshold=args.category_threshold,
            priority_threshold=args.priority_threshold,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(format_prediction(result))


if __name__ == "__main__":
    main()
