"""Prediction entry point for TicketSense (Day 1 skeleton).

Prediction (with input validation, a confidence score and a
low-confidence warning) is scheduled for Day 11-13, once a saved
model exists. Day 1 only validates the input text, using the same
``clean_text`` helper the model path will use.
"""

from __future__ import annotations

import argparse

from .features import clean_text


def predict_text(text: str) -> dict:
    """Validate input now; model prediction is added on Day 11-13.

    Returns:
        A dict with the cleaned text. The ``category``, ``priority``
        and ``confidence`` keys are intentionally absent until a model
        exists — returning a made-up prediction would be dishonest.
    """
    return {"text": clean_text(text), "category": None, "priority": None, "confidence": None}


def main() -> None:
    """Validate one ticket from the command line."""
    parser = argparse.ArgumentParser(description="Predict a ticket category and priority.")
    parser.add_argument("text", help="Ticket text to classify.")
    args = parser.parse_args()
    print(predict_text(args.text))


if __name__ == "__main__":
    main()
