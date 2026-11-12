"""Text features for TicketSense (Day 1 skeleton — baseline in Day 6).

The baseline planned in the 30-day plan is TF-IDF + Logistic
Regression. This module will own the vectoriser so training and
prediction always use the same preprocessing (no leakage).

Day 1: only the shared text-cleaning helper is implemented.
The TF-IDF pipeline itself lands with the baseline on Day 5-6.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lower-case text and collapse whitespace; reject empty input.

    Raises:
        ValueError: If ``text`` is not a string or is blank.
    """
    if not isinstance(text, str):
        raise ValueError("Ticket text must be a string.")
    cleaned = _WS.sub(" ", text.strip().lower())
    if not cleaned:
        raise ValueError("Ticket text must not be empty.")
    return cleaned
