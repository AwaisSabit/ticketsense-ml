"""Text features for TicketSense.

The shared text-cleaning helper lives here, and it is also the
preprocessor the TF-IDF vectoriser in ``ticketsense.train`` uses,
so training and prediction always clean text the same way.
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
