"""Sample ticket texts for the TicketSense demo page and API.

One realistic ticket per category, plus a vague ticket that the
model should not be sure about. The demo home page shows these
as clickable buttons that fill the form, so a visitor can try
the demo without typing a ticket first.

The ``expected_category`` values were checked against the saved
baseline (TF-IDF + Logistic Regression, trained on the 200
synthetic demo tickets): each category sample predicts its own
category. The vague sample predicts ``other`` at 0.27
confidence — below the 0.30 category threshold — so it triggers
the low-confidence warning on both halves. These are
expectations for the demo, not a test set: the priority half of
the baseline is weak, so no priority is promised here.
"""

from __future__ import annotations

# Each sample: a short label for the button, the ticket text,
# the category the saved baseline predicts for it (``None`` for
# the vague ticket, which has no right answer), and whether the
# low-confidence warning should fire for it.
SAMPLE_TICKETS = [
    {
        "label": "Billing — charged twice",
        "text": "My card was charged twice for the same order, please refund it",
        "expected_category": "billing",
        "should_warn": True,
    },
    {
        "label": "Technical — app crashes",
        "text": "The app crashes every time I try to open the checkout page",
        "expected_category": "technical",
        "should_warn": True,
    },
    {
        "label": "Account — cannot log in",
        "text": "I cannot log in to my account because the password reset email never arrives",
        "expected_category": "account",
        "should_warn": True,
    },
    {
        "label": "Delivery — parcel stuck",
        "text": "My parcel has been stuck in transit for five days and still has not arrived",
        "expected_category": "delivery",
        "should_warn": True,
    },
    {
        "label": "Other — thank you",
        "text": "Thank you for the quick reply, everything works great now",
        "expected_category": "other",
        "should_warn": False,
    },
    {
        "label": "Vague — should warn",
        "text": "hello I need some help please",
        "expected_category": None,
        "should_warn": True,
    },
]


def sample_texts() -> list[str]:
    """Return just the ticket texts, in display order."""
    return [sample["text"] for sample in SAMPLE_TICKETS]
