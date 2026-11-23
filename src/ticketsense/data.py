"""Data loading and validation for TicketSense.

Target labels (frozen for the baseline):

Categories (``category``):
    * ``billing``   -- payments, invoices, refunds, charges, subscriptions
    * ``technical`` -- crashes, errors, slowness, broken features, login loops
    * ``account``   -- password, email/phone change, lockout, deletion, security
    * ``delivery``  -- shipping status, address, courier, damaged/lost parcels
    * ``other``     -- thanks, suggestions, general questions, feedback

Priorities (``priority``):
    * ``high``   -- money lost / charged wrongly, service unusable, security
      risk, parcel marked delivered but never received.
    * ``medium`` -- a real problem, but a workaround exists or it can wait
      a day (for example a wrong address, a missing receipt, slow loading).
    * ``low``    -- a question, a small request, or a nice-to-have
      (for example opening hours, a profile-photo update, a suggestion).

The demo file is a hand-written synthetic demonstration set of
200 tickets (see ``data/README.md``). It is NOT real customer
data, and a model trained on it must not be presented as
production-ready.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import CATEGORIES, PRIORITIES

REQUIRED_COLUMNS = ("ticket_id", "text", "category", "priority")
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "demo_tickets.csv"


def load_tickets(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load a tickets CSV and validate its schema and labels.

    Args:
        path: Path to a CSV with columns ``ticket_id``, ``text``,
            ``category`` and ``priority``.

    Returns:
        The loaded dataframe, with text stripped of surrounding whitespace.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a required column is missing, a ticket id is
            duplicated, a text is empty, or a category/priority label
            is outside the allowed sets.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tickets file not found: {path}")

    df = pd.read_csv(path, dtype={"ticket_id": str, "text": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[list(REQUIRED_COLUMNS)].copy()
    df["text"] = df["text"].astype(str).str.strip()

    if df["ticket_id"].duplicated().any():
        dups = df.loc[df["ticket_id"].duplicated(), "ticket_id"].unique().tolist()
        raise ValueError(f"Duplicate ticket_id values: {dups}")
    if df["text"].eq("").any() or df["text"].isna().any():
        raise ValueError("Empty ticket text found.")
    if df["category"].isna().any() or df["priority"].isna().any():
        raise ValueError("Missing category or priority labels found.")

    bad_categories = sorted(set(df["category"]) - set(CATEGORIES))
    if bad_categories:
        raise ValueError(f"Unknown categories: {bad_categories}. Allowed: {CATEGORIES}")
    bad_priorities = sorted(set(df["priority"]) - set(PRIORITIES))
    if bad_priorities:
        raise ValueError(f"Unknown priorities: {bad_priorities}. Allowed: {PRIORITIES}")
    return df


def summarise(df: pd.DataFrame) -> dict:
    """Report row count, missing values, duplicates and class balance.

    Returns a plain dict (no printing) so tests and other modules can
    reuse it. Duplicates here are counted on the full row, after the
    strict ``load_tickets`` checks have already rejected duplicate ids.
    """
    return {
        "n_rows": int(len(df)),
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_texts": int(df["text"].duplicated().sum()),
        "category_counts": df["category"].value_counts().to_dict(),
        "priority_counts": df["priority"].value_counts().to_dict(),
    }


def print_summary(df: pd.DataFrame) -> dict:
    """Print a human-readable summary and return it (see ``summarise``)."""
    summary = summarise(df)
    print(f"Rows: {summary['n_rows']}")
    print(f"Missing values: {summary['missing_values']}")
    print(f"Duplicate rows: {summary['duplicate_rows']}")
    print(f"Duplicate texts: {summary['duplicate_texts']}")
    print(f"Category balance: {summary['category_counts']}")
    print(f"Priority balance: {summary['priority_counts']}")
    return summary


if __name__ == "__main__":
    tickets = load_tickets()
    print_summary(tickets)
