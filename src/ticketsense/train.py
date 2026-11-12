"""Training entry point for TicketSense (Day 1 skeleton).

Full baseline (TF-IDF + Logistic Regression, stratified train/test
split, saved pipeline) is scheduled for Day 5-6. Day 1 only
establishes the CLI surface so later days do not change the interface.
"""

from __future__ import annotations

import argparse


def main() -> None:
    """Parse arguments and report that training is not yet implemented."""
    parser = argparse.ArgumentParser(description="Train the TicketSense baseline model.")
    parser.add_argument("--data", default="data/demo_tickets.csv", help="Tickets CSV to train on.")
    parser.add_argument("--out", default="models/ticketsense.joblib", help="Where to save the pipeline.")
    args = parser.parse_args()
    # Deliberately not a silent no-op: Day 1 has no model to train yet.
    raise NotImplementedError(
        f"Training is scheduled for Day 5-6 of the 30-day plan. "
        f"Refusing to pretend to train on {args.data} -> {args.out}."
    )


if __name__ == "__main__":
    main()
