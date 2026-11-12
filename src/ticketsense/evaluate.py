"""Evaluation entry point for TicketSense (Day 1 skeleton).

Full evaluation (accuracy, precision, recall, F1, confusion matrix,
per-class results on a held-out test set) is scheduled for Day 7.
Day 1 only establishes the CLI surface.
"""

from __future__ import annotations

import argparse


def main() -> None:
    """Parse arguments and report that evaluation is not yet implemented."""
    parser = argparse.ArgumentParser(description="Evaluate a trained TicketSense model.")
    parser.add_argument("--model", default="models/ticketsense.joblib", help="Saved pipeline to evaluate.")
    parser.add_argument("--data", default="data/demo_tickets.csv", help="Held-out tickets CSV.")
    args = parser.parse_args()
    raise NotImplementedError(
        f"Evaluation is scheduled for Day 7 of the 30-day plan. "
        f"No model exists yet at {args.model}."
    )


if __name__ == "__main__":
    main()
