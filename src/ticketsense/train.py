"""Training for the TicketSense baseline (TF-IDF + Logistic Regression).

The baseline trains two separate pipelines on the same ticket text:
one predicts the ticket ``category``, the other predicts the
``priority``. Both pipelines are TF-IDF followed by Logistic
Regression, fitted only on the training split — the vectoriser
never sees the held-out test tickets, so there is no leakage from
test text into the vocabulary.

The split is stratified by category with a fixed seed (42) and a
20% test share, so the same data and the same code always produce
the same 160-train / 40-test split. Saved pipelines are joblib
binaries under ``models/`` and are git-ignored (they stay local);
the text evaluation report in ``reports/evaluation.md`` is what
gets committed.

Run it with::

    PYTHONPATH=src python -m ticketsense.train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data import DEFAULT_PATH, REQUIRED_COLUMNS, load_tickets
from .features import clean_text

RANDOM_SEED = 42
TEST_SIZE = 0.2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATEGORY_MODEL = PROJECT_ROOT / "models" / "category_pipeline.joblib"
DEFAULT_PRIORITY_MODEL = PROJECT_ROOT / "models" / "priority_pipeline.joblib"


def split_tickets(
    df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split tickets into train and test sets, stratified by category.

    The split is stratified so every category keeps its share in
    both halves (with 200 tickets and 40 per category, the test set
    holds 8 tickets of each category). The same seed always gives
    the same split.

    Args:
        df: Tickets dataframe as returned by ``load_tickets``.
        test_size: Share of tickets held out for testing.
        seed: Random seed, fixed so the split is reproducible.

    Returns:
        ``(train, test)`` dataframes.

    Raises:
        ValueError: If ``df`` is not a usable tickets table, if a
            category has fewer than 2 tickets (stratification needs
            at least one per side), or if any ticket id or text
            appears on both sides of the split.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Train/test split needs a pandas dataframe.")
    if df.empty:
        raise ValueError("Cannot split an empty dataframe: no tickets to split.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if (df["category"].value_counts() < 2).any():
        raise ValueError(
            "Stratified split needs at least 2 tickets per category; "
            f"smallest category counts: {df['category'].value_counts().to_dict()}"
        )

    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["category"]
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # The demo set has no duplicate ids or texts, so any overlap
    # here would mean the split itself went wrong — fail loudly.
    if not set(train["ticket_id"]).isdisjoint(set(test["ticket_id"])):
        raise ValueError("Train/test split leaked ticket ids across the split.")
    if not set(train["text"]).isdisjoint(set(test["text"])):
        raise ValueError("Train/test split leaked ticket texts across the split.")
    return train, test


def build_pipeline() -> Pipeline:
    """Build the TF-IDF + Logistic Regression pipeline.

    ``clean_text`` is used as the vectoriser preprocessor so the
    model sees exactly the same cleaned text the rest of the
    project uses. Word pairs (1-2 grams) are included because
    short phrases like "charged twice" carry the signal in
    one-sentence tickets.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text, ngram_range=(1, 2), min_df=1
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            ),
        ]
    )


def train_baseline(
    data_path: str | Path = DEFAULT_PATH,
    category_out: str | Path = DEFAULT_CATEGORY_MODEL,
    priority_out: str | Path = DEFAULT_PRIORITY_MODEL,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> dict:
    """Train both baseline pipelines and save them to disk.

    Args:
        data_path: Tickets CSV to train on.
        category_out: Where to save the category pipeline.
        priority_out: Where to save the priority pipeline.
        test_size: Share of tickets held out for testing.
        seed: Random seed for the stratified split.

    Returns:
        A plain dict with the split sizes, the label counts in each
        half, and the paths the pipelines were saved to.
    """
    tickets = load_tickets(data_path)
    train, test = split_tickets(tickets, test_size=test_size, seed=seed)

    category_model = build_pipeline()
    category_model.fit(train["text"], train["category"])
    category_out = Path(category_out)
    category_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(category_model, category_out)

    priority_model = build_pipeline()
    priority_model.fit(train["text"], train["priority"])
    priority_out = Path(priority_out)
    priority_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(priority_model, priority_out)

    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_size": test_size,
        "seed": seed,
        "train_category_counts": train["category"].value_counts().to_dict(),
        "test_category_counts": test["category"].value_counts().to_dict(),
        "train_priority_counts": train["priority"].value_counts().to_dict(),
        "test_priority_counts": test["priority"].value_counts().to_dict(),
        "category_model": str(category_out),
        "priority_model": str(priority_out),
    }


def main() -> None:
    """Train both pipelines from the command line."""
    parser = argparse.ArgumentParser(
        description="Train the TicketSense baseline model."
    )
    parser.add_argument(
        "--data", default=str(DEFAULT_PATH), help="Tickets CSV to train on."
    )
    parser.add_argument(
        "--category-out",
        default=str(DEFAULT_CATEGORY_MODEL),
        help="Where to save the category pipeline.",
    )
    parser.add_argument(
        "--priority-out",
        default=str(DEFAULT_PRIORITY_MODEL),
        help="Where to save the priority pipeline.",
    )
    parser.add_argument(
        "--test-size", type=float, default=TEST_SIZE, help="Share held out for testing."
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help="Random seed for the split."
    )
    args = parser.parse_args()

    summary = train_baseline(
        data_path=args.data,
        category_out=args.category_out,
        priority_out=args.priority_out,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(f"Train: {summary['n_train']} tickets; test: {summary['n_test']} tickets")
    print(f"Test category counts: {summary['test_category_counts']}")
    print(f"Test priority counts: {summary['test_priority_counts']}")
    print(f"Category model saved to {summary['category_model']}")
    print(f"Priority model saved to {summary['priority_model']}")


if __name__ == "__main__":
    main()
