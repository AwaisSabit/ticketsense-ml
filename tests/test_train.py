"""Tests for src/ticketsense/train.py — split and baseline training."""

from pathlib import Path

import joblib
import pandas as pd
import pytest

from ticketsense.data import load_tickets
from ticketsense.train import build_pipeline, split_tickets, train_baseline

DATA = Path(__file__).resolve().parents[1] / "data" / "demo_tickets.csv"


def _small_frame() -> pd.DataFrame:
    """16 tickets, 4 per category, so a 25% split gives 1 test ticket each."""
    rows = []
    for i, category in enumerate(("billing", "technical", "account", "delivery")):
        for j in range(4):
            rows.append(
                {
                    "ticket_id": f"T{i}{j}",
                    "text": f"please help with my {category} problem number {j}",
                    "category": category,
                    "priority": ("low", "medium", "high")[j % 3],
                }
            )
    return pd.DataFrame(rows)


def test_demo_split_is_160_train_40_test():
    df = load_tickets(DATA)
    train, test = split_tickets(df)
    assert len(train) == 160
    assert len(test) == 40


def test_demo_split_is_stratified_by_category():
    df = load_tickets(DATA)
    train, test = split_tickets(df)
    assert test["category"].value_counts().to_dict() == {
        "billing": 8,
        "technical": 8,
        "account": 8,
        "delivery": 8,
        "other": 8,
    }
    assert train["category"].value_counts().to_dict() == {
        "billing": 32,
        "technical": 32,
        "account": 32,
        "delivery": 32,
        "other": 32,
    }


def test_split_has_no_overlap_and_is_reproducible():
    df = load_tickets(DATA)
    train_a, test_a = split_tickets(df, seed=42)
    train_b, test_b = split_tickets(df, seed=42)
    assert set(train_a["ticket_id"]).isdisjoint(set(test_a["ticket_id"]))
    assert set(train_a["text"]).isdisjoint(set(test_a["text"]))
    assert list(test_a["ticket_id"]) == list(test_b["ticket_id"])
    assert list(train_a["ticket_id"]) == list(train_b["ticket_id"])


def test_different_seed_gives_a_different_split():
    df = load_tickets(DATA)
    _train, test_42 = split_tickets(df, seed=42)
    _train, test_7 = split_tickets(df, seed=7)
    assert set(test_42["ticket_id"]) != set(test_7["ticket_id"])


def test_small_frame_split_is_stratified():
    train, test = split_tickets(_small_frame(), test_size=0.25, seed=42)
    assert len(train) == 12
    assert len(test) == 4
    assert test["category"].value_counts().to_dict() == {
        "billing": 1,
        "technical": 1,
        "account": 1,
        "delivery": 1,
    }


def test_empty_dataframe_raises():
    with pytest.raises(ValueError, match="empty dataframe"):
        split_tickets(pd.DataFrame(columns=["ticket_id", "text", "category", "priority"]))


def test_category_with_one_ticket_raises():
    rows = _small_frame().to_dict("records")
    rows.append(
        {
            "ticket_id": "T99",
            "text": "a lone ticket about something else",
            "category": "other",
            "priority": "low",
        }
    )
    with pytest.raises(ValueError, match="at least 2 tickets per category"):
        split_tickets(pd.DataFrame(rows))


def test_pipeline_fits_and_predicts_valid_labels():
    df = load_tickets(DATA)
    train, test = split_tickets(df)
    pipeline = build_pipeline()
    pipeline.fit(train["text"], train["category"])
    predicted = pipeline.predict(test["text"])
    assert len(predicted) == 40
    assert set(predicted) <= {"billing", "technical", "account", "delivery", "other"}
    # The hand-written demo set is keyword-heavy, so the baseline
    # should do far better than guessing (20% for 5 categories).
    assert (predicted == test["category"]).mean() >= 0.7


def test_train_baseline_writes_both_pipelines(tmp_path):
    category_out = tmp_path / "category_pipeline.joblib"
    priority_out = tmp_path / "priority_pipeline.joblib"
    summary = train_baseline(
        data_path=DATA, category_out=category_out, priority_out=priority_out
    )
    assert summary["n_train"] == 160
    assert summary["n_test"] == 40
    assert summary["seed"] == 42
    assert category_out.exists()
    assert priority_out.exists()

    category_model = joblib.load(category_out)
    priority_model = joblib.load(priority_out)
    df = load_tickets(DATA)
    _train, test = split_tickets(df)
    category_pred = category_model.predict(test["text"])
    priority_pred = priority_model.predict(test["text"])
    assert set(category_pred) <= {
        "billing",
        "technical",
        "account",
        "delivery",
        "other",
    }
    assert set(priority_pred) <= {"low", "medium", "high"}
