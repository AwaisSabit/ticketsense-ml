"""Tests for src/ticketsense/data.py — loading, validation, summary."""

from pathlib import Path

import pandas as pd
import pytest

from ticketsense.data import REQUIRED_COLUMNS, load_tickets, summarise

DATA = Path(__file__).resolve().parents[1] / "data" / "demo_tickets.csv"


def test_demo_dataset_loads_with_expected_schema():
    df = load_tickets(DATA)
    assert list(df.columns) == list(REQUIRED_COLUMNS)
    assert len(df) == 200
    assert df["ticket_id"].is_unique


def test_demo_dataset_labels_are_all_valid():
    df = load_tickets(DATA)
    assert set(df["category"]) == {"billing", "technical", "account", "delivery", "other"}
    assert set(df["priority"]) == {"low", "medium", "high"}
    # 40 tickets per category by construction; priority is intentionally uneven.
    assert df["category"].value_counts().to_dict() == {
        "billing": 40,
        "technical": 40,
        "account": 40,
        "delivery": 40,
        "other": 40,
    }
    assert df["priority"].value_counts().to_dict() == {
        "high": 48,
        "medium": 75,
        "low": 77,
    }
    assert df["priority"].value_counts().sum() == 200


def test_summary_reports_balance_and_no_missing_or_duplicates():
    df = load_tickets(DATA)
    summary = summarise(df)
    assert summary["n_rows"] == 200
    assert summary["missing_values"] == {c: 0 for c in REQUIRED_COLUMNS}
    assert summary["duplicate_rows"] == 0
    assert summary["duplicate_texts"] == 0
    assert summary["category_counts"]["billing"] == 40
    assert summary["priority_counts"] == {"high": 48, "medium": 75, "low": 77}
    assert sum(summary["priority_counts"].values()) == 200


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_tickets("data/does-not-exist.csv")


def _write(tmp_path, rows):
    path = tmp_path / "tickets.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_missing_column_raises(tmp_path):
    path = _write(tmp_path, [{"ticket_id": "T1", "text": "help", "category": "billing"}])
    with pytest.raises(ValueError, match="Missing required columns"):
        load_tickets(path)


def test_unknown_category_raises(tmp_path):
    path = _write(
        tmp_path,
        [{"ticket_id": "T1", "text": "help please", "category": "shipping", "priority": "low"}],
    )
    with pytest.raises(ValueError, match="Unknown categories"):
        load_tickets(path)


def test_unknown_priority_raises(tmp_path):
    path = _write(
        tmp_path,
        [{"ticket_id": "T1", "text": "help please", "category": "billing", "priority": "urgent"}],
    )
    with pytest.raises(ValueError, match="Unknown priorities"):
        load_tickets(path)


def test_duplicate_ticket_id_raises(tmp_path):
    rows = [
        {"ticket_id": "T1", "text": "first ticket", "category": "billing", "priority": "low"},
        {"ticket_id": "T1", "text": "second ticket", "category": "billing", "priority": "low"},
    ]
    with pytest.raises(ValueError, match="Duplicate ticket_id"):
        load_tickets(_write(tmp_path, rows))


def test_empty_text_raises(tmp_path):
    rows = [{"ticket_id": "T1", "text": "   ", "category": "billing", "priority": "low"}]
    with pytest.raises(ValueError, match="Empty ticket text"):
        load_tickets(_write(tmp_path, rows))
