"""Tests for src/ticketsense/explore.py — analysis, charts, report."""

from pathlib import Path

import pandas as pd
import pytest

from ticketsense.data import REQUIRED_COLUMNS, load_tickets
from ticketsense.explore import (
    cross_tab,
    explore,
    length_stats,
    make_charts,
    top_words,
    write_report,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "demo_tickets.csv"


def _small_frame():
    """A tiny hand-made dataframe with lengths we can check by hand."""
    return pd.DataFrame(
        [
            {"ticket_id": "T1", "text": "please refund my payment", "category": "billing", "priority": "high"},
            {"ticket_id": "T2", "text": "payment failed twice today", "category": "billing", "priority": "medium"},
            {"ticket_id": "T3", "text": "the app crashes on login", "category": "technical", "priority": "high"},
            {"ticket_id": "T4", "text": "app is slow", "category": "technical", "priority": "low"},
        ]
    )


def test_length_stats_are_correct_on_small_frame():
    stats = length_stats(_small_frame())
    # Word counts: 4, 4, 5, 3 -> mean 4.0, median 4.0, min 3, max 5.
    assert stats["overall"]["word_count"] == {"mean": 4.0, "median": 4.0, "min": 3, "max": 5}
    # Char counts on cleaned text: 24, 26, 24, 11 -> mean 21.25, median 24.0.
    assert stats["overall"]["char_count"] == {"mean": 21.25, "median": 24.0, "min": 11, "max": 26}
    # Billing word counts are 4 and 4; technical are 5 and 3.
    assert stats["by_category"]["billing"]["word_count"] == {
        "mean": 4.0,
        "median": 4.0,
        "min": 4,
        "max": 4,
    }
    assert stats["by_category"]["technical"]["word_count"] == {
        "mean": 4.0,
        "median": 4.0,
        "min": 3,
        "max": 5,
    }


def test_top_words_exclude_stopwords():
    words = top_words(_small_frame())
    billing = dict(words["billing"])
    # "please", "my" and "the" are stopwords, so they must not appear.
    assert "please" not in billing
    assert "my" not in billing
    assert "the" not in dict(words["technical"])
    # Real content words survive, counted correctly.
    assert billing["payment"] == 2
    assert billing["refund"] == 1
    assert dict(words["technical"])["app"] == 2


def test_top_words_strip_trailing_punctuation():
    df = pd.DataFrame(
        [
            {"ticket_id": "T1", "text": "please update, the app", "category": "technical", "priority": "low"},
            {"ticket_id": "T2", "text": "update the app now", "category": "technical", "priority": "low"},
        ]
    )
    words = dict(top_words(df)["technical"])
    assert words["update"] == 2
    assert words["app"] == 2
    assert "update," not in words


def test_cross_tab_counts_add_up_to_row_count():
    df = load_tickets(DATA)
    table = cross_tab(df)
    total = sum(sum(row.values()) for row in table.values())
    assert total == len(df) == 200
    # Each category row adds up to its 40 tickets.
    assert sum(table["billing"].values()) == 40
    assert sum(table["technical"].values()) == 40
    # Priority columns add up to the known priority split.
    assert sum(row["high"] for row in table.values()) == 48
    assert sum(row["medium"] for row in table.values()) == 75
    assert sum(row["low"] for row in table.values()) == 77
    # Small frame: every cell is present, even the zeros.
    small = cross_tab(_small_frame())
    assert small["billing"] == {"low": 0, "medium": 1, "high": 1}
    assert small["technical"] == {"low": 1, "medium": 0, "high": 1}
    assert small["other"] == {"low": 0, "medium": 0, "high": 0}


def test_explore_returns_plain_dicts_with_expected_keys():
    result = explore(load_tickets(DATA))
    assert result["n_rows"] == 200
    assert result["missing_values"] == {c: 0 for c in REQUIRED_COLUMNS}
    assert result["duplicate_rows"] == 0
    assert result["duplicate_texts"] == 0
    assert set(result["length_stats"]["by_category"]) == {
        "billing",
        "technical",
        "account",
        "delivery",
        "other",
    }
    assert set(result["top_words"]) == {
        "billing",
        "technical",
        "account",
        "delivery",
        "other",
    }
    assert len(result["top_words"]["billing"]) == 10


def test_report_is_written_and_contains_key_sections(tmp_path):
    df = load_tickets(DATA)
    report = write_report(df, tmp_path / "eda.md")
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    for section in (
        "# TicketSense — exploratory analysis",
        "## Dataset overview",
        "## Label distribution",
        "## Text length",
        "## Most common words per category",
        "## Category by priority",
        "## What this means",
    ):
        assert section in text
    assert "200" in text
    assert "perfectly balanced" in text
    assert "Nothing in this report is a performance claim" in text


def test_report_references_charts_when_they_exist(tmp_path):
    df = load_tickets(DATA)
    make_charts(df, tmp_path)
    report = write_report(df, tmp_path / "eda.md")
    text = report.read_text(encoding="utf-8")
    assert "category_balance.png" in text
    assert "priority_balance.png" in text


def test_charts_are_written_as_small_pngs(tmp_path):
    written = make_charts(load_tickets(DATA), tmp_path)
    assert len(written) == 2
    for path in written:
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0


def test_empty_dataframe_raises_clear_error():
    empty = pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    with pytest.raises(ValueError, match="empty dataframe"):
        length_stats(empty)
    with pytest.raises(ValueError, match="empty dataframe"):
        top_words(empty)
    with pytest.raises(ValueError, match="empty dataframe"):
        cross_tab(empty)
    with pytest.raises(ValueError, match="empty dataframe"):
        write_report(empty, Path("/tmp/never-written.md"))


def test_missing_column_raises_clear_error():
    df = _small_frame().drop(columns=["priority"])
    with pytest.raises(ValueError, match="Missing required columns"):
        explore(df)
