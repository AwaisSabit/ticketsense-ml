"""Exploratory analysis for TicketSense.

Pure analysis functions that take a loaded tickets dataframe (see
``ticketsense.data.load_tickets``) and return plain dicts, so they
can be reused from notebooks, tests and the report writer without
any printing. The report writer and the charts live in this module
too, and ``python -m ticketsense.explore`` runs the whole thing
against the demo dataset.

The demo set is small (200 hand-written tickets), clean and
perfectly balanced by construction, so this module describes the
data — it makes no performance claim about any model.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

import pandas as pd

from .data import DEFAULT_PATH, REQUIRED_COLUMNS, load_tickets, summarise
from .features import clean_text

# Small built-in English stopword list, so the "most common words"
# tables show ticket content instead of filler words. No new
# dependency is needed for this.
STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "else",
        "when", "while", "of", "to", "in", "on", "at", "for", "with",
        "without", "by", "from", "as", "is", "am", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "can", "could", "should",
        "may", "might", "must", "shall", "i", "me", "my", "we",
        "our", "you", "your", "he", "she", "it", "its", "they",
        "them", "their", "this", "that", "these", "those", "there",
        "here", "what", "which", "who", "whom", "how", "why", "not",
        "no", "yes", "please", "just", "very", "so", "too", "also",
        "about", "after", "before", "over", "under", "into", "out",
        "up", "down", "more", "most", "some", "any", "all", "each",
        "every", "other", "another", "one", "two", "get", "got",
        "like", "want", "need", "needs", "know", "still", "now",
        "today", "yesterday", "ago",
    }
)

DEFAULT_REPORT_PATH = Path(__file__).resolve().parents[2] / "reports" / "eda.md"
TOP_WORDS_PER_CATEGORY = 10

# Words are pulled out of the cleaned text with this pattern so
# punctuation stuck to a word ("update," at the end of a clause)
# counts as the word itself, not as a separate token.
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check the dataframe is a usable tickets table.

    Raises:
        ValueError: If ``df`` is not a dataframe, is empty, or is
            missing a required column.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Exploratory analysis needs a pandas dataframe.")
    if df.empty:
        raise ValueError("Cannot explore an empty dataframe: no tickets to analyse.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def _length_stats(values: list[int]) -> dict:
    """Mean, median, min and max of a list of lengths."""
    return {
        "mean": round(statistics.fmean(values), 2),
        "median": float(statistics.median(values)),
        "min": int(min(values)),
        "max": int(max(values)),
    }


def length_stats(df: pd.DataFrame) -> dict:
    """Word-count and character-count stats, overall and per category.

    Lengths are measured on the cleaned text (see
    ``ticketsense.features.clean_text``), so upper case and extra
    whitespace do not inflate the numbers.
    """
    df = _validate(df)
    cleaned = df["text"].astype(str).map(clean_text)
    words = cleaned.str.split().str.len()
    chars = cleaned.str.len()

    result = {
        "overall": {
            "word_count": _length_stats(words.tolist()),
            "char_count": _length_stats(chars.tolist()),
        },
        "by_category": {},
    }
    for category, group in df.groupby("category", sort=True):
        mask = df["category"] == category
        result["by_category"][category] = {
            "word_count": _length_stats(words[mask].tolist()),
            "char_count": _length_stats(chars[mask].tolist()),
        }
    return result


def top_words(df: pd.DataFrame, n: int = TOP_WORDS_PER_CATEGORY) -> dict:
    """Most common words per category, stopwords removed.

    Returns ``{category: [(word, count), ...]}`` sorted by count
    (highest first), then alphabetically so ties are stable.
    Single letters and pure numbers are dropped along with the
    stopwords, so the tables show real ticket vocabulary.
    """
    df = _validate(df)
    result: dict[str, list[tuple[str, int]]] = {}
    for category, group in df.groupby("category", sort=True):
        counts: dict[str, int] = {}
        for text in group["text"].astype(str):
            for word in _WORD.findall(clean_text(text)):
                if word in STOPWORDS or len(word) < 2 or not any(c.isalpha() for c in word):
                    continue
                counts[word] = counts.get(word, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        result[category] = ranked[:n]
    return result


def cross_tab(df: pd.DataFrame) -> dict:
    """Category-by-priority counts as a nested dict.

    Every category and priority appears, even with a count of 0,
    so the table always adds up to the row count.
    """
    df = _validate(df)
    table = pd.crosstab(df["category"], df["priority"])
    return {
        category: {
            priority: int(table.at[category, priority])
            if priority in table.columns and category in table.index
            else 0
            for priority in ("low", "medium", "high")
        }
        for category in ("billing", "technical", "account", "delivery", "other")
    }


def explore(df: pd.DataFrame) -> dict:
    """Run every analysis and return one plain dict."""
    df = _validate(df)
    summary = summarise(df)
    lengths = length_stats(df)
    words = top_words(df)
    table = cross_tab(df)
    return {
        "n_rows": summary["n_rows"],
        "missing_values": summary["missing_values"],
        "duplicate_rows": summary["duplicate_rows"],
        "duplicate_texts": summary["duplicate_texts"],
        "category_counts": summary["category_counts"],
        "priority_counts": summary["priority_counts"],
        "length_stats": lengths,
        "top_words": words,
        "cross_tab": table,
    }


def _stats_table(stats: dict) -> str:
    """One Markdown table row per stats dict (word and char counts)."""
    lines = [
        "| Measure | Mean | Median | Min | Max |",
        "|---|---|---|---|---|",
    ]
    for label, key in (("Word count", "word_count"), ("Character count", "char_count")):
        s = stats[key]
        lines.append(
            f"| {label} | {s['mean']} | {s['median']:.0f} | {s['min']} | {s['max']} |"
        )
    return "\n".join(lines)


def write_report(df: pd.DataFrame, path: str | Path = DEFAULT_REPORT_PATH) -> Path:
    """Write a readable Markdown exploration report for the tickets.

    If the balance charts (``category_balance.png`` and
    ``priority_balance.png``) sit next to the report file, they are
    referenced from it; otherwise the report is written without
    them. Returns the path written.
    """
    df = _validate(df)
    result = explore(df)
    path = Path(path)
    report_dir = path.parent
    category_chart = report_dir / "category_balance.png"
    priority_chart = report_dir / "priority_balance.png"

    lines = ["# TicketSense — exploratory analysis", ""]
    lines.append(
        f"This report describes the {result['n_rows']} demo tickets in "
        "`data/demo_tickets.csv`. It is a description of the data, "
        "not an evaluation of any model — for the baseline results, "
        "see `reports/evaluation.md`."
    )
    lines.append("")

    lines.append("## Dataset overview")
    lines.append("")
    lines.append(f"- Rows: **{result['n_rows']}**")
    lines.append("- Missing values per column:")
    for column, count in result["missing_values"].items():
        lines.append(f"  - `{column}`: {count}")
    lines.append(f"- Duplicate rows: {result['duplicate_rows']}")
    lines.append(f"- Duplicate texts: {result['duplicate_texts']}")
    lines.append("")

    lines.append("## Label distribution")
    lines.append("")
    lines.append("### Category")
    lines.append("")
    lines.append("| Category | Tickets |")
    lines.append("|---|---:|")
    for category in ("billing", "technical", "account", "delivery", "other"):
        lines.append(f"| {category} | {result['category_counts'].get(category, 0)} |")
    lines.append("")
    lines.append("### Priority")
    lines.append("")
    lines.append("| Priority | Tickets |")
    lines.append("|---|---:|")
    for priority in ("low", "medium", "high"):
        lines.append(f"| {priority} | {result['priority_counts'].get(priority, 0)} |")
    lines.append("")

    if category_chart.exists() or priority_chart.exists():
        lines.append("## Balance charts")
        lines.append("")
        if category_chart.exists():
            lines.append("![Tickets per category](category_balance.png)")
            lines.append("")
        if priority_chart.exists():
            lines.append("![Tickets per priority](priority_balance.png)")
            lines.append("")

    lines.append("## Text length")
    lines.append("")
    lines.append("Word and character counts are measured on the cleaned text")
    lines.append("(lower-cased, extra whitespace collapsed).")
    lines.append("")
    lines.append("### Overall")
    lines.append("")
    lines.append(_stats_table(result["length_stats"]["overall"]))
    lines.append("")
    for category in ("billing", "technical", "account", "delivery", "other"):
        lines.append(f"### {category}")
        lines.append("")
        lines.append(_stats_table(result["length_stats"]["by_category"][category]))
        lines.append("")

    lines.append("## Most common words per category")
    lines.append("")
    lines.append(
        "Words are lower-cased and cleaned, common English stopwords "
        "are removed, and only the top 10 per category are shown."
    )
    lines.append("")
    for category in ("billing", "technical", "account", "delivery", "other"):
        words = result["top_words"].get(category, [])
        listed = ", ".join(f"{word} ({count})" for word, count in words)
        lines.append(f"- **{category}**: {listed}")
    lines.append("")

    lines.append("## Category by priority")
    lines.append("")
    lines.append("| Category | low | medium | high | Total |")
    lines.append("|---|---:|---:|---:|---:|")
    for category in ("billing", "technical", "account", "delivery", "other"):
        row = result["cross_tab"][category]
        total = sum(row.values())
        lines.append(
            f"| {category} | {row['low']} | {row['medium']} | {row['high']} | {total} |"
        )
    lines.append(f"| **Total** | {result['priority_counts'].get('low', 0)} | "
                 f"{result['priority_counts'].get('medium', 0)} | "
                 f"{result['priority_counts'].get('high', 0)} | "
                 f"{result['n_rows']} |")
    lines.append("")

    lines.append("## What this means")
    lines.append("")
    lines.append(
        "- The set is **small**: 200 tickets cannot show how a model "
        "would behave on a real support queue. Any model trained on "
        "it is a pipeline smoke test only."
    )
    lines.append(
        "- The set is **clean and hand-written**: every ticket is one "
        "tidy English sentence, with no typos, no order numbers and "
        "no personal data. Real tickets are messier than this."
    )
    lines.append(
        "- The categories are **perfectly balanced** (40 tickets each) "
        "because the set was built that way, not because real queues "
        "look like that. A model that does well here has only shown "
        "it can separate tidy, evenly spread examples."
    )
    lines.append(
        "- Priority labels are one person's judgement, not a support "
        "team's agreed policy, so the priority split (high 48, "
        "medium 75, low 77) is a starting point, not a ground truth."
    )
    lines.append(
        "- Nothing in this report is a performance claim. The baseline "
        "trained on this set is judged in `reports/evaluation.md`, "
        "with those limits in mind."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_charts(df: pd.DataFrame, reports_dir: str | Path | None = None) -> list[Path]:
    """Draw the category and priority balance bar charts.

    Returns the two PNG paths written. Matplotlib is used with the
    non-interactive Agg backend, so this works without a display.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _validate(df)
    summary = summarise(df)
    reports_dir = Path(reports_dir) if reports_dir else DEFAULT_REPORT_PATH.parent
    reports_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for counts, title, filename in (
        (summary["category_counts"], "Tickets per category", "category_balance.png"),
        (summary["priority_counts"], "Tickets per priority", "priority_balance.png"),
    ):
        labels = sorted(counts)
        values = [counts[label] for label in labels]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, values)
        ax.set_title(title)
        ax.set_ylabel("Tickets")
        for label, value in zip(labels, values):
            ax.text(label, value + 0.5, str(value), ha="center", va="bottom")
        ax.set_ylim(0, max(values) * 1.15)
        fig.tight_layout()
        out = reports_dir / filename
        fig.savefig(out)
        plt.close(fig)
        written.append(out)
    return written


def main() -> Path:
    """Load the demo data, chart it and write reports/eda.md."""
    tickets = load_tickets(DEFAULT_PATH)
    charts = make_charts(tickets)
    report = write_report(tickets)
    summary = summarise(tickets)
    print(f"Rows: {summary['n_rows']}")
    print(f"Category balance: {summary['category_counts']}")
    print(f"Priority balance: {summary['priority_counts']}")
    for chart in charts:
        print(f"Chart written: {chart}")
    print(f"Report written: {report}")
    return report


if __name__ == "__main__":
    main()
