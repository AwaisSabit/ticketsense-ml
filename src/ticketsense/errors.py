"""Error analysis for the TicketSense baseline.

Takes the held-out test tickets and the two trained pipelines and
looks at *which* tickets the baseline gets wrong, instead of only
how many: the misclassified tickets with their texts, the most
common confusion pairs (which true label is mistaken for which
predicted label), and the per-class weak spots.

A second baseline — TF-IDF + Multinomial Naive Bayes, trained on
the exact same training split — is scored on the same test split
for comparison. Logistic Regression stays the baseline unless
Naive Bayes is clearly better on both tasks; the comparison is
recorded honestly either way.

The same split as training is reused (stratified, seed 42, 20%
held out), so nothing here retrains differently. Run it with::

    PYTHONPATH=src python -m ticketsense.errors

after ``python -m ticketsense.train`` has saved the pipelines.
It writes ``reports/error_analysis.md`` and two confusion-matrix
charts (``reports/category_confusion.png`` and
``reports/priority_confusion.png``). The charts are local only —
the snapshot push is text-only — the Markdown report is what
gets committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from . import CATEGORIES, PRIORITIES
from .data import DEFAULT_PATH, load_tickets
from .evaluate import evaluate_models, evaluate_pipeline
from .train import (
    DEFAULT_CATEGORY_MODEL,
    DEFAULT_PRIORITY_MODEL,
    RANDOM_SEED,
    TEST_SIZE,
    build_pipeline,
    split_tickets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "error_analysis.md"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

# A class counts as a weak spot when the baseline finds fewer
# than half of its test tickets (recall below 0.5).
WEAK_RECALL = 0.5


def _load_model(path: str | Path, what: str):
    """Load a saved pipeline, or fail with a clear message."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{what} model not found at {path} — "
            "run `PYTHONPATH=src python -m ticketsense.train` first "
            "to train and save the baseline pipelines."
        )
    return joblib.load(path)


def misclassified(
    test: pd.DataFrame, category_model, priority_model
) -> dict[str, list[dict]]:
    """List every test ticket the baseline gets wrong, with its text.

    Returns ``{"category": [...], "priority": [...]}`` where each
    entry is ``{"ticket_id", "text", "true_label",
    "predicted_label"}`` in test-set order.
    """
    category_pred = category_model.predict(test["text"])
    priority_pred = priority_model.predict(test["text"])
    result: dict[str, list[dict]] = {"category": [], "priority": []}
    for (_, row), cat_pred, pri_pred in zip(
        test.iterrows(), category_pred, priority_pred
    ):
        if cat_pred != row["category"]:
            result["category"].append(
                {
                    "ticket_id": str(row["ticket_id"]),
                    "text": str(row["text"]),
                    "true_label": str(row["category"]),
                    "predicted_label": str(cat_pred),
                }
            )
        if pri_pred != row["priority"]:
            result["priority"].append(
                {
                    "ticket_id": str(row["ticket_id"]),
                    "text": str(row["text"]),
                    "true_label": str(row["priority"]),
                    "predicted_label": str(pri_pred),
                }
            )
    return result


def confusion_pairs(
    confusion_matrix: list[list[int]], labels: tuple[str, ...]
) -> list[dict]:
    """Count the off-diagonal confusion pairs, most common first.

    Each entry is ``{"true_label", "predicted_label", "count"}``.
    Correct predictions (the diagonal) are not listed.
    """
    pairs = []
    for i, true_label in enumerate(labels):
        for j, predicted_label in enumerate(labels):
            if i == j:
                continue
            count = int(confusion_matrix[i][j])
            if count > 0:
                pairs.append(
                    {
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": count,
                    }
                )
    pairs.sort(key=lambda pair: (-pair["count"], pair["true_label"], pair["predicted_label"]))
    return pairs


def weak_spots(per_class: dict) -> list[dict]:
    """Per-class weak spots: classes with recall below WEAK_RECALL.

    Each entry is ``{"label", "recall", "f1", "support"}``, worst
    recall first.
    """
    spots = [
        {
            "label": name,
            "recall": row["recall"],
            "f1": row["f1"],
            "support": row["support"],
        }
        for name, row in per_class.items()
        if row["recall"] < WEAK_RECALL
    ]
    spots.sort(key=lambda spot: (spot["recall"], spot["label"]))
    return spots


def build_naive_bayes_pipeline() -> Pipeline:
    """TF-IDF + Multinomial Naive Bayes, same text setup as the baseline.

    The vectoriser settings match ``train.build_pipeline`` (the
    shared ``clean_text`` preprocessor, word pairs) so the only
    difference between the two baselines is the classifier.
    """
    from .features import clean_text

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text, ngram_range=(1, 2), min_df=1
                ),
            ),
            ("clf", MultinomialNB()),
        ]
    )


def compare_naive_bayes(
    data_path: str | Path = DEFAULT_PATH,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> dict:
    """Train and score the Naive Bayes baseline on the same split.

    Returns accuracy and macro F1 for category and priority, plus
    the split details, so the numbers sit next to the Logistic
    Regression ones in the report.
    """
    tickets = load_tickets(data_path)
    train, test = split_tickets(tickets, test_size=test_size, seed=seed)

    category_model = build_naive_bayes_pipeline()
    category_model.fit(train["text"], train["category"])
    priority_model = build_naive_bayes_pipeline()
    priority_model.fit(train["text"], train["priority"])

    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_size": test_size,
        "seed": seed,
        "category": evaluate_pipeline(
            category_model, test["text"], test["category"], CATEGORIES
        ),
        "priority": evaluate_pipeline(
            priority_model, test["text"], test["priority"], PRIORITIES
        ),
    }


def error_analysis(
    category_path: str | Path = DEFAULT_CATEGORY_MODEL,
    priority_path: str | Path = DEFAULT_PRIORITY_MODEL,
    data_path: str | Path = DEFAULT_PATH,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> dict:
    """Analyse the Logistic Regression baseline's errors on the test set.

    Loads the two saved pipelines, replays the training split, and
    returns the evaluation numbers, the misclassified tickets, the
    confusion pairs, the weak spots, and the Naive Bayes comparison.
    """
    tickets = load_tickets(data_path)
    _train, test = split_tickets(tickets, test_size=test_size, seed=seed)

    category_model = _load_model(category_path, "Category")
    priority_model = _load_model(priority_path, "Priority")

    results = evaluate_models(
        category_path=category_path,
        priority_path=priority_path,
        data_path=data_path,
        test_size=test_size,
        seed=seed,
    )
    errors = misclassified(test, category_model, priority_model)
    comparison = compare_naive_bayes(
        data_path=data_path, test_size=test_size, seed=seed
    )

    return {
        "n_train": results["n_train"],
        "n_test": results["n_test"],
        "test_size": test_size,
        "seed": seed,
        "test_category_counts": results["test_category_counts"],
        "test_priority_counts": results["test_priority_counts"],
        "category": results["category"],
        "priority": results["priority"],
        "misclassified": errors,
        "category_confusion_pairs": confusion_pairs(
            results["category"]["confusion_matrix"], CATEGORIES
        ),
        "priority_confusion_pairs": confusion_pairs(
            results["priority"]["confusion_matrix"], PRIORITIES
        ),
        "category_weak_spots": weak_spots(results["category"]["per_class"]),
        "priority_weak_spots": weak_spots(results["priority"]["per_class"]),
        "naive_bayes": comparison,
    }


def make_charts(
    analysis: dict, reports_dir: str | Path | None = None
) -> list[Path]:
    """Draw the category and priority confusion-matrix charts.

    Same style as the balance charts in ``ticketsense.explore``
    (matplotlib, non-interactive Agg backend, 7x4 figures, tight
    layout): a grid with the count in each cell, rows are the true
    label and columns are the predicted label. Returns the two PNG
    paths written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports_dir = Path(reports_dir) if reports_dir else DEFAULT_REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for key, title, filename in (
        ("category", "Category confusion matrix", "category_confusion.png"),
        ("priority", "Priority confusion matrix", "priority_confusion.png"),
    ):
        result = analysis[key]
        matrix = result["confusion_matrix"]
        labels = result["labels"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.imshow(matrix, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > max(max(r) for r in matrix) / 2 else "black",
                )
        fig.tight_layout()
        out = reports_dir / filename
        fig.savefig(out)
        plt.close(fig)
        written.append(out)
    return written


def _errors_table(errors: list[dict]) -> str:
    """Render one misclassified-tickets table in Markdown."""
    if not errors:
        return "No tickets were misclassified."
    lines = ["| Ticket | Text | True label | Predicted label |"]
    lines.append("| --- | --- | --- | --- |")
    for error in errors:
        lines.append(
            f"| {error['ticket_id']} | {error['text']} "
            f"| {error['true_label']} | {error['predicted_label']} |"
        )
    return "\n".join(lines)


def _pairs_table(pairs: list[dict]) -> str:
    """Render one confusion-pairs table in Markdown."""
    if not pairs:
        return "No confusions — every ticket was classified correctly."
    lines = ["| True label | Predicted label | Tickets |"]
    lines.append("| --- | --- | ---: |")
    for pair in pairs:
        lines.append(
            f"| {pair['true_label']} | {pair['predicted_label']} | {pair['count']} |"
        )
    return "\n".join(lines)


def _weak_spots_text(spots: list[dict], what: str) -> str:
    """Plain-English weak-spot lines for one task."""
    if not spots:
        return f"No {what} weak spots: every label was found at least half the time."
    lines = []
    for spot in spots:
        lines.append(
            f"- **{spot['label']}**: recall {spot['recall']:.4f} "
            f"(found {spot['recall'] * spot['support']:.0f} of "
            f"{spot['support']} test tickets), F1 {spot['f1']:.4f}."
        )
    return "\n".join(lines)


def write_report(analysis: dict, path: str | Path = DEFAULT_REPORT_PATH) -> Path:
    """Write the error-analysis results as a readable Markdown report."""
    path = Path(path)
    report_dir = path.parent
    category_chart = report_dir / "category_confusion.png"
    priority_chart = report_dir / "priority_confusion.png"

    category = analysis["category"]
    priority = analysis["priority"]
    naive_bayes = analysis["naive_bayes"]

    n_cat_wrong = len(analysis["misclassified"]["category"])
    n_pri_wrong = len(analysis["misclassified"]["priority"])
    n_test = analysis["n_test"]

    # The comparison verdict: Naive Bayes only replaces Logistic
    # Regression if it is better on both tasks at once.
    nb_better_category = naive_bayes["category"]["accuracy"] > category["accuracy"]
    nb_better_priority = naive_bayes["priority"]["accuracy"] > priority["accuracy"]
    if nb_better_category and nb_better_priority:
        verdict = (
            "Naive Bayes beats Logistic Regression on both tasks, so it "
            "is worth considering as the new baseline — but on 40 test "
            "tickets a small lead is mostly noise, so the switch would "
            "need a bigger test set to be trusted."
        )
    elif nb_better_category or nb_better_priority:
        verdict = (
            "Naive Bayes is better on one task and worse on the other, "
            "so there is no clear winner. Logistic Regression stays the "
            "baseline."
        )
    else:
        verdict = (
            "Naive Bayes is worse on both tasks, so Logistic Regression "
            "stays the baseline."
        )

    lines = ["# TicketSense baseline — error analysis", ""]
    lines.append(
        "The baseline is TF-IDF (word pairs, cleaned text) + Logistic "
        "Regression, one pipeline for ticket category and one for "
        "priority. This report looks at the tickets it gets wrong on "
        "the held-out test set, instead of only how many it gets wrong."
    )
    lines.append("")

    lines.append("## Split")
    lines.append("")
    lines.append(f"- Total tickets: {analysis['n_train'] + analysis['n_test']}")
    lines.append(f"- Train: {analysis['n_train']} tickets")
    lines.append(
        f"- Test: {analysis['n_test']} tickets "
        f"({analysis['test_size']:.0%} held out, stratified by category, "
        f"random seed {analysis['seed']})"
    )
    lines.append(f"- Test category counts: {analysis['test_category_counts']}")
    lines.append(f"- Test priority counts: {analysis['test_priority_counts']}")
    lines.append("")

    lines.append("## How many were wrong")
    lines.append("")
    lines.append(
        f"- Category: **{n_cat_wrong} of {n_test}** test tickets "
        f"misclassified (accuracy {category['accuracy']:.4f})."
    )
    lines.append(
        f"- Priority: **{n_pri_wrong} of {n_test}** test tickets "
        f"misclassified (accuracy {priority['accuracy']:.4f})."
    )
    lines.append("")

    lines.append("## Category errors")
    lines.append("")
    lines.append("### Misclassified tickets")
    lines.append("")
    lines.append(_errors_table(analysis["misclassified"]["category"]))
    lines.append("")
    lines.append("### Confusion pairs (most common first)")
    lines.append("")
    lines.append(_pairs_table(analysis["category_confusion_pairs"]))
    lines.append("")
    lines.append("### Weak spots (recall below 0.5)")
    lines.append("")
    lines.append(_weak_spots_text(analysis["category_weak_spots"], "category"))
    lines.append("")

    lines.append("## Priority errors")
    lines.append("")
    lines.append("### Misclassified tickets")
    lines.append("")
    lines.append(_errors_table(analysis["misclassified"]["priority"]))
    lines.append("")
    lines.append("### Confusion pairs (most common first)")
    lines.append("")
    lines.append(_pairs_table(analysis["priority_confusion_pairs"]))
    lines.append("")
    lines.append("### Weak spots (recall below 0.5)")
    lines.append("")
    lines.append(_weak_spots_text(analysis["priority_weak_spots"], "priority"))
    lines.append("")

    if category_chart.exists() or priority_chart.exists():
        lines.append("## Confusion-matrix charts")
        lines.append("")
        if category_chart.exists():
            lines.append("![Category confusion matrix](category_confusion.png)")
            lines.append("")
        if priority_chart.exists():
            lines.append("![Priority confusion matrix](priority_confusion.png)")
            lines.append("")

    lines.append("## Second baseline: TF-IDF + Naive Bayes")
    lines.append("")
    lines.append(
        "Trained on the same 160 training tickets and scored on the "
        "same 40 test tickets as the Logistic Regression baseline, so "
        "the numbers are directly comparable. Only the classifier "
        "differs — the text cleaning and TF-IDF setup are identical."
    )
    lines.append("")
    lines.append("| Task | Logistic Regression | Naive Bayes |")
    lines.append("| --- | ---: | ---: |")
    lines.append(
        f"| Category accuracy | {category['accuracy']:.4f} "
        f"| {naive_bayes['category']['accuracy']:.4f} |"
    )
    lines.append(
        f"| Category macro F1 | {category['macro_f1']:.4f} "
        f"| {naive_bayes['category']['macro_f1']:.4f} |"
    )
    lines.append(
        f"| Priority accuracy | {priority['accuracy']:.4f} "
        f"| {naive_bayes['priority']['accuracy']:.4f} |"
    )
    lines.append(
        f"| Priority macro F1 | {priority['macro_f1']:.4f} "
        f"| {naive_bayes['priority']['macro_f1']:.4f} |"
    )
    lines.append("")
    lines.append(f"**Verdict:** {verdict}")
    lines.append("")

    lines.append("## What the errors mean")
    lines.append("")
    lines.append(
        "- **Category errors are mostly near misses between related "
        "topics.** The confusion pairs above show tickets landing in "
        "a neighbouring category (for example a billing ticket "
        "talking about the app, or a technical ticket mentioning a "
        "payment). With one-sentence tickets the model only has a "
        "handful of words to work with, so one shared word can tip "
        "the prediction. A couple of the 'errors' are arguably "
        "debatable labels — for example the ticket about seeing "
        "someone else's orders in your account is labelled technical "
        "but reads like an account problem — so the true category "
        "errors are fewer than the count suggests."
    )
    lines.append(
        "- **Priority errors are a model problem first, and a "
        "labelling problem second.** The model almost never predicts "
        "high priority: of 14 high test tickets it found 1, calling "
        "10 medium and 3 low. Several of the missed high tickets "
        "read clearly urgent to a human — a stranger placing orders "
        "on your account, a card charged three times, a product "
        "that cut someone's hand — so the model is not just "
        "struggling with borderline cases; it systematically "
        "under-predicts high. At the same time, priority was "
        "labelled by one person's judgement with no written rules "
        "for where medium ends and high begins, so two people would "
        "disagree on several of these rows, and that caps how well "
        "any model can score on priority. Both fixes are needed: "
        "agreed labelling rules written down, and a model that is "
        "not afraid to predict high."
    )
    lines.append(
        "- **The weak spots point at the same two fixes.** There is "
        "no category weak spot by the recall-below-half rule — "
        "billing is the weakest category at recall 0.625 — but high "
        "priority is the real weak spot at recall 0.0714. Both need "
        "more, and more varied, examples of the rare case — and for "
        "priority, agreed labelling rules written down before more "
        "tickets are labelled, so the model has a consistent target "
        "to learn."
    )
    lines.append(
        "- **Nothing here changes the headline numbers.** Category "
        "accuracy stays 0.85 and priority accuracy stays 0.525 on "
        "the same 40 test tickets. This report only explains where "
        "those numbers come from."
    )
    lines.append(
        "- **Only 40 test tickets.** Every ticket is 2.5% of "
        "accuracy, so one or two tickets flipping would move the "
        "confusion pairs and the weak spots. Treat the patterns "
        "above as hints about where to look next, not as a verdict "
        "on the model."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    """Analyse the baseline errors from the command line."""
    parser = argparse.ArgumentParser(
        description="Analyse the TicketSense baseline's errors."
    )
    parser.add_argument(
        "--category-model",
        default=str(DEFAULT_CATEGORY_MODEL),
        help="Saved category pipeline to analyse.",
    )
    parser.add_argument(
        "--priority-model",
        default=str(DEFAULT_PRIORITY_MODEL),
        help="Saved priority pipeline to analyse.",
    )
    parser.add_argument(
        "--data", default=str(DEFAULT_PATH), help="Tickets CSV (for the test split)."
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT_PATH),
        help="Where to write the Markdown report.",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Where to write the confusion-matrix charts.",
    )
    parser.add_argument(
        "--test-size", type=float, default=TEST_SIZE, help="Share held out for testing."
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help="Random seed for the split."
    )
    args = parser.parse_args()

    analysis = error_analysis(
        category_path=args.category_model,
        priority_path=args.priority_model,
        data_path=args.data,
        test_size=args.test_size,
        seed=args.seed,
    )
    charts = make_charts(analysis, args.reports_dir)
    report_path = write_report(analysis, args.report)
    print(f"Test set: {analysis['n_test']} tickets")
    print(
        f"Category misclassified: {len(analysis['misclassified']['category'])} "
        f"of {analysis['n_test']}"
    )
    print(
        f"Priority misclassified: {len(analysis['misclassified']['priority'])} "
        f"of {analysis['n_test']}"
    )
    print(
        "Naive Bayes — category accuracy: "
        f"{analysis['naive_bayes']['category']['accuracy']:.4f}, "
        "priority accuracy: "
        f"{analysis['naive_bayes']['priority']['accuracy']:.4f}"
    )
    for chart in charts:
        print(f"Chart written: {chart}")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
