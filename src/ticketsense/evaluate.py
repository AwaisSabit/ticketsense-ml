"""Evaluation for the TicketSense baseline.

Loads the two saved pipelines, replays the exact same stratified
split used for training (same seed, same test share), and evaluates
both models on the held-out test tickets only — accuracy,
precision, recall and F1 (macro and per class) plus a confusion
matrix for category and for priority.

The numbers are written to ``reports/evaluation.md`` as plain text,
so they can be committed and read on GitHub (the saved pipelines
are binaries and stay local). Run it with::

    PYTHONPATH=src python -m ticketsense.evaluate

after ``python -m ticketsense.train`` has saved the pipelines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from . import CATEGORIES, PRIORITIES
from .data import DEFAULT_PATH, load_tickets
from .train import (
    DEFAULT_CATEGORY_MODEL,
    DEFAULT_PRIORITY_MODEL,
    RANDOM_SEED,
    TEST_SIZE,
    split_tickets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "evaluation.md"


def evaluate_pipeline(pipeline, texts, labels, label_names: tuple[str, ...]) -> dict:
    """Evaluate one saved pipeline on held-out texts.

    Args:
        pipeline: A fitted scikit-learn pipeline.
        texts: Ticket texts to predict on.
        labels: True labels, in the same order as ``texts``.
        label_names: The full label set, in display order.

    Returns:
        A plain dict with accuracy, macro precision/recall/F1,
        per-class precision/recall/F1/support, and the confusion
        matrix as nested lists (rows are true labels, columns are
        predicted labels, in ``label_names`` order).
    """
    predicted = pipeline.predict(texts)
    report = classification_report(
        labels,
        predicted,
        labels=list(label_names),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predicted, labels=list(label_names))
    return {
        "accuracy": round(float(accuracy_score(labels, predicted)), 4),
        "macro_precision": round(float(report["macro avg"]["precision"]), 4),
        "macro_recall": round(float(report["macro avg"]["recall"]), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "per_class": {
            name: {
                "precision": round(float(report[name]["precision"]), 4),
                "recall": round(float(report[name]["recall"]), 4),
                "f1": round(float(report[name]["f1-score"]), 4),
                "support": int(report[name]["support"]),
            }
            for name in label_names
        },
        "confusion_matrix": [[int(v) for v in row] for row in matrix],
        "labels": list(label_names),
    }


def evaluate_models(
    category_path: str | Path = DEFAULT_CATEGORY_MODEL,
    priority_path: str | Path = DEFAULT_PRIORITY_MODEL,
    data_path: str | Path = DEFAULT_PATH,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> dict:
    """Evaluate both saved pipelines on the held-out test set.

    The test set is rebuilt with the same split training used, so
    the models are scored on exactly the tickets they never saw.

    Returns:
        A plain dict with the split details and one evaluation
        result each for category and priority.
    """
    tickets = load_tickets(data_path)
    _train, test = split_tickets(tickets, test_size=test_size, seed=seed)

    category_model = joblib.load(category_path)
    priority_model = joblib.load(priority_path)

    return {
        "n_train": int(len(tickets) - len(test)),
        "n_test": int(len(test)),
        "test_size": test_size,
        "seed": seed,
        "test_category_counts": test["category"].value_counts().to_dict(),
        "test_priority_counts": test["priority"].value_counts().to_dict(),
        "category": evaluate_pipeline(
            category_model, test["text"], test["category"], CATEGORIES
        ),
        "priority": evaluate_pipeline(
            priority_model, test["text"], test["priority"], PRIORITIES
        ),
    }


def _per_class_table(result: dict) -> str:
    """Render one per-class metrics table in Markdown."""
    lines = ["| Label | Precision | Recall | F1 | Support |"]
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name in result["labels"]:
        row = result["per_class"][name]
        lines.append(
            f"| {name} | {row['precision']:.4f} | {row['recall']:.4f} "
            f"| {row['f1']:.4f} | {row['support']} |"
        )
    return "\n".join(lines)


def _confusion_table(result: dict) -> str:
    """Render one confusion matrix in Markdown (rows = true, columns = predicted)."""
    labels = result["labels"]
    header = "| True \\ Predicted | " + " | ".join(labels) + " |"
    divider = "| --- | " + " | ".join("---:" for _ in labels) + " |"
    rows = [header, divider]
    for name, row in zip(labels, result["confusion_matrix"]):
        rows.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def write_report(results: dict, path: str | Path = DEFAULT_REPORT_PATH) -> Path:
    """Write the evaluation results as a readable Markdown report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def section(title: str, result: dict) -> str:
        return (
            f"## {title}\n\n"
            f"- Accuracy: **{result['accuracy']:.4f}**\n"
            f"- Macro precision: {result['macro_precision']:.4f}\n"
            f"- Macro recall: {result['macro_recall']:.4f}\n"
            f"- Macro F1: {result['macro_f1']:.4f}\n\n"
            f"### Per-class results\n\n"
            f"{_per_class_table(result)}\n\n"
            f"### Confusion matrix (rows are the true label, columns are the predicted label)\n\n"
            f"{_confusion_table(result)}\n"
        )

    text = (
        "# TicketSense baseline evaluation\n\n"
        "Baseline: TF-IDF (word pairs, cleaned text) + Logistic Regression, "
        "one pipeline for ticket category and one for priority. Both were "
        "trained on the training split only and scored on the held-out "
        "test tickets below.\n\n"
        "## Split\n\n"
        f"- Total tickets: {results['n_train'] + results['n_test']}\n"
        f"- Train: {results['n_train']} tickets\n"
        f"- Test: {results['n_test']} tickets ({results['test_size']:.0%} held out, "
        f"stratified by category, random seed {results['seed']})\n"
        f"- Test category counts: {results['test_category_counts']}\n"
        f"- Test priority counts: {results['test_priority_counts']}\n"
        "- The vectoriser was fitted on the training split only, so no "
        "test text entered the vocabulary.\n\n"
        f"{section('Category', results['category'])}\n"
        f"{section('Priority', results['priority'])}\n"
        "## Limitations — read before trusting these numbers\n\n"
        "1. **Only 40 test tickets.** Every single ticket is 2.5% of "
        "accuracy, so the numbers above move a lot with a handful of "
        "tickets. They show the pipeline works end to end; they are "
        "not a performance claim.\n"
        "2. **Hand-written, clean demo data.** Real tickets have "
        "typos, mixed languages (including Roman Urdu/Urdu), slang, "
        "order numbers and personal details. None of that is in this "
        "set, so real-world accuracy will be lower.\n"
        "3. **Categories are perfectly balanced by construction.** "
        "Real support queues are not, so the per-class scores here "
        "flatter the model.\n"
        "4. **Priority labels are one person's judgement**, not a "
        "support team's agreed policy. Two people would disagree on "
        "several rows, which caps how well any model can score on "
        "priority.\n"
        "5. **A larger, properly licensed dataset must replace this "
        "one** before any performance claim is made.\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    """Evaluate both saved pipelines from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained TicketSense model."
    )
    parser.add_argument(
        "--category-model",
        default=str(DEFAULT_CATEGORY_MODEL),
        help="Saved category pipeline to evaluate.",
    )
    parser.add_argument(
        "--priority-model",
        default=str(DEFAULT_PRIORITY_MODEL),
        help="Saved priority pipeline to evaluate.",
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
        "--test-size", type=float, default=TEST_SIZE, help="Share held out for testing."
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help="Random seed for the split."
    )
    args = parser.parse_args()

    results = evaluate_models(
        category_path=args.category_model,
        priority_path=args.priority_model,
        data_path=args.data,
        test_size=args.test_size,
        seed=args.seed,
    )
    report_path = write_report(results, args.report)
    print(
        f"Test set: {results['n_test']} tickets "
        f"(train {results['n_train']}, seed {results['seed']})"
    )
    print(f"Category accuracy: {results['category']['accuracy']:.4f}")
    print(f"Priority accuracy: {results['priority']['accuracy']:.4f}")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
