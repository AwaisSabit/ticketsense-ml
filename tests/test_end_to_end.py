"""End-to-end tests for TicketSense: the full user flow.

These tests run the real command-line entry points, not the
library functions directly: train the baseline, evaluate it,
predict one ticket, and then drive the demo web page and its
JSON endpoint. Training uses the fixed seed and data, so the
saved pipelines are the same baseline the reports describe —
retraining here reproduces them byte-for-byte.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ticketsense.app import app

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFUND_TICKET = "my card was charged twice, please refund me"


def _run_cli(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a TicketSense module as a user would, from a shell."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def trained_models(tmp_path_factory):
    """Train and evaluate once, through the CLI, for the module.

    The models are saved to the usual ``models/`` files, so the
    demo leg below exercises the same files a user gets. The
    evaluation report goes to a temporary file so the committed
    report is not rewritten.
    """
    train = _run_cli("ticketsense.train")
    assert train.returncode == 0, train.stderr
    assert "Train: 160 tickets; test: 40 tickets" in train.stdout
    assert "Category model saved to" in train.stdout

    report = tmp_path_factory.mktemp("e2e") / "evaluation.md"
    evaluate = _run_cli(
        "ticketsense.evaluate", "--report", str(report)
    )
    assert evaluate.returncode == 0, evaluate.stderr
    assert "Category accuracy: 0.8500" in evaluate.stdout
    assert "Priority accuracy: 0.5250" in evaluate.stdout
    assert "0.8500" in report.read_text(encoding="utf-8")
    return {
        "category": PROJECT_ROOT / "models" / "category_pipeline.joblib",
        "priority": PROJECT_ROOT / "models" / "priority_pipeline.joblib",
    }


def test_train_cli_saves_both_pipelines(trained_models) -> None:
    assert trained_models["category"].is_file()
    assert trained_models["priority"].is_file()
    assert trained_models["category"].stat().st_size > 0
    assert trained_models["priority"].stat().st_size > 0


def test_predict_cli_answers_a_refund_ticket(trained_models) -> None:
    result = _run_cli("ticketsense.predict", REFUND_TICKET)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(
        "This looks like a billing ticket, medium priority."
    )
    assert "Category: billing (confidence 0.44)" in result.stdout
    assert "Priority: medium (confidence 0.42)" in result.stdout
    assert "Warning:" in result.stdout


def test_predict_cli_rejects_empty_text(trained_models) -> None:
    result = _run_cli("ticketsense.predict", "   ")
    assert result.returncode == 1
    assert "must not be empty" in result.stderr


def test_demo_end_to_end(trained_models) -> None:
    client = TestClient(app)
    home = client.get("/")
    assert home.status_code == 200
    assert "Predict" in home.text

    page = client.post("/predict", data={"text": REFUND_TICKET})
    assert page.status_code == 200
    assert "This looks like a billing ticket, medium priority." in page.text
    assert "billing" in page.text
    assert "Try another ticket" in page.text

    api = client.get("/api/predict", params={"text": REFUND_TICKET})
    assert api.status_code == 200
    data = api.json()
    assert data["category"] == "billing"
    assert data["priority"] == "medium"
    assert data["category_threshold"] == 0.30
    assert data["priority_threshold"] == 0.45
    assert data["low_confidence"] is True
    assert data["warning"]


def test_demo_and_cli_agree_on_the_same_ticket(trained_models) -> None:
    cli = _run_cli("ticketsense.predict", REFUND_TICKET)
    assert cli.returncode == 0, cli.stderr
    api = TestClient(app).get("/api/predict", params={"text": REFUND_TICKET})
    assert api.status_code == 200
    assert api.json()["category"] in cli.stdout
    assert api.json()["priority"] in cli.stdout
