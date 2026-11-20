# TicketSense

Classify support tickets and predict priority — a first-pass
suggestion for a small support team, not an automatic decision.

**Status: work in progress.** Built as part of a 30-day portfolio
challenge, Nov–Dec 2026. Repo skeleton, target labels, a synthetic
demo dataset, a tested data-loading module, and an exploratory
analysis with a report (`reports/eda.md`) and balance charts exist.
A first baseline model is trained: TF-IDF + Logistic Regression,
one pipeline for category and one for priority, evaluated on a
held-out test set — see `reports/evaluation.md` for the numbers.
The saved pipelines live in `models/` (local only, git-ignored).
The baseline's errors are analysed in `reports/error_analysis.md`
(which tickets it gets wrong, the confusion pairs and the weak
spots), with confusion-matrix charts in `reports/`, and a second
baseline (TF-IDF + Naive Bayes, same split) is compared there —
Logistic Regression stays the baseline. Prediction now works:
`PYTHONPATH=src python -m ticketsense.predict "ticket text"`
prints the predicted category and priority with a confidence
score for each, plus a plain-English low-confidence warning
when either score is below its threshold (category 0.30,
priority 0.45 — measured on the held-out test set, because
this model's probabilities are compressed and a single 0.6
threshold would flag every prediction). Sample predictions
with their confidences are in `reports/prediction_examples.md`. A small
demo web page serves the same predictions — each confidence as
a percentage with a visual bar and the threshold it is judged
against in plain words, plus the low-confidence warning — and
the result page offers a "Try another ticket" link and the six
sample tickets to try another in one click; see the demo section
below. The project is also packaged for install: `pip install -e .`
installs it as `ticketsense` with the `ticketsense-train`,
`ticketsense-predict` and `ticketsense-demo` commands.

## Problem

Small support teams receive many text tickets and need a first-pass
suggestion for (a) what the ticket is about and (b) how urgent it
is, with a confidence score and a clear low-confidence warning.

## Target labels

### Category — `category`

| Label | Meaning |
|---|---|
| `billing` | Payments, invoices, refunds, wrong charges, subscriptions |
| `technical` | Crashes, errors, slowness, broken features, login loops |
| `account` | Password, email/phone change, lockout, deletion, security |
| `delivery` | Shipping status, address, courier, damaged/lost parcels |
| `other` | Thanks, suggestions, general questions, feedback |

### Priority — `priority`

| Label | Meaning |
|---|---|
| `high` | Money lost / charged wrongly, service unusable, security risk, parcel marked delivered but never received |
| `medium` | A real problem, but a workaround exists or it can wait a day |
| `low` | A question, a small request, or a nice-to-have |

The same definitions live in `src/ticketsense/data.py` (module
docstring) and are enforced by `load_tickets()` — unknown labels
are rejected, not silently accepted.

## Stack

Python 3.12, pandas, NumPy, scikit-learn (baseline: TF-IDF +
Logistic Regression), joblib, pytest. Demo interface: FastAPI
with uvicorn, tested with the FastAPI test client.

## Project layout

```text
ticketsense-ml/
├── data/
│   ├── demo_tickets.csv   # 200 hand-written synthetic demo tickets
│   └── README.md          # schema, labels, licensing, limitations
├── notebooks/             # exploration
├── src/ticketsense/
│   ├── data.py            # load + validate + summarise (tested)
│   ├── features.py        # text cleaning; TF-IDF to follow
│   ├── explore.py         # exploratory analysis + report (tested)
│   ├── train.py           # stratified split + TF-IDF + Logistic Regression baseline (tested)
│   ├── evaluate.py        # held-out evaluation + report (tested)
│   ├── errors.py          # error analysis + Naive Bayes comparison + report (tested)
│   ├── predict.py         # prediction with confidence score + low-confidence warning (tested)
│   ├── samples.py         # sample ticket texts for the demo page (one per category + a vague one)
│   └── app.py             # FastAPI demo web page + JSON endpoint (tested)
├── models/                # saved pipeline — binaries are git-ignored
├── reports/               # exploration report (eda.md) + balance charts
│                          # + evaluation (evaluation.md) + error analysis
│                          # (error_analysis.md) + confusion charts
│                          # + sample predictions (prediction_examples.md)
├── tests/                 # pytest tests
├── pyproject.toml         # package metadata, dependencies, console commands
├── requirements.txt       # pinned dev install (same versions as pyproject.toml, plus pytest)
├── .gitignore
├── README.md
└── PROGRESS.md
```

## Install and run

```bash
cd ticketsense-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Or install the package itself (editable, from this folder) —
# this is the same runtime set, pinned to the same versions, and
# it adds the ticketsense-train, ticketsense-predict and
# ticketsense-demo commands. requirements.txt stays as the
# pinned dev install path: it lists the same versions plus
# pytest, which the package itself does not need at runtime.
pip install -e .

# Load the demo data and print class balance
PYTHONPATH=src python -m ticketsense.data

# Run the exploratory analysis and write reports/eda.md
PYTHONPATH=src python -m ticketsense.explore

# Train the baseline (stratified 160/40 split, seed 42) and save
# the two pipelines under models/ (local only, git-ignored)
PYTHONPATH=src python -m ticketsense.train

# Evaluate both pipelines on the held-out test set and write
# reports/evaluation.md
PYTHONPATH=src python -m ticketsense.evaluate

# Analyse the baseline's errors and compare it with a Naive Bayes
# second baseline, and write reports/error_analysis.md plus the
# confusion-matrix charts in reports/ (charts are local only)
PYTHONPATH=src python -m ticketsense.errors

# Predict one ticket: category and priority with a confidence
# score for each, and a low-confidence warning when either score
# is below its threshold (category 0.30, priority 0.45)
PYTHONPATH=src python -m ticketsense.predict "my card was charged twice, please refund me"

# Start the demo web page (FastAPI + uvicorn) on
# http://127.0.0.1:8000 — a form with one text box and a Predict
# button, showing the category and priority with their confidence
# scores as percentages with a visual bar and the threshold each
# score is judged against in plain words, the low-confidence
# warning when it applies, and the submitted text back. The
# result page also offers a "Try another ticket" link back to
# the form and the six sample tickets as clickable buttons.
# Six sample tickets (one per category plus a vague one, from
# `src/ticketsense/samples.py`) are shown as clickable buttons
# above the form so the demo can be tried without typing.
# Empty text shows a clear error on the page instead of a crash,
# and missing model files show a message telling you to train
# the baseline first.
PYTHONPATH=src python -m ticketsense.app

# The same prediction as JSON, without the page:
# curl "http://127.0.0.1:8000/api/predict?text=my%20card%20was%20charged%20twice"
# The JSON also carries the thresholds each confidence is judged
# against (category 0.30, priority 0.45) and a plain-English
# confidence note for each task. Empty text answers 400 and
# missing model files answer 503.

# After `pip install -e .`, the same three commands are
# available without PYTHONPATH=src, from any folder:
#   ticketsense-train
#   ticketsense-predict "my card was charged twice, please refund me"
#   ticketsense-demo
# They call the same module entry points above — train, predict
# and the demo app — so the behaviour is identical. The data
# file and the saved models are still read from this project
# folder, wherever the command is run from.

# Run the tests
PYTHONPATH=src python -m pytest -q
```

Expected output: 200 rows, 0 missing values, 0 duplicates,
category 40 each, priority high 48 / medium 75 / low 77. Baseline
results on the 40 held-out tickets: category accuracy 0.85,
priority accuracy 0.525 (see `reports/evaluation.md` — small test
set, so treat these as pipeline smoke-test numbers, not a
performance claim).

## Dataset and licensing

This project currently uses **only** the hand-written synthetic set in
`data/demo_tickets.csv` — 200 example tickets, no real customer
data, no external download. See `data/README.md` for the schema,
the public datasets considered and rejected so far (Kaggle
re-uploads with unclear licences, Twitter support data with
noisy/PII risk, GitHub issues as the wrong text type), and the
limitations. No dataset may be added later without recording its
exact source, licence and permitted use in `data/README.md` first.

## Limitations

- **The baseline is trained on 200 synthetic rows only.**
  Category accuracy 0.85 and priority accuracy 0.525 on 40
  held-out tickets show the pipeline works end to end; they are
  not a performance claim. Priority is the weak half — most
  high-priority tickets are missed (see `reports/evaluation.md`).
  The error analysis (`reports/error_analysis.md`) shows why: the
  model found only 1 of 14 high-priority test tickets, calling 10
  medium and 3 low, including tickets a human would clearly call
  urgent — so it systematically under-predicts high, and the
  priority labels themselves have no written rules for where
  medium ends and high begins. A Naive Bayes second baseline on
  the same split scored lower on both tasks (category 0.825,
  priority 0.500), so Logistic Regression stays the baseline.
  A larger, properly licensed dataset must replace the demo set
  before any real claim is made.
- **200 synthetic rows** cannot support a real performance claim.
  Any model trained on them is a pipeline smoke test only.
- Priority labels are one person's judgement, not a support
  team's agreed policy. Real tickets have typos, Roman Urdu/Urdu,
  order numbers and PII — none of that is in the demo set.
- Predictions, when they exist, will be suggestions, not
  automatic decisions. Low-confidence cases must be flagged
  for a human.
- **Confidence scores are compressed and must be read as a
  ranking, not as a probability of being right.** On the
  held-out test tickets the clearest predictions still score
  under 0.5 (category confidence only ranges 0.22–0.47), so
  "0.45" does not mean "45% likely right" — it means surer
  than a ticket scoring 0.27. The low-confidence thresholds
  (category 0.30, priority 0.45) were measured from that test
  set: predictions above them were right far more often than
  those below them. The warning fired on 4 of 5 sample
  predictions in `reports/prediction_examples.md`, every time
  on the priority half — which matches the weak priority
  accuracy above. The scores need recalibration on a larger
  dataset before they can be quoted as probabilities.
