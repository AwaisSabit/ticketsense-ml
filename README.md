# TicketSense

Classify support tickets and predict priority — a first-pass
suggestion for a small support team, not an automatic decision.

**Status: work in progress.** Built as part of a 30-day portfolio
challenge, Nov–Dec 2026. Repo skeleton, target labels, a synthetic
demo dataset, a tested data-loading module, and an exploratory
analysis with a report (`reports/eda.md`) and balance charts exist.
No model is trained yet.

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
Logistic Regression), joblib, pytest. Demo interface:
Streamlit *or* FastAPI — to be chosen after a model exists.

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
│   ├── train.py           # skeleton; baseline to follow
│   ├── evaluate.py        # skeleton; evaluation to follow
│   ├── predict.py         # input validation; model to follow
│   └── app.py             # skeleton; demo interface to follow
├── models/                # saved pipeline — binaries are git-ignored
├── reports/               # exploration report (eda.md) + balance charts
├── tests/                 # pytest tests
├── requirements.txt
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

# Load the demo data and print class balance
PYTHONPATH=src python -m ticketsense.data

# Run the exploratory analysis and write reports/eda.md
PYTHONPATH=src python -m ticketsense.explore

# Run the tests
PYTHONPATH=src python -m pytest -q
```

Expected output: 200 rows, 0 missing values, 0 duplicates,
category 40 each, priority high 48 / medium 75 / low 77.

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

- **No model yet.** Nothing in this repo predicts anything yet;
  `predict.py` deliberately returns no category/priority.
- **200 synthetic rows** cannot support a real performance claim.
  Any model trained on them is a pipeline smoke test only.
- Priority labels are one person's judgement, not a support
  team's agreed policy. Real tickets have typos, Roman Urdu/Urdu,
  order numbers and PII — none of that is in the demo set.
- Predictions, when they exist, will be suggestions, not
  automatic decisions. Low-confidence cases must be flagged
  for a human.
