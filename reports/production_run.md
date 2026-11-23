# Production-like run — TicketSense

A full run of the real commands from a normal shell, as a user
would run them after installing the project, done before the
1.0.0 release. No test harness, no mocks: the actual
`train`, `evaluate`, `predict` and demo commands against the
real demo data and the real saved models.

## Commands and results

All commands run from the project folder with the project
virtualenv (`.venv`, Python 3.12.3):

```
PYTHONPATH=src .venv/bin/python -m ticketsense.train
```

Train: 160 tickets; test: 40 tickets (8 per category).
Both pipelines saved to `models/`. Took 6.2 seconds.

```
PYTHONPATH=src .venv/bin/python -m ticketsense.evaluate
```

Test set: 40 tickets (train 160, seed 42).
Category accuracy: 0.8500. Priority accuracy: 0.5250.
These are exactly the numbers in `reports/evaluation.md` —
the regenerated report is unchanged.

```
PYTHONPATH=src .venv/bin/python -m ticketsense.predict \
  "my card was charged twice, please refund me"
```

"This looks like a billing ticket, medium priority."
Category billing (0.44), priority medium (0.42), with the
low-confidence warning on priority. Two more tickets checked
the same way: an app crash reads technical / medium, and a
thank-you note reads other / low — all three sensible.

```
PYTHONPATH=src .venv/bin/python -m ticketsense.app
```

Demo server on http://127.0.0.1:8000, checked with curl:

- `GET /` → 200, the form page with the Predict button.
- `GET /api/predict?text=my%20card%20was%20charged%20twice`
  → billing 0.38 / medium 0.40, with the priority warning
  and both threshold fields.
- `GET /api/predict?text=` → 400, empty text rejected.
- `POST /predict` with a thank-you ticket → 200.

The server was stopped after the checks.

## What was verified

- Retraining from scratch reproduces the saved pipelines
  byte-for-byte (sha256 of both `models/*.joblib` files
  unchanged before and after the run), so the release models
  are exactly the evaluated baseline.
- Evaluation, prediction and the demo all agree on the same
  numbers: category 0.85, priority 0.525.
- The demo answers real HTTP requests with the right status
  codes — 200 for a prediction, 400 for empty text — and the
  JSON carries the same labels, confidences and warnings as
  the command line.
