# TicketSense — Progress

## Day 1 — 2026-11-12 (TicketSense only; ClientFlow and RepoDoctor not touched)

### What was done
- Created the repo structure from the plan: `data/`, `notebooks/`,
  `src/ticketsense/` (`data.py`, `features.py`, `train.py`,
  `evaluate.py`, `predict.py`, `app.py`), `models/`, `reports/`,
  `tests/`, `requirements.txt`, `.gitignore`.
- Defined target labels in code (`src/ticketsense/data.py` docstring
  and `CATEGORIES`/`PRIORITIES` in `__init__.py`) and in docs:
  5 categories (billing, technical, account, delivery, other) and
  3 priorities (low, medium, high) with plain-English meanings.
- Wrote a real data loader: `load_tickets()` validates required
  columns, duplicate ticket ids, empty text, missing labels and
  unknown category/priority values; `summarise()` /
  `print_summary()` report missing values, duplicates and class
  balance.
- Created a small synthetic demo dataset, `data/demo_tickets.csv`:
  50 hand-written tickets, 10 per category, priority high 12 /
  medium 19 / low 19, 0 missing values, 0 duplicate rows/texts.
  Documented in `data/README.md` that it is synthetic, its schema
  and limitations, and the public datasets considered and why none
  was downloaded on Day 1 (unclear Kaggle licences, noisy/PII
  risk in Twitter support data, GitHub issues are the wrong text
  type, real customer data is private).
- `train.py` / `evaluate.py` / `app.py` are honest skeletons:
  they raise `NotImplementedError` with the scheduled day instead
  of pretending to train/evaluate/serve. `predict.py` validates
  input but returns no category/priority until a model exists.
- Added 12 pytest tests for loading/validation/summary and for
  text cleaning/prediction validation.

### Tests / checks run (Day 1, local venv `.venv`, Python 3.12.3)
- `PYTHONPATH=src .venv/bin/python -m pytest -q`
  → **12 passed in 0.59s** (exit 0)
- `PYTHONPATH=src .venv/bin/python -m ticketsense.data`
  → Rows: 50; Missing values: all 0; Duplicate rows: 0;
  Duplicate texts: 0; Category balance: billing 10, technical 10,
  account 10, delivery 10, other 10; Priority balance:
  medium 19, low 19, high 12
- `git status` / `git diff --check` reviewed before committing;
  no secrets, no large files, `.venv/` ignored.
- Installed (pinned): pandas 2.2.3, numpy 1.26.4,
  scikit-learn 1.5.2, joblib 1.4.2, pytest 8.3.3.

### Commits (local only — nothing pushed)
- See `git log --oneline` in this repo. 2 real commits on Day 1.
- **Not pushed:** GitHub is not connected yet, and this repo has
  no remote. Local `git init` + commits only, as instructed.

### Next steps — Day 2 (TicketSense)
- Pick the real dataset (or deliberately build a larger synthetic
  set): record exact source, licence and permitted use in
  `data/README.md` *before* downloading anything.
- Write the data documentation / schema module work from the
  plan (Day 2: dataset source/schema documentation; data loading
  module is already done, so use Day 2 to validate the chosen
  dataset against the loader and record its balance/missing/
  duplicate report in `reports/`).
- Then Day 3-4: validation and exploratory analysis; Day 5-6:
  train/test split and TF-IDF + Logistic Regression baseline.

## Day 2 — 2026-11-13 (TicketSense)

### What was done
- Dataset decision: enlarge the synthetic set instead of downloading
  a public set. No public support-ticket dataset was cleanly usable
  (Kaggle re-uploads with unclear licences, Twitter/X data noisy
  with PII risk, GitHub issues the wrong text type, real customer
  data private). The enlarged set stays hand-written synthetic
  data, so there is no licence restriction and no real customer
  data anywhere in the project.
- `data/demo_tickets.csv` grew from 50 to 200 hand-written tickets:
  150 new tickets (TS-051–TS-200), 40 per category, priorities
  high 48 / medium 75 / low 77.
- `data/README.md` records the decision, the source and licence
  ("hand-written for this project, no licence restriction"), the
  new schema (TS-001…TS-200), the new balance, and the updated
  limitations (still small at 200 rows; still clean hand-written
  text; still perfectly balanced categories; priority labels are
  judgement, not team policy).
- `README.md` updated with the new totals.
- Tests in `tests/test_data.py` updated for the 200-row set.

### Tests / checks run (Day 2, local venv `.venv`, Python 3.12.3)
- `PYTHONPATH=src .venv/bin/python -m pytest -q`
  → **12 passed**
- `PYTHONPATH=src .venv/bin/python -m ticketsense.data`
  → Rows: 200; Missing values: all 0; Duplicate rows: 0;
  Duplicate texts: 0; Category balance: 40 each;
  Priority balance: low 77, medium 75, high 48
- `git diff --check` — clean; no secrets in the diff.

### Commits (local only — nothing pushed yet)
- See `git log` in this repo. 2 real commits on Day 2.

### Next steps — Day 3 (TicketSense)
- Data validation, missing values, duplicates, label distribution
  — largely done by the loader; next is exploratory analysis and
  initial visualisations.
- Then: train/test split and the TF-IDF + Logistic Regression
  baseline.
