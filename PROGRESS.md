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
