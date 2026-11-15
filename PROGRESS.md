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

## Day 3 — 2026-11-14 (TicketSense)

### What was done
- New module `src/ticketsense/explore.py` with pure analysis
  functions that take a loaded dataframe and return plain dicts
  (no printing inside them): text-length stats per category and
  overall (word and character count: mean, median, min, max),
  most common words per category (lower-cased, cleaned with the
  existing `clean_text` helper, punctuation stripped so "update,"
  counts as "update", and a small built-in English stopword list
  — no new dependency), and a category-by-priority cross-tab.
- `write_report()` writes a readable Markdown report to
  `reports/eda.md` (row count, missing values, duplicates, label
  distribution, length stats, top words per category, cross-tab,
  and a plain-English "what this means" section stating the set
  is small, clean, hand-written and perfectly balanced, so no
  performance claim can be made from it). Runnable entry point:
  `PYTHONPATH=src python -m ticketsense.explore`.
- Initial visualisations: matplotlib installed cleanly in the
  project `.venv` (3.9.2, pinned in `requirements.txt` with its
  dependencies; numpy kept at the pinned 1.26.4 and contourpy
  pinned to 1.2.1, the version compatible with it — `pip check`
  is clean). Two bar charts generated into `reports/`:
  `category_balance.png` (40 each) and `priority_balance.png`
  (low 77, medium 75, high 48), both about 14 KB, and referenced
  from `reports/eda.md`. `.gitignore` now allows exactly those
  two PNGs (the general `reports/*.png` rule still stands).
- Key numbers from the exploration (full detail in `reports/eda.md`):
  200 rows, 0 missing values, 0 duplicate rows/texts. Ticket
  length overall: mean 15.7 words (median 16, min 10, max 21),
  mean 81.0 characters (median 81, min 57, max 114). Longest
  category on average is billing (16.45 words); shortest is
  technical (15.0 words). Top words: billing — order (11),
  payment (9); technical — app (12), website (8); account —
  account (26), email (13); delivery — order (16), delivery (13),
  parcel (13); other — thank (5), watch (5). Cross-tab: high
  priority is spread evenly (10–11 per category) except `other`
  (6 high, 21 low) — the polite/thanks tickets sit in `other`.
- Fixed the stale README line that still said "50 synthetic
  rows" — it now says 200. README also mentions the new
  exploration report and charts and the `ticketsense.explore`
  command. No day numbers in any README.

### Tests / checks run (Day 3, local venv `.venv`, Python 3.12.3)
- `PYTHONPATH=src .venv/bin/python -m pytest -q`
  → **22 passed** (12 existing + 10 new in `tests/test_explore.py`:
  length stats correct on a hand-made frame, stopwords excluded,
  punctuation stripped from top words, cross-tab adds up to the
  row count, report written with all key sections, charts written
  as small PNGs, empty dataframe and missing column raise clear
  errors)
- `PYTHONPATH=src .venv/bin/python -m ticketsense.explore`
  → Rows: 200; category 40 each; priority low 77, medium 75,
  high 48; both charts and `reports/eda.md` written
- `git diff --check` — clean; no secrets in the diff; `.venv/`,
  `__pycache__/`, models binaries ignored as before.

### Commits (local only — nothing pushed)
- See `git log --oneline` in this repo. 2 real commits on Day 3.
- **Not pushed:** local commits only, as instructed; the parent
  run does the push.

### Next steps — Day 4 (TicketSense)
- Train/test split and the TF-IDF + Logistic Regression baseline.

## Day 4 — 2026-11-14 (TicketSense)

### What was done
- Real training in `src/ticketsense/train.py` (was an honest
  skeleton): `split_tickets()` makes a stratified split of the 200
  tickets — 160 train / 40 test, 20% held out, fixed seed 42, 8
  test tickets per category — and fails loudly if any ticket id or
  text appears on both sides. `build_pipeline()` is TF-IDF (word
  pairs, shared `clean_text` preprocessor) + Logistic Regression.
  `train_baseline()` trains two pipelines on the same training
  texts — one for category, one for priority — and saves them
  with joblib to `models/category_pipeline.joblib` and
  `models/priority_pipeline.joblib` (already git-ignored, so the
  binaries stay local; only the text report is committed). The
  vectoriser is fitted on the training split only, so no test text
  enters the vocabulary. Runnable: `PYTHONPATH=src python -m
  ticketsense.train`.
- Real evaluation in `src/ticketsense/evaluate.py` (was an honest
  skeleton): `evaluate_models()` replays the exact same split
  (seed 42, 20%) and scores both saved pipelines on the held-out
  test tickets — accuracy, macro precision/recall/F1, per-class
  precision/recall/F1/support, and a confusion matrix for category
  and for priority. `write_report()` writes the results to
  `reports/evaluation.md` with the split details and honest
  limitations. Runnable: `PYTHONPATH=src python -m
  ticketsense.evaluate`.
- Results on the 40 held-out tickets (full detail in
  `reports/evaluation.md`): **category accuracy 0.85**, macro F1
  0.8449 (delivery and `other` perfect, billing recall 0.625 the
  weakest); **priority accuracy 0.525**, macro F1 0.4535 — the weak
  half of the baseline. High-priority recall is only 0.0714: of 14
  high test tickets, 10 were called medium and 3 low. Priority
  labels are one person's judgement, so this is partly a labelling
  problem, not just a model problem — recorded as a limitation,
  not hidden.
- README updated (no day numbers): status now says the baseline
  exists, the train/evaluate commands are documented with the
  headline numbers, and the limitations say what the numbers do
  and do not mean.

### Tests / checks run (Day 4, local venv `.venv`, Python 3.12.3)
- `PYTHONPATH=src .venv/bin/python -m pytest -q`
  → **36 passed** (22 existing + 14 new: 8 in
  `tests/test_train.py` — split sizes, stratification, no
  overlap, same-seed reproducibility, different-seed difference,
  empty and too-small inputs raise, pipeline fits and predicts
  valid labels, `train_baseline` writes both loadable pipelines;
  6 in `tests/test_evaluate.py` — metrics in range, confusion
  matrix sums to 40, exact baseline numbers for the pinned demo
  set and seed, report covers split/metrics/limits, missing model
  file raises)
- `PYTHONPATH=src .venv/bin/python -m ticketsense.train`
  → Train: 160 tickets; test: 40 tickets; both pipelines saved
- `PYTHONPATH=src .venv/bin/python -m ticketsense.evaluate`
  → Category accuracy: 0.8500; Priority accuracy: 0.5250;
  `reports/evaluation.md` written
- `git diff --check` — clean; no secrets in the diff; model
  binaries (`.joblib`) confirmed git-ignored and not staged.

### Commits (local only — nothing pushed)
- See `git log --oneline` in this repo. 3 real commits on Day 4.
- **Not pushed:** local commits only, as instructed; the parent
  run does the push.

### Next steps — Day 5 (TicketSense)
- Error analysis of the baseline: which tickets it gets wrong,
  confusion pairs, and a second-baseline comparison.
- Then: prediction with a confidence score and a low-confidence
  warning (`predict.py`), using the saved pipelines; then the demo
  interface (`app.py`).

## Day 5 — 2026-11-15 (TicketSense)

### What was done
- New module `src/ticketsense/errors.py` with pure analysis
  functions that take the held-out test tickets and the two
  trained pipelines: `misclassified()` lists every wrong ticket
  for category and for priority (ticket id, text, true label,
  predicted label), `confusion_pairs()` counts the off-diagonal
  pairs most common first, and `weak_spots()` flags classes with
  recall below 0.5. The same split is reused (stratified, seed
  42, 160 train / 40 test) and the existing train/evaluate
  helpers are reused — nothing is retrained differently.
- `write_report()` writes `reports/error_analysis.md` (via a
  `PYTHONPATH=src python -m ticketsense.errors` entry point)
  covering the misclassified tickets with their texts, the
  confusion pairs, the per-class weak spots, a plain-English
  reading of what the errors mean, and the second-baseline
  comparison. Two confusion-matrix charts are drawn into
  `reports/` (`category_confusion.png`, 22 KB, and
  `priority_confusion.png`, 16 KB) in the same matplotlib style
  as the balance charts. The charts are local only — the
  snapshot push is text-only, so they are git-ignored and the
  Markdown report is what gets committed.
- Second baseline comparison: TF-IDF + Multinomial Naive Bayes,
  trained on the same 160 training tickets and scored on the
  same 40 test tickets, with identical text cleaning and TF-IDF
  settings so only the classifier differs.
- Key numbers (full detail in `reports/error_analysis.md`):
  category **6 of 40** tickets wrong, priority **19 of 40**
  wrong. Top confusion pair is high priority called medium
  (10 tickets); low called medium is next (5). No category weak
  spot by the recall-below-half rule — billing is the weakest
  category at recall 0.625 — but high priority is the real weak
  spot at recall 0.0714 (1 of 14 found). Naive Bayes scored
  category accuracy 0.825 / macro F1 0.8213 and priority
  accuracy 0.500 / macro F1 0.4040, worse than Logistic
  Regression on both tasks (0.85 / 0.8449 and 0.525 / 0.4535),
  so Logistic Regression stays the baseline.
- Plain-English reading, recorded honestly: category errors are
  mostly near misses between related topics, and a couple of
  the "errors" are arguably debatable labels. Priority errors
  are a model problem first — several missed high tickets read
  clearly urgent to a human (a stranger placing orders on your
  account, a card charged three times, a product that cut
  someone's hand) — and a labelling problem second, because
  priority has no written rules for where medium ends and high
  begins. The fix is both: agreed labelling rules written down,
  and a model that is not afraid to predict high.
- README updated (no day numbers): status, project layout,
  run commands and limitations now cover the error analysis,
  the charts and the Naive Bayes comparison.

### Tests / checks run (Day 5, local venv `.venv`, Python 3.12.3)
- `PYTHONPATH=src .venv/bin/python -m pytest -q`
  → **48 passed** (36 existing + 12 new in
  `tests/test_errors.py`: misclassified list correct on a
  hand-made frame, confusion pairs counted correctly with the
  diagonal excluded, perfect predictions yield no errors, weak
  spots flagged only below recall 0.5, exact error counts and
  confusion pairs for the pinned demo set and seed, Naive Bayes
  comparison runs with metrics in range and the real numbers,
  report written with all key sections, both charts written as
  PNGs, missing model files raise a clear error)
- `PYTHONPATH=src .venv/bin/python -m ticketsense.errors`
  → Category misclassified: 6 of 40; Priority misclassified:
  19 of 40; Naive Bayes category 0.8250, priority 0.5000;
  both charts and `reports/error_analysis.md` written
- `git diff --check` — clean; no secrets in the diff; model
  binaries (`.joblib`) confirmed git-ignored and not staged;
  the new confusion PNGs are git-ignored (local only) as
  intended. The DeprecationWarnings in the test run (scipy
  L-BFGS-B options, matplotlib text conversion) come from the
  pinned library versions and already appear in the Day 3 chart
  tests — they are not caused by the new code.

### Commits (local only — nothing pushed)
- See `git log --oneline` in this repo. 3 real commits on Day 5.
- **Not pushed:** local commits only, as instructed; the parent
  run does the push.

### Next steps — Day 6 (TicketSense)
- Prediction with a confidence score and a low-confidence warning
  (`predict.py`), using the saved pipelines; then the demo
  interface (`app.py`).
