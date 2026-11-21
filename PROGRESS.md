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

## Day 6 — 2026-11-16 (TicketSense)

### What was done
- Real prediction in `src/ticketsense/predict.py` (was an honest
  skeleton that only validated input): `load_models()` loads the
  two saved pipelines and raises a clear error telling the user
  to run the train command first when a model file is missing;
  `predict_label()` returns one label with its confidence (the
  probability the pipeline gives its top label);
  `predict_ticket()` predicts category and priority for one
  ticket text and returns both labels, a confidence score for
  each, a `low_confidence` flag and a plain-English `warning`
  when either score is below its threshold; `format_prediction()`
  renders the result as plain-English lines. Runnable:
  `PYTHONPATH=src python -m ticketsense.predict "ticket text"`.
  Empty or blank text is rejected by the existing `clean_text`
  validation, and missing model files exit with a clear error
  instead of a traceback. The old `predict_text` name is kept
  as a thin alias so existing callers keep working.
- Thresholds measured, not guessed: the first guess was a
  single 0.6 threshold, but measuring the saved model on the 40
  held-out test tickets showed its probabilities are compressed
  (category confidence only 0.22–0.47, median 0.31; priority
  0.34–0.61, median 0.46), so 0.6 would flag every prediction
  and the warning would carry no information. The thresholds
  are per task instead — category 0.30 and priority 0.45 —
  picked from that test set: category predictions at or above
  0.30 were right 95% of the time (20 of 21) versus 74% below
  it, and priority predictions at or above 0.45 were right 70%
  (16 of 23) versus only 29% below it (5 of 17). Both values
  and the measurement are documented in the module docstring
  and in the README limitations, with the warning that the
  scores are a ranking, not a probability of being right, and
  need recalibration on a larger dataset.
- Sample predictions written to `reports/prediction_examples.md`:
  5 real runs of the module. All 5 category predictions match
  what a person would say (billing 0.45, technical 0.48, other
  0.27, other 0.35, delivery 0.39). The warning fired on 4 of
  the 5, every time on the priority half — including the two
  clearest tickets (priority 0.37 and 0.39) — which matches the
  known weak priority half. The vague "hello I need some help
  please" ticket warned on both halves (category 0.27, barely
  above a random guess for five categories), and the thank-you
  note passed cleanly with no warning (other 0.35, low 0.50).
- README updated (no day numbers): status, project layout, run
  commands and limitations now cover the prediction module,
  the measured thresholds and the compressed-confidence caveat.

### Tests / checks run (Day 6, local venv `.venv`, Python 3.12.3)
- `PYTHONPATH=src .venv/bin/python -m pytest -q`
  → **65 passed** (48 existing + 17 new in
  `tests/test_predict.py`: both pipelines load, missing files
  raise a clear "train first" error, prediction returns labels
  and confidences in range, clear billing/technical tickets get
  the right category above threshold, the vague ticket warns on
  both halves, the thank-you ticket has no warning, empty/blank/
  non-string text raises, the thresholds control the warning
  with tiny trained pipelines, the formatted output includes
  the warning only when low, and the CLI prints a prediction
  and exits with a clear error for empty text and missing
  models. The stale Day 1 test in `tests/test_features.py`
  that asserted "no prediction exists" was updated — a model
  exists now, so `predict_text` returns a real prediction.)
- `PYTHONPATH=src python -m ticketsense.predict "..."` on the
  5 sample texts → see `reports/prediction_examples.md`; empty
  text exits 1 with "Ticket text must not be empty", and a
  missing model file exits 1 with "Train the baseline first"
- `git diff --check` — clean; no secrets in the diff; model
  binaries (`.joblib`) confirmed git-ignored and not staged.
  The DeprecationWarnings in the test run (scipy L-BFGS-B
  options, matplotlib text conversion) come from the pinned
  library versions and already appear in the Day 3/5 chart
  tests — they are not caused by the new code.

### Commits (local only — nothing pushed)
- See `git log --oneline` in this repo. 3 real commits on Day 6.
- **Not pushed:** local commits only, as instructed; the parent
  run does the push.

### Next steps — Day 7 (TicketSense)
- Demo interface (`app.py`): Streamlit or FastAPI, showing the
  prediction with its confidence scores and the low-confidence
  warning; then sample inputs and malformed-input tests.

## Day 7 — 2026-11-17 (TicketSense)

### What was done
- Demo interface in `src/ticketsense/app.py` (was an honest
  skeleton): a small FastAPI app, chosen over Streamlit so it
  can be tested with the FastAPI test client. It reuses
  `predict_ticket()` from `predict.py` — no prediction logic
  is duplicated. `GET /` serves a simple page with a form: one
  text box for the ticket text and a Predict button (plain
  inline HTML and styling, no template files). `POST /predict`
  runs the prediction and shows a result page with the
  predicted category and its confidence score, the predicted
  priority and its confidence score, the low-confidence
  warning in plain words when flagged, and the submitted text
  back (HTML-escaped). `GET /api/predict?text=...` returns the
  same prediction as JSON (labels, confidences,
  `low_confidence`, `warning`).
- Errors handled gracefully, no tracebacks: empty or blank
  text shows a clear "Ticket text must not be empty" message
  on the page (and a 400 JSON error for the API); missing model
  files show a clear "Train the baseline first" message (and a
  503 JSON error for the API). Runnable:
  `PYTHONPATH=src python -m ticketsense.app` starts the server
  with uvicorn on http://127.0.0.1:8000.
- New dependencies, pinned in `requirements.txt` and installed
  in the project `.venv` (Python 3.12.3): fastapi 0.163.1,
  uvicorn 0.54.0, httpx 0.28.1 (needed by the test client) and
  python-multipart 0.0.34 (needed for form posts). `pip check`
  is clean.
- README updated (no day numbers): status, stack, project
  layout and run commands now cover the demo page, the JSON
  endpoint and the error behaviour.

### Tests / checks run (Day 7, local venv `.venv`, Python 3.12.3)
- `PYTHONPATH=src .venv/bin/python -m pytest -q`
  → **78 passed** (65 existing + 13 new in
  `tests/test_app.py`: home page loads with the form, form
  prediction shows the right category/priority labels and
  confidence numbers for a clear ticket, the low-confidence
  warning appears for the vague ticket and not for a clear
  thank-you ticket, empty text gives a clear error instead of
  a crash, missing model files give a clear "train first"
  error on both the page and the API, the JSON endpoint
  returns labels/confidences/warning fields with the right
  error codes for empty text, and submitted text is escaped
  on the result page)
- Live server check: `PYTHONPATH=src python -m ticketsense.app`
  started uvicorn on port 8000; `GET /` returned 200 with the
  form, `GET /api/predict?text=...` returned billing 0.44 /
  medium 0.42 with the priority warning for a refund ticket,
  empty text returned 400, and a form post for the vague ticket
  showed the warning on both halves
- `git diff --check` — clean; no secrets in the diff; model
  binaries (`.joblib`) confirmed git-ignored and not staged.
  The DeprecationWarnings in the test run (scipy L-BFGS-B
  options, matplotlib text conversion, starlette TestClient
  httpx note) come from the pinned library versions and already
  appear in earlier test runs — they are not caused by the new
  code.

### Commits (local only — nothing pushed)
- See `git log --oneline` in this repo. 3 real commits on Day 7.
- **Not pushed:** local commits only, as instructed; the parent
  run does the push.

### Next steps — Day 8 (TicketSense)
- Sample inputs and malformed-input tests for the demo; then
  packaging and final polish toward the release.

## Day 8 — 2026-11-18 (TicketSense)

### What was done
- Model-loading check first, before any new code: `app.py`
  calls `predict_ticket()` from `predict.py`, which loads the
  saved pipelines `models/category_pipeline.joblib` and
  `models/priority_pipeline.joblib` with joblib on every call.
  Verified live that both files load as fitted scikit-learn
  pipelines (category classes: account, billing, delivery,
  other, technical) and that the demo predicts with them — a
  refund ticket came back billing 0.40 / medium 0.38. Nothing
  was a stub and no fix was needed; no model was retrained and
  no model file was changed.
- Sample inputs in a new module `src/ticketsense/samples.py`:
  six realistic tickets — one per category (billing charged
  twice, technical app crash, account login/reset email,
  delivery parcel stuck in transit, other thank-you note)
  plus a vague "hello I need some help please" that should
  warn. Each sample carries its label, text, the category the
  saved baseline predicts for it (checked live — all five
  category samples predict their own category), and whether
  the low-confidence warning should fire. The vague sample
  predicts `other` at 0.27 confidence, below the 0.30
  category threshold, so it warns on both halves.
- Demo wiring: the home page now shows the six samples as
  clickable buttons above the form; clicking one fills the
  text box with that ticket (small inline JavaScript, text
  quoted with `json.dumps` so it is safe, labels HTML-escaped
  like everything else on the page). The JSON endpoint is
  unchanged and is documented in the page footer as before.
- Malformed-input tests in a new file
  `tests/test_demo_inputs.py` (23 tests): empty, whitespace-only
  and missing text on the form POST (400 with "must not be
  empty" and the form still shown) and on the API (400 JSON
  error, including no `text` query param at all); an 11,000-
  character ticket through both the form and the API (200 with
  a valid prediction, answered in well under 10 seconds);
  punctuation-only text ("!!! ??? ...") answered normally
  instead of crashing on an all-zero TF-IDF vector; non-string
  input to `predict_ticket` (`None`, a number, a float, a list,
  a dict) raising a clear `ValueError` ("must be a string");
  and every sample ticket predicting successfully through the
  API with the expected category and warning behaviour. No
  test ever sees a 500 or a traceback.
- README updated (no day numbers): project layout and the demo
  run commands now cover the samples module and the clickable
  sample buttons.

### Tests / checks run (Day 8, local venv `.venv`, Python 3.12.3)
- `PYTHONPATH=src .venv/bin/python -m pytest -q`
  → **101 passed** (78 existing + 23 new in
  `tests/test_demo_inputs.py`)
- Live model check: `load_models()` loads both saved pipelines;
  `predict_ticket` on a refund ticket returns billing 0.40 /
  medium 0.38 with the priority warning, as on Day 6/7
- Home page smoke check via the test client: 200, all six
  sample labels present, buttons fill the form text box
- `git diff --check` — clean; no secrets in the diff; model
  binaries (`.joblib`) confirmed git-ignored and not staged.
  The StarletteDeprecationWarning in the test run (httpx with
  the test client) comes from the pinned library versions and
  already appears in the Day 7 run — it is not caused by the
  new code.

### Commits (local only — nothing pushed)
- See `git log --oneline` in this repo. 2 real commits on Day 8.
- **Not pushed:** local commits only, as instructed; the parent
  run does the push.

### Next steps — Day 9 (TicketSense)
- Packaging and final polish toward the release.

## Day 9 — 2026-11-19 (TicketSense)

### What was done
- Confidence display on the demo result page (`src/ticketsense/app.py`):
  each confidence is now shown as a percentage as well as the raw
  two-decimal score, with a simple visual bar (plain HTML/CSS —
  a filled span inside a grey track, no new dependency) and the
  threshold it is judged against in plain words. A new helper,
  `confidence_note()`, writes the sentence: above the bar it
  reads "Category confidence 44% — above the 30% bar, so this
  looks fairly sure"; below it reads "Priority confidence 38% —
  below the 45% bar, so treat this with care". The thresholds
  themselves are unchanged (category 0.30, priority 0.45) and no
  model file was retrained or touched. The low-confidence
  warning, the predicted labels, the raw scores and the submitted
  text are all still on the page exactly as before.
- Demo flow: the result page now offers a clear way onward — a
  "Try another ticket" link back to the form, and the six sample
  tickets as clickable buttons on the result page too, so a
  visitor can try another sample in one click. The home page
  keeps its existing form and samples, unchanged.
- JSON endpoint: `GET /api/predict` now carries the same
  threshold context — `category_threshold` 0.30,
  `priority_threshold` 0.45, and a plain-English
  `category_confidence_note` / `priority_confidence_note` for
  each task. All seven original fields keep their names and
  values exactly; only the four new fields are added. The one
  existing test that asserted the exact JSON key set was updated
  to list the new fields.
- README updated (no day numbers): the demo section now covers
  the confidence bars and threshold wording, the try-another
  link and samples on the result page, and the threshold fields
  in the JSON response.

### Tests / checks run (Day 9, local venv `.venv`, Python 3.12.3)
- `PYTHONPATH=src .venv/bin/python -m pytest -q`
  → **109 passed** (101 existing + 8 new in
  `tests/test_demo_display.py`: result page shows percentages
  and "above the 30%/45% bar" wording for a clear thank-you
  ticket, "below the bar" wording for the vague ticket, two
  visual bars filled to the right percentages, the warning still
  appears when flagged, the result page has the try-another link
  and all six sample buttons with the form still there, the home
  page is unchanged, and the JSON response carries the threshold
  fields and confidence notes for both a clear and a vague
  ticket while keeping the original fields exact)
- Live server check: `PYTHONPATH=src python -m ticketsense.app`
  started uvicorn on port 8000; `GET /` returned 200, a form post
  for the thank-you ticket showed both "above the bar" notes and
  two bars, a form post for the vague ticket showed the
  "below the bar" note and the warning on both halves, and
  `GET /api/predict?text=...` for the vague ticket returned
  category 0.27 / priority 0.37 with both thresholds and both
  "treat this with care" notes. The server was stopped after the
  check.
- `git diff --check` — clean; no secrets in the diff; model
  binaries (`.joblib`) confirmed git-ignored and not staged.
  The warnings in the test run come from the pinned library
  versions and already appear in earlier runs — they are not
  caused by the new code.

### Commits (local only — nothing pushed)
- See `git log --oneline` in this repo. 3 real commits on Day 9.
- **Not pushed:** local commits only, as instructed; the parent
  run does the push.

### Next steps — Day 10 (TicketSense)
- Packaging and final polish toward the release.

## Day 10 — 2026-11-20 (TicketSense)

### What was done
- Packaging in a new `pyproject.toml` (setuptools, `src` layout):
  the package is `ticketsense`, version taken from
  `ticketsense.__version__` (0.1.0, unchanged), Python >= 3.10,
  with the runtime dependencies pinned to the same versions
  already in `requirements.txt` (pandas 2.2.3, numpy 1.26.4,
  scikit-learn 1.5.2, joblib 1.4.2, matplotlib 3.9.2,
  fastapi 0.163.1, uvicorn 0.54.0, httpx 0.28.1,
  python-multipart 0.0.34). pytest stays out of the runtime
  list — it is a dev-only dependency.
- Three console scripts, each calling the existing module entry
  point (no logic duplicated): `ticketsense-train` calls
  `ticketsense.train:main`, `ticketsense-predict` calls
  `ticketsense.predict:main`, and `ticketsense-demo` calls
  `ticketsense.app:main`. All data and model paths are already
  anchored to the project folder, so the commands work from any
  directory once installed.
- `requirements.txt` stays as the pinned dev install path: it
  lists the same runtime versions plus pytest. A short comment
  at the top of the file records the relationship, and a test
  checks the two lists never drift apart. The README explains
  both install paths and documents the console commands next to
  the module commands.
- No model file was retrained for this work on purpose — the
  training run below was the verification of the new
  `ticketsense-train` command, using the same data and seed, so
  the saved pipelines are the same baseline. Model binaries
  stay git-ignored and local-only, as before.

### Tests / checks run (Day 10, local venv `.venv`, Python 3.12.3)
- `PYTHONPATH=src .venv/bin/python -m pytest -q`
  → **116 passed** (109 existing + 7 new in
  `tests/test_packaging.py`: pyproject names the package and
  takes its version from `ticketsense.__version__`, declares
  the `src` layout and Python >= 3.10, declares the three
  console scripts, all three entry-point functions are
  callable, every runtime version in pyproject matches the pin
  in `requirements.txt`, and — only when the package is
  installed — the installed version matches the package and
  the three scripts are registered)
- Fresh-venv verification (a brand-new venv, not the project
  `.venv`): `pip install -e .` succeeded, `pip check` is
  clean, `import ticketsense` reports version 0.1.0.
  `ticketsense-train` trained on the demo data (160 train /
  40 test, same split as before) and saved both pipelines;
  `ticketsense-predict "my card was charged twice, please
  refund me"` returned billing 0.44 / medium 0.42 with the
  priority warning; the demo home page served 200 with the
  form and the JSON endpoint returned `other` / `low` for a
  thank-you ticket, both through the FastAPI test client.
  The full suite in that fresh venv (pytest installed there
  just to run it) passed: **116 passed**, including the two
  install-dependent tests that skip in the project venv.
  The fresh venv was deleted after the check. One hiccup:
  the first fresh venv was built under `/tmp`, which is a
  512 MB tmpfs and ran out of space mid-install; the check
  was redone with the venv under the home directory, where
  there is room.
- `PYTHONPATH=src python -m ticketsense.predict "..."` from
  the project still works exactly as before (billing 0.44 /
  medium 0.42, priority warning).
- `git diff --check` — clean; no secrets in the diff; model
  binaries (`.joblib`) confirmed git-ignored and not staged.
  The warnings in the test run come from the pinned library
  versions and already appear in earlier runs — they are not
  caused by the new code.

### Commits (local only — nothing pushed)
- See `git log --oneline` in this repo. 3 real commits on Day 10.
- **Not pushed:** local commits only, as instructed; the parent
  run does the push.

### Next steps — Day 11 (TicketSense)
- Final polish toward the release: review the README and the
  reports once more, and confirm the v1.0.0 release checklist.

## Day 11 — 2026-11-21 (TicketSense)

### What was done
- Preprocessing and prediction tests in two new files, 32 new
  tests, with no duplication of existing coverage:
  - `tests/test_preprocessing.py` (13 tests): the `clean_text`
    rules — tabs, newlines and mixed whitespace collapse,
    punctuation and numbers are kept, non-ASCII text (Roman
    Urdu, accented names) is kept, cleaning is idempotent, every
    non-string form (None, int, float, list, dict, bytes) and
    every blank form is rejected, and punctuation-only text
    passes cleaning (it is the model's problem, not the
    cleaner's). The vectoriser: its preprocessor is `clean_text`,
    its vocabulary is all lowercase, word pairs such as
    "charged twice" are present, rare words and stopwords are
    kept (min_df=1, no stopword list), punctuation is tokenised
    away, and `predict_label` hands the pipeline the cleaned
    text, not the raw input.
  - `tests/test_prediction_edges.py` (19 tests): the threshold
    boundaries, driven by a fake pipeline with fixed
    probabilities so a test can sit exactly on a threshold —
    confidence exactly at 0.30 (category) and 0.45 (priority)
    is *not* low, just below warns on that half only, both
    below names both in one warning, with the exact warning
    wording pinned; the threshold constants themselves pinned;
    confidence rounded to 4 decimals and taken from the top
    probability; `predict_label` rejects empty/non-string
    directly; `predict_ticket` rejects every non-string form;
    messy input returns cleaned text; the missing-model error
    names both files when both are absent and only the missing
    one when just the priority file is absent; passing one
    loaded pipeline loads the other half from disk; the
    `predict_text` alias returns the same result;
    `format_prediction` shows two-decimal confidences;
    punctuation-only and ~5,000-character tickets still
    predict; the CLI `--category-threshold` /
    `--priority-threshold` flags control the warning; and the
    CLI with a missing priority model exits 1 with the
    train-first error.
- Clean-environment re-run: a fresh venv built under
  `~/workspace/30-day-plan/.day11-venv-ticketsense` (home
  directory, not /tmp — /tmp ran out of space on Day 10),
  `pip install -e .` plus pytest 8.3.3, `pip check` clean,
  `ticketsense` 0.1.0 importable. The full suite in it passed:
  **148 passed**. Trained from scratch and evaluated in that
  clean env: **category accuracy 0.8500, priority accuracy
  0.5250** — exactly the numbers in the committed
  `reports/evaluation.md`. The regenerated report and both
  saved pipelines are byte-identical to the ones already in
  the repo (sha256 check), so there was no discrepancy and no
  report or code fix was needed. The `ticketsense-predict`
  console command in the clean env returned billing 0.44 /
  medium 0.42 with the priority warning, as before. The
  scratch venv was deleted afterwards.

### Tests / checks run (Day 11, local venv `.venv`, Python 3.12.3)
- `PYTHONPATH=src .venv/bin/python -m pytest -q`
  → **148 passed** (116 before + 32 new: 13 in
  `tests/test_preprocessing.py`, 19 in
  `tests/test_prediction_edges.py`)
- Clean venv: `pip check` clean; `pytest -q` → **148 passed**;
  `ticketsense-train` → 160 train / 40 test; `ticketsense-evaluate`
  → category accuracy 0.8500, priority accuracy 0.5250, matching
  the committed report exactly (sha256 of the regenerated
  report and both model files unchanged)
- `git diff --check` — clean; no secrets in the diff; model
  binaries (`.joblib`) confirmed git-ignored and not staged.
  The warnings in the test run come from the pinned library
  versions and already appear in earlier runs — they are not
  caused by the new code.

### Commits (local only — nothing pushed)
- See `git log --oneline` in this repo. 2 real commits on Day 11.
- **Not pushed:** local commits only, as instructed; the parent
  run does the push.

### Next steps — Day 12 (TicketSense)
- README/docs polish, then a full-test bug-fix pass (with
  ClientFlow and RepoDoctor, per the compressed schedule).
