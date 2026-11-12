# TicketSense data

## What is here

| File | Contents |
|---|---|
| `demo_tickets.csv` | 50 hand-written example support tickets (synthetic demo data) |

## The demo dataset

`demo_tickets.csv` was written by hand. It is **synthetic
demonstration data, not real customer tickets**. No real customer,
student, or company data is used anywhere in this project.

### Schema

| Column | Meaning | Allowed values |
|---|---|---|
| `ticket_id` | Unique id, `TS-001` … `TS-050` | unique, non-empty |
| `text` | The ticket message, one sentence | non-empty English text |
| `category` | What the ticket is about | `billing`, `technical`, `account`, `delivery`, `other` |
| `priority` | How urgent it is | `low`, `medium`, `high` |

### Label definitions

See `src/ticketsense/data.py` — the same definitions live in code
and in the main `README.md`, so the loader, the docs and the tests
cannot silently drift apart.

### Balance (from `summarise()`)

- Rows: **50**
- Category: 10 each — billing 10, technical 10, account 10,
  delivery 10, other 10 (perfectly balanced *by construction*).
- Priority: high 12, medium 19, low 19 — uneven on purpose, because
  most real tickets are not high priority. Exact counts are printed
  by `python -m ticketsense.data` (see main README).
- Missing values: 0. Duplicate rows: 0. Duplicate texts: 0.

### Limitations — read before trusting any model trained on this

1. **Tiny (50 rows).** A model trained on 50 examples will not
   generalise. This set exists only to build and test the pipeline
   (loading, validation, split, baseline) end-to-end.
2. **Hand-written and clean.** Real tickets have typos, mixed
   languages (including Roman Urdu/Urdu), slang, screenshots,
   order numbers and PII. None of that is in this set.
3. **Categories are perfectly balanced.** Real support queues are
   not — `technical` and `billing` usually dominate, and `other`
   is a rag-bag, not a real class.
4. **Priority labels are my judgement**, not a support team's
   agreed policy. Two people would disagree on several rows.
5. **No leakage check can be meaningful at this size** — there are
   no near-duplicate real tickets to catch yet.

This set will be kept for pipeline tests, and used only for the
first baseline smoke test. A larger, properly
licensed dataset (or a clearly documented larger synthetic set)
must replace it before any performance claim is made.

## Public datasets considered (none downloaded)

No external dataset was downloaded, deliberately, because
the plan forbids using a dataset without checking its licence and
permitted use. What was considered:

| Option | Why it was NOT used |
|---|---|
| Kaggle "customer support tickets" sets | Many are re-uploads with no clear licence, no source, and no label policy. Downloading one would need a Kaggle account and a per-file licence check. |
| Twitter / X customer-support conversations | Public posts, but noisy, short, full of handles/URLs, and the terms for redistribution/training need checking. Also a PII risk if real users are included. |
| GitHub issues as "tickets" | Freely licensed in many repos, but issues are developer bug reports, not customer support tickets — wrong text style and wrong label meaning. |
| Real customer tickets from any business | Private/confidential data. Never used without explicit owner permission, and never committed to a public repo. |

**Decision:** use the small synthetic demo set above,
document the options honestly, and make the dataset choice the
next task: pick one public set, record its exact source URL,
licence, and permitted use in this file *before* downloading it,
or deliberately build a larger synthetic set and label it as such.

## Rules

- Never commit a large dataset, a model binary, or any file with
  real personal data. Small, licensed, clearly labelled data only.
- If a dataset is added, update this file with source, licence,
  download date, row count, schema, and known limitations.
- Keep train/test separation strict and check for duplicates
  across the split.
