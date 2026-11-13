# TicketSense data

## What is here

| File | Contents |
|---|---|
| `demo_tickets.csv` | 200 hand-written example support tickets (synthetic demo data) |

## The demo dataset

`demo_tickets.csv` was written by hand. It is **synthetic
demonstration data, not real customer tickets**. No real customer,
student, or company data is used anywhere in this project.

**Source and licence:** hand-written for this project. There is
no licence restriction on it, because no outside work was copied
into it, and it contains no real customer data. It may be used,
shared and changed freely as part of this project.

### Schema

| Column | Meaning | Allowed values |
|---|---|---|
| `ticket_id` | Unique id, `TS-001` … `TS-200` | unique, non-empty |
| `text` | The ticket message, one sentence | non-empty English text |
| `category` | What the ticket is about | `billing`, `technical`, `account`, `delivery`, `other` |
| `priority` | How urgent it is | `low`, `medium`, `high` |

### Label definitions

See `src/ticketsense/data.py` — the same definitions live in code
and in the main `README.md`, so the loader, the docs and the tests
cannot silently drift apart.

### Balance (from `summarise()`)

- Rows: **200**
- Category: 40 each — billing 40, technical 40, account 40,
  delivery 40, other 40 (perfectly balanced *by construction*).
- Priority: high 48, medium 75, low 77 — uneven on purpose, because
  most real tickets are not high priority. Exact counts are printed
  by `python -m ticketsense.data` (see main README).
- Missing values: 0. Duplicate rows: 0. Duplicate texts: 0.

### Limitations — read before trusting any model trained on this

1. **Still small (200 rows).** A model trained on 200 examples
   will not generalise to real support queues. This set exists
   only to build and test the pipeline (loading, validation,
   split, baseline) end-to-end.
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

This set will be kept for pipeline tests and for the first
baseline smoke test. A larger, properly licensed dataset must
replace it before any performance claim is made.

## Public datasets considered (none downloaded)

No external dataset was downloaded, deliberately, because
the plan forbids using a dataset without checking its licence and
permitted use. What was considered:

| Option | Source and licence | Why it was NOT used |
|---|---|---|
| Kaggle "customer support tickets" sets | Kaggle re-uploads, licence unclear or missing | Many are re-uploads with no clear licence, no source, and no label policy. Downloading one would need a Kaggle account and a per-file licence check. |
| Twitter / X customer-support conversations | Public posts on X, redistribution terms unclear | Noisy, short, full of handles/URLs, and the terms for redistribution/training need checking. Also a PII risk if real users are included. |
| GitHub issues as "tickets" | Freely licensed in many repos | Issues are developer bug reports, not customer support tickets — wrong text style and wrong label meaning. |
| Bitext customer support dataset | bitext.com / Hugging Face mirror, licence CDLA-Sharing-1.0 | About 27,000 question-answer pairs for intent detection, with 27 intents. It is not single tickets, it has no priority labels, and mapping its 27 intents to our 5 categories would mean inventing labels. |
| "Enterprise NLP Classification 10K" | opendatabay.com listing, no licence stated | 10,000 tickets, but the listing states no licence and no clear original source, so permitted use cannot be verified. |
| Hugging Face "ticket-classification" model | huggingface.co, model under Apache 2.0 | This is a trained model, not a dataset, and its training data has no stated source or licence. |
| Real customer tickets from any business | Private/confidential data | Never used without explicit owner permission, and never committed to a public repo. |

**Decision:** no real public dataset was cleanly usable — the
ones with a clear licence do not fit our labels, and the ones
that look like tickets have no clear licence. So the synthetic
set was deliberately grown from 50 tickets to 200 tickets
(40 per category), all hand-written for this project, with no
real customer data in it. Nothing was downloaded.

## Rules

- Never commit a large dataset, a model binary, or any file with
  real personal data. Small, licensed, clearly labelled data only.
- If a dataset is added, update this file with source, licence,
  download date, row count, schema, and known limitations.
- Keep train/test separation strict and check for duplicates
  across the split.
