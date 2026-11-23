# TicketSense — exploratory analysis

This report describes the 200 demo tickets in `data/demo_tickets.csv`. It is a description of the data, not an evaluation of any model — for the baseline results, see `reports/evaluation.md`.

## Dataset overview

- Rows: **200**
- Missing values per column:
  - `ticket_id`: 0
  - `text`: 0
  - `category`: 0
  - `priority`: 0
- Duplicate rows: 0
- Duplicate texts: 0

## Label distribution

### Category

| Category | Tickets |
|---|---:|
| billing | 40 |
| technical | 40 |
| account | 40 |
| delivery | 40 |
| other | 40 |

### Priority

| Priority | Tickets |
|---|---:|
| low | 77 |
| medium | 75 |
| high | 48 |

## Balance charts

![Tickets per category](category_balance.png)

![Tickets per priority](priority_balance.png)

## Text length

Word and character counts are measured on the cleaned text
(lower-cased, extra whitespace collapsed).

### Overall

| Measure | Mean | Median | Min | Max |
|---|---|---|---|---|
| Word count | 15.7 | 16 | 10 | 21 |
| Character count | 81.0 | 81 | 57 | 114 |

### billing

| Measure | Mean | Median | Min | Max |
|---|---|---|---|---|
| Word count | 16.45 | 16 | 12 | 20 |
| Character count | 85.83 | 87 | 65 | 107 |

### technical

| Measure | Mean | Median | Min | Max |
|---|---|---|---|---|
| Word count | 15.0 | 15 | 12 | 19 |
| Character count | 77.53 | 77 | 63 | 93 |

### account

| Measure | Mean | Median | Min | Max |
|---|---|---|---|---|
| Word count | 15.45 | 15 | 12 | 20 |
| Character count | 78.95 | 80 | 57 | 98 |

### delivery

| Measure | Mean | Median | Min | Max |
|---|---|---|---|---|
| Word count | 15.93 | 16 | 11 | 20 |
| Character count | 82.83 | 84 | 61 | 99 |

### other

| Measure | Mean | Median | Min | Max |
|---|---|---|---|---|
| Word count | 15.68 | 16 | 10 | 21 |
| Character count | 79.88 | 80 | 57 | 114 |

## Most common words per category

Words are lower-cased and cleaned, common English stopwords are removed, and only the top 10 per category are shown.

- **billing**: order (11), payment (9), charged (6), invoice (6), money (6), plan (6), refund (6), shows (6), card (5), delivery (5)
- **technical**: app (12), website (8), shows (7), cannot (6), order (6), page (6), update (6), even (5), open (4), phone (4)
- **account**: account (26), email (13), old (7), profile (6), cannot (5), change (5), changed (4), code (4), name (4), never (4)
- **delivery**: order (19), delivery (15), parcel (14), arrived (7), home (7), rider (7), address (4), courier (4), days (4), deliver (4)
- **other**: help (5), order (5), shop (5), thank (5), watch (5), bought (4), gift (4), new (4), products (4), delivery (3)

## Category by priority

| Category | low | medium | high | Total |
|---|---:|---:|---:|---:|
| billing | 14 | 16 | 10 | 40 |
| technical | 13 | 16 | 11 | 40 |
| account | 16 | 14 | 10 | 40 |
| delivery | 13 | 16 | 11 | 40 |
| other | 21 | 13 | 6 | 40 |
| **Total** | 77 | 75 | 48 | 200 |

## What this means

- The set is **small**: 200 tickets cannot show how a model would behave on a real support queue. Any model trained on it is a pipeline smoke test only.
- The set is **clean and hand-written**: every ticket is one tidy English sentence, with no typos, no order numbers and no personal data. Real tickets are messier than this.
- The categories are **perfectly balanced** (40 tickets each) because the set was built that way, not because real queues look like that. A model that does well here has only shown it can separate tidy, evenly spread examples.
- Priority labels are one person's judgement, not a support team's agreed policy, so the priority split (high 48, medium 75, low 77) is a starting point, not a ground truth.
- Nothing in this report is a performance claim. The baseline trained on this set is judged in `reports/evaluation.md`, with those limits in mind.
