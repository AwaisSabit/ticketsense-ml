# TicketSense sample predictions

Predictions from the saved baseline pipelines (TF-IDF + Logistic
Regression, trained on 160 tickets), run with:

```bash
PYTHONPATH=src python -m ticketsense.predict "ticket text here"
```

Confidence is the probability the pipeline gives its top label.
A prediction counts as low-confidence when category confidence
is below 0.30 or priority confidence is below 0.45 — thresholds
measured on the 40 held-out test tickets (see `predict.py` for
why a single 0.6 threshold would flag every prediction: this
model's probabilities are compressed, category only 0.22–0.47).

## The examples

### 1. Charged twice — billing, but check the priority

```text
Ticket: my card was charged twice for the same order, please refund the extra payment
Category: billing (confidence 0.45)
Priority: high (confidence 0.37)
Warning: Low confidence on priority (0.37) — please check this ticket by hand instead of trusting the suggestion.
```

The category is right and reasonably confident. The priority
warning is honest: the model says high, but at 0.37 it is far
from sure.

### 2. App crashes — technical, but check the priority

```text
Ticket: the app crashes every time i try to open it
Category: technical (confidence 0.48)
Priority: high (confidence 0.39)
Warning: Low confidence on priority (0.39) — please check this ticket by hand instead of trusting the suggestion.
```

Again the category is right; again the priority is the unsure
half.

### 3. Vague ticket — low confidence on both

```text
Ticket: hello i need some help please
Category: other (confidence 0.27)
Priority: high (confidence 0.37)
Warning: Low confidence on category (0.27) and priority (0.37) — please check this ticket by hand instead of trusting the suggestion.
```

This is the warning working as intended: the ticket says
almost nothing, the category score (0.27) is barely above a
random guess for five categories (0.20), and the "high"
priority is a guess — the model calls most vague tickets high
because it has no signal to go on. A human must read this one.

### 4. Thank-you note — no warning

```text
Ticket: thank you for the quick reply, everything works great now
Category: other (confidence 0.35)
Priority: low (confidence 0.50)
```

The one example that passes cleanly: the category and the low
priority both match what a person would say, and both scores
clear their thresholds.

### 5. Parcel not arrived — delivery, but check the priority

```text
Ticket: my parcel still has not arrived
Category: delivery (confidence 0.39)
Priority: medium (confidence 0.39)
Warning: Low confidence on priority (0.39) — please check this ticket by hand instead of trusting the suggestion.
```

The category is right. The priority is a genuine judgement
call — a missing parcel could be medium or high depending on
the team's rules — and the model is not sure, so it warns.

## What this shows, honestly

- All five category predictions match what a person would say,
  and the two clearest tickets score highest (0.45, 0.48).
  The category half is usable as a first-pass suggestion.
- The warning fired on 4 of the 5 examples, and every time it
  was the priority half (once the category too). That matches
  the evaluation: priority accuracy is only 0.525 and the
  model found just 1 of 14 high-priority test tickets, so
  priority suggestions should be treated as rough guesses
  until the labelling rules are written down and the model is
  retrained on more data.
- The absolute confidence numbers are low even when the
  prediction is right (the clearest ticket scored 0.48), so
  read them as a ranking — higher means surer — not as "48%
  likely to be right". On the held-out test set, predictions
  above the thresholds were right far more often than those
  below them, which is what the warning is built on.
