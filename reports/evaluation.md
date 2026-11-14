# TicketSense baseline evaluation

Baseline: TF-IDF (word pairs, cleaned text) + Logistic Regression, one pipeline for ticket category and one for priority. Both were trained on the training split only and scored on the held-out test tickets below.

## Split

- Total tickets: 200
- Train: 160 tickets
- Test: 40 tickets (20% held out, stratified by category, random seed 42)
- Test category counts: {'billing': 8, 'delivery': 8, 'account': 8, 'technical': 8, 'other': 8}
- Test priority counts: {'low': 17, 'high': 14, 'medium': 9}
- The vectoriser was fitted on the training split only, so no test text entered the vocabulary.

## Category

- Accuracy: **0.8500**
- Macro precision: 0.8628
- Macro recall: 0.8500
- Macro F1: 0.8449

### Per-class results

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| billing | 1.0000 | 0.6250 | 0.7692 | 8 |
| technical | 0.7500 | 0.7500 | 0.7500 | 8 |
| account | 0.8750 | 0.8750 | 0.8750 | 8 |
| delivery | 0.8889 | 1.0000 | 0.9412 | 8 |
| other | 0.8000 | 1.0000 | 0.8889 | 8 |

### Confusion matrix (rows are the true label, columns are the predicted label)

| True \ Predicted | billing | technical | account | delivery | other |
| --- | ---: | ---: | ---: | ---: | ---: |
| billing | 5 | 1 | 0 | 1 | 1 |
| technical | 0 | 6 | 1 | 0 | 1 |
| account | 0 | 1 | 7 | 0 | 0 |
| delivery | 0 | 0 | 0 | 8 | 0 |
| other | 0 | 0 | 0 | 0 | 8 |

## Priority

- Accuracy: **0.5250**
- Macro precision: 0.6993
- Macro recall: 0.5554
- Macro F1: 0.4535

### Per-class results

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| low | 0.7500 | 0.7059 | 0.7273 | 17 |
| medium | 0.3478 | 0.8889 | 0.5000 | 9 |
| high | 1.0000 | 0.0714 | 0.1333 | 14 |

### Confusion matrix (rows are the true label, columns are the predicted label)

| True \ Predicted | low | medium | high |
| --- | ---: | ---: | ---: |
| low | 12 | 5 | 0 |
| medium | 1 | 8 | 0 |
| high | 3 | 10 | 1 |

## Limitations — read before trusting these numbers

1. **Only 40 test tickets.** Every single ticket is 2.5% of accuracy, so the numbers above move a lot with a handful of tickets. They show the pipeline works end to end; they are not a performance claim.
2. **Hand-written, clean demo data.** Real tickets have typos, mixed languages (including Roman Urdu/Urdu), slang, order numbers and personal details. None of that is in this set, so real-world accuracy will be lower.
3. **Categories are perfectly balanced by construction.** Real support queues are not, so the per-class scores here flatter the model.
4. **Priority labels are one person's judgement**, not a support team's agreed policy. Two people would disagree on several rows, which caps how well any model can score on priority.
5. **A larger, properly licensed dataset must replace this one** before any performance claim is made.
