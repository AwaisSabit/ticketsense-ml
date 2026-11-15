# TicketSense baseline — error analysis

The baseline is TF-IDF (word pairs, cleaned text) + Logistic Regression, one pipeline for ticket category and one for priority. This report looks at the tickets it gets wrong on the held-out test set, instead of only how many it gets wrong.

## Split

- Total tickets: 200
- Train: 160 tickets
- Test: 40 tickets (20% held out, stratified by category, random seed 42)
- Test category counts: {'billing': 8, 'delivery': 8, 'account': 8, 'technical': 8, 'other': 8}
- Test priority counts: {'low': 17, 'high': 14, 'medium': 9}

## How many were wrong

- Category: **6 of 40** test tickets misclassified (accuracy 0.8500).
- Priority: **19 of 40** test tickets misclassified (accuracy 0.5250).

## Category errors

### Misclassified tickets

| Ticket | Text | True label | Predicted label |
| --- | --- | --- | --- |
| TS-066 | Do you charge extra for paying by card instead of cash on delivery? | billing | other |
| TS-095 | My account shows someone else's orders, I can see another customer's name and address. | technical | account |
| TS-065 | The payment went through but I never got an order confirmation, and now I am worried my money is lost. | billing | technical |
| TS-058 | What is the service fee on a cash on delivery order, I want to know before I order. | billing | delivery |
| TS-111 | I cannot log in at all, the app says my account does not exist even though I registered last week. | account | technical |
| TS-100 | Is there a way to use the website in Urdu, my father finds English difficult? | technical | other |

### Confusion pairs (most common first)

| True label | Predicted label | Tickets |
| --- | --- | ---: |
| account | technical | 1 |
| billing | delivery | 1 |
| billing | other | 1 |
| billing | technical | 1 |
| technical | account | 1 |
| technical | other | 1 |

### Weak spots (recall below 0.5)

No category weak spots: every label was found at least half the time.

## Priority errors

### Misclassified tickets

| Ticket | Text | True label | Predicted label |
| --- | --- | --- | --- |
| TS-119 | I received an email saying my account will be closed, is this real or a scam? | high | low |
| TS-148 | My parcel was delivered to the wrong address, a neighbour signed for it and I cannot find it. | high | medium |
| TS-090 | Can the app work without internet, or does it always need a connection? | low | medium |
| TS-095 | My account shows someone else's orders, I can see another customer's name and address. | high | low |
| TS-197 | Thank you, the issue with my order was solved and I am happy with the service. | low | medium |
| TS-028 | Someone else may have used my account, I see a login from a city I have never visited. | high | medium |
| TS-183 | I need urgent help, a product I bought from you has a sharp edge and cut my hand. | high | medium |
| TS-153 | Can I track my order without the tracking number, I lost the message with the details? | low | medium |
| TS-115 | Can I use the same email for two accounts, one for my shop and one for personal use? | low | medium |
| TS-123 | A stranger is using my account to place orders, please block the account immediately. | high | low |
| TS-079 | My card was charged three times for one order, please check and return the extra money. | high | medium |
| TS-109 | Can I change the app language to Roman Urdu, I find it easier to read? | low | medium |
| TS-009 | I paid for the yearly plan yesterday, but my account still shows the free plan. | high | medium |
| TS-111 | I cannot log in at all, the app says my account does not exist even though I registered last week. | high | medium |
| TS-108 | The payment page keeps loading forever and never shows the confirmation screen. | high | medium |
| TS-004 | My card payment failed three times but the money was still deducted from my bank account. | high | medium |
| TS-145 | The parcel arrived with half the items missing, two items from my order are not in the box. | high | medium |
| TS-174 | What is your return policy for clothes that do not fit, I want to know before buying? | medium | low |
| TS-081 | The app crashes as soon as I open it after the new update, I cannot use it at all. | high | medium |

### Confusion pairs (most common first)

| True label | Predicted label | Tickets |
| --- | --- | ---: |
| high | medium | 10 |
| low | medium | 5 |
| high | low | 3 |
| medium | low | 1 |

### Weak spots (recall below 0.5)

- **high**: recall 0.0714 (found 1 of 14 test tickets), F1 0.1333.

## Confusion-matrix charts

![Category confusion matrix](category_confusion.png)

![Priority confusion matrix](priority_confusion.png)

## Second baseline: TF-IDF + Naive Bayes

Trained on the same 160 training tickets and scored on the same 40 test tickets as the Logistic Regression baseline, so the numbers are directly comparable. Only the classifier differs — the text cleaning and TF-IDF setup are identical.

| Task | Logistic Regression | Naive Bayes |
| --- | ---: | ---: |
| Category accuracy | 0.8500 | 0.8250 |
| Category macro F1 | 0.8449 | 0.8213 |
| Priority accuracy | 0.5250 | 0.5000 |
| Priority macro F1 | 0.4535 | 0.4040 |

**Verdict:** Naive Bayes is worse on both tasks, so Logistic Regression stays the baseline.

## What the errors mean

- **Category errors are mostly near misses between related topics.** The confusion pairs above show tickets landing in a neighbouring category (for example a billing ticket talking about the app, or a technical ticket mentioning a payment). With one-sentence tickets the model only has a handful of words to work with, so one shared word can tip the prediction. A couple of the 'errors' are arguably debatable labels — for example the ticket about seeing someone else's orders in your account is labelled technical but reads like an account problem — so the true category errors are fewer than the count suggests.
- **Priority errors are a model problem first, and a labelling problem second.** The model almost never predicts high priority: of 14 high test tickets it found 1, calling 10 medium and 3 low. Several of the missed high tickets read clearly urgent to a human — a stranger placing orders on your account, a card charged three times, a product that cut someone's hand — so the model is not just struggling with borderline cases; it systematically under-predicts high. At the same time, priority was labelled by one person's judgement with no written rules for where medium ends and high begins, so two people would disagree on several of these rows, and that caps how well any model can score on priority. Both fixes are needed: agreed labelling rules written down, and a model that is not afraid to predict high.
- **The weak spots point at the same two fixes.** There is no category weak spot by the recall-below-half rule — billing is the weakest category at recall 0.625 — but high priority is the real weak spot at recall 0.0714. Both need more, and more varied, examples of the rare case — and for priority, agreed labelling rules written down before more tickets are labelled, so the model has a consistent target to learn.
- **Nothing here changes the headline numbers.** Category accuracy stays 0.85 and priority accuracy stays 0.525 on the same 40 test tickets. This report only explains where those numbers come from.
- **Only 40 test tickets.** Every ticket is 2.5% of accuracy, so one or two tickets flipping would move the confusion pairs and the weak spots. Treat the patterns above as hints about where to look next, not as a verdict on the model.
