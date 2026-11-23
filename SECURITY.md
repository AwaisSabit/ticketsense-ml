# Security notes for TicketSense

This file records the final security, configuration and leakage
review done before the 1.0.0 release, and the rules that keep
this repo safe afterwards. The checks are also enforced by
`tests/test_release_hygiene.py`, so a future change that breaks
one of them fails the test suite instead of slipping through.

## What was reviewed

- Every tracked file in the repo (code, tests, docs, config,
  reports, data) was scanned for secret-shaped text: API keys,
  tokens, passwords assigned in code, cloud keys and private
  keys. **None found.**
- `data/demo_tickets.csv` was checked for real contact details:
  no email addresses and no phone-number-like sequences. The
  set is 200 hand-written synthetic tickets; there is no real
  customer data anywhere in the project.
- The saved models in `models/` are joblib binaries that stay
  local only — they are git-ignored and are not part of any
  commit or release file. The same goes for the generated
  report charts (`reports/*.png`, except the two small balance
  charts that are part of the EDA report).
- No code in this project makes network calls at runtime. The
  only server is the demo web page, which listens on
  `127.0.0.1` (this machine only) and never sends ticket text
  anywhere.
- Dependencies are pinned to exact versions in both
  `requirements.txt` and `pyproject.toml`, and the two lists are
  checked against each other by a test, so a build cannot
  silently pull in a different version.

## Rules for later changes

- Never commit a real API key, token, password or private key.
  There is no place in this project that needs one.
- Never commit a `.env` file, a model binary
  (`models/*.joblib`), or a real customer ticket. New data must
  be synthetic, or its exact source, licence and permitted use
  must be recorded in `data/README.md` first.
- Keep the dependency pins exact. If a version changes, change
  it in both `requirements.txt` and `pyproject.toml` together.

## If a secret is ever committed by mistake

Rotate the secret immediately where it was issued, then tell
the maintainer so the file can be cleaned from the history.
Do not just delete the line and commit again — the old value
stays visible in the earlier commit.
