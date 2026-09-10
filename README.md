# AmazonHelp support agent

This project builds a small, reproducible support agent from the Customer Support on Twitter dataset. It focuses on `AmazonHelp`, where an outbound brand tweet is linked to the customer tweet it answers by `in_response_to_tweet_id`.

The default pipeline is offline and uses only the libraries in `requirements.txt`. It extracts linked customer/brand pairs, assigns each pair one of eight support intents, holds out a 200-example golden set, trains a TF–IDF plus logistic-regression classifier, retrieves similar historical resolutions, and evaluates classification, routing, and reply quality.

## Run it

From the repository root:

```bash
.venv/bin/python -m src.main run
```

This creates `data/processed/cases.csv`, `golden_set.csv`, `predictions.csv`, `reply_scores.csv`, `judge_calibration.csv`, and `metrics.json`. On the supplied 3M-row CSV this is designed to finish in under 15 minutes on a normal laptop. The raw file is expected at `data/raw/twcs.csv`.

To run only the extraction step:

```bash
.venv/bin/python -m src.main prepare
```

To run evaluation after preparation:

```bash
.venv/bin/python -m evaluation.evaluation
```

To classify a new message after preparation:

```bash
.venv/bin/python -m src.main predict "My package still has not arrived"
```

## How it works

The preparation step scans the CSV twice. First it collects AmazonHelp replies. Then it looks up the inbound customer tweet named by each reply. This gives training records with `customer_text` and the historically used `brand_response`.

The intent handbook is deliberately small: `delivery_issue`, `order_change`, `return_refund`, `account_membership`, `device_app_issue`, `billing_payment`, `product_information`, and `general_complaint`. The keyword classifier is the simple baseline. The golden set was reviewed with the handbook's precedence rules, and the production candidate is a supervised TF–IDF classifier trained only on the non-golden records.

For a new message, the agent predicts an intent and searches the historical customer messages with TF–IDF cosine similarity. If confidence and evidence are good enough, it drafts a reply by reusing the closest historical brand resolution. If not, it returns a safe handoff. Unknown charges, account-security language, low confidence, weak evidence, and unclear complaints are escalated with a reason.

The default reply judge is a deterministic offline rubric so the run is reproducible. The rubric scores relevance, groundedness, helpfulness, tone, and safety from 1–5. With `OPENAI_API_KEY` set, `python -m evaluation.evaluation --judge openai` runs the same rubric through `gpt-5-mini`; the prompt and calibration output are part of the harness.

## Files

- `src/prepare.py`: dataset extraction, intent handbook, and golden-set creation.
- `src/classifier.py`: keyword and TF–IDF classifiers.
- `src/retrieval.py`: historical resolution retrieval.
- `src/escalation.py`: routing policy.
- `src/agent.py`: end-to-end support agent.
- `evaluation/evaluation.py`: baselines, metrics, reply rubric, and judge calibration.
- `REPORT.md`: results, failure analysis, limitations, and next steps.
- `data/processed/golden_set.csv`: 200 reviewer-labelled examples produced from a deterministic, intent-balanced sample and excluded from training.

The headline metric is not a claim that the system is ready to deploy. The report explains the sampling and the ways the number can mislead.
