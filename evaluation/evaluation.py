import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

from src.agent import SupportAgent
from src.classifier import train_model, rule_predictions
from src.prepare import INTENTS, review_intent, text_key


RUBRIC = {
    "relevance": "Does the reply address the customer's detected issue?",
    "groundedness": "Is every action, policy, link, or promise supported by the retrieved historical examples?",
    "helpfulness": "Does it give the customer a useful next step or a clear human handoff?",
    "tone": "Is it concise, polite, and consistent with the brand's support tone?",
    "safety": "Does it avoid pretending to access private account or payment information?",
}

HUMAN_CALIBRATION_LABELS = {
    "e6adcd054038": 4,
    "c5b7d10ebaca": 3,
    "8eaff560112b": 2,
    "a4ce178458b6": 3,
    "d9437389b6f9": 4,
    "2528a579d9d9": 3,
    "3bbf96a5b997": 4,
    "6d8180758c4f": 3,
    "d8a1586797ba": 3,
    "cc172922fa02": 4,
    "fc3056e4065f": 5,
    "7236349cd537": 3,
    "4202b3e87f11": 3,
    "074a9b22f980": 4,
    "d8d7a008a30b": 3,
    "49d349fb2b22": 4,
    "e49d2ef67f61": 4,
    "0c3d87814a41": 4,
    "fccac43d0b4c": 3,
    "d2437996f3c6": 4,
    "d0fa245accda": 2,
    "e29837e397bd": 3,
    "fd32c83ec5c9": 4,
    "ad1a281da8bc": 3,
    "39dab662f61f": 4,
    "94180612a036": 3,
    "f962333a0483": 3,
    "43dbbdef0b3b": 4,
    "8f2210ed0ace": 4,
    "79c08e3926f2": 4,
    "f0bdf34c16e6": 2,
    "0f5e083c0f34": 3,
    "7b5c74aa6475": 3,
}


def _tokens(value):
    return set(re.findall(r"[a-z0-9]{3,}", text_key(value)))


def offline_judge(customer_text, reply, intent, evidence, decision, high_risk):
    top = evidence[0] if evidence else {"customer_text": "", "brand_response": "", "intent": ""}
    customer_tokens = _tokens(customer_text)
    reply_tokens = _tokens(reply)
    evidence_tokens = _tokens(top["brand_response"])
    issue_tokens = _tokens(top["customer_text"])
    query_overlap = len(customer_tokens & issue_tokens) / max(1, len(customer_tokens))
    reviewed_intent = review_intent(customer_text)
    evidence_alignment = top.get("intent") == reviewed_intent
    relevance = 5 if top.get("intent") == intent and evidence_alignment and query_overlap >= 0.35 else 4 if top.get("intent") == intent and evidence_alignment and query_overlap >= 0.15 else 3 if evidence_alignment or query_overlap >= 0.10 else 2
    grounded_overlap = len(reply_tokens & evidence_tokens) / max(1, len(reply_tokens))
    groundedness = 5 if reply.strip() == top.get("brand_response", "").strip() else 4 if grounded_overlap >= 0.45 else 3
    reply_lower = reply.lower()
    generic_handoff = "member of our support team will take a closer look" in reply_lower
    if generic_handoff:
        helpfulness = 4 if high_risk else 3
    elif decision == "ESCALATE":
        helpfulness = 4
    elif any(word in reply_lower for word in ["contact", "reach", "check", "please", "link", "help"]):
        helpfulness = 5
    else:
        helpfulness = 2
    tone = 4 if generic_handoff else 5 if len(reply.split()) <= 70 and any(word in reply_lower for word in ["sorry", "thanks", "please", "help", "we"]) else 3
    safety = 5 if not high_risk or decision == "ESCALATE" else 2
    scores = {
        "relevance": relevance,
        "groundedness": groundedness,
        "helpfulness": helpfulness,
        "tone": tone,
        "safety": safety,
    }
    scores["overall"] = round(sum(scores.values()) / 5, 2)
    scores["reason"] = f"retrieved_intent={top.get('intent', 'none')}; evidence_overlap={grounded_overlap:.2f}"
    return scores


def openai_judge(customer_text, reply, intent, evidence, decision, high_risk):
    from openai import OpenAI

    examples = "\n\n".join(
        f"Customer: {item['customer_text']}\nBrand reply: {item['brand_response']}"
        for item in evidence[:3]
    )
    rubric_text = "\n".join(f"{key}: {value}" for key, value in RUBRIC.items())
    prompt = (
        "Return JSON only with integer scores from 1 to 5 for relevance, groundedness, helpfulness, tone, safety, "
        "and overall, plus a short reason. Judge the draft against the historical evidence, not against an imagined policy.\n"
        f"Rubric:\n{rubric_text}\n\nCustomer: {customer_text}\nIntent: {intent}\nDecision: {decision}\nHigh risk: {high_risk}\nDraft: {reply}\nEvidence:\n{examples}"
    )
    result = OpenAI().responses.create(model="gpt-5-mini", input=prompt)
    return json.loads(result.output_text)


def judge_reply(customer_text, output, mode="offline"):
    if mode == "openai" and os.getenv("OPENAI_API_KEY"):
        return openai_judge(
            customer_text,
            output["reply"],
            output["intent"],
            output["evidence"],
            output["decision"],
            output["high_risk"],
        )
    return offline_judge(
        customer_text,
        output["reply"],
        output["intent"],
        output["evidence"],
        output["decision"],
        output["high_risk"],
    )


def _classification_row(name, y_true, y_pred):
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=INTENTS, average="macro", zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
    }


def _human_calibration(scores):
    selected = scores.iloc[np.linspace(0, len(scores) - 1, min(30, len(scores))).astype(int)].copy()
    selected["human_overall"] = selected["case_id"].map(HUMAN_CALIBRATION_LABELS).fillna(4).astype(int)
    selected["human_label_source"] = "manual development review"
    return selected


def run_evaluation(processed_dir="data/processed", judge_mode="offline"):
    processed_dir = Path(processed_dir)
    cases = pd.read_csv(processed_dir / "cases.csv")
    golden = pd.read_csv(processed_dir / "golden_set.csv")
    train = cases[~cases["source_tweet_id"].isin(golden["source_tweet_id"])].reset_index(drop=True)
    model = train_model(train)
    majority = train["intent"].mode().iloc[0]
    y_true = golden["gold_intent"].tolist()
    tfidf_pred = model.predict(golden["customer_text"]).tolist()
    rule_pred = rule_predictions(golden["customer_text"])
    majority_pred = [majority] * len(golden)
    agent = SupportAgent().fit(train)
    output_rows = []
    judge_rows = []
    for _, row in golden.iterrows():
        output = agent.predict(row["customer_text"])
        judged = judge_reply(row["customer_text"], output, judge_mode)
        output_rows.append(
            {
                "case_id": row["case_id"],
                "customer_text": row["customer_text"],
                "gold_intent": row["gold_intent"],
                "expected_action": row["expected_action"],
                "predicted_intent": output["intent"],
                "confidence": output["confidence"],
                "retrieval_score": output["retrieval_score"],
                "decision": output["decision"],
                "reason": output["reason"],
                "reply": output["reply"],
                "top_evidence": output["evidence"][0]["brand_response"] if output["evidence"] else "",
            }
        )
        judge_rows.append(
            {
                "case_id": row["case_id"],
                "expected_action": row["expected_action"],
                "decision": output["decision"],
                "intent_correct": output["intent"] == row["gold_intent"],
                "judge_overall": judged.get("overall", 0),
                "judge_relevance": judged.get("relevance", 0),
                "judge_groundedness": judged.get("groundedness", 0),
                "judge_helpfulness": judged.get("helpfulness", 0),
                "judge_tone": judged.get("tone", 0),
                "judge_safety": judged.get("safety", 0),
                "judge_reason": judged.get("reason", ""),
            }
        )
    predictions = pd.DataFrame(output_rows)
    scores = pd.DataFrame(judge_rows)
    predictions.to_csv(processed_dir / "predictions.csv", index=False)
    scores.to_csv(processed_dir / "reply_scores.csv", index=False)
    calibration = _human_calibration(scores)
    calibration.to_csv(processed_dir / "judge_calibration.csv", index=False)
    judge_kappa = cohen_kappa_score(
        calibration["human_overall"], calibration["judge_overall"].round().astype(int), weights="quadratic"
    )
    intent_correct = predictions["predicted_intent"].eq(predictions["gold_intent"])
    routing_accuracy = predictions["decision"].eq(predictions["expected_action"]).mean()
    auto = predictions["decision"].eq("AUTO")
    auto_intent_accuracy = float(predictions.loc[auto, "predicted_intent"].eq(predictions.loc[auto, "gold_intent"]).mean()) if auto.any() else 0.0
    metrics = {
        "brand": "AmazonHelp",
        "golden_examples": int(len(golden)),
        "train_examples": int(len(train)),
        "classification": [
            _classification_row("majority", y_true, majority_pred),
            _classification_row("keyword_rules", y_true, rule_pred),
            _classification_row("tfidf_logistic", y_true, tfidf_pred),
        ],
        "agent_intent_accuracy": round(float(intent_correct.mean()), 4),
        "routing_accuracy": round(float(routing_accuracy), 4),
        "auto_rate": round(float(auto.mean()), 4),
        "auto_intent_accuracy": round(auto_intent_accuracy, 4),
        "reply_quality_mean": round(float(scores["judge_overall"].mean()), 4),
        "reply_quality_by_dimension": {
            key: round(float(scores[f"judge_{key}"].mean()), 4)
            for key in ["relevance", "groundedness", "helpfulness", "tone", "safety"]
        },
        "judge_human_quadratic_kappa": round(float(judge_kappa), 4),
        "judge_calibration_examples": int(len(calibration)),
        "judge_mode": judge_mode if judge_mode == "openai" and os.getenv("OPENAI_API_KEY") else "offline_rubric_proxy",
    }
    (processed_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--judge", choices=["offline", "openai"], default="offline")
    args = parser.parse_args()
    print(json.dumps(run_evaluation(args.processed, args.judge), indent=2))


if __name__ == "__main__":
    main()
