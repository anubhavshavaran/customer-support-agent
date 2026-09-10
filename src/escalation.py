def decide_escalation(intent, intent_confidence, retrieval_score, high_risk):
    reasons = []
    if high_risk:
        reasons.append("Sensitive account or payment signal needs a human review")
    if intent == "general_complaint":
        reasons.append("No stable support intent was detected")
    if intent_confidence < 0.34:
        reasons.append("Intent confidence is below the auto-handling threshold")
    if retrieval_score < 0.14:
        reasons.append("Historical resolution evidence is weak")
    if reasons:
        return {"decision": "ESCALATE", "reason": "; ".join(reasons)}
    return {
        "decision": "AUTO",
        "reason": "Intent confidence and historical resolution evidence clear the thresholds",
    }
