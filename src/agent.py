import os
import re

from .classifier import train_model
from .escalation import decide_escalation
from .prepare import is_high_risk, normalize_text
from .retrieval import HistoricalRetriever


def _safe_reply(value):
    value = normalize_text(value)
    value = re.sub(r"\s+\^[A-Z]{1,4}$", "", value).strip()
    return value


class SupportAgent:
    def __init__(self, use_openai=False):
        self.use_openai = use_openai and bool(os.getenv("OPENAI_API_KEY"))
        self.classifier = None
        self.retriever = None

    def fit(self, cases):
        self.classifier = train_model(cases)
        self.retriever = HistoricalRetriever(cases)
        return self

    def _openai_reply(self, customer_message, intent, evidence):
        from openai import OpenAI

        examples = "\n\n".join(
            f"Customer: {item['customer_text']}\nBrand reply: {item['brand_response']}"
            for item in evidence[:3]
        )
        prompt = (
            "You draft a concise support reply for AmazonHelp. "
            "Use only the facts and actions shown in the historical examples. "
            "Do not promise a refund, timeline, or outcome not shown. "
            "If the message needs account-specific investigation, ask the customer to use the official support channel.\n\n"
            f"Customer message: {customer_message}\nDetected intent: {intent}\nHistorical examples:\n{examples}"
        )
        response = OpenAI().responses.create(model="gpt-5-mini", input=prompt)
        return response.output_text.strip()

    def predict(self, message, top_k=5):
        if self.classifier is None or self.retriever is None:
            raise RuntimeError("The agent must be fitted before prediction")
        probabilities = self.classifier.predict_proba([message])[0]
        classes = self.classifier.classes_
        best = int(probabilities.argmax())
        intent = str(classes[best])
        confidence = float(probabilities[best])
        candidates = self.retriever.search(message, max(top_k, 20))
        same_intent = [item for item in candidates if item["intent"] == intent]
        evidence = (same_intent or candidates)[:top_k]
        retrieval_score = evidence[0]["score"] if evidence else 0.0
        risk = is_high_risk(message)
        routing = decide_escalation(intent, confidence, retrieval_score, risk)
        if routing["decision"] == "AUTO" and evidence:
            reply = _safe_reply(evidence[0]["brand_response"])
            if self.use_openai:
                reply = self._openai_reply(message, intent, evidence)
        else:
            reply = "Thanks for reaching out. A member of our support team will take a closer look at this."
        return {
            "intent": intent,
            "confidence": confidence,
            "retrieval_score": retrieval_score,
            "decision": routing["decision"],
            "reason": routing["reason"],
            "reply": reply,
            "evidence": evidence,
            "high_risk": risk,
        }


def run_agent(message, cases, use_openai=False):
    return SupportAgent(use_openai=use_openai).fit(cases).predict(message)
