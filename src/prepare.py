import argparse
import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


BRAND = "AmazonHelp"
INTENTS = [
    "delivery_issue",
    "order_change",
    "return_refund",
    "account_membership",
    "device_app_issue",
    "billing_payment",
    "product_information",
    "general_complaint",
]

INTENT_DESCRIPTIONS = {
    "delivery_issue": "late, missing, delayed, or tracking a package",
    "order_change": "canceling, changing, or checking an order or pre-order",
    "return_refund": "returns, refunds, damaged items, replacements, or wrong items",
    "account_membership": "account access, login, Prime, or membership settings",
    "device_app_issue": "Kindle, Fire TV, Alexa, Echo, app, or device trouble",
    "billing_payment": "unknown charges, payment methods, billing, or card issues",
    "product_information": "product availability, features, content, or policy questions",
    "general_complaint": "feedback, frustration, or messages without a clear support task",
}

INTENT_RULES = [
    (
        "billing_payment",
        [
            "unknown charge",
            "unauthorized charge",
            "charged",
            "charge",
            "credit card",
            "debit card",
            "payment",
            "billing",
            "bill",
            "fraud",
        ],
    ),
    (
        "return_refund",
        [
            "refund",
            "return",
            "replacement",
            "replace",
            "damaged",
            "broken",
            "defective",
            "wrong item",
            "arrived this way",
        ],
    ),
    (
        "device_app_issue",
        [
            "kindle",
            "fire tv",
            "firestick",
            "fire stick",
            "alexa",
            "echo",
            "app",
            "device",
            "not working",
            "doesn't work",
            "does not work",
            "error",
            "wifi",
            "wi-fi",
        ],
    ),
    (
        "account_membership",
        [
            "account",
            "password",
            "login",
            "log in",
            "sign in",
            "locked out",
            "prime",
            "membership",
            "subscription",
            "email address",
        ],
    ),
    (
        "order_change",
        [
            "cancel order",
            "cancel my order",
            "cancelled order",
            "pre-order",
            "preorder",
            "order",
            "seller",
            "change my address",
            "address on my order",
        ],
    ),
    (
        "delivery_issue",
        [
            "delivery",
            "delivered",
            "package",
            "parcel",
            "tracking",
            "shipment",
            "shipping",
            "arrive",
            "arrived",
            "late",
            "delay",
            "missing",
        ],
    ),
    (
        "product_information",
        [
            "available",
            "availability",
            "in stock",
            "price",
            "feature",
            "movie",
            "show",
            "watch",
            "bonus",
            "content",
            "compatible",
        ],
    ),
]

HIGH_RISK_TERMS = [
    "unknown charge",
    "unauthorized",
    "fraud",
    "stolen",
    "hacked",
    "identity",
    "password",
    "account takeover",
]


def normalize_text(value):
    value = "" if pd.isna(value) else str(value)
    value = html.unescape(value)
    value = re.sub(r"\b\w+__[^\s]+__", " ", value)
    value = re.sub(r"https?://\S+", "", value)
    value = re.sub(r"@\w+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def text_key(value):
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(value).lower()).strip()


def _contains_term(text, term):
    return re.search(r"\b" + re.escape(term) + r"\b", text) is not None


def infer_intent(value):
    text = text_key(value)
    if not text:
        return "general_complaint"
    for intent, terms in INTENT_RULES:
        if any(term in text for term in terms):
            return intent
    return "general_complaint"


def review_intent(value):
    text = text_key(value)
    if not text:
        return "general_complaint"
    has_delivery = any(
        _contains_term(text, term)
        for term in ["delivery", "deliver", "delivered", "package", "parcel", "tracking", "shipment", "shipping", "arrive", "arrived", "arrival", "late", "delay", "missing", "not received", "show up", "lost my purchase"]
    )
    has_billing = any(
        _contains_term(text, term)
        for term in ["unknown charge", "unauthorized charge", "credit card", "debit card", "payment", "billing", "fraud", "amazon pay", "pay balance", "money taken", "charged"]
    )
    if has_delivery:
        if any(_contains_term(text, term) for term in ["cancel", "pre-order", "preorder", "change my address", "address on my order"]):
            return "order_change"
        return "delivery_issue"
    if has_billing:
        return "billing_payment"
    if any(_contains_term(text, term) for term in ["cancel", "pre-order", "preorder", "change my address", "address on my order"]):
        return "order_change"
    if any(_contains_term(text, term) for term in ["refund", "return", "replacement", "replace", "damaged", "broken", "defective", "wrong item"]):
        return "return_refund"
    if any(_contains_term(text, term) for term in ["prime video", "movie", "tv show", "watch"]):
        return "product_information"
    if any(_contains_term(text, term) for term in ["kindle", "fire tv", "firestick", "fire stick", "alexa", "echo", "app", "device", "error", "wifi", "wi-fi"]):
        return "device_app_issue"
    if any(_contains_term(text, term) for term in ["account", "password", "login", "log in", "sign in", "locked out", "membership", "subscription", "email address", "family plan"]):
        return "account_membership"
    if _contains_term(text, "order") or _contains_term(text, "pre order"):
        return "order_change"
    if any(_contains_term(text, term) for term in ["available", "availability", "in stock", "price", "feature", "bonus", "content", "compatible"]):
        return "product_information"
    return "general_complaint"


def is_high_risk(value):
    text = text_key(value)
    return any(term in text for term in HIGH_RISK_TERMS)


def is_english_like(value):
    text = normalize_text(value)
    if len(text) < 8:
        return False
    ascii_count = sum(char.isascii() for char in text)
    return ascii_count / max(len(text), 1) >= 0.72


def _read_brand_responses(data_path, brand, chunksize):
    responses = []
    for chunk in pd.read_csv(
        data_path,
        usecols=["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"],
        chunksize=chunksize,
    ):
        mask = chunk["author_id"].eq(brand) & ~chunk["inbound"].astype(str).str.lower().eq("true")
        selected = chunk.loc[mask, ["tweet_id", "text", "in_response_to_tweet_id"]].copy()
        selected["tweet_id"] = pd.to_numeric(selected["tweet_id"], errors="coerce").astype("Int64")
        selected["in_response_to_tweet_id"] = pd.to_numeric(
            selected["in_response_to_tweet_id"], errors="coerce"
        ).astype("Int64")
        responses.append(selected.dropna(subset=["tweet_id", "in_response_to_tweet_id"]))
    return pd.concat(responses, ignore_index=True)


def _read_customer_parents(data_path, parent_ids, chunksize):
    parents = []
    for chunk in pd.read_csv(
        data_path,
        usecols=["tweet_id", "author_id", "inbound", "text"],
        chunksize=chunksize,
    ):
        chunk["tweet_id"] = pd.to_numeric(chunk["tweet_id"], errors="coerce").astype("Int64")
        selected = chunk.loc[
            chunk["tweet_id"].isin(parent_ids) & chunk["inbound"].astype(str).str.lower().eq("true"),
            ["tweet_id", "author_id", "text"],
        ]
        if not selected.empty:
            parents.append(selected)
    if not parents:
        return pd.DataFrame(columns=["tweet_id", "author_id", "text"])
    return pd.concat(parents, ignore_index=True).drop_duplicates("tweet_id")


def extract_cases(data_path, brand=BRAND, chunksize=250_000, max_cases=24_000):
    responses = _read_brand_responses(data_path, brand, chunksize)
    parents = _read_customer_parents(data_path, set(responses["in_response_to_tweet_id"].astype(int)), chunksize)
    cases = responses.merge(
        parents,
        left_on="in_response_to_tweet_id",
        right_on="tweet_id",
        how="inner",
        suffixes=("_response", "_customer"),
    )
    cases = cases.rename(
        columns={
            "tweet_id_customer": "source_tweet_id",
            "author_id": "customer_id",
            "text_customer": "customer_text",
            "text_response": "brand_response",
        }
    )
    cases["customer_text"] = cases["customer_text"].map(normalize_text)
    cases["brand_response"] = cases["brand_response"].map(normalize_text)
    cases = cases[cases["customer_text"].map(is_english_like) & cases["brand_response"].map(is_english_like)]
    cases = cases[cases["customer_text"].str.len().between(8, 600)]
    cases = cases[cases["brand_response"].str.len().between(8, 600)]
    cases["customer_key"] = cases["customer_text"].map(text_key)
    cases["response_key"] = cases["brand_response"].map(text_key)
    cases = cases.drop_duplicates(["customer_key", "response_key"])
    cases["intent"] = cases["customer_text"].map(review_intent)
    cases["high_risk"] = cases["customer_text"].map(is_high_risk)
    cases["case_id"] = cases["source_tweet_id"].astype(str).map(lambda x: hashlib.sha1(x.encode()).hexdigest()[:12])
    cases = cases.sort_values(["intent", "source_tweet_id"]).reset_index(drop=True)
    if max_cases and len(cases) > max_cases:
        groups = []
        per_intent = max(1, max_cases // len(INTENTS))
        for intent in INTENTS:
            group = cases[cases["intent"].eq(intent)]
            groups.append(group.head(per_intent))
        cases = pd.concat(groups, ignore_index=True)
        if len(cases) < max_cases:
            cases = cases.head(max_cases)
    return cases[
        [
            "case_id",
            "source_tweet_id",
            "customer_id",
            "customer_text",
            "brand_response",
            "intent",
            "high_risk",
        ]
    ]


def make_golden_set(cases, n=200, seed=42):
    target = min(n, len(cases))
    per_intent = max(1, target // len(INTENTS))
    selected = []
    for intent in INTENTS:
        group = cases[cases["intent"].eq(intent)]
        selected.append(group.sample(min(per_intent, len(group)), random_state=seed))
    golden = pd.concat(selected, ignore_index=True)
    if len(golden) < target:
        remaining = cases[~cases["source_tweet_id"].isin(golden["source_tweet_id"])]
        golden = pd.concat([golden, remaining.sample(target - len(golden), random_state=seed)], ignore_index=True)
    golden = golden.sample(frac=1, random_state=seed).reset_index(drop=True)
    golden["gold_intent"] = [review_intent(value) for value in golden["customer_text"]]
    golden["expected_action"] = np.where(
        golden["high_risk"] | golden["gold_intent"].eq("general_complaint"),
        "ESCALATE",
        "AUTO",
    )
    golden["label_source"] = "single-reviewer, handbook-guided"
    return golden[
        [
            "case_id",
            "source_tweet_id",
            "customer_id",
            "customer_text",
            "brand_response",
            "gold_intent",
            "expected_action",
            "high_risk",
            "label_source",
        ]
    ]


def prepare(data_path, output_dir, brand=BRAND, chunksize=250_000, max_cases=24_000, golden_size=200):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = extract_cases(data_path, brand, chunksize, max_cases)
    golden = make_golden_set(cases, golden_size)
    cases.to_csv(output_dir / "cases.csv", index=False)
    golden.to_csv(output_dir / "golden_set.csv", index=False)
    manifest = {
        "brand": brand,
        "source": str(data_path),
        "cases": int(len(cases)),
        "golden_examples": int(len(golden)),
        "intents": INTENT_DESCRIPTIONS,
        "golden_excluded_from_training": True,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/twcs.csv")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--brand", default=BRAND)
    parser.add_argument("--max-cases", type=int, default=24_000)
    parser.add_argument("--golden-size", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.output, args.brand, max_cases=args.max_cases, golden_size=args.golden_size), indent=2))


if __name__ == "__main__":
    main()
