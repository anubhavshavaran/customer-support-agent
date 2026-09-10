from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .prepare import infer_intent


def build_classifier():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=80_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=400, class_weight="balanced"),
            ),
        ]
    )


def train_model(train_df):
    model = build_classifier()
    model.fit(train_df["customer_text"], train_df["intent"])
    return model


def rule_predictions(texts):
    return [infer_intent(value) for value in texts]
