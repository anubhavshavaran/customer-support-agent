import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class HistoricalRetriever:
    def __init__(self, cases):
        self.cases = cases.reset_index(drop=True).copy()
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_features=100_000,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.cases["customer_text"])

    def search(self, query, k=5):
        query_matrix = self.vectorizer.transform([query])
        scores = (query_matrix @ self.matrix.T).toarray().ravel()
        indexes = np.argsort(-scores)[:k]
        results = []
        for index in indexes:
            row = self.cases.iloc[int(index)]
            results.append(
                {
                    "case_id": row["case_id"],
                    "customer_text": row["customer_text"],
                    "brand_response": row["brand_response"],
                    "intent": row["intent"],
                    "score": float(scores[index]),
                }
            )
        return results
