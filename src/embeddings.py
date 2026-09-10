from .retrieval import HistoricalRetriever


def build_retriever(cases):
    return HistoricalRetriever(cases)
