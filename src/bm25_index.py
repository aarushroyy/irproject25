from rank_bm25 import BM25Okapi
from typing import Dict, List, Tuple
try:
    from .preprocess import basic_tokenize
except ImportError:
    from preprocess import basic_tokenize

class BM25Index:
    def __init__(self, docs: Dict[str, str]):
        """
        docs: dict[docid] = text
        """
        self.doc_ids = list(docs.keys())
        tokenized_docs = [basic_tokenize(docs[d]) for d in self.doc_ids]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Returns top-k (doc_id, score)
        """
        tokens = basic_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = [(self.doc_ids[i], float(scores[i])) for i in ranked_indices]
        return results