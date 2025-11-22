from typing import Dict, List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
try:
    from .preprocess import basic_clean
except ImportError:
    from preprocess import basic_clean

class DenseIndex:
    def __init__(self, docs: Dict[str, str], model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.doc_ids = list(docs.keys())
        self.model = SentenceTransformer(model_name)

        texts = [basic_clean(docs[d]) for d in self.doc_ids]
        embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        self.embeddings = embeddings
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        q_emb = self.model.encode([basic_clean(query)], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(q_emb, k)
        scores = scores[0]
        indices = indices[0]
        results = [(self.doc_ids[i], float(scores[i])) for i in indices]
        return results