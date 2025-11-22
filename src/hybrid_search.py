"""
Hybrid search combining BM25 and Dense retrieval using Reciprocal Rank Fusion
"""
from typing import List, Tuple, Dict

def reciprocal_rank_fusion(
    bm25_results: List[Tuple[str, float]],
    dense_results: List[Tuple[str, float]],
    k: int = 60,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Combine BM25 and Dense results using Reciprocal Rank Fusion (RRF)
    
    Args:
        bm25_results: List of (doc_id, score) from BM25
        dense_results: List of (doc_id, score) from Dense retrieval
        k: RRF constant (default 60)
        top_k: Number of results to return
        
    Returns:
        List of (doc_id, rrf_score) ranked by RRF score
    """
    def create_rank_map(results):
        return {doc_id: rank for rank, (doc_id, _) in enumerate(results, start=1)}

    bm25_ranks = create_rank_map(bm25_results)
    dense_ranks = create_rank_map(dense_results)

    # Get all unique documents
    all_docs = set(bm25_ranks.keys()) | set(dense_ranks.keys())
    
    rrf_scores: Dict[str, float] = {}

    for doc_id in all_docs:
        score = 0.0
        if doc_id in bm25_ranks:
            score += 1.0 / (k + bm25_ranks[doc_id])
        if doc_id in dense_ranks:
            score += 1.0 / (k + dense_ranks[doc_id])
        rrf_scores[doc_id] = score

    # Sort by RRF score and return top_k
    ranked_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked_results

class HybridSearch:
    """Hybrid search system combining BM25 and Dense retrieval"""
    
    def __init__(self, bm25_index, dense_index, rrf_k: int = 60):
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.rrf_k = rrf_k
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Perform hybrid search combining BM25 and Dense retrieval
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, rrf_score) ranked by hybrid score
        """
        # Get results from both methods
        bm25_results = self.bm25_index.search(query, k=top_k * 2)
        dense_results = self.dense_index.search(query, k=top_k * 2)
        
        # Combine using RRF
        hybrid_results = reciprocal_rank_fusion(
            bm25_results, dense_results, k=self.rrf_k, top_k=top_k
        )
        
        return hybrid_results

def hybrid_search(bm25_index, dense_index, query: str, top_k: int = 10, rrf_k: int = 60):
    """
    Convenience function for hybrid search
    
    Args:
        bm25_index: BM25 index instance
        dense_index: Dense index instance  
        query: Search query
        top_k: Number of results to return
        rrf_k: RRF constant
        
    Returns:
        List of (doc_id, rrf_score)
    """
    hybrid = HybridSearch(bm25_index, dense_index, rrf_k)
    return hybrid.search(query, top_k)