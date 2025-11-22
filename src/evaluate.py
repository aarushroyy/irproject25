from typing import Dict, List
import math

def accuracy_at_1(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]]) -> float:
    """
    run[qid] = [docid1, docid2, ...]
    """
    correct = 0
    total = 0
    for qid, ranked_docs in run.items():
        rel_docs = qrels.get(qid, {})
        if not rel_docs:
            continue
        total += 1
        if ranked_docs and ranked_docs[0] in rel_docs:
            correct += 1
    return correct / total if total > 0 else 0.0

def recall_at_k(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], k: int = 10) -> float:
    """
    Check if any relevant doc is in top-k
    """
    correct = 0
    total = 0
    for qid, ranked_docs in run.items():
        rel_docs = set(qrels.get(qid, {}).keys())
        if not rel_docs:
            continue
        total += 1
        top_k_docs = set(ranked_docs[:k])
        if rel_docs & top_k_docs:
            correct += 1
    return correct / total if total > 0 else 0.0

def mrr(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]]) -> float:
    """
    Mean Reciprocal Rank
    """
    reciprocal_ranks = []
    for qid, ranked_docs in run.items():
        rel_docs = set(qrels.get(qid, {}).keys())
        if not rel_docs:
            continue
        
        rr = 0.0
        for i, doc_id in enumerate(ranked_docs, 1):
            if doc_id in rel_docs:
                rr = 1.0 / i
                break
        reciprocal_ranks.append(rr)
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

def dcg_at_k(relevances: List[int], k: int) -> float:
    """
    Discounted Cumulative Gain at k
    """
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += (2**relevances[i] - 1) / math.log2(i + 2)
    return dcg

def ndcg_at_k(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], k: int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain at k
    """
    ndcg_scores = []
    for qid, ranked_docs in run.items():
        rel_docs = qrels.get(qid, {})
        if not rel_docs:
            continue
        
        relevances = []
        for doc_id in ranked_docs[:k]:
            relevances.append(rel_docs.get(doc_id, 0))
        
        dcg = dcg_at_k(relevances, k)
        
        ideal_relevances = sorted(rel_docs.values(), reverse=True)[:k]
        idcg = dcg_at_k(ideal_relevances, k)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    
    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0