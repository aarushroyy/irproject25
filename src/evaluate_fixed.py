from typing import Dict, List
import math

def accuracy_at_1_fixed(run: Dict[str, List[str]], qrels: Dict[str, List[str]]) -> float:
    """
    run[qid] = [docid1, docid2, ...]
    qrels[qid] = [relevant_docid1, relevant_docid2, ...]
    """
    correct = 0
    total = 0
    for qid, ranked_docs in run.items():
        rel_docs = qrels.get(qid, [])
        if not rel_docs:
            continue
        total += 1
        if ranked_docs and ranked_docs[0] in rel_docs:
            correct += 1
    return correct / total if total > 0 else 0.0

def recall_at_k_fixed(run: Dict[str, List[str]], qrels: Dict[str, List[str]], k: int = 10) -> float:
    """
    Check if any relevant doc is in top-k
    """
    correct = 0
    total = 0
    for qid, ranked_docs in run.items():
        rel_docs = set(qrels.get(qid, []))
        if not rel_docs:
            continue
        total += 1
        top_k_docs = set(ranked_docs[:k])
        if rel_docs & top_k_docs:
            correct += 1
    return correct / total if total > 0 else 0.0

def mrr_fixed(run: Dict[str, List[str]], qrels: Dict[str, List[str]]) -> float:
    """
    Compute MRR (Mean Reciprocal Rank)
    """
    rr_sum = 0.0
    total = 0
    for qid, ranked_docs in run.items():
        rel_docs = set(qrels.get(qid, []))
        if not rel_docs:
            continue
        total += 1
        for rank, doc in enumerate(ranked_docs, 1):
            if doc in rel_docs:
                rr_sum += 1.0 / rank
                break
    return rr_sum / total if total > 0 else 0.0

def ndcg_at_k_fixed(run: Dict[str, List[str]], qrels: Dict[str, List[str]], k: int = 10) -> float:
    """
    Compute nDCG@k
    """
    dcg_sum = 0.0
    idcg_sum = 0.0
    total = 0
    
    for qid, ranked_docs in run.items():
        rel_docs = set(qrels.get(qid, []))
        if not rel_docs:
            continue
        total += 1
        
        # DCG
        dcg = 0.0
        for i, doc in enumerate(ranked_docs[:k]):
            if doc in rel_docs:
                dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
        
        # IDCG (assuming binary relevance)
        idcg = 0.0
        for i in range(min(len(rel_docs), k)):
            idcg += 1.0 / math.log2(i + 2)
        
        if idcg > 0:
            dcg_sum += dcg / idcg
            idcg_sum += 1
    
    return dcg_sum / idcg_sum if idcg_sum > 0 else 0.0