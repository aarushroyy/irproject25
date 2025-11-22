import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_queries, load_qrels, load_docs, filter_queries_with_docs
from src.hinglish_convert import hindi_to_hinglish
from src.bm25_index import BM25Index
from src.dense_index import DenseIndex
from src.hybrid_search import hybrid_search
from src.evaluate import accuracy_at_1, recall_at_k, mrr, ndcg_at_k

def main():
    print("Loading data...")
    queries = load_queries()
    qrels = load_qrels()
    docs = load_docs(limit=10000)
    
    print(f"Original queries: {len(queries)}, qrels: {len(qrels)}, docs: {len(docs)}")
    
    queries, qrels = filter_queries_with_docs(queries, qrels, docs)
    print(f"After filtering: {len(queries)} queries")
    
    queries_hi = queries
    print("Converting to Hinglish...")
    queries_hing = {qid: hindi_to_hinglish(q) for qid, q in queries_hi.items()}
    
    print("Building BM25 index...")
    bm25 = BM25Index(docs)
    
    print("Building Dense index...")
    dense = DenseIndex(docs)
    
    runs = {}
    
    print("Running Hindi + BM25...")
    run_hi_bm25 = {}
    for qid, qtext in queries_hi.items():
        res = bm25.search(qtext, k=10)
        run_hi_bm25[qid] = [docid for docid, _ in res]
    runs["hi_bm25"] = run_hi_bm25
    
    print("Running Hindi + Dense...")
    run_hi_dense = {}
    for qid, qtext in queries_hi.items():
        res = dense.search(qtext, k=10)
        run_hi_dense[qid] = [docid for docid, _ in res]
    runs["hi_dense"] = run_hi_dense
    
    print("Running Hindi + Hybrid...")
    run_hi_hybrid = {}
    for qid, qtext in queries_hi.items():
        res = hybrid_search(bm25, dense, qtext, top_k=10)
        run_hi_hybrid[qid] = [docid for docid, _ in res]
    runs["hi_hybrid"] = run_hi_hybrid
    
    print("Running Hinglish + BM25...")
    run_hing_bm25 = {}
    for qid, qtext in queries_hing.items():
        res = bm25.search(qtext, k=10)
        run_hing_bm25[qid] = [docid for docid, _ in res]
    runs["hing_bm25"] = run_hing_bm25
    
    print("Running Hinglish + Dense...")
    run_hing_dense = {}
    for qid, qtext in queries_hing.items():
        res = dense.search(qtext, k=10)
        run_hing_dense[qid] = [docid for docid, _ in res]
    runs["hing_dense"] = run_hing_dense
    
    print("Running Hinglish + Hybrid...")
    run_hing_hybrid = {}
    for qid, qtext in queries_hing.items():
        res = hybrid_search(bm25, dense, qtext, top_k=10)
        run_hing_hybrid[qid] = [docid for docid, _ in res]
    runs["hing_hybrid"] = run_hing_hybrid
    
    print("\nEvaluation Results:")
    print("=" * 80)
    print(f"{'Method':<15} {'Query Type':<10} {'Acc@1':<8} {'Recall@10':<12} {'MRR':<8} {'nDCG@10':<10}")
    print("=" * 80)
    
    results = {}
    for name, run in runs.items():
        acc1 = accuracy_at_1(run, qrels)
        rec10 = recall_at_k(run, qrels, k=10)
        mrr_val = mrr(run, qrels)
        ndcg10 = ndcg_at_k(run, qrels, k=10)
        
        method, query_type = name.split('_', 1)
        query_type = query_type.replace('_', '+')
        
        results[name] = {
            'acc_at_1': acc1,
            'recall_at_10': rec10,
            'mrr': mrr_val,
            'ndcg_at_10': ndcg10
        }
        
        print(f"{method.upper():<15} {query_type:<10} {acc1:<8.3f} {rec10:<12.3f} {mrr_val:<8.3f} {ndcg10:<10.3f}")
    
    os.makedirs("experiments/runs", exist_ok=True)
    with open("experiments/runs/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to experiments/runs/results.json")
    print(f"Total queries evaluated: {len(queries)}")

if __name__ == "__main__":
    main()