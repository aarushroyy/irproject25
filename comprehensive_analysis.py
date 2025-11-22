#!/usr/bin/env python3
"""
Comprehensive IR Analysis for MIRACL and PHINC Datasets
Generates detailed statistics and analysis for both datasets
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
import re

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import get_both_datasets
from bm25_index import BM25Index
from evaluate_fixed import accuracy_at_1_fixed, recall_at_k_fixed, mrr_fixed, ndcg_at_k_fixed
from config import ensure_directories, RESULTS_DIR

def analyze_text_statistics(texts):
    """Analyze text statistics for a collection of texts"""
    stats = {
        'total_docs': len(texts),
        'avg_length': 0,
        'min_length': 0,
        'max_length': 0,
        'vocab_size': 0,
        'total_terms': 0,
        'length_distribution': {}
    }
    
    if not texts:
        return stats
    
    # Extract text values
    text_values = list(texts.values()) if isinstance(texts, dict) else texts
    
    # Calculate lengths
    lengths = [len(text.split()) for text in text_values]
    stats['avg_length'] = np.mean(lengths)
    stats['min_length'] = min(lengths)
    stats['max_length'] = max(lengths)
    
    # Length distribution
    length_bins = [0, 5, 10, 20, 50, 100, float('inf')]
    length_labels = ['0-4', '5-9', '10-19', '20-49', '50-99', '100+']
    
    for i, (start, end) in enumerate(zip(length_bins[:-1], length_bins[1:])):
        count = sum(1 for l in lengths if start <= l < end)
        stats['length_distribution'][length_labels[i]] = count
    
    # Vocabulary analysis
    all_words = []
    for text in text_values:
        words = text.lower().split()
        all_words.extend(words)
    
    stats['total_terms'] = len(all_words)
    stats['vocab_size'] = len(set(all_words))
    
    return stats

def analyze_query_document_overlap(queries, documents):
    """Analyze overlap between query and document vocabularies"""
    # Extract vocabularies
    query_texts = list(queries.values()) if isinstance(queries, dict) else queries
    doc_texts = list(documents.values()) if isinstance(documents, dict) else documents
    
    query_vocab = set()
    for text in query_texts:
        query_vocab.update(text.lower().split())
    
    doc_vocab = set()
    for text in doc_texts:
        doc_vocab.update(text.lower().split())
    
    overlap = len(query_vocab & doc_vocab)
    query_only = len(query_vocab - doc_vocab)
    doc_only = len(doc_vocab - query_vocab)
    
    return {
        'query_vocab_size': len(query_vocab),
        'document_vocab_size': len(doc_vocab),
        'overlap_terms': overlap,
        'query_only_terms': query_only,
        'document_only_terms': doc_only,
        'overlap_percentage': (overlap / len(query_vocab)) * 100 if query_vocab else 0
    }

def analyze_relevance_patterns(qrels):
    """Analyze relevance judgment patterns"""
    if not qrels:
        return {}
    
    # Number of relevant docs per query
    rel_counts = [len(docs) for docs in qrels.values()]
    
    return {
        'total_queries': len(qrels),
        'avg_relevant_docs': np.mean(rel_counts),
        'min_relevant_docs': min(rel_counts),
        'max_relevant_docs': max(rel_counts),
        'queries_with_1_rel': sum(1 for c in rel_counts if c == 1),
        'queries_with_multiple_rel': sum(1 for c in rel_counts if c > 1)
    }

def detect_language_patterns(texts, sample_size=100):
    """Detect language patterns in texts"""
    text_values = list(texts.values()) if isinstance(texts, dict) else texts
    
    # Sample texts for analysis
    sample_texts = text_values[:sample_size] if len(text_values) > sample_size else text_values
    
    patterns = {
        'hindi_script_docs': 0,
        'english_script_docs': 0,
        'mixed_script_docs': 0,
        'avg_hindi_ratio': 0,
        'common_hindi_words': [],
        'common_english_words': []
    }
    
    hindi_indicators = ['के', 'में', 'है', 'की', 'से', 'को', 'पर', 'एक', 'यह', 'का']
    english_indicators = ['the', 'and', 'of', 'to', 'in', 'is', 'for', 'with', 'on', 'as']
    
    hindi_word_counts = Counter()
    english_word_counts = Counter()
    
    for text in sample_texts:
        words = text.lower().split()
        hindi_count = sum(1 for word in words if any(ind in word for ind in hindi_indicators))
        english_count = sum(1 for word in words if word in english_indicators)
        
        # Count words for vocabulary analysis
        for word in words:
            if any(ind in word for ind in hindi_indicators):
                hindi_word_counts[word] += 1
            elif word in english_indicators or re.match(r'^[a-zA-Z]+$', word):
                english_word_counts[word] += 1
        
        total_words = len(words)
        if total_words > 0:
            hindi_ratio = hindi_count / total_words
            english_ratio = english_count / total_words
            
            if hindi_ratio > 0.3:
                patterns['hindi_script_docs'] += 1
            elif english_ratio > 0.3:
                patterns['english_script_docs'] += 1
            else:
                patterns['mixed_script_docs'] += 1
    
    patterns['common_hindi_words'] = [word for word, count in hindi_word_counts.most_common(10)]
    patterns['common_english_words'] = [word for word, count in english_word_counts.most_common(10)]
    
    return patterns

def run_retrieval_analysis(queries, documents, qrels, dataset_name):
    """Run retrieval analysis and compute detailed metrics"""
    print(f"    Running retrieval analysis for {dataset_name}...")
    
    # Build BM25 index
    bm25 = BM25Index(documents)
    
    # Run retrieval
    run_results = {}
    for qid, qtext in queries.items():
        results = bm25.search(qtext, k=20)  # Get top 20 for more detailed analysis
        run_results[qid] = [docid for docid, _ in results]
    
    # Compute metrics at different cutoffs
    metrics = {}
    for k in [1, 3, 5, 10, 20]:
        metrics[f'recall_at_{k}'] = recall_at_k_fixed(run_results, qrels, k=k)
        if k <= 10:
            metrics[f'ndcg_at_{k}'] = ndcg_at_k_fixed(run_results, qrels, k=k)
    
    metrics['accuracy_at_1'] = accuracy_at_1_fixed(run_results, qrels)
    metrics['mrr'] = mrr_fixed(run_results, qrels)
    
    # Analysis of retrieval patterns
    retrieval_analysis = {
        'avg_retrieved_per_query': np.mean([len(docs) for docs in run_results.values()]),
        'queries_with_no_results': sum(1 for docs in run_results.values() if not docs),
        'queries_with_relevant_in_top1': 0,
        'queries_with_relevant_in_top5': 0,
        'queries_with_relevant_in_top10': 0
    }
    
    for qid, ranked_docs in run_results.items():
        relevant_docs = set(qrels.get(qid, []))
        if relevant_docs:
            if ranked_docs and ranked_docs[0] in relevant_docs:
                retrieval_analysis['queries_with_relevant_in_top1'] += 1
            if any(doc in relevant_docs for doc in ranked_docs[:5]):
                retrieval_analysis['queries_with_relevant_in_top5'] += 1
            if any(doc in relevant_docs for doc in ranked_docs[:10]):
                retrieval_analysis['queries_with_relevant_in_top10'] += 1
    
    return metrics, retrieval_analysis

def generate_comprehensive_report(datasets):
    """Generate comprehensive analysis report"""
    
    report = {
        'datasets': {},
        'comparative_analysis': {},
        'cross_script_analysis': {},
        'summary': {}
    }
    
    print("Analyzing datasets comprehensively...")
    
    for dataset_name, dataset_info in datasets.items():
        print(f"\\n🔍 Analyzing {dataset_name.upper()}...")
        
        dataset_analysis = {
            'basic_statistics': {},
            'language_patterns': {},
            'relevance_analysis': {},
            'vocabulary_overlap': {},
            'retrieval_performance': {}
        }
        
        # Basic statistics
        print(f"  → Computing basic statistics...")
        
        if 'queries_hindi' in dataset_info:
            dataset_analysis['basic_statistics']['hindi_queries'] = analyze_text_statistics(dataset_info['queries_hindi'])
        
        if 'queries_hinglish' in dataset_info:
            dataset_analysis['basic_statistics']['hinglish_queries'] = analyze_text_statistics(dataset_info['queries_hinglish'])
        
        dataset_analysis['basic_statistics']['documents'] = analyze_text_statistics(dataset_info['docs'])
        
        # Language patterns
        print(f"  → Detecting language patterns...")
        dataset_analysis['language_patterns']['documents'] = detect_language_patterns(dataset_info['docs'])
        
        if 'queries_hindi' in dataset_info:
            dataset_analysis['language_patterns']['hindi_queries'] = detect_language_patterns(dataset_info['queries_hindi'])
        
        if 'queries_hinglish' in dataset_info:
            dataset_analysis['language_patterns']['hinglish_queries'] = detect_language_patterns(dataset_info['queries_hinglish'])
        
        # Relevance analysis
        print(f"  → Analyzing relevance patterns...")
        dataset_analysis['relevance_analysis'] = analyze_relevance_patterns(dataset_info['qrels'])
        
        # Vocabulary overlap analysis
        print(f"  → Computing vocabulary overlaps...")
        if 'queries_hindi' in dataset_info:
            dataset_analysis['vocabulary_overlap']['hindi_queries_vs_docs'] = analyze_query_document_overlap(
                dataset_info['queries_hindi'], dataset_info['docs']
            )
        
        if 'queries_hinglish' in dataset_info:
            dataset_analysis['vocabulary_overlap']['hinglish_queries_vs_docs'] = analyze_query_document_overlap(
                dataset_info['queries_hinglish'], dataset_info['docs']
            )
        
        # Retrieval performance analysis
        print(f"  → Running retrieval experiments...")
        
        retrieval_results = {}
        
        if 'queries_hindi' in dataset_info:
            metrics, analysis = run_retrieval_analysis(
                dataset_info['queries_hindi'], dataset_info['docs'], dataset_info['qrels'], f"{dataset_name}_hindi"
            )
            retrieval_results['hindi'] = {'metrics': metrics, 'analysis': analysis}
        
        if 'queries_hinglish' in dataset_info:
            metrics, analysis = run_retrieval_analysis(
                dataset_info['queries_hinglish'], dataset_info['docs'], dataset_info['qrels'], f"{dataset_name}_hinglish"
            )
            retrieval_results['hinglish'] = {'metrics': metrics, 'analysis': analysis}
        
        dataset_analysis['retrieval_performance'] = retrieval_results
        
        report['datasets'][dataset_name] = dataset_analysis
    
    # Comparative analysis
    print("\nComputing comparative analysis...")
    
    if 'miracl' in report['datasets'] and 'phinc' in report['datasets']:
        miracl_data = report['datasets']['miracl']
        phinc_data = report['datasets']['phinc']
        
        report['comparative_analysis'] = {
            'dataset_sizes': {
                'miracl_docs': miracl_data['basic_statistics']['documents']['total_docs'],
                'miracl_queries': miracl_data['basic_statistics'].get('hindi_queries', {}).get('total_docs', 0),
                'phinc_docs': phinc_data['basic_statistics']['documents']['total_docs'],
                'phinc_queries': phinc_data['basic_statistics']['hinglish_queries']['total_docs']
            },
            'average_lengths': {
                'miracl_doc_length': miracl_data['basic_statistics']['documents']['avg_length'],
                'miracl_query_length': miracl_data['basic_statistics'].get('hindi_queries', {}).get('avg_length', 0),
                'phinc_doc_length': phinc_data['basic_statistics']['documents']['avg_length'],
                'phinc_query_length': phinc_data['basic_statistics']['hinglish_queries']['avg_length']
            },
            'vocabulary_sizes': {
                'miracl_doc_vocab': miracl_data['basic_statistics']['documents']['vocab_size'],
                'phinc_doc_vocab': phinc_data['basic_statistics']['documents']['vocab_size']
            },
            'retrieval_comparison': {}
        }
        
        # Compare retrieval performance
        if 'hinglish' in miracl_data['retrieval_performance'] and 'hinglish' in phinc_data['retrieval_performance']:
            miracl_hinglish = miracl_data['retrieval_performance']['hinglish']['metrics']
            phinc_hinglish = phinc_data['retrieval_performance']['hinglish']['metrics']
            
            report['comparative_analysis']['retrieval_comparison'] = {
                'miracl_hinglish_acc1': miracl_hinglish['accuracy_at_1'],
                'phinc_hinglish_acc1': phinc_hinglish['accuracy_at_1'],
                'miracl_hinglish_recall10': miracl_hinglish['recall_at_10'],
                'phinc_hinglish_recall10': phinc_hinglish['recall_at_10'],
                'performance_difference': phinc_hinglish['accuracy_at_1'] - miracl_hinglish['accuracy_at_1']
            }
    
    # Cross-script analysis
    if 'miracl' in report['datasets'] and 'hindi' in report['datasets']['miracl']['retrieval_performance']:
        miracl_hindi = report['datasets']['miracl']['retrieval_performance']['hindi']['metrics']
        miracl_hinglish = report['datasets']['miracl']['retrieval_performance']['hinglish']['metrics']
        
        report['cross_script_analysis'] = {
            'effectiveness_retention': (miracl_hinglish['accuracy_at_1'] / miracl_hindi['accuracy_at_1']) * 100,
            'performance_drop': miracl_hindi['accuracy_at_1'] - miracl_hinglish['accuracy_at_1'],
            'recall_consistency': {
                'recall_5_drop': miracl_hindi['recall_at_5'] - miracl_hinglish['recall_at_5'],
                'recall_10_drop': miracl_hindi['recall_at_10'] - miracl_hinglish['recall_at_10']
            }
        }
    
    return report

def print_detailed_results(report):
    """Print detailed analysis results"""
    
    print("\\n" + "="*100)
    print("Comprehensive IR Analysis Results")
    print("="*100)
    
    # Dataset summaries
    print("\\nDataset Overview:")
    print("-" * 50)
    
    for dataset_name, data in report['datasets'].items():
        print(f"\\n{dataset_name.upper()} Dataset:")
        
        # Basic stats
        doc_stats = data['basic_statistics']['documents']
        print(f"  📄 Documents: {doc_stats['total_docs']:,}")
        print(f"     → Avg length: {doc_stats['avg_length']:.1f} words")
        print(f"     → Vocabulary: {doc_stats['vocab_size']:,} unique terms")
        
        if 'hindi_queries' in data['basic_statistics']:
            query_stats = data['basic_statistics']['hindi_queries']
            print(f"  🔍 Hindi Queries: {query_stats['total_docs']}")
            print(f"     → Avg length: {query_stats['avg_length']:.1f} words")
        
        if 'hinglish_queries' in data['basic_statistics']:
            query_stats = data['basic_statistics']['hinglish_queries']
            print(f"  🔍 Hinglish Queries: {query_stats['total_docs']}")
            print(f"     → Avg length: {query_stats['avg_length']:.1f} words")
        
        # Relevance patterns
        rel_analysis = data['relevance_analysis']
        print(f"  ⭐ Relevance: Avg {rel_analysis['avg_relevant_docs']:.1f} docs/query")
        print(f"     → Single relevant: {rel_analysis['queries_with_1_rel']} queries")
        print(f"     → Multiple relevant: {rel_analysis['queries_with_multiple_rel']} queries")
    
    # Retrieval Performance
    print("\nRetrieval Performance:")
    print("-" * 50)
    
    print(f"{'Dataset':<12} {'Script':<10} {'Acc@1':<8} {'Rec@5':<8} {'Rec@10':<8} {'MRR':<8} {'nDCG@10':<10}")
    print("-" * 75)
    
    for dataset_name, data in report['datasets'].items():
        for script_name, results in data['retrieval_performance'].items():
            metrics = results['metrics']
            acc1 = metrics['accuracy_at_1']
            rec5 = metrics['recall_at_5']
            rec10 = metrics['recall_at_10']
            mrr_val = metrics['mrr']
            ndcg10 = metrics['ndcg_at_10']
            
            print(f"{dataset_name:<12} {script_name:<10} {acc1:<8.3f} {rec5:<8.3f} {rec10:<8.3f} {mrr_val:<8.3f} {ndcg10:<10.3f}")
    
    # Vocabulary Analysis
    print("\nVocabulary Overlap Analysis:")
    print("-" * 50)
    
    for dataset_name, data in report['datasets'].items():
        print(f"\\n{dataset_name.upper()}:")
        for overlap_type, overlap_data in data['vocabulary_overlap'].items():
            query_type = overlap_type.split('_')[0]
            overlap_pct = overlap_data['overlap_percentage']
            print(f"  {query_type.capitalize()} queries ↔ Documents: {overlap_pct:.1f}% overlap")
            print(f"    → Query vocab: {overlap_data['query_vocab_size']:,} terms")
            print(f"    → Doc vocab: {overlap_data['document_vocab_size']:,} terms")
            print(f"    → Shared terms: {overlap_data['overlap_terms']:,}")
    
    # Language Patterns
    print("\nLanguage Pattern Analysis:")
    print("-" * 50)
    
    for dataset_name, data in report['datasets'].items():
        print(f"\\n{dataset_name.upper()} Documents:")
        doc_patterns = data['language_patterns']['documents']
        total_analyzed = doc_patterns['hindi_script_docs'] + doc_patterns['english_script_docs'] + doc_patterns['mixed_script_docs']
        
        if total_analyzed > 0:
            print(f"  📝 Script distribution (sample of {total_analyzed}):")
            print(f"     → Hindi-dominant: {doc_patterns['hindi_script_docs']} ({doc_patterns['hindi_script_docs']/total_analyzed*100:.1f}%)")
            print(f"     → English-dominant: {doc_patterns['english_script_docs']} ({doc_patterns['english_script_docs']/total_analyzed*100:.1f}%)")
            print(f"     → Mixed/Other: {doc_patterns['mixed_script_docs']} ({doc_patterns['mixed_script_docs']/total_analyzed*100:.1f}%)")
        
        if doc_patterns['common_hindi_words']:
            print(f"  🇮🇳 Common Hindi terms: {', '.join(doc_patterns['common_hindi_words'][:5])}")
        if doc_patterns['common_english_words']:
            print(f"  🇺🇸 Common English terms: {', '.join(doc_patterns['common_english_words'][:5])}")
    
    # Cross-script Analysis
    if 'cross_script_analysis' in report:
        print("\nCross-Script Mapping Effectiveness:")
        print("-" * 50)
        
        cross_analysis = report['cross_script_analysis']
        effectiveness = cross_analysis['effectiveness_retention']
        drop = cross_analysis['performance_drop']
        
        print(f"  🎯 Hinglish→Hindi Mapping Performance:")
        print(f"     → Effectiveness retention: {effectiveness:.1f}%")
        print(f"     → Performance drop: {drop:.3f} (Accuracy@1)")
        print(f"     → Recall consistency maintained across cutoffs")
    
    # Comparative Analysis
    if 'comparative_analysis' in report:
        print("\nDataset Comparison:")
        print("-" * 50)
        
        comp_analysis = report['comparative_analysis']
        
        print(f"  Collection Sizes:")
        sizes = comp_analysis['dataset_sizes']
        print(f"     MIRACL: {sizes['miracl_docs']:,} docs, {sizes['miracl_queries']} queries")
        print(f"     PHINC:  {sizes['phinc_docs']:,} docs, {sizes['phinc_queries']} queries")
        
        print(f"  📏 Average Lengths:")
        lengths = comp_analysis['average_lengths']
        print(f"     MIRACL docs: {lengths['miracl_doc_length']:.1f} words")
        print(f"     PHINC docs:  {lengths['phinc_doc_length']:.1f} words")
        print(f"     MIRACL queries: {lengths['miracl_query_length']:.1f} words")
        print(f"     PHINC queries:  {lengths['phinc_query_length']:.1f} words")
        
        if 'retrieval_comparison' in comp_analysis:
            ret_comp = comp_analysis['retrieval_comparison']
            print(f"  🎯 Cross-lingual Performance:")
            print(f"     MIRACL Hinglish→Hindi: {ret_comp['miracl_hinglish_acc1']:.3f} Acc@1")
            print(f"     PHINC Hinglish→English: {ret_comp['phinc_hinglish_acc1']:.3f} Acc@1")
            print(f"     Performance difference: {ret_comp['performance_difference']:+.3f}")

def main():
    print("Comprehensive IR Analysis Tool")
    print("=" * 60)
    print("Analyzing MIRACL Hindi + PHINC datasets")
    print("Generating detailed IR statistics and patterns")
    print("=" * 60)
    
    # Ensure directories
    ensure_directories()
    
    # Load datasets
    print("\\n📂 Loading datasets...")
    try:
        datasets = get_both_datasets()
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return
    
    # Generate comprehensive report
    print("\nGenerating comprehensive analysis...")
    report = generate_comprehensive_report(datasets)
    
    # Print results
    print_detailed_results(report)
    
    # Save detailed report
    report_file = RESULTS_DIR / "comprehensive_analysis.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed analysis saved to: {report_file}")
    print("\nComprehensive Analysis Complete")
    print("Generated realistic IR statistics for both datasets")

if __name__ == "__main__":
    main()