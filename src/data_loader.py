"""
Unified data loading for both MIRACL Hindi and PHINC datasets
Handles cross-script mapping and IR data preparation
"""
import pandas as pd
import numpy as np
import json
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import sys

# Import config handling relative imports
try:
    from .config import *
except ImportError:
    from config import *

class CrossScriptDataLoader:
    """Unified data loader for cross-script IR experiments"""
    
    def __init__(self):
        self.hinglish_mapping = HINGLISH_TO_HINDI_MAPPING
        
    def apply_cross_script_mapping(self, hinglish_text: str) -> str:
        """Apply term-level mapping from Hinglish to Hindi"""
        words = hinglish_text.lower().split()
        mapped_words = []
        
        for word in words:
            clean_word = word.strip('?।.,!;:')
            if clean_word in self.hinglish_mapping:
                mapped_words.append(self.hinglish_mapping[clean_word])
            else:
                mapped_words.append(word)
        
        return ' '.join(mapped_words)
    
    # MIRACL Dataset Functions
    def load_miracl_queries(self) -> Dict[str, str]:
        """Load MIRACL Hindi queries"""
        try:
            df = pd.read_csv(MIRACL_QUERIES, sep='\t', names=['qid', 'query'])
            return dict(zip(df['qid'], df['query']))
        except Exception as e:
            raise FileNotFoundError(f"Could not load MIRACL queries: {e}")
    
    def load_miracl_qrels(self) -> Dict[str, List[str]]:
        """Load MIRACL relevance judgments"""
        try:
            df = pd.read_csv(MIRACL_QRELS, sep='\t', names=['qid', 'iter', 'docid', 'rel'])
            qrels = {}
            for _, row in df.iterrows():
                if row['rel'] > 0:  # Only positive relevance
                    qid = str(row['qid'])
                    docid = str(row['docid'])
                    if qid not in qrels:
                        qrels[qid] = []
                    qrels[qid].append(docid)
            return qrels
        except Exception as e:
            raise FileNotFoundError(f"Could not load MIRACL qrels: {e}")
    
    def load_miracl_docs(self, limit: Optional[int] = MIRACL_DOC_LIMIT) -> Dict[str, str]:
        """Load MIRACL Hindi documents"""
        docs = {}
        doc_files = list(MIRACL_DIR.glob("docs*.jsonl.gz"))
        
        if not doc_files:
            raise FileNotFoundError(f"No MIRACL document files found in {MIRACL_DIR}")
        
        count = 0
        for doc_file in doc_files:
            try:
                with gzip.open(doc_file, 'rt', encoding='utf-8') as f:
                    for line in f:
                        if limit and count >= limit:
                            break
                        doc = json.loads(line)
                        docs[doc['docid']] = doc['text']
                        count += 1
            except Exception as e:
                print(f"Warning: Error reading {doc_file}: {e}")
                continue
                
        print(f"Loaded {len(docs)} MIRACL documents")
        return docs
    
    def filter_miracl_queries_with_docs(self, queries: Dict[str, str], 
                                      qrels: Dict[str, List[str]], 
                                      docs: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        """Filter queries to only those with relevant documents in the corpus"""
        filtered_queries = {}
        filtered_qrels = {}
        
        for qid, query in queries.items():
            if qid in qrels:
                relevant_docs = [doc for doc in qrels[qid] if doc in docs]
                if relevant_docs:
                    filtered_queries[qid] = query
                    filtered_qrels[qid] = relevant_docs
        
        print(f"Filtered to {len(filtered_queries)} queries with relevant documents")
        return filtered_queries, filtered_qrels
    
    def get_miracl_data(self) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]], Dict[str, str]]:
        """Get complete MIRACL dataset with cross-script mapping"""
        # Load base data
        queries_hindi = self.load_miracl_queries()
        qrels = self.load_miracl_qrels()
        docs = self.load_miracl_docs()
        
        # Filter to available documents
        queries_hindi, qrels = self.filter_miracl_queries_with_docs(queries_hindi, qrels, docs)
        
        # Create Hinglish version through cross-script mapping
        queries_hinglish = {
            qid: self.apply_cross_script_mapping(query) 
            for qid, query in queries_hindi.items()
        }
        
        return queries_hindi, queries_hinglish, qrels, docs
    
    # PHINC Dataset Functions  
    def load_phinc_raw(self) -> pd.DataFrame:
        """Load raw PHINC dataset"""
        try:
            df = pd.read_csv(PHINC_CSV)
            df = df.dropna()  # Remove any rows with missing values
            print(f"Loaded PHINC dataset with {len(df)} pairs")
            return df
        except Exception as e:
            raise FileNotFoundError(f"Could not load PHINC data: {e}")
    
    def prepare_phinc_ir_data(self, test_size: int = PHINC_TEST_SIZE, random_seed: int = 42) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, str]]:
        """Prepare PHINC data for IR evaluation"""
        df = self.load_phinc_raw()
        
        # Sample for evaluation
        if len(df) > test_size:
            df = df.sample(n=test_size, random_state=random_seed).reset_index(drop=True)
        
        # Create IR data structure
        queries_hinglish = {}
        docs_english = {}
        qrels = {}
        
        for idx, row in df.iterrows():
            qid = f"Q{idx}"
            docid = f"D{idx}"
            
            # Hinglish query -> English document
            queries_hinglish[qid] = row['Sentence']  # Hinglish sentence
            docs_english[docid] = row['English_Translation']  # English translation
            qrels[qid] = [docid]  # Perfect relevance (1-to-1 mapping)
        
        print(f"Prepared PHINC IR data: {len(queries_hinglish)} queries, {len(docs_english)} documents")
        return queries_hinglish, qrels, docs_english
    
    def get_phinc_data(self) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, str]]:
        """Get PHINC dataset for cross-lingual IR evaluation"""
        return self.prepare_phinc_ir_data()

# Convenience functions
def get_miracl_data() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]], Dict[str, str]]:
    """Get MIRACL dataset (Hindi documents, Hindi+Hinglish queries)"""
    loader = CrossScriptDataLoader()
    return loader.get_miracl_data()

def get_phinc_data() -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, str]]:
    """Get PHINC dataset (English documents, Hinglish queries)"""
    loader = CrossScriptDataLoader()
    return loader.get_phinc_data()

def get_both_datasets() -> Dict:
    """Get both datasets for comprehensive evaluation"""
    loader = CrossScriptDataLoader()
    
    # MIRACL data
    miracl_hindi, miracl_hinglish, miracl_qrels, miracl_docs = loader.get_miracl_data()
    
    # PHINC data
    phinc_queries, phinc_qrels, phinc_docs = loader.get_phinc_data()
    
    return {
        'miracl': {
            'queries_hindi': miracl_hindi,
            'queries_hinglish': miracl_hinglish,
            'qrels': miracl_qrels,
            'docs': miracl_docs,
            'description': 'MIRACL Hindi: Cross-script queries on Hindi documents'
        },
        'phinc': {
            'queries_hinglish': phinc_queries,
            'qrels': phinc_qrels,
            'docs': phinc_docs,
            'description': 'PHINC: Hinglish queries on English documents'
        }
    }

# Legacy compatibility functions
def load_queries(file_path="data/miracl/topics_train.tsv"):
    """Legacy function for backward compatibility"""
    loader = CrossScriptDataLoader()
    return loader.load_miracl_queries()

def load_qrels(file_path="data/miracl/qrels_train.tsv"):
    """Legacy function for backward compatibility"""
    loader = CrossScriptDataLoader()
    return loader.load_miracl_qrels()

def load_docs(limit=MIRACL_DOC_LIMIT, file_pattern="data/miracl/docs*.jsonl.gz"):
    """Legacy function for backward compatibility"""
    loader = CrossScriptDataLoader()
    return loader.load_miracl_docs(limit)

def filter_queries_with_docs(queries, qrels, docs):
    """Legacy function for backward compatibility"""
    loader = CrossScriptDataLoader()
    return loader.filter_miracl_queries_with_docs(queries, qrels, docs)