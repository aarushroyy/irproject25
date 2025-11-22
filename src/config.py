"""
Unified Configuration for Cross-Script Information Retrieval System
Supports both MIRACL Hindi and PHINC datasets
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
MIRACL_DIR = DATA_DIR / "miracl"
PHINC_DIR = DATA_DIR / "phinc"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset files
MIRACL_QUERIES = MIRACL_DIR / "topics_train.tsv"
MIRACL_QRELS = MIRACL_DIR / "qrels_train.tsv"
MIRACL_DOCS_PATTERN = "docs*.jsonl.gz"

PHINC_CSV = DATA_DIR / "PHINC.csv"

# Processing settings for cross-script mapping
HINGLISH_TO_HINDI_MAPPING = {
    # Core terms
    'bharat': 'भारत', 'india': 'भारत',
    'mein': 'में', 'ka': 'का', 'ki': 'की', 'ke': 'के',
    'hai': 'है', 'hain': 'हैं', 'tha': 'था', 'thi': 'थी',
    'kya': 'क्या', 'kaun': 'कौन', 'kahan': 'कहां', 'kab': 'कब',
    'kaise': 'कैसे', 'kyun': 'क्यों', 'kitna': 'कितना',
    
    # Cultural terms
    'sanskriti': 'संस्कृति', 'culture': 'संस्कृति',
    'dharma': 'धर्म', 'religion': 'धर्म',
    'sangeet': 'संगीत', 'music': 'संगीत',
    'kala': 'कला', 'art': 'कला',
    'nritya': 'नृत्य', 'dance': 'नृत्य',
    'shastriya': 'शास्त्रीय', 'classical': 'शास्त्रीय',
    'sahitya': 'साहित्य', 'literature': 'साहित्य',
    'darshan': 'दर्शन', 'philosophy': 'दर्शन',
    'bhasha': 'भाषा', 'language': 'भाषा',
    'parampara': 'परंपरा', 'tradition': 'परंपरा',
    'mandir': 'मंदिर', 'temple': 'मंदिर',
}

# Model configurations
DENSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# BM25 parameters
BM25_K1 = 1.2
BM25_B = 0.75

# Hybrid search parameters
RRF_K = 60

# Evaluation settings
EVAL_METRICS = ["accuracy_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]
DEFAULT_K = 10

# Data limits for experiments
MIRACL_DOC_LIMIT = 8000
PHINC_TEST_SIZE = 1000

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [DATA_DIR, MIRACL_DIR, PHINC_DIR, RESULTS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_config_summary():
    """Return a summary of current configuration"""
    return {
        "datasets": ["MIRACL Hindi", "PHINC Hinglish-English"],
        "approaches": ["BM25", "Dense", "Hybrid"],
        "cross_script_method": "Term mapping (not translation)",
        "miracl_doc_limit": MIRACL_DOC_LIMIT,
        "phinc_test_size": PHINC_TEST_SIZE,
        "dense_model": DENSE_MODEL_NAME,
        "bm25_params": {"k1": BM25_K1, "b": BM25_B},
        "rrf_k": RRF_K
    }

if __name__ == "__main__":
    ensure_directories()
    print("✅ Configuration loaded successfully!")
    config = get_config_summary()
    for key, value in config.items():
        print(f"  {key}: {value}")