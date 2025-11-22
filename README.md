# Cross-Script Information Retrieval System

A comprehensive information retrieval system implementing cross-script and cross-lingual search capabilities for Hindi-English multilingual scenarios.

## Overview

This system demonstrates practical cross-script information retrieval using term mapping approaches rather than machine translation. It supports two distinct multilingual retrieval scenarios:

1. **MIRACL Dataset**: Hinglish queries on Hindi documents (cross-script mapping)
2. **PHINC Dataset**: Hinglish queries on English documents (cross-lingual retrieval)

## System Architecture

- **Cross-script mapping**: Dictionary-based term substitution for Hinglish→Hindi
- **BM25 retrieval**: Efficient lexical matching for both datasets
- **Unified evaluation**: Comprehensive metrics across multiple IR scenarios

## Datasets

### MIRACL Hindi
- **Documents**: 8,000 Hindi Wikipedia passages
- **Queries**: 107 Hindi queries with Hinglish variants
- **Task**: Cross-script retrieval with term mapping

### PHINC
- **Documents**: 1,000 English documents  
- **Queries**: 1,000 Hinglish social media queries
- **Task**: Direct cross-lingual retrieval

## Performance Results

| Dataset | Script   | Accuracy@1 | Recall@10 | MRR   |
|---------|----------|------------|-----------|-------|
| MIRACL  | Hindi    | 0.336      | 0.654     | 0.438 |
| MIRACL  | Hinglish | 0.290      | 0.654     | 0.420 |
| PHINC   | Hinglish | 0.566      | 0.752     | 0.635 |

## Key Findings

- Cross-script mapping retains 86% effectiveness on MIRACL
- Direct cross-lingual retrieval achieves 57% accuracy on PHINC
- Term mapping approach avoids translation overhead
- Query characteristics significantly impact retrieval effectiveness

## Usage

### Run Comprehensive Evaluation
```bash
python final_evaluation.py
```

### Generate Detailed Analysis
```bash
python comprehensive_analysis.py
```

### View Quick Results Summary
```bash
python quick_results.py
```

## Requirements

See `requirements.txt` for dependencies.

## Results

Detailed evaluation results and analysis are available in:
- `FINAL_RESULTS.md` - Comprehensive results and analysis
- `results/` - JSON output files with detailed metrics

## Technical Implementation

The system uses a modular architecture with:
- `src/data_loader.py` - Unified dataset loading
- `src/bm25_index.py` - BM25 retrieval implementation
- `src/evaluate_fixed.py` - IR evaluation metrics
- `src/config.py` - System configuration

This implementation demonstrates practical viability of cross-script information retrieval without machine translation infrastructure.