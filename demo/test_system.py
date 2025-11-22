#!/usr/bin/env python3
"""
Simple demo for the cross-script IR system
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import get_config_summary, ensure_directories

def main():
    print("Cross-Script Information Retrieval System Demo")
    print("=" * 50)
    
    # Initialize
    ensure_directories()
    
    # Show configuration
    config = get_config_summary()
    print("\\n📋 System Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Test data loading
    print("\nTesting data loading...")
    try:
        from data_loader import get_both_datasets
        datasets = get_both_datasets()
        
        print("Data loading successful!")
        print("\nAvailable datasets:")
        for name, data in datasets.items():
            print(f"  {name.upper()}: {data['description']}")
            if 'queries_hindi' in data:
                print(f"    Hindi queries: {len(data['queries_hindi'])}")
            if 'queries_hinglish' in data:
                print(f"    Hinglish queries: {len(data['queries_hinglish'])}")
            print(f"    Documents: {len(data['docs'])}")
        
        # Show sample queries
        print("\nSample queries:")
        
        # MIRACL samples
        if 'miracl' in datasets:
            miracl_data = datasets['miracl']
            if miracl_data['queries_hindi']:
                sample_qid = list(miracl_data['queries_hindi'].keys())[0]
                hindi_query = miracl_data['queries_hindi'][sample_qid]
                hinglish_query = miracl_data['queries_hinglish'][sample_qid]
                print(f"  MIRACL - Hindi: {hindi_query}")
                print(f"  MIRACL - Hinglish: {hinglish_query}")
        
        # PHINC samples
        if 'phinc' in datasets:
            phinc_data = datasets['phinc']
            if phinc_data['queries_hinglish']:
                sample_qid = list(phinc_data['queries_hinglish'].keys())[0]
                hinglish_query = phinc_data['queries_hinglish'][sample_qid]
                print(f"  PHINC - Hinglish: {hinglish_query}")
        
        print("\nDemo completed successfully!")
        print("\\nNext steps:")
        print("  • Run 'python main_evaluation.py' for full evaluation")
        print("  • Check 'results/' folder for outputs")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\\nTroubleshooting:")
        print("  • Ensure data files are in data/miracl/ and data/ directories")
        print("  • Check if PHINC.csv is available")
        print("  • Verify all dependencies are installed")

if __name__ == "__main__":
    main()