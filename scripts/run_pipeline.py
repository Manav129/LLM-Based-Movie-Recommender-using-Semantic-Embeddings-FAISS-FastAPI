"""
Pipeline Runner Script
Orchestrates the complete offline pipeline for building the recommendation system.

This script runs everything in sequence:
1. Load and clean data (movies + ratings)
2. Generate embeddings using SentenceTransformers
3. Build FAISS index for fast search
4. Save all artifacts to disk

Run this script once to set up everything, then start the API server.
"""

import sys
import time
from pathlib import Path

# Add src directory to Python path so we can import our modules
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_all_data, print_data_statistics
from src.build_embeddings import build_and_save_embeddings
from src.build_faiss import build_and_save_faiss_index


# ============================================
# PIPELINE EXECUTION
# ============================================

def run_pipeline():
    """
    Run the complete offline pipeline.
    
    This function executes all steps needed to prepare the recommendation system:
    1. Data loading and cleaning
    2. Embedding generation
    3. FAISS index building
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("Starting pipeline...")
    start_time = time.time()
    
    try:
        step_start = time.time()
        movies_df, ratings_df = load_all_data()
        step_time = time.time() - step_start
        print(f"Step 1/3: Loaded {len(movies_df)} movies, {len(ratings_df)} ratings ({step_time:.1f}s)")
        
        step_start = time.time()
        embeddings = build_and_save_embeddings(movies_df)
        step_time = time.time() - step_start
        print(f"Step 2/3: Built embeddings ({step_time:.1f}s)")
        
        step_start = time.time()
        index, normalized_embeddings = build_and_save_faiss_index(embeddings)
        step_time = time.time() - step_start
        print(f"Step 3/3: Built FAISS index ({step_time:.1f}s)")
        
        total_time = time.time() - start_time
        print(f"\nPipeline completed successfully.")
        print(f"Movies: {len(movies_df)}, Ratings: {len(ratings_df)}, Total time: {total_time:.1f}s")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return False
    except ValueError as e:
        print(f"\nERROR: {e}")
        return False
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================
# VALIDATION FUNCTION
# ============================================

def validate_artifacts():
    """
    Validate that all required artifacts exist and are valid.
    
    This is useful to check if the pipeline ran successfully.
    
    Returns:
        bool: True if all artifacts are valid, False otherwise
    """
    from src.config import EMBEDDINGS_FILE, MOVIE_METADATA_FILE, FAISS_INDEX_FILE
    
    print("\n" + "=" * 70)
    print("🔍 VALIDATING ARTIFACTS")
    print("=" * 70)
    
    all_valid = True
    
    # Check embeddings file
    if EMBEDDINGS_FILE.exists():
        print(f"\n✅ Embeddings file exists: {EMBEDDINGS_FILE.name}")
        size_mb = EMBEDDINGS_FILE.stat().st_size / 1024 / 1024
        print(f"   Size: {size_mb:.2f} MB")
    else:
        print(f"\n❌ Embeddings file missing: {EMBEDDINGS_FILE}")
        all_valid = False
    
    # Check metadata file
    if MOVIE_METADATA_FILE.exists():
        print(f"\n✅ Metadata file exists: {MOVIE_METADATA_FILE.name}")
        size_mb = MOVIE_METADATA_FILE.stat().st_size / 1024 / 1024
        print(f"   Size: {size_mb:.2f} MB")
    else:
        print(f"\n❌ Metadata file missing: {MOVIE_METADATA_FILE}")
        all_valid = False
    
    # Check FAISS index file
    if FAISS_INDEX_FILE.exists():
        print(f"\n✅ FAISS index exists: {FAISS_INDEX_FILE.name}")
        size_mb = FAISS_INDEX_FILE.stat().st_size / 1024 / 1024
        print(f"   Size: {size_mb:.2f} MB")
    else:
        print(f"\n❌ FAISS index missing: {FAISS_INDEX_FILE}")
        all_valid = False
    
    print("\n" + "=" * 70)
    
    if all_valid:
        print("✅ All artifacts are present and valid!")
        print("=" * 70)
        return True
    else:
        print("❌ Some artifacts are missing. Run the pipeline:")
        print("   python scripts/run_pipeline.py")
        print("=" * 70)
        return False


# ============================================
# COMMAND LINE INTERFACE
# ============================================

def main():
    """
    Main entry point for the pipeline script.
    
    Supports command line arguments:
    - (no args): Run the pipeline
    - validate: Validate artifacts exist
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Movie Recommendation System - Offline Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py           # Run the full pipeline
  python scripts/run_pipeline.py validate  # Validate artifacts exist
        """
    )
    
    parser.add_argument(
        'action',
        nargs='?',
        default='run',
        choices=['run', 'validate'],
        help='Action to perform (default: run)'
    )
    
    args = parser.parse_args()
    
    if args.action == 'validate':
        # Validate artifacts
        success = validate_artifacts()
        sys.exit(0 if success else 1)
    else:
        # Run pipeline
        success = run_pipeline()
        sys.exit(0 if success else 1)


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
