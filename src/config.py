"""
Configuration file for the Movie Recommendation System.
This file centralizes all paths, settings, and hyperparameters.
"""

import os
from pathlib import Path

# ============================================
# PROJECT PATHS
# ============================================

# Get the root directory of the project (parent of 'src' folder)
# __file__ gives the path to this config.py file
# .parent goes up one level to 'src' folder
# .parent again goes up to the root project folder
ROOT_DIR = Path(__file__).parent.parent

# Data folder - where raw CSV files are stored
DATA_DIR = ROOT_DIR / "data"
MOVIES_CSV = DATA_DIR / "movies.csv"
RATINGS_CSV = DATA_DIR / "ratings.csv"

# Artifacts folder - where we save generated files (embeddings, FAISS index)
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
EMBEDDINGS_FILE = ARTIFACTS_DIR / "movie_embeddings.npy"  # NumPy array file
FAISS_INDEX_FILE = ARTIFACTS_DIR / "faiss_index.index"    # FAISS index file
MOVIE_METADATA_FILE = ARTIFACTS_DIR / "movie_metadata.pkl"  # Pickled DataFrame

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# MODEL SETTINGS
# ============================================

# Sentence Transformer model from HuggingFace
# 'all-MiniLM-L6-v2' is a lightweight, fast model that creates 384-dimensional embeddings
# Good balance between speed and quality for beginners
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Embedding dimension (output size of the model)
EMBEDDING_DIM = 384


# ============================================
# RECOMMENDATION SETTINGS
# ============================================

# Default number of recommendations to return
DEFAULT_K = 10

# Minimum rating threshold for collaborative filtering
# Only consider ratings >= this value as "positive" ratings
MIN_RATING_THRESHOLD = 3.5


# ============================================
# API SETTINGS
# ============================================

# FastAPI server configuration
API_HOST = "127.0.0.1"  # localhost
API_PORT = 8000

# CORS settings (allow frontend to call backend from different origin)
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5500",  # For Live Server VS Code extension
    "http://127.0.0.1:5500",
]


# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_config_summary():
    """
    Returns a summary of the configuration settings.
    Useful for debugging and logging.
    """
    return {
        "root_dir": str(ROOT_DIR),
        "data_dir": str(DATA_DIR),
        "artifacts_dir": str(ARTIFACTS_DIR),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "default_k": DEFAULT_K,
    }


if __name__ == "__main__":
    # Test the configuration by printing paths
    print("=" * 50)
    print("CONFIGURATION SUMMARY")
    print("=" * 50)
    
    config = get_config_summary()
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    
    print("\n" + "=" * 50)
    print("FILE CHECKS")
    print("=" * 50)
    print(f"Data directory exists: {DATA_DIR.exists()}")
    print(f"Artifacts directory exists: {ARTIFACTS_DIR.exists()}")
    print(f"movies.csv exists: {MOVIES_CSV.exists()}")
    print(f"ratings.csv exists: {RATINGS_CSV.exists()}")
