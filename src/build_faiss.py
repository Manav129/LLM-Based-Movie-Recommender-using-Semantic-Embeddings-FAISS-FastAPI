"""
FAISS Index Builder Module
Builds a FAISS index for fast similarity search on movie embeddings.

What is FAISS?
- FAISS = Facebook AI Similarity Search (library by Meta)
- Allows searching through millions of vectors in milliseconds
- Much faster than computing similarity with every vector manually
- Think of it like a "search engine" for embeddings
"""

import numpy as np
import faiss
from pathlib import Path
from src.config import FAISS_INDEX_FILE, EMBEDDING_DIM


# ============================================
# BUILD FAISS INDEX
# ============================================

def build_faiss_index(embeddings):
    """
    Build a FAISS index from movie embeddings.
    
    We use IndexFlatL2 which:
    - "Flat" = brute force search (checks all vectors, but very fast in FAISS)
    - "L2" = uses L2 distance (Euclidean distance)
    - Perfect for small to medium datasets (up to ~1 million vectors)
    - Exact search (not approximate, returns truly closest matches)
    
    For larger datasets, you could use IndexIVFFlat or IndexHNSW (faster but approximate).
    
    Args:
        embeddings: Numpy array of shape (num_movies, embedding_dim)
    
    Returns:
        faiss.Index: FAISS index ready for searching
    """
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    if normalized_embeddings.dtype != np.float32:
        normalized_embeddings = normalized_embeddings.astype(np.float32)
    index.add(normalized_embeddings)
    return index, normalized_embeddings


# ============================================
# SAVE FAISS INDEX
# ============================================

def save_faiss_index(index):
    """
    Save FAISS index to disk.
    
    FAISS has its own binary format (.index file) which is:
    - Very compact
    - Fast to load
    - Preserves the index structure
    
    Args:
        index: FAISS index object
    """
    faiss.write_index(index, str(FAISS_INDEX_FILE))


# ============================================
# LOAD FAISS INDEX
# ============================================

def load_faiss_index():
    """
    Load FAISS index from disk.
    
    This is used by the recommendation system to load the pre-built index
    instead of building it every time.
    
    Returns:
        faiss.Index: Loaded FAISS index
    
    Raises:
        FileNotFoundError: If index hasn't been built yet
    """
    if not FAISS_INDEX_FILE.exists():
        raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_FILE}")
    index = faiss.read_index(str(FAISS_INDEX_FILE))
    return index


# ============================================
# SEARCH FUNCTIONS
# ============================================

def search_similar_movies(index, query_embedding, k=10):
    """
    Search for k most similar movies using FAISS index.
    
    How it works:
    1. Takes a query embedding (vector representing one movie)
    2. FAISS finds the k nearest neighbors in the index
    3. Returns distances and indices of the k closest movies
    
    Args:
        index: FAISS index
        query_embedding: Vector of shape (embedding_dim,) or (1, embedding_dim)
        k: Number of similar movies to retrieve
    
    Returns:
        tuple: (distances, indices)
            - distances: Array of shape (k,) with similarity scores
            - indices: Array of shape (k,) with movie indices
    """
    # Ensure query is 2D: shape (1, embedding_dim)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Normalize query embedding (same as we did for index)
    query_norm = np.linalg.norm(query_embedding)
    query_normalized = query_embedding / query_norm
    
    # Ensure float32 dtype
    if query_normalized.dtype != np.float32:
        query_normalized = query_normalized.astype(np.float32)
    
    # Search the index
    # Returns: distances (lower = more similar) and indices
    distances, indices = index.search(query_normalized, k)
    
    # Return as 1D arrays
    return distances[0], indices[0]


def search_by_movie_index(index, embeddings, movie_idx, k=10):
    """
    Find similar movies given a movie index.
    
    This is a convenience function that:
    1. Gets the embedding for the movie at movie_idx
    2. Searches for k+1 similar movies (includes the query movie itself)
    3. Excludes the query movie from results
    4. Returns k similar movies
    
    Args:
        index: FAISS index
        embeddings: All movie embeddings
        movie_idx: Index of the query movie
        k: Number of similar movies to return
    
    Returns:
        tuple: (distances, indices) excluding the query movie
    """
    # Get the query embedding
    query_embedding = embeddings[movie_idx]
    
    # Search for k+1 movies (we'll remove the query movie itself)
    distances, indices = search_similar_movies(index, query_embedding, k + 1)
    
    # Remove the query movie from results (it will be the first one with distance ~0)
    # Find where the query movie appears in results
    mask = indices != movie_idx
    distances = distances[mask][:k]
    indices = indices[mask][:k]
    
    return distances, indices


# ============================================
# TESTING FUNCTION
# ============================================

def test_faiss_search(index, embeddings, movies_df, movie_title, k=5):
    """
    Test FAISS index by searching for similar movies.
    
    Args:
        index: FAISS index
        embeddings: Movie embeddings
        movies_df: Movies DataFrame
        movie_title: Title of movie to search for
        k: Number of results
    """
    print(f"\n🔍 Testing FAISS search for: '{movie_title}'")
    
    # Find the movie
    movie_match = movies_df[movies_df['title'].str.contains(movie_title, case=False, na=False)]
    
    if len(movie_match) == 0:
        print(f"   ❌ Movie '{movie_title}' not found")
        return
    
    movie_idx = movie_match.index[0]
    movie_id = movies_df.iloc[movie_idx]['movie_id']
    genres = movies_df.iloc[movie_idx]['genres']
    
    print(f"\n   Query Movie:")
    print(f"   - Title: {movies_df.iloc[movie_idx]['title']}")
    print(f"   - Genres: {genres}")
    print(f"   - Movie ID: {movie_id}")
    print(f"   - Index: {movie_idx}")
    
    # Search using FAISS
    distances, indices = search_by_movie_index(index, embeddings, movie_idx, k)
    
    # Convert distances to similarity scores
    # Since we normalized, L2 distance relates to cosine similarity
    # similarity = 1 - (distance^2 / 2)
    similarities = 1 - (distances ** 2) / 2
    
    print(f"\n   Top {k} similar movies:")
    for i, (idx, dist, sim) in enumerate(zip(indices, distances, similarities), 1):
        title = movies_df.iloc[idx]['title']
        genres = movies_df.iloc[idx]['genres']
        print(f"   {i}. {title}")
        print(f"      Genres: {genres}")
        print(f"      Similarity: {sim:.3f} (Distance: {dist:.4f})")


# ============================================
# FULL PIPELINE
# ============================================

def build_and_save_faiss_index(embeddings):
    """
    Complete pipeline: build and save FAISS index.
    
    This is the main function to call when building the index.
    
    Args:
        embeddings: Movie embeddings array
    
    Returns:
        faiss.Index: Built FAISS index
    """
    # Build index
    index, normalized_embeddings = build_faiss_index(embeddings)
    
    # Save to disk
    save_faiss_index(index)
    
    return index, normalized_embeddings


# ============================================
# TEST THE MODULE
# ============================================

if __name__ == "__main__":
    """
    Test the FAISS index builder.
    Run this file directly to test: python src/build_faiss.py
    """
    from src.build_embeddings import load_embeddings
    
    try:
        print("=" * 60)
        print("TESTING FAISS INDEX BUILDER")
        print("=" * 60)
        
        # Load embeddings
        embeddings, movies_df = load_embeddings()
        
        # Build and save FAISS index
        index, normalized_embeddings = build_and_save_faiss_index(embeddings)
        
        # Test search if we have movies
        if len(movies_df) > 0:
            test_movie = movies_df.iloc[0]['title']
            test_faiss_search(index, normalized_embeddings, movies_df, test_movie, k=5)
        
        print("\n✅ Test complete!")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 TIP: Run build_embeddings.py first to generate embeddings")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
