"""
Embeddings Builder Module
Generates vector embeddings for movies using SentenceTransformers.

What are embeddings?
- Embeddings are numerical representations of text
- Similar movies have similar embeddings (closer together in vector space)
- The model converts text like "Action adventure movie" into a list of 384 numbers
- These numbers capture the MEANING of the text
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
from src.config import (
    EMBEDDING_MODEL_NAME, 
    EMBEDDING_DIM,
    EMBEDDINGS_FILE,
    MOVIE_METADATA_FILE
)


# ============================================
# PREPARE TEXT FOR EMBEDDING
# ============================================

def prepare_movie_text(movies_df):
    """
    Combine movie information into a single text string for each movie.
    
    We combine: title + genres + overview
    This gives the model more context to understand what the movie is about.
    
    Example:
    Title: "The Dark Knight"
    Genres: "Action|Crime|Drama"
    Overview: "Batman must face the Joker..."
    
    Combined text: "The Dark Knight. Genres: Action, Crime, Drama. Batman must face the Joker..."
    
    Args:
        movies_df: DataFrame with columns [movie_id, title, genres, overview]
    
    Returns:
        list: List of text strings, one per movie
    """
    movie_texts = []
    for idx, row in movies_df.iterrows():
        title = row['title']
        genres = row['genres'].replace('|', ', ')
        overview = row['overview']
        if genres:
            combined_text = f"{title}. Genres: {genres}. {overview}"
        else:
            combined_text = f"{title}. {overview}"
        movie_texts.append(combined_text)
    return movie_texts


# ============================================
# GENERATE EMBEDDINGS
# ============================================

def generate_embeddings(movies_df):
    """
    Generate vector embeddings for all movies using SentenceTransformers.
    
    This function:
    1. Loads the pre-trained language model
    2. Prepares movie text (title + genres + overview)
    3. Converts text to embeddings (vectors of numbers)
    4. Returns embeddings as a numpy array
    
    Args:
        movies_df: DataFrame with movie information
    
    Returns:
        np.ndarray: Array of shape (num_movies, embedding_dim)
                   Each row is a vector representing one movie
    """
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    movie_texts = prepare_movie_text(movies_df)
    embeddings = model.encode(
        movie_texts,
        show_progress_bar=False,
        batch_size=32,
        convert_to_numpy=True
    )
    return embeddings


# ============================================
# SAVE EMBEDDINGS
# ============================================

def save_embeddings(embeddings, movies_df):
    """
    Save embeddings and movie metadata to disk.
    
    We save:
    1. Embeddings as a .npy file (NumPy binary format - fast and efficient)
    2. Movie metadata as a .pkl file (Pickle format - preserves DataFrame structure)
    
    Args:
        embeddings: Numpy array of embeddings
        movies_df: DataFrame with movie information
    """
    np.save(EMBEDDINGS_FILE, embeddings)
    movies_df.to_pickle(MOVIE_METADATA_FILE)


# ============================================
# LOAD EMBEDDINGS (FOR LATER USE)
# ============================================

def load_embeddings():
    """
    Load previously saved embeddings and metadata from disk.
    
    This is used by the recommendation system to load the pre-computed embeddings
    instead of generating them every time (which would be slow).
    
    Returns:
        tuple: (embeddings array, movies DataFrame)
    
    Raises:
        FileNotFoundError: If embeddings haven't been generated yet
    """
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")
    if not MOVIE_METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {MOVIE_METADATA_FILE}")
    embeddings = np.load(EMBEDDINGS_FILE)
    movies_df = pd.read_pickle(MOVIE_METADATA_FILE)
    return embeddings, movies_df


# ============================================
# FULL PIPELINE
# ============================================

def build_and_save_embeddings(movies_df):
    """
    Complete pipeline: generate and save embeddings.
    
    This is the main function to call when building embeddings.
    
    Args:
        movies_df: DataFrame with movie information
    
    Returns:
        np.ndarray: Generated embeddings
    """
    # Generate embeddings
    embeddings = generate_embeddings(movies_df)
    
    # Save to disk
    save_embeddings(embeddings, movies_df)
    
    return embeddings


# ============================================
# SIMILARITY TESTING (HELPER FUNCTION)
# ============================================

def test_similarity(embeddings, movies_df, movie_title, top_k=5):
    """
    Test the embeddings by finding similar movies.
    
    This is a simple similarity test using cosine similarity.
    (Later we'll use FAISS for faster search)
    
    Args:
        embeddings: Movie embeddings array
        movies_df: Movies DataFrame
        movie_title: Title of the movie to find similar movies for
        top_k: Number of similar movies to return
    """
    print(f"\n🔍 Finding movies similar to: '{movie_title}'")
    
    # Find the movie in the dataframe
    movie_match = movies_df[movies_df['title'].str.contains(movie_title, case=False, na=False)]
    
    if len(movie_match) == 0:
        print(f"   ❌ Movie '{movie_title}' not found")
        return
    
    # Get the index and embedding of the query movie
    movie_idx = movie_match.index[0]
    query_embedding = embeddings[movie_idx]
    
    # Calculate cosine similarity between query and all movies
    # Cosine similarity = dot product of normalized vectors
    # Values close to 1 = very similar, close to 0 = different
    
    # Normalize embeddings (convert to unit vectors)
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_normalized = query_embedding / np.linalg.norm(query_embedding)
    
    # Calculate similarity scores
    similarities = np.dot(normalized_embeddings, query_normalized)
    
    # Get top-k most similar movies (excluding the query movie itself)
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]  # Skip index 0 (itself)
    
    # Display results
    print(f"\n   Top {top_k} similar movies:")
    for i, idx in enumerate(top_indices, 1):
        title = movies_df.iloc[idx]['title']
        score = similarities[idx]
        genres = movies_df.iloc[idx]['genres']
        print(f"   {i}. {title} (Score: {score:.3f}, Genres: {genres})")


# ============================================
# TEST THE MODULE
# ============================================

if __name__ == "__main__":
    """
    Test the embeddings builder.
    Run this file directly to test: python src/build_embeddings.py
    """
    from src.data_loader import load_movies
    
    try:
        # Load movies
        print("=" * 60)
        print("TESTING EMBEDDINGS BUILDER")
        print("=" * 60)
        
        movies_df = load_movies()
        
        # Generate and save embeddings
        embeddings = build_and_save_embeddings(movies_df)
        
        # Test similarity if we have movies
        if len(movies_df) > 0:
            # Test with the first movie in the dataset
            test_movie = movies_df.iloc[0]['title']
            test_similarity(embeddings, movies_df, test_movie, top_k=5)
        
        print("\n✅ Test complete!")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
