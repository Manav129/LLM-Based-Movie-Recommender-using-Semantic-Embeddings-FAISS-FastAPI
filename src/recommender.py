"""
Recommender Module
Core recommendation logic combining content-based and collaborative filtering.

This module provides two main recommendation functions:
1. recommend_by_movie() - Content-based: Find similar movies
2. recommend_for_user() - Hybrid: Personalized recommendations based on user history
"""

import numpy as np
import pandas as pd
from src.config import DEFAULT_K, MIN_RATING_THRESHOLD
from src.build_faiss import search_similar_movies, load_faiss_index
from src.build_embeddings import load_embeddings
from src.data_loader import load_ratings


# ============================================
# GLOBAL VARIABLES (LOADED ONCE)
# ============================================

# These will be loaded once when the module is imported
# This avoids loading them every time we make a recommendation
_faiss_index = None
_embeddings = None
_movies_df = None
_ratings_df = None


def initialize_recommender():
    """
    Initialize the recommender by loading all required data.
    
    This function loads:
    - FAISS index
    - Movie embeddings
    - Movie metadata
    - Ratings data
    
    Call this once at startup (e.g., when API server starts).
    
    Raises:
        FileNotFoundError: If artifacts haven't been generated
    """
    global _faiss_index, _embeddings, _movies_df, _ratings_df
    
    _faiss_index = load_faiss_index()
    _embeddings, _movies_df = load_embeddings()
    try:
        _ratings_df = load_ratings()
    except FileNotFoundError:
        _ratings_df = None
    print(f"Recommender initialized: {len(_movies_df)} movies loaded")


def get_loaded_data():
    """
    Get the loaded data (for internal use).
    
    Returns:
        tuple: (faiss_index, embeddings, movies_df, ratings_df)
    """
    if _faiss_index is None:
        raise RuntimeError(
            "Recommender not initialized. Call initialize_recommender() first."
        )
    
    return _faiss_index, _embeddings, _movies_df, _ratings_df


# ============================================
# CONTENT-BASED RECOMMENDATION
# ============================================

def recommend_by_movie(movie_title, k=DEFAULT_K):
    """
    Recommend movies similar to a given movie (content-based).
    
    How it works:
    1. Find the movie in the database by title (fuzzy match)
    2. Get its embedding vector
    3. Search FAISS index for k most similar movies
    4. Return list of similar movies with scores
    
    Args:
        movie_title: Title of the movie (partial match works)
        k: Number of recommendations to return
    
    Returns:
        dict: {
            'query_movie': {...},  # Information about the query movie
            'recommendations': [...]  # List of k similar movies
        }
    
    Raises:
        ValueError: If movie not found
    """
    # Get loaded data
    index, embeddings, movies_df, _ = get_loaded_data()
    
    # Find the movie (case-insensitive partial match)
    movie_matches = movies_df[
        movies_df['title'].str.contains(movie_title, case=False, na=False)
    ]
    
    if len(movie_matches) == 0:
        raise ValueError(
            f"Movie '{movie_title}' not found in database. "
            f"Try a different title or check spelling."
        )
    
    # If multiple matches, take the first one
    if len(movie_matches) > 1:
        print(f"⚠️  Found {len(movie_matches)} matches for '{movie_title}'. Using: {movie_matches.iloc[0]['title']}")
    
    # Get the query movie
    movie_idx = movie_matches.index[0]
    query_movie = movies_df.iloc[movie_idx]
    
    # Get the embedding
    query_embedding = embeddings[movie_idx]
    
    # Search for similar movies using FAISS
    distances, indices = search_similar_movies(index, query_embedding, k + 1)
    
    # Remove the query movie itself from results
    mask = indices != movie_idx
    distances = distances[mask][:k]
    indices = indices[mask][:k]
    
    # Convert distances to similarity scores
    # For normalized vectors: similarity = 1 - (distance^2 / 2)
    similarities = 1 - (distances ** 2) / 2
    
    # Build recommendations list
    recommendations = []
    for idx, sim in zip(indices, similarities):
        movie = movies_df.iloc[idx]
        recommendations.append({
            'movie_id': int(movie['movie_id']),
            'title': movie['title'],
            'genres': movie['genres'],
            'overview': movie['overview'],
            'similarity_score': float(sim)
        })
    
    # Build result
    result = {
        'query_movie': {
            'movie_id': int(query_movie['movie_id']),
            'title': query_movie['title'],
            'genres': query_movie['genres'],
            'overview': query_movie['overview']
        },
        'recommendations': recommendations
    }
    
    return result


# ============================================
# COLLABORATIVE FILTERING HELPERS
# ============================================

def get_user_profile(user_id):
    """
    Build a user profile based on their rating history.
    
    How it works:
    1. Get all movies the user rated highly (>= threshold)
    2. Get embeddings for those movies
    3. Compute average embedding (user's taste profile)
    
    Args:
        user_id: User identifier
    
    Returns:
        tuple: (user_profile_embedding, rated_movie_ids, num_ratings)
    
    Raises:
        ValueError: If user not found or has no ratings
    """
    _, embeddings, movies_df, ratings_df = get_loaded_data()
    
    if ratings_df is None:
        raise ValueError("Ratings data not loaded. User recommendations unavailable.")
    
    # Get user's ratings
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    
    if len(user_ratings) == 0:
        raise ValueError(f"User {user_id} not found or has no ratings.")
    
    # Get highly rated movies (above threshold)
    liked_ratings = user_ratings[user_ratings['rating'] >= MIN_RATING_THRESHOLD]
    
    if len(liked_ratings) == 0:
        raise ValueError(
            f"User {user_id} has no ratings above {MIN_RATING_THRESHOLD}. "
            f"Cannot generate recommendations."
        )
    
    # Get movie IDs and find their indices in movies_df
    liked_movie_ids = set(liked_ratings['movie_id'].values)
    
    # Find indices in movies_df
    movie_indices = []
    for movie_id in liked_movie_ids:
        matches = movies_df[movies_df['movie_id'] == movie_id]
        if len(matches) > 0:
            movie_indices.append(matches.index[0])
    
    if len(movie_indices) == 0:
        raise ValueError(f"None of user {user_id}'s liked movies found in movie database.")
    
    # Get embeddings for liked movies
    liked_embeddings = embeddings[movie_indices]
    
    # Compute user profile as average of liked movie embeddings
    # This represents the user's general taste
    user_profile = np.mean(liked_embeddings, axis=0)
    
    # Get all rated movie IDs (to exclude from recommendations)
    rated_movie_ids = set(user_ratings['movie_id'].values)
    
    return user_profile, rated_movie_ids, len(user_ratings)


# ============================================
# HYBRID RECOMMENDATION
# ============================================

def recommend_for_user(user_id, k=DEFAULT_K):
    """
    Recommend movies for a specific user (hybrid approach).
    
    How it works (Collaborative + Content-based):
    1. Get user's rating history
    2. Build user profile (average embedding of highly-rated movies)
    3. Search FAISS for movies similar to user's taste
    4. Exclude movies the user has already rated
    5. Return top k recommendations
    
    Args:
        user_id: User identifier
        k: Number of recommendations to return
    
    Returns:
        dict: {
            'user_id': int,
            'num_ratings': int,  # Total ratings by user
            'num_liked': int,    # Number of highly-rated movies
            'recommendations': [...]  # List of k recommended movies
        }
    
    Raises:
        ValueError: If user not found or has insufficient data
    """
    index, embeddings, movies_df, ratings_df = get_loaded_data()
    
    # Build user profile
    user_profile, rated_movie_ids, num_ratings = get_user_profile(user_id)
    
    # Search for movies similar to user's profile
    # We search for more than k to account for filtering
    search_k = min(k * 3, len(movies_df))  # Search 3x more, but not more than total movies
    distances, indices = search_similar_movies(index, user_profile, search_k)
    
    # Convert distances to similarity scores
    similarities = 1 - (distances ** 2) / 2
    
    # Filter out already rated movies
    recommendations = []
    for idx, sim in zip(indices, similarities):
        movie = movies_df.iloc[idx]
        movie_id = movie['movie_id']
        
        # Skip if user already rated this movie
        if movie_id in rated_movie_ids:
            continue
        
        recommendations.append({
            'movie_id': int(movie_id),
            'title': movie['title'],
            'genres': movie['genres'],
            'overview': movie['overview'],
            'similarity_score': float(sim)
        })
        
        # Stop once we have k recommendations
        if len(recommendations) >= k:
            break
    
    # If we don't have enough recommendations, warn user
    if len(recommendations) < k:
        print(f"⚠️  Only {len(recommendations)} new movies to recommend (user has rated most movies)")
    
    # Build result
    result = {
        'user_id': int(user_id),
        'num_ratings': num_ratings,
        'num_liked': len([r for r in rated_movie_ids if r]),  # Approximate
        'recommendations': recommendations
    }
    
    return result


# ============================================
# TASTE PROFILE RECOMMENDATION
# ============================================

def recommend_by_taste_profile(ratings_dict, k=DEFAULT_K):
    """
    Recommend movies based on a temporary taste profile.
    
    This allows users to rate movies on-the-spot and get instant
    personalized recommendations without needing a user account.
    
    Args:
        ratings_dict: Dictionary {movie_id: rating} of user's ratings
        k: Number of recommendations to return
    
    Returns:
        dict: Recommendations based on taste profile
    
    Example:
        ratings = {1: 5.0, 260: 4.5, 318: 5.0}
        result = recommend_by_taste_profile(ratings, k=10)
    """
    index, embeddings, movies_df, _ = get_loaded_data()
    
    if len(ratings_dict) < 3:
        raise ValueError("Please rate at least 3 movies to get recommendations.")
    
    # Filter highly-rated movies (>= 3.5)
    liked_movie_ids = [mid for mid, rating in ratings_dict.items() if rating >= 3.5]
    
    if len(liked_movie_ids) < 2:
        raise ValueError("Please rate at least 2 movies with 3.5 stars or higher.")
    
    # Find indices in movies_df
    movie_indices = []
    for movie_id in liked_movie_ids:
        matches = movies_df[movies_df['movie_id'] == movie_id]
        if len(matches) > 0:
            movie_indices.append(matches.index[0])
    
    if len(movie_indices) == 0:
        raise ValueError("None of the rated movies found in database.")
    
    # Get embeddings for liked movies
    liked_embeddings = embeddings[movie_indices]
    
    # Compute user profile as average of liked movie embeddings
    user_profile = np.mean(liked_embeddings, axis=0)
    
    # Search FAISS for similar movies
    distances, indices = search_similar_movies(index, user_profile, k * 3)
    
    # Convert distances to similarities
    similarities = 1 - (distances ** 2) / 2
    
    # Exclude already-rated movies
    rated_ids = set(ratings_dict.keys())
    recommendations = []
    
    for idx, sim in zip(indices, similarities):
        movie = movies_df.iloc[idx]
        movie_id = int(movie['movie_id'])
        
        # Skip if already rated
        if movie_id in rated_ids:
            continue
        
        recommendations.append({
            'movie_id': movie_id,
            'title': movie['title'],
            'genres': movie['genres'],
            'overview': movie['overview'],
            'similarity_score': float(sim)
        })
        
        if len(recommendations) >= k:
            break
    
    result = {
        'recommendations': recommendations
    }
    
    return result


# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_movie_by_id(movie_id):
    """
    Get movie details by movie_id.
    
    Args:
        movie_id: Movie identifier
    
    Returns:
        dict: Movie details
    
    Raises:
        ValueError: If movie not found
    """
    _, _, movies_df, _ = get_loaded_data()
    
    movie_match = movies_df[movies_df['movie_id'] == movie_id]
    
    if len(movie_match) == 0:
        raise ValueError(f"Movie with ID {movie_id} not found.")
    
    movie = movie_match.iloc[0]
    return {
        'movie_id': int(movie['movie_id']),
        'title': movie['title'],
        'genres': movie['genres'],
        'overview': movie['overview']
    }


def search_movies(query, limit=10):
    """
    Search for movies by title (for autocomplete/search features).
    
    Args:
        query: Search query string
        limit: Maximum number of results
    
    Returns:
        list: List of matching movies
    """
    _, _, movies_df, _ = get_loaded_data()
    
    # Case-insensitive partial match
    matches = movies_df[
        movies_df['title'].str.contains(query, case=False, na=False)
    ].head(limit)
    
    results = []
    for _, movie in matches.iterrows():
        results.append({
            'movie_id': int(movie['movie_id']),
            'title': movie['title'],
            'genres': movie['genres']
        })
    
    return results


# ============================================
# TEST THE MODULE
# ============================================

if __name__ == "__main__":
    """
    Test the recommender module.
    Run this file directly to test: python src/recommender.py
    """
    try:
        print("=" * 70)
        print("TESTING RECOMMENDER MODULE")
        print("=" * 70)
        
        # Initialize
        initialize_recommender()
        
        # Test 1: Content-based recommendation
        print("\n" + "=" * 70)
        print("TEST 1: Content-Based Recommendations")
        print("=" * 70)
        
        _, _, movies_df, _ = get_loaded_data()
        test_movie = movies_df.iloc[0]['title']
        
        print(f"\nSearching for movies similar to: '{test_movie}'")
        result = recommend_by_movie(test_movie, k=3)
        
        print(f"\nQuery Movie: {result['query_movie']['title']}")
        print(f"Genres: {result['query_movie']['genres']}")
        print(f"\nRecommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"\n{i}. {rec['title']}")
            print(f"   Genres: {rec['genres']}")
            print(f"   Similarity: {rec['similarity_score']:.3f}")
        
        # Test 2: User-based recommendation
        print("\n\n" + "=" * 70)
        print("TEST 2: User-Based Recommendations")
        print("=" * 70)
        
        _, _, _, ratings_df = get_loaded_data()
        
        if ratings_df is not None and len(ratings_df) > 0:
            test_user = ratings_df['user_id'].iloc[0]
            
            print(f"\nGenerating recommendations for User {test_user}")
            result = recommend_for_user(test_user, k=3)
            
            print(f"\nUser ID: {result['user_id']}")
            print(f"Total ratings: {result['num_ratings']}")
            print(f"\nRecommendations:")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"\n{i}. {rec['title']}")
                print(f"   Genres: {rec['genres']}")
                print(f"   Match Score: {rec['similarity_score']:.3f}")
        else:
            print("\n⚠️  Ratings data not available. Skipping user-based test.")
        
        # Test 3: Movie search
        print("\n\n" + "=" * 70)
        print("TEST 3: Movie Search")
        print("=" * 70)
        
        search_query = movies_df.iloc[0]['title'].split()[0]  # First word of first movie
        print(f"\nSearching for: '{search_query}'")
        
        results = search_movies(search_query, limit=5)
        print(f"\nFound {len(results)} movies:")
        for movie in results:
            print(f"  - {movie['title']} ({movie['genres']})")
        
        print("\n" + "=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 TIP: Run the pipeline first: python scripts/run_pipeline.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
