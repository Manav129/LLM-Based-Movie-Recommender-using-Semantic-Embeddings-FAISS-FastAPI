"""
Data Loader Module
Handles loading and preprocessing of movies and ratings data.
"""

import pandas as pd
from pathlib import Path
from src.config import MOVIES_CSV, RATINGS_CSV


# ============================================
# LOAD MOVIES DATA
# ============================================

def load_movies():
    """
    Load and clean the movies dataset.
    
    Required columns in movies.csv:
    - movie_id: Unique identifier for each movie
    - title: Movie title
    - genres: Pipe-separated genres (e.g., "Action|Thriller")
    - overview: Movie description/plot summary
    
    Returns:
        pd.DataFrame: Cleaned movies dataframe
        
    Raises:
        FileNotFoundError: If movies.csv doesn't exist
        ValueError: If required columns are missing
    """
    # Check if file exists
    if not MOVIES_CSV.exists():
        raise FileNotFoundError(
            f"movies.csv not found at: {MOVIES_CSV}\n"
            f"Please place your movies.csv file in the data/ folder."
        )
    
    print(f"📂 Loading movies from: {MOVIES_CSV}")
    
    # Load CSV file into pandas DataFrame
    movies_df = pd.read_csv(MOVIES_CSV)
    
    print(f"   Loaded {len(movies_df)} movies")
    
    # Check for required columns
    required_columns = ['movie_id', 'title', 'genres', 'overview']
    missing_columns = [col for col in required_columns if col not in movies_df.columns]
    
    if missing_columns:
        raise ValueError(
            f"Missing required columns in movies.csv: {missing_columns}\n"
            f"Found columns: {list(movies_df.columns)}"
        )
    
    # Clean the data
    print("   Cleaning movies data...")
    
    # 1. Remove duplicates based on movie_id
    movies_df = movies_df.drop_duplicates(subset=['movie_id'], keep='first')
    
    # 2. Remove rows with missing critical fields
    # We need title and overview to create embeddings
    movies_df = movies_df.dropna(subset=['title', 'overview'])
    
    # 3. Fill missing genres with empty string
    movies_df['genres'] = movies_df['genres'].fillna('')
    
    # 4. Convert movie_id to integer (in case it's stored as float)
    movies_df['movie_id'] = movies_df['movie_id'].astype(int)
    
    # 5. Strip whitespace from text columns
    movies_df['title'] = movies_df['title'].str.strip()
    movies_df['overview'] = movies_df['overview'].str.strip()
    movies_df['genres'] = movies_df['genres'].str.strip()
    
    # 6. Remove rows with empty title or overview after stripping
    movies_df = movies_df[
        (movies_df['title'].str.len() > 0) & 
        (movies_df['overview'].str.len() > 0)
    ]
    
    # 7. Reset index to have clean sequential indices
    movies_df = movies_df.reset_index(drop=True)
    
    print(f"   ✅ Cleaned: {len(movies_df)} movies remaining")
    
    return movies_df


# ============================================
# LOAD RATINGS DATA
# ============================================

def load_ratings():
    """
    Load and clean the ratings dataset.
    
    Required columns in ratings.csv:
    - user_id: User identifier
    - movie_id: Movie identifier
    - rating: Rating value (e.g., 1.0 to 5.0)
    
    Returns:
        pd.DataFrame: Cleaned ratings dataframe
        
    Raises:
        FileNotFoundError: If ratings.csv doesn't exist
        ValueError: If required columns are missing
    """
    # Check if file exists
    if not RATINGS_CSV.exists():
        raise FileNotFoundError(
            f"ratings.csv not found at: {RATINGS_CSV}\n"
            f"Please place your ratings.csv file in the data/ folder."
        )
    
    print(f"📂 Loading ratings from: {RATINGS_CSV}")
    
    # Load CSV file
    ratings_df = pd.read_csv(RATINGS_CSV)
    
    print(f"   Loaded {len(ratings_df)} ratings")
    
    # Check for required columns
    required_columns = ['user_id', 'movie_id', 'rating']
    missing_columns = [col for col in required_columns if col not in ratings_df.columns]
    
    if missing_columns:
        raise ValueError(
            f"Missing required columns in ratings.csv: {missing_columns}\n"
            f"Found columns: {list(ratings_df.columns)}"
        )
    
    # Clean the data
    print("   Cleaning ratings data...")
    
    # 1. Remove duplicates (same user rating same movie multiple times)
    # Keep the last rating
    ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'movie_id'], keep='last')
    
    # 2. Remove rows with missing values
    ratings_df = ratings_df.dropna(subset=['user_id', 'movie_id', 'rating'])
    
    # 3. Convert to appropriate data types
    ratings_df['user_id'] = ratings_df['user_id'].astype(int)
    ratings_df['movie_id'] = ratings_df['movie_id'].astype(int)
    ratings_df['rating'] = ratings_df['rating'].astype(float)
    
    # 4. Filter out invalid ratings (assuming rating scale is 0.5 to 5.0)
    ratings_df = ratings_df[
        (ratings_df['rating'] >= 0.5) & 
        (ratings_df['rating'] <= 5.0)
    ]
    
    # 5. Reset index
    ratings_df = ratings_df.reset_index(drop=True)
    
    print(f"   ✅ Cleaned: {len(ratings_df)} ratings remaining")
    print(f"   Users: {ratings_df['user_id'].nunique()}")
    print(f"   Movies rated: {ratings_df['movie_id'].nunique()}")
    
    return ratings_df


# ============================================
# COMBINED LOADER
# ============================================

def load_all_data():
    """
    Load both movies and ratings data.
    
    This is a convenience function that loads both datasets
    and ensures they are compatible (i.e., ratings reference valid movies).
    
    Returns:
        tuple: (movies_df, ratings_df)
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    # Load both datasets
    movies_df = load_movies()
    ratings_df = load_ratings()
    
    # Filter ratings to only include movies that exist in movies_df
    print("\n🔗 Filtering ratings to match available movies...")
    valid_movie_ids = set(movies_df['movie_id'].values)
    original_count = len(ratings_df)
    
    ratings_df = ratings_df[ratings_df['movie_id'].isin(valid_movie_ids)]
    filtered_count = original_count - len(ratings_df)
    
    if filtered_count > 0:
        print(f"   Removed {filtered_count} ratings for movies not in dataset")
    
    print(f"   ✅ Final ratings count: {len(ratings_df)}")
    
    print("\n" + "=" * 60)
    print("DATA LOADING COMPLETE")
    print("=" * 60)
    print(f"Movies: {len(movies_df)}")
    print(f"Ratings: {len(ratings_df)}")
    print(f"Users: {ratings_df['user_id'].nunique()}")
    print("=" * 60)
    
    return movies_df, ratings_df


# ============================================
# DATA STATISTICS (HELPER FUNCTION)
# ============================================

def print_data_statistics(movies_df, ratings_df):
    """
    Print detailed statistics about the loaded data.
    Useful for understanding your dataset.
    
    Args:
        movies_df: Movies dataframe
        ratings_df: Ratings dataframe
    """
    print("\n" + "=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)
    
    # Movies statistics
    print("\n📽️  MOVIES:")
    print(f"   Total movies: {len(movies_df)}")
    print(f"   Genres present: {movies_df['genres'].str.split('|').explode().nunique()}")
    print(f"   Average overview length: {movies_df['overview'].str.len().mean():.0f} characters")
    
    # Ratings statistics
    print("\n⭐ RATINGS:")
    print(f"   Total ratings: {len(ratings_df)}")
    print(f"   Unique users: {ratings_df['user_id'].nunique()}")
    print(f"   Unique movies rated: {ratings_df['movie_id'].nunique()}")
    print(f"   Average rating: {ratings_df['rating'].mean():.2f}")
    print(f"   Rating distribution:")
    print(f"      Min: {ratings_df['rating'].min()}")
    print(f"      Max: {ratings_df['rating'].max()}")
    print(f"      Median: {ratings_df['rating'].median()}")
    
    # User activity
    ratings_per_user = ratings_df.groupby('user_id').size()
    print(f"\n👤 USER ACTIVITY:")
    print(f"   Average ratings per user: {ratings_per_user.mean():.1f}")
    print(f"   Most active user rated: {ratings_per_user.max()} movies")
    print(f"   Least active user rated: {ratings_per_user.min()} movies")
    
    print("=" * 60)


# ============================================
# TEST THE MODULE
# ============================================

if __name__ == "__main__":
    """
    Test the data loader by loading and displaying sample data.
    Run this file directly to test: python src/data_loader.py
    """
    try:
        # Load all data
        movies_df, ratings_df = load_all_data()
        
        # Print statistics
        print_data_statistics(movies_df, ratings_df)
        
        # Show sample data
        print("\n" + "=" * 60)
        print("SAMPLE DATA")
        print("=" * 60)
        print("\n📽️  First 3 movies:")
        print(movies_df[['movie_id', 'title', 'genres']].head(3))
        
        print("\n⭐ First 5 ratings:")
        print(ratings_df.head(5))
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 TIP: Make sure you have movies.csv and ratings.csv in the data/ folder")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
